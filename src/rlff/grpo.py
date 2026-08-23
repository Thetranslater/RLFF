"""Role-level GRPO grouping, advantages, masks, and training-batch conversion.

SUBAGENT BRIEF — PHASE D (role GRPO and training assembly)
===========================================================
Read ``src/rlff/IMPLEMENTATION.md`` and reuse the established contracts. Build
subgroups keyed by `(group_id, character)`, combine each completion's local
reward with its trajectory-global contribution, normalize across trajectories,
and broadcast the resulting advantage only onto that completion's generated
tokens. Context/prompt/other-character tokens must have zero loss mask.

Implement numerical edge cases explicitly: minimum subgroup size, zero
variance, incomplete groups, invalid trajectories, multiple completions by the
same character, and deterministic ordering. Convert validated results into the
minimal AReaL batch representation without retokenizing text. Unit-test the
math on CPU against hand-calculated examples.

Do not implement rollout, RM calls, a full optimizer loop, or an alternative
distributed trainer here. Phase D owns this file together with `runtime.py` and
`cli.py`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

from .contracts import (
    CompletionTrace,
    RoleAdvantage,
    RoleEffectiveReward,
    RolloutGroup,
    TerminationReason,
    Trajectory,
)


class RoleGRPOError(ValueError):
    """Raised when a rollout group cannot produce a safe role-level batch."""


_DEFAULT_REWARD_STD_EPSILON: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class RoleGRPOResult:
    """The CPU result of role-level normalization and token-mask conversion.

    ``advantages`` and ``samples`` are ordered by the source trajectory order,
    then by completion order.  ``samples`` are plain dictionaries so an
    AReaL adapter can turn them into its native tensors without this module
    importing torch or fabricating a tokenizer boundary.
    """

    advantages: tuple[RoleAdvantage, ...]
    samples: tuple[dict[str, Any], ...]
    dropped_trajectory_ids: tuple[str, ...] = ()

    def as_batch(self) -> tuple[dict[str, Any], ...]:
        """Return the framework-neutral batch records."""

        return self.samples


TrainingSample: TypeAlias = dict[str, Any]


def _is_complete_trajectory(trajectory: Trajectory) -> bool:
    """Return whether a trajectory is a successful fixed-horizon rollout."""

    return (
        trajectory.valid
        and not trajectory.truncated
        and trajectory.termination_reason is TerminationReason.MAX_ROUNDS
        and trajectory.completed_rounds == trajectory.planned_rounds
    )


def _eligible_trajectories(
    group: RolloutGroup,
    *,
    drop_incomplete: bool,
) -> tuple[tuple[Trajectory, ...], tuple[str, ...]]:
    complete: list[Trajectory] = []
    dropped: list[str] = []
    for trajectory in group.trajectories:
        if _is_complete_trajectory(trajectory):
            complete.append(trajectory)
        else:
            if not drop_incomplete:
                raise RoleGRPOError(
                    f"trajectory {trajectory.trajectory_id!r} is invalid or incomplete"
                )
            dropped.append(trajectory.trajectory_id)
    # GRPO's normalization denominator is the fixed sampling group. Keeping
    # only the valid subset would silently change that group, so one failed
    # trajectory invalidates the whole group when dropping is enabled.
    if dropped:
        return (), tuple(item.trajectory_id for item in group.trajectories)
    return tuple(complete), ()


def _finite_float(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise RoleGRPOError(f"{name} must be finite")
    return float(value)


def _reward_map(
    group: RolloutGroup,
    effective_rewards: Iterable[RoleEffectiveReward],
) -> dict[tuple[str, str], RoleEffectiveReward]:
    """Validate rewards against exact rollout completions and index them."""

    trajectories = {item.trajectory_id: item for item in group.trajectories}
    completions: dict[str, dict[str, CompletionTrace]] = {}
    for trajectory in group.trajectories:
        by_id: dict[str, CompletionTrace] = {}
        for completion in trajectory.completions:
            if completion.completion_id in by_id:
                raise RoleGRPOError(
                    f"trajectory {trajectory.trajectory_id!r} has duplicate completion ID"
                )
            by_id[completion.completion_id] = completion
        completions[trajectory.trajectory_id] = by_id

    result: dict[tuple[str, str], RoleEffectiveReward] = {}
    for reward in effective_rewards:
        if reward.group_id != group.group_id:
            raise RoleGRPOError(
                f"role reward {reward.trajectory_id!r}/{reward.character!r} belongs "
                f"to group {reward.group_id!r}, expected {group.group_id!r}"
            )
        matched_trajectory = trajectories.get(reward.trajectory_id)
        if matched_trajectory is None:
            raise RoleGRPOError(
                f"role reward references unknown trajectory {reward.trajectory_id!r}"
            )
        key = (reward.trajectory_id, reward.character)
        if key in result:
            raise RoleGRPOError(f"duplicate role reward for {key!r}")
        _finite_float(reward.effective_reward, name="effective reward")
        trace_map = completions[reward.trajectory_id]
        for completion_id in reward.completion_ids:
            matched_completion = trace_map.get(completion_id)
            if matched_completion is None:
                raise RoleGRPOError(
                    f"role reward {key!r} references unknown completion {completion_id!r}"
                )
            if matched_completion.character != reward.character:
                raise RoleGRPOError(
                    f"role reward {key!r} references completion for "
                    f"{matched_completion.character!r}"
                )
        result[key] = reward
    return result


def _validate_role_reward_coverage(
    eligible: Sequence[Trajectory],
    reward_map: Mapping[tuple[str, str], RoleEffectiveReward],
    *,
    drop_incomplete: bool,
) -> tuple[str, ...]:
    """Require every eligible trajectory/role to have an exact reward span."""

    if not eligible:
        return ()
    first = eligible[0]
    expected_characters = tuple(
        dict.fromkeys(completion.character for completion in first.completions)
    )
    if not expected_characters:
        message = "eligible trajectory has no character completions"
        if drop_incomplete:
            return ()
        raise RoleGRPOError(message)
    expected_set = set(expected_characters)
    problems: list[str] = []
    for trajectory in eligible:
        for completion in trajectory.completions:
            if not completion.completion_token_ids:
                problems.append(
                    f"trajectory {trajectory.trajectory_id!r} has zero-token completion "
                    f"{completion.completion_id!r}"
                )
        actual_characters = tuple(
            dict.fromkeys(completion.character for completion in trajectory.completions)
        )
        if set(actual_characters) != expected_set:
            problems.append(
                f"trajectory {trajectory.trajectory_id!r} role set {actual_characters!r} "
                f"does not match {expected_characters!r}"
            )
        for character in expected_characters:
            key = (trajectory.trajectory_id, character)
            reward = reward_map.get(key)
            actual_ids = tuple(
                completion.completion_id
                for completion in trajectory.completions
                if completion.character == character
            )
            if reward is None:
                problems.append(f"missing role reward for {key!r}")
            elif reward.completion_ids != actual_ids:
                problems.append(
                    f"role reward {key!r} completion_ids {reward.completion_ids!r} "
                    f"do not exactly cover {actual_ids!r}"
                )
        # A reward for a character that did not occur in the trajectory would
        # otherwise be silently ignored by the subgroup construction.
        for reward_trajectory, character in reward_map:
            if reward_trajectory == trajectory.trajectory_id and character not in expected_set:
                problems.append(f"unexpected role reward for {(reward_trajectory, character)!r}")
    if problems:
        message = "incomplete role reward coverage: " + "; ".join(problems)
        if drop_incomplete:
            return ()
        raise RoleGRPOError(message)
    return expected_characters


def compute_role_advantages(
    group: RolloutGroup,
    effective_rewards: Sequence[RoleEffectiveReward],
    *,
    min_group_size: int = 2,
    reward_std_epsilon: float = _DEFAULT_REWARD_STD_EPSILON,
    drop_incomplete_trajectory: bool = True,
) -> tuple[RoleAdvantage, ...]:
    """Normalize effective rewards independently for every ``(group, role)``.

    The standard deviation is the population standard deviation.  A subgroup
    with fewer than ``min_group_size`` eligible trajectories is omitted.  A
    zero-variance subgroup is valid but receives an all-zero advantage; it is
    not divided by a fabricated epsilon.  Rewards are never mixed between
    characters or between groups.
    """

    if type(min_group_size) is not int or min_group_size <= 0:
        raise ValueError("min_group_size must be a positive integer")
    epsilon = _finite_float(reward_std_epsilon, name="reward_std_epsilon")
    if epsilon < 0:
        raise ValueError("reward_std_epsilon must be non-negative")
    reward_map = _reward_map(group, effective_rewards)
    eligible, _ = _eligible_trajectories(
        group,
        drop_incomplete=drop_incomplete_trajectory,
    )
    characters = _validate_role_reward_coverage(
        eligible,
        reward_map,
        drop_incomplete=drop_incomplete_trajectory,
    )
    if not characters:
        return ()
    grouped: dict[str, list[RoleEffectiveReward]] = {}
    for character in characters:
        grouped[character] = [
            reward_map[(trajectory.trajectory_id, character)] for trajectory in eligible
        ]

    result: list[RoleAdvantage] = []
    for character in characters:
        records = grouped[character]
        if len(records) < min_group_size:
            continue
        values = [_finite_float(item.effective_reward, name="effective reward") for item in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        # Epsilon is a numerical guard, not a denominator that changes the
        # scale of a genuinely constant reward subgroup.
        constant = std <= epsilon
        for reward, value in zip(records, values, strict=True):
            advantage = 0.0 if constant else (value - mean) / std
            result.append(
                RoleAdvantage(
                    group_id=reward.group_id,
                    trajectory_id=reward.trajectory_id,
                    character=character,
                    completion_ids=reward.completion_ids,
                    effective_reward=value,
                    advantage=advantage,
                    valid_group_size=len(records),
                    reward_mean=mean,
                    reward_std=std,
                )
            )
    # Keep the role records in the same trajectory/character order as the
    # rollouts, rather than in the input order of reward records.
    by_key = {(item.trajectory_id, item.character): item for item in result}
    ordered_keys = [
        (trajectory.trajectory_id, character)
        for trajectory in eligible
        for character in dict.fromkeys(item.character for item in trajectory.completions)
        if (trajectory.trajectory_id, character) in by_key
    ]
    return tuple(by_key[key] for key in ordered_keys)


def _advantage_by_completion(
    advantages: Sequence[RoleAdvantage],
) -> dict[str, RoleAdvantage]:
    result: dict[str, RoleAdvantage] = {}
    for role_advantage in advantages:
        for completion_id in role_advantage.completion_ids:
            if completion_id in result:
                raise RoleGRPOError(f"completion {completion_id!r} has multiple advantages")
            result[completion_id] = role_advantage
    return result


def completion_to_training_sample(
    completion: CompletionTrace,
    role_advantage: RoleAdvantage,
) -> TrainingSample:
    """Convert one exact completion to a framework-neutral RLFF training record.

    The prompt IDs and generated IDs are concatenated as already supplied by
    the rollout engine. No text is tokenized here. ``rollout_logprobs`` stays
    completion-aligned (there are no fabricated prompt logprobs), while
    ``loss_mask`` and ``advantages`` are full-sequence aligned. The result is
    deliberately RLFF/framework-neutral: scalar reward and policy version are
    metadata; a runtime adapter performs any pinned AReaL-native conversion.
    """

    if completion.group_id != role_advantage.group_id:
        raise RoleGRPOError("completion group differs from role advantage")
    if completion.trajectory_id != role_advantage.trajectory_id:
        raise RoleGRPOError("completion trajectory differs from role advantage")
    if completion.completion_id not in role_advantage.completion_ids:
        raise RoleGRPOError("role advantage does not contain this completion")
    if completion.character != role_advantage.character:
        raise RoleGRPOError("completion character differs from role advantage")
    prompt_ids = tuple(completion.prompt_token_ids)
    generated_ids = tuple(completion.completion_token_ids)
    input_ids = prompt_ids + generated_ids
    prompt_length = len(prompt_ids)
    completion_length = len(generated_ids)
    if completion_length == 0:
        raise RoleGRPOError("zero-token completions cannot be used for training")
    advantage = float(role_advantage.advantage)
    full_loss_mask = (0.0,) * prompt_length + (1.0,) * completion_length
    full_advantages = (0.0,) * prompt_length + (advantage,) * completion_length
    full_logprobs = (0.0,) * prompt_length + tuple(completion.rollout_logprobs)
    return {
        "input_ids": input_ids,
        "attention_mask": (1,) * len(input_ids),
        "loss_mask": full_loss_mask,
        "logprobs": full_logprobs,
        "advantages": full_advantages,
        "rollout_logprobs": tuple(completion.rollout_logprobs),
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": generated_ids,
        "completion_start": completion.completion_start,
        "completion_end": completion.completion_end,
        "completion_id": completion.completion_id,
        "trajectory_id": completion.trajectory_id,
        "group_id": completion.group_id,
        "character": completion.character,
        "turn_index": completion.turn_index,
        "advantage": advantage,
        "effective_reward": float(role_advantage.effective_reward),
        "policy_version": completion.policy_version,
        "tokenizer_fingerprint": completion.tokenizer_fingerprint,
    }


def build_role_grpo_batch(
    group: RolloutGroup,
    effective_rewards: Sequence[RoleEffectiveReward],
    *,
    min_group_size: int = 2,
    reward_std_epsilon: float = _DEFAULT_REWARD_STD_EPSILON,
    drop_incomplete_trajectory: bool = True,
) -> RoleGRPOResult:
    """Build role advantages and completion-token-masked training samples."""

    advantages = compute_role_advantages(
        group,
        effective_rewards,
        min_group_size=min_group_size,
        reward_std_epsilon=reward_std_epsilon,
        drop_incomplete_trajectory=drop_incomplete_trajectory,
    )
    by_completion = _advantage_by_completion(advantages)
    samples: list[TrainingSample] = []
    eligible, dropped = _eligible_trajectories(
        group,
        drop_incomplete=drop_incomplete_trajectory,
    )
    for trajectory in eligible:
        for completion in trajectory.completions:
            role_advantage = by_completion.get(completion.completion_id)
            if role_advantage is not None:
                samples.append(completion_to_training_sample(completion, role_advantage))
    return RoleGRPOResult(
        advantages=advantages,
        samples=tuple(samples),
        dropped_trajectory_ids=dropped,
    )


__all__ = [
    "RoleGRPOError",
    "RoleGRPOResult",
    "TrainingSample",
    "build_role_grpo_batch",
    "completion_to_training_sample",
    "compute_role_advantages",
]
