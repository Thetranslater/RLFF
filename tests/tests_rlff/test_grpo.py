from __future__ import annotations

import pytest

from rlff.contracts import (
    CompletionTrace,
    RoleEffectiveReward,
    RolloutGroup,
    TerminationReason,
    Trajectory,
)
from rlff.grpo import (
    RoleGRPOError,
    build_role_grpo_batch,
    completion_to_training_sample,
    compute_role_advantages,
)


def completion(
    *,
    trajectory_id: str,
    completion_id: str,
    character: str,
    turn_index: int,
    prompt: tuple[int, ...] = (10, 11),
    generated: tuple[int, ...] = (20, 21),
) -> CompletionTrace:
    return CompletionTrace(
        completion_id=completion_id,
        episode_id="episode-1",
        group_id="group-1",
        trajectory_id=trajectory_id,
        character=character,
        turn_index=turn_index,
        prompt_token_ids=prompt,
        completion_token_ids=generated,
        completion_start=len(prompt),
        completion_end=len(prompt) + len(generated),
        rollout_logprobs=(-0.1,) * len(generated),
        text="text is never tokenized",
        finish_reason="stop",
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )


def trajectory(
    trajectory_id: str,
    *,
    complete: bool = True,
    alice_twice: bool = False,
) -> Trajectory:
    completions = [
        completion(
            trajectory_id=trajectory_id,
            completion_id=f"{trajectory_id}-alice-0",
            character="Alice",
            turn_index=0,
        )
    ]
    if alice_twice:
        completions.append(
            completion(
                trajectory_id=trajectory_id,
                completion_id=f"{trajectory_id}-alice-1",
                character="Alice",
                turn_index=1,
                prompt=(1,),
                generated=(99,),
            )
        )
    completions.append(
        completion(
            trajectory_id=trajectory_id,
            completion_id=f"{trajectory_id}-bob-0",
            character="Bob",
            turn_index=len(completions),
        )
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        episode_id="episode-1",
        group_id="group-1",
        completions=completions,
        termination_reason=(
            TerminationReason.MAX_ROUNDS if complete else TerminationReason.TRUNCATED
        ),
        planned_rounds=1,
        completed_rounds=1 if complete else 0,
        completion_count=len(completions),
        valid=complete,
        truncated=not complete,
        invalid_reason=None if complete else "test truncation",
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )


def group(*trajectories: Trajectory) -> RolloutGroup:
    return RolloutGroup(
        group_id="group-1",
        episode_id="episode-1",
        trajectories=trajectories,
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )


def role_reward(
    trajectory_id: str,
    character: str,
    reward: float,
    *completion_ids: str,
) -> RoleEffectiveReward:
    return RoleEffectiveReward(
        group_id="group-1",
        trajectory_id=trajectory_id,
        character=character,
        completion_ids=completion_ids,
        aggregated_local_reward=reward,
        trajectory_role_contribution=0.0,
        effective_reward=reward,
    )


def test_role_normalization_is_independent_and_hand_calculated() -> None:
    first = trajectory("trajectory-0")
    second = trajectory("trajectory-1")
    rollout = group(first, second)
    rewards = [
        role_reward("trajectory-0", "Alice", 1.0, "trajectory-0-alice-0"),
        role_reward("trajectory-1", "Alice", 3.0, "trajectory-1-alice-0"),
        role_reward("trajectory-0", "Bob", 4.0, "trajectory-0-bob-0"),
        role_reward("trajectory-1", "Bob", 4.0, "trajectory-1-bob-0"),
    ]

    result = compute_role_advantages(rollout, rewards)

    assert [(item.trajectory_id, item.character) for item in result] == [
        ("trajectory-0", "Alice"),
        ("trajectory-0", "Bob"),
        ("trajectory-1", "Alice"),
        ("trajectory-1", "Bob"),
    ]
    assert [item.advantage for item in result] == [-1.0, 0.0, 1.0, 0.0]
    assert all(item.valid_group_size == 2 for item in result)
    assert result[0].reward_mean == 2.0
    assert result[0].reward_std == 1.0


def test_zero_variance_is_valid_but_has_zero_advantage() -> None:
    rollout = group(trajectory("trajectory-0"), trajectory("trajectory-1"))
    rewards = [
        role_reward("trajectory-0", "Alice", 2.0, "trajectory-0-alice-0"),
        role_reward("trajectory-1", "Alice", 2.0, "trajectory-1-alice-0"),
        role_reward("trajectory-0", "Bob", 2.0, "trajectory-0-bob-0"),
        role_reward("trajectory-1", "Bob", 2.0, "trajectory-1-bob-0"),
    ]

    result = compute_role_advantages(rollout, rewards)

    assert len(result) == 4
    assert [item.advantage for item in result if item.character == "Alice"] == [0.0, 0.0]
    assert [item.reward_std for item in result if item.character == "Alice"] == [0.0, 0.0]


def test_multiple_completions_same_role_receive_same_role_advantage() -> None:
    first = trajectory("trajectory-0", alice_twice=True)
    second = trajectory("trajectory-1", alice_twice=True)
    rollout = group(first, second)
    rewards = [
        role_reward(
            "trajectory-0",
            "Alice",
            1.0,
            "trajectory-0-alice-0",
            "trajectory-0-alice-1",
        ),
        role_reward(
            "trajectory-1",
            "Alice",
            3.0,
            "trajectory-1-alice-0",
            "trajectory-1-alice-1",
        ),
        role_reward("trajectory-0", "Bob", 0.0, "trajectory-0-bob-0"),
        role_reward("trajectory-1", "Bob", 0.0, "trajectory-1-bob-0"),
    ]

    result = build_role_grpo_batch(rollout, rewards)

    assert len(result.advantages) == 4
    assert [item.advantage for item in result.advantages if item.character == "Alice"] == [
        -1.0,
        1.0,
    ]
    assert [sample["completion_id"] for sample in result.samples] == [
        "trajectory-0-alice-0",
        "trajectory-0-alice-1",
        "trajectory-0-bob-0",
        "trajectory-1-alice-0",
        "trajectory-1-alice-1",
        "trajectory-1-bob-0",
    ]
    first_sample = result.samples[0]
    assert first_sample["input_ids"] == (10, 11, 20, 21)
    assert first_sample["loss_mask"] == (0.0, 0.0, 1.0, 1.0)
    assert first_sample["advantages"] == (0.0, 0.0, -1.0, -1.0)
    assert first_sample["logprobs"] == (0.0, 0.0, -0.1, -0.1)
    assert first_sample["effective_reward"] == 1.0
    assert "rewards" not in first_sample
    assert "versions" not in first_sample
    assert result.samples[1]["input_ids"] == (1, 99)


def test_incomplete_trajectory_is_dropped_and_minimum_role_group_applies() -> None:
    rollout = group(trajectory("trajectory-0"), trajectory("trajectory-1", complete=False))
    rewards = [
        role_reward("trajectory-0", "Alice", 1.0, "trajectory-0-alice-0"),
        role_reward("trajectory-1", "Alice", 3.0, "trajectory-1-alice-0"),
    ]

    assert compute_role_advantages(rollout, rewards) == ()
    with pytest.raises(RoleGRPOError, match="invalid or incomplete"):
        compute_role_advantages(rollout, rewards, drop_incomplete_trajectory=False)


def test_one_failure_invalidates_the_fixed_group_instead_of_shrinking_it() -> None:
    rollout = group(
        trajectory("trajectory-0"),
        trajectory("trajectory-1"),
        trajectory("trajectory-2"),
        trajectory("trajectory-3", complete=False),
    )
    rewards = [
        role_reward(
            trajectory_id,
            "Alice",
            float(index),
            f"{trajectory_id}-alice-0",
        )
        for index, trajectory_id in enumerate(
            ("trajectory-0", "trajectory-1", "trajectory-2", "trajectory-3")
        )
    ]
    rewards.extend(
        role_reward(
            trajectory_id,
            "Bob",
            0.0,
            f"{trajectory_id}-bob-0",
        )
        for trajectory_id in (
            "trajectory-0",
            "trajectory-1",
            "trajectory-2",
            "trajectory-3",
        )
    )
    result = build_role_grpo_batch(rollout, rewards)
    assert result.advantages == ()
    assert result.samples == ()
    assert result.dropped_trajectory_ids == (
        "trajectory-0",
        "trajectory-1",
        "trajectory-2",
        "trajectory-3",
    )


def test_unknown_completion_or_group_reward_is_rejected() -> None:
    rollout = group(trajectory("trajectory-0"), trajectory("trajectory-1"))
    bad_completion = role_reward("trajectory-0", "Alice", 1.0, "missing")
    with pytest.raises(RoleGRPOError, match="unknown completion"):
        compute_role_advantages(rollout, [bad_completion])

    bad_group = bad_completion.model_copy(update={"group_id": "other-group"})
    with pytest.raises(RoleGRPOError, match="belongs to group"):
        compute_role_advantages(rollout, [bad_group])


def test_each_eligible_role_needs_exact_completion_coverage() -> None:
    first = trajectory("trajectory-0", alice_twice=True)
    second = trajectory("trajectory-1", alice_twice=True)
    rollout = group(first, second)
    incomplete = [
        role_reward("trajectory-0", "Alice", 1.0, "trajectory-0-alice-0"),
        role_reward(
            "trajectory-1",
            "Alice",
            3.0,
            "trajectory-1-alice-0",
            "trajectory-1-alice-1",
        ),
        role_reward("trajectory-0", "Bob", 0.0, "trajectory-0-bob-0"),
        role_reward("trajectory-1", "Bob", 0.0, "trajectory-1-bob-0"),
    ]
    assert compute_role_advantages(rollout, incomplete) == ()
    with pytest.raises(RoleGRPOError, match="do not exactly cover"):
        compute_role_advantages(rollout, incomplete, drop_incomplete_trajectory=False)


def test_zero_token_completion_is_not_a_training_sample() -> None:
    first = trajectory("trajectory-0")
    empty = first.completions[0].model_copy(
        update={
            "completion_token_ids": (),
            "completion_end": first.completions[0].completion_start,
            "rollout_logprobs": (),
        }
    )
    first = first.model_copy(update={"completions": (empty, *first.completions[1:])})
    rollout = group(first, trajectory("trajectory-1"))
    rewards = [
        role_reward("trajectory-0", "Alice", 1.0, "trajectory-0-alice-0"),
        role_reward("trajectory-1", "Alice", 3.0, "trajectory-1-alice-0"),
        role_reward("trajectory-0", "Bob", 0.0, "trajectory-0-bob-0"),
        role_reward("trajectory-1", "Bob", 0.0, "trajectory-1-bob-0"),
    ]
    assert build_role_grpo_batch(rollout, rewards).samples == ()
    with pytest.raises(RoleGRPOError, match="zero-token"):
        compute_role_advantages(rollout, rewards, drop_incomplete_trajectory=False)


def test_completion_conversion_rejects_identity_mismatch() -> None:
    trace = trajectory("trajectory-0").completions[0]
    reward = role_reward("trajectory-0", "Alice", 1.0, trace.completion_id)
    role_advantage = compute_role_advantages(
        group(trajectory("trajectory-0"), trajectory("trajectory-1")),
        [
            reward,
            role_reward("trajectory-1", "Alice", 3.0, "trajectory-1-alice-0"),
            role_reward("trajectory-0", "Bob", 0.0, "trajectory-0-bob-0"),
            role_reward("trajectory-1", "Bob", 0.0, "trajectory-1-bob-0"),
        ],
    )[0]
    with pytest.raises(RoleGRPOError, match="trajectory"):
        completion_to_training_sample(
            trace,
            role_advantage.model_copy(update={"trajectory_id": "other"}),
        )


def test_reward_std_epsilon_and_minimum_size_are_validated() -> None:
    rollout = group(trajectory("trajectory-0"), trajectory("trajectory-1"))
    rewards = [
        role_reward("trajectory-0", "Alice", 1.0, "trajectory-0-alice-0"),
        role_reward("trajectory-1", "Alice", 2.0, "trajectory-1-alice-0"),
    ]
    with pytest.raises(ValueError, match="positive integer"):
        compute_role_advantages(rollout, rewards, min_group_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_role_advantages(rollout, rewards, reward_std_epsilon=-1.0)
