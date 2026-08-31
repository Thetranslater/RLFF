"""Proxy episode decoding and role-level advantage normalization.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .types import ProxyCompletionView, ProxyGroupError, ProxyTrajectoryView

logger = logging.getLogger(__name__)


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProxyGroupError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProxyGroupError(f"{field} must be finite")
    return result


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def decode_proxy_episode_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Decode an episode transported either as a mapping or opaque JSON.

    Hugging Face Datasets normalizes nested mapping schemas across all rows and
    inserts null values for missing keys.  Canonical episodes are therefore
    transported through AReaL as JSON strings so their fingerprints survive
    the Arrow round trip unchanged.  Mapping input remains supported for the
    direct smoke-test and unit-test paths.
    """

    raw: Any = data.get("episode", data)
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProxyGroupError("rollout data episode JSON is invalid") from exc
    if not isinstance(raw, Mapping):
        raise ProxyGroupError("rollout data episode must be a mapping or JSON object")
    return cast(Mapping[str, Any], dict(raw))


def normalize_proxy_role_advantages(
    trajectories: Sequence[ProxyTrajectoryView],
    completion_rewards: Mapping[str, float],
    trajectory_role_rewards: Mapping[tuple[str, str], float],
    *,
    completion_weight: float = 0.6,
    global_weight: float = 0.4,
    min_group_size: int = 2,
    reward_std_epsilon: float = 1e-8,
    completion_advantage_gate_enabled: bool = False,
    completion_bad_reward_threshold: float = 2.5,
    completion_good_reward_threshold: float = 4.5,
    completion_bad_density_threshold: float = 0.6,
    completion_bad_to_good_ratio: float = 2.0,
    completion_min_bad_completions: int = 3,
) -> dict[str, float]:
    """Aggregate then normalize role rewards for one complete proxy group.

    Local completion scores are first averaged by ``(trajectory, character)``;
    the same character's full-trajectory task score is then combined.  Only after that step are
    values normalized independently by ``(group_id, character)``.  Every
    completion belonging to one trajectory/character initially receives the
    same resulting scalar.  The optional conservative gate then zeroes a
    completion whose absolute quality clearly conflicts with that direction.
    """

    if not trajectories:
        raise ProxyGroupError("proxy group must contain at least one trajectory")
    if type(min_group_size) is not int or min_group_size <= 0:
        raise ProxyGroupError("min_group_size must be a positive integer")
    completion_weight = _finite(completion_weight, field="completion_weight")
    global_weight = _finite(global_weight, field="global_weight")
    epsilon = _finite(reward_std_epsilon, field="reward_std_epsilon")
    bad_threshold = _finite(
        completion_bad_reward_threshold,
        field="completion_bad_reward_threshold",
    )
    good_threshold = _finite(
        completion_good_reward_threshold,
        field="completion_good_reward_threshold",
    )
    bad_density_threshold = _finite(
        completion_bad_density_threshold,
        field="completion_bad_density_threshold",
    )
    bad_to_good_ratio = _finite(
        completion_bad_to_good_ratio,
        field="completion_bad_to_good_ratio",
    )
    if type(completion_advantage_gate_enabled) is not bool:
        raise ProxyGroupError("completion_advantage_gate_enabled must be a boolean")
    if (
        type(completion_min_bad_completions) is not int
        or completion_min_bad_completions <= 0
    ):
        raise ProxyGroupError("completion_min_bad_completions must be a positive integer")
    if completion_weight < 0 or global_weight < 0:
        raise ProxyGroupError("reward weights must be non-negative")
    if epsilon < 0:
        raise ProxyGroupError("reward_std_epsilon must be non-negative")
    if not 1 <= bad_threshold < good_threshold <= 5:
        raise ProxyGroupError(
            "completion reward thresholds must satisfy 1 <= bad < good <= 5"
        )
    if not 0 < bad_density_threshold <= 1:
        raise ProxyGroupError("completion_bad_density_threshold must be in (0, 1]")
    if bad_to_good_ratio < 1:
        raise ProxyGroupError("completion_bad_to_good_ratio must be at least 1")

    first_group = trajectories[0].group_id
    if any(item.group_id != first_group for item in trajectories):
        raise ProxyGroupError("all proxy trajectories must belong to one group")
    role_values: dict[tuple[str, str], float] = {}
    role_completion_rewards: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {}
    role_order: list[str] = []
    seen_trajectory_ids: set[str] = set()
    seen_completion_ids: set[str] = set()
    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise ProxyGroupError("proxy group contains duplicate trajectory IDs")
        seen_trajectory_ids.add(trajectory.trajectory_id)
        if not trajectory.valid or trajectory.truncated:
            raise ProxyGroupError(
                f"trajectory {trajectory.trajectory_id!r} is invalid or truncated"
            )
        if (
            trajectory.termination_reason != "max_rounds"
            or trajectory.completed_rounds != trajectory.planned_rounds
        ):
            raise ProxyGroupError(
                f"trajectory {trajectory.trajectory_id!r} did not complete max_rounds"
            )
        by_character: dict[str, list[ProxyCompletionView]] = {}
        for completion in trajectory.completions:
            if completion.group_id != first_group:
                raise ProxyGroupError("completion group differs from trajectory group")
            if completion.trajectory_id != trajectory.trajectory_id:
                raise ProxyGroupError("completion trajectory differs from trajectory")
            if not completion.character or completion.character.casefold() == "environment":
                raise ProxyGroupError("Environment/narrator completions are forbidden")
            if completion.completion_id in seen_completion_ids:
                raise ProxyGroupError("proxy group contains duplicate completion IDs")
            seen_completion_ids.add(completion.completion_id)
            if completion.completion_id in completion_rewards:
                by_character.setdefault(completion.character, []).append(completion)
            else:
                raise ProxyGroupError(f"missing completion reward {completion.completion_id!r}")
        if not by_character:
            raise ProxyGroupError(f"trajectory {trajectory.trajectory_id!r} has no completions")
        for character, completions in by_character.items():
            if character not in role_order:
                role_order.append(character)
            local_values = [
                _finite(
                    completion_rewards[completion.completion_id],
                    field=f"completion reward {completion.completion_id}",
                )
                for completion in completions
            ]
            local_mean = sum(local_values) / len(local_values)
            key = (trajectory.trajectory_id, character)
            trajectory_role_reward = _finite(
                trajectory_role_rewards.get(key),
                field=f"trajectory role reward {trajectory.trajectory_id}/{character}",
            )
            role_values[key] = (
                completion_weight * local_mean + global_weight * trajectory_role_reward
            )
            role_completion_rewards[key] = tuple(
                (item.completion_id, value)
                for item, value in zip(completions, local_values, strict=True)
            )

    expected_roles = set(role_order)
    if len(trajectories) < min_group_size:
        raise ProxyGroupError(
            f"proxy group has {len(trajectories)} trajectories; minimum is {min_group_size}"
        )
    for trajectory in trajectories:
        actual = {
            character
            for (trajectory_id, character) in role_values
            if trajectory_id == trajectory.trajectory_id
        }
        if actual != expected_roles:
            raise ProxyGroupError(
                f"trajectory {trajectory.trajectory_id!r} role set differs from group"
            )

    result: dict[str, float] = {}
    positive_bad_masked = 0
    negative_good_masked = 0
    for character in role_order:
        records = [
            (trajectory.trajectory_id, role_values[(trajectory.trajectory_id, character)])
            for trajectory in trajectories
        ]
        mean = sum(value for _, value in records) / len(records)
        variance = sum((value - mean) ** 2 for _, value in records) / len(records)
        std = math.sqrt(variance)
        for trajectory_id, value in records:
            advantage = 0.0 if std <= epsilon else (value - mean) / std
            key = (trajectory_id, character)
            completion_records = role_completion_rewards[key]
            bad_count = sum(
                reward <= bad_threshold for _, reward in completion_records
            )
            good_count = sum(
                reward >= good_threshold for _, reward in completion_records
            )
            bad_density = bad_count / len(completion_records)
            bad_majority = (
                bad_count >= completion_min_bad_completions
                and bad_density >= bad_density_threshold
                and bad_count >= bad_to_good_ratio * good_count
            )
            for completion_id, reward in completion_records:
                gated_advantage = advantage
                if completion_advantage_gate_enabled:
                    if advantage > 0 and reward <= bad_threshold:
                        gated_advantage = 0.0
                        positive_bad_masked += 1
                    elif advantage < 0 and reward >= good_threshold and bad_majority:
                        gated_advantage = 0.0
                        negative_good_masked += 1
                result[completion_id] = gated_advantage
    masked = positive_bad_masked + negative_good_masked
    if completion_advantage_gate_enabled:
        logger.info(
            "RLFF completion advantage gate group=%s masked=%d/%d "
            "positive_bad=%d negative_good_dense_bad=%d",
            first_group,
            masked,
            len(result),
            positive_bad_masked,
            negative_good_masked,
        )
    expected_ids = {
        completion.completion_id
        for trajectory in trajectories
        for completion in trajectory.completions
    }
    if set(result) != expected_ids:
        raise ProxyGroupError("normalized reward mapping does not cover exactly the group")
    return result
