"""Proxy episode decoding and role-level advantage normalization.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .types import ProxyCompletionView, ProxyGroupError, ProxyTrajectoryView


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
) -> dict[str, float]:
    """Aggregate then normalize role rewards for one complete proxy group.

    Local completion scores are first averaged by ``(trajectory, character)``;
    the same character's full-trajectory task score is then combined.  Only after that step are
    values normalized independently by ``(group_id, character)``.  Every
    completion belonging to one trajectory/character receives exactly the
    same resulting scalar.
    """

    if not trajectories:
        raise ProxyGroupError("proxy group must contain at least one trajectory")
    if type(min_group_size) is not int or min_group_size <= 0:
        raise ProxyGroupError("min_group_size must be a positive integer")
    completion_weight = _finite(completion_weight, field="completion_weight")
    global_weight = _finite(global_weight, field="global_weight")
    epsilon = _finite(reward_std_epsilon, field="reward_std_epsilon")
    if completion_weight < 0 or global_weight < 0:
        raise ProxyGroupError("reward weights must be non-negative")
    if epsilon < 0:
        raise ProxyGroupError("reward_std_epsilon must be non-negative")

    first_group = trajectories[0].group_id
    if any(item.group_id != first_group for item in trajectories):
        raise ProxyGroupError("all proxy trajectories must belong to one group")
    role_values: dict[tuple[str, str], float] = {}
    role_completion_ids: dict[tuple[str, str], tuple[str, ...]] = {}
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
            role_completion_ids[key] = tuple(item.completion_id for item in completions)

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
            for completion_id in role_completion_ids[(trajectory_id, character)]:
                result[completion_id] = advantage
    expected_ids = {
        completion.completion_id
        for trajectory in trajectories
        for completion in trajectory.completions
    }
    if set(result) != expected_ids:
        raise ProxyGroupError("normalized reward mapping does not cover exactly the group")
    return result
