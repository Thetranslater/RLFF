"""Reward batch orchestration, aggregation, and provider factory.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any, cast

from ..contracts import (
    CompletionReward,
    EpisodeRecord,
    RewardStatus,
    RoleEffectiveReward,
    Trajectory,
    TrajectoryReward,
)
from .prompts import load_reward_prompts
from .protocol import QWEN_DASHSCOPE_PROVIDER, RewardAggregationError
from .provider_base import PlaceholderRewardProvider
from .providers import DeepSeekRewardProvider, QwenDashScopeRewardProvider
from .transport_types import RewardBatch, RewardProvider, RewardTransport, RewardTransportCallable


def _ensure_sequence(value: Trajectory | Sequence[Trajectory]) -> tuple[Trajectory, ...]:
    return (value,) if isinstance(value, Trajectory) else tuple(value)


async def score_completions(
    provider: RewardProvider,
    episode: EpisodeRecord,
    trajectories: Trajectory | Sequence[Trajectory],
) -> tuple[CompletionReward, ...]:
    """Score each trajectory/character once and return all reply rewards."""

    active = _ensure_sequence(trajectories)
    calls = [
        provider.score_completion_role(episode, trajectory, character)
        for trajectory in active
        for character in dict.fromkeys(
            completion.character for completion in trajectory.completions
        )
    ]
    batches = tuple(await asyncio.gather(*calls)) if calls else ()
    return tuple(reward for batch in batches for reward in batch)


async def score_trajectories(
    provider: RewardProvider,
    episode: EpisodeRecord,
    trajectories: Trajectory | Sequence[Trajectory],
) -> tuple[TrajectoryReward, ...]:
    """Score every trajectory/character pair in stable source order."""

    active = _ensure_sequence(trajectories)
    calls = [
        provider.score_trajectory(episode, trajectory, character)
        for trajectory in active
        for character in dict.fromkeys(
            completion.character for completion in trajectory.completions
        )
    ]
    return tuple(await asyncio.gather(*calls)) if calls else ()


async def score_rollout_group(
    provider: RewardProvider,
    episode: EpisodeRecord,
    trajectories: Trajectory | Sequence[Trajectory],
) -> RewardBatch:
    """Score completion-local and trajectory-role rewards in stable order."""

    local, trajectory_role_rewards = await asyncio.gather(
        score_completions(provider, episode, trajectories),
        score_trajectories(provider, episode, trajectories),
    )
    return RewardBatch(local, trajectory_role_rewards)


def _invalid_reward_message(reward: Any, scope: str) -> str:
    return (
        f"invalid {scope} reward for "
        f"{getattr(reward, 'trajectory_id', '?')}/{getattr(reward, 'character', '')}: "
        f"{getattr(reward, 'error', None) or 'unspecified error'}"
    )


def aggregate_role_rewards(
    completion_rewards: Sequence[CompletionReward],
    trajectory_rewards: Sequence[TrajectoryReward],
    *,
    completion_weight: float = 0.6,
    global_weight: float = 0.4,
) -> tuple[RoleEffectiveReward, ...]:
    """Aggregate local means and weighted trajectory-role task rewards."""

    if not math.isfinite(completion_weight) or completion_weight < 0:
        raise RewardAggregationError("completion_weight must be finite and non-negative")
    if not math.isfinite(global_weight) or global_weight < 0:
        raise RewardAggregationError("global_weight must be finite and non-negative")

    role_rewards_by_key: dict[tuple[str, str], TrajectoryReward] = {}
    for trajectory_reward in trajectory_rewards:
        if trajectory_reward.status is not RewardStatus.VALID:
            raise RewardAggregationError(_invalid_reward_message(trajectory_reward, "trajectory"))
        if trajectory_reward.reward is None or not math.isfinite(trajectory_reward.reward):
            raise RewardAggregationError("valid trajectory reward has no finite scalar")
        key = (trajectory_reward.trajectory_id, trajectory_reward.character)
        if key in role_rewards_by_key:
            raise RewardAggregationError(
                "duplicate trajectory role reward "
                f"{trajectory_reward.trajectory_id!r}/{trajectory_reward.character!r}"
            )
        role_rewards_by_key[key] = trajectory_reward

    grouped: dict[tuple[str, str], list[CompletionReward]] = {}
    key_order: list[tuple[str, str]] = []
    completion_ids: set[str] = set()
    for completion_reward in completion_rewards:
        if completion_reward.status is not RewardStatus.VALID:
            raise RewardAggregationError(_invalid_reward_message(completion_reward, "completion"))
        if completion_reward.reward is None or not math.isfinite(completion_reward.reward):
            raise RewardAggregationError("valid completion reward has no finite scalar")
        if completion_reward.completion_id in completion_ids:
            raise RewardAggregationError(
                f"duplicate completion reward {completion_reward.completion_id!r}"
            )
        completion_ids.add(completion_reward.completion_id)
        key = (completion_reward.trajectory_id, completion_reward.character)
        if key not in grouped:
            grouped[key] = []
            key_order.append(key)
        grouped[key].append(completion_reward)

    records: list[RoleEffectiveReward] = []
    for trajectory_id, character in key_order:
        local = grouped[(trajectory_id, character)]
        trajectory_role_reward = role_rewards_by_key.get((trajectory_id, character))
        if trajectory_role_reward is None:
            raise RewardAggregationError(
                f"missing valid trajectory role reward for {trajectory_id!r}/{character!r}"
            )
        mean_local = sum(cast(float, item.reward) for item in local) / len(local)
        weighted_local = mean_local * completion_weight
        weighted_trajectory_role = cast(float, trajectory_role_reward.reward) * global_weight
        effective = weighted_local + weighted_trajectory_role
        if not all(
            math.isfinite(value)
            for value in (mean_local, weighted_local, weighted_trajectory_role, effective)
        ):
            raise RewardAggregationError("reward aggregation produced a non-finite value")
        records.append(
            RoleEffectiveReward(
                group_id=local[0].group_id,
                trajectory_id=trajectory_id,
                character=character,
                completion_ids=tuple(item.completion_id for item in local),
                aggregated_local_reward=weighted_local,
                trajectory_role_contribution=weighted_trajectory_role,
                effective_reward=effective,
            )
        )
    return tuple(records)


def create_reward_provider(
    config: Any,
    *,
    api_key: str | None = None,
    transport: RewardTransportCallable | RewardTransport | None = None,
    tracer: Any = None,
    langsmith_api_key: str | None = None,
    development: bool = False,
) -> RewardProvider:
    """Instantiate the explicitly configured reward provider."""

    provider = getattr(config, "provider", None)
    if provider == "placeholder":
        # Prompt paths are still a required, auditable boundary in a
        # development run.  The files must exist and be non-empty; their
        # marker remains visible rather than being silently replaced.
        load_reward_prompts(
            config.completion.prompt_path,
            config.global_reward.prompt_path,
            provider="placeholder",
            development=True,
        )
        return PlaceholderRewardProvider(config=config)
    if provider == "deepseek":
        return DeepSeekRewardProvider(
            config=config,
            api_key=api_key,
            transport=transport,
            tracer=tracer,
            langsmith_api_key=langsmith_api_key,
            development=development,
        )
    if provider == QWEN_DASHSCOPE_PROVIDER:
        return QwenDashScopeRewardProvider(
            config=config,
            api_key=api_key,
            transport=transport,
            tracer=tracer,
            langsmith_api_key=langsmith_api_key,
            development=development,
        )
    raise ValueError(f"unsupported reward provider {provider!r}")
