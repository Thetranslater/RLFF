"""Shared and placeholder reward-provider implementations.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..contracts import (
    CompletionReward,
    CompletionTrace,
    EpisodeRecord,
    Trajectory,
    TrajectoryReward,
)
from .payloads import _character, _trajectory_character_completions
from .protocol import PLACEHOLDER_PROVIDER, REWARD_RESPONSE_SCHEMA_VERSION, RewardResponseError
from .transport_types import RewardBatch


class _RewardProviderBase:
    """Shared ordered batch helpers for the two concrete providers."""

    provider_name: str

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]:
        raise NotImplementedError

    async def score_completion(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        completion: CompletionTrace,
    ) -> CompletionReward:
        """Compatibility helper; batch paths call ``score_completion_role`` directly."""

        rewards = await self.score_completion_role(
            episode,
            trajectory,
            completion.character,
        )
        return next(
            reward for reward in rewards if reward.completion_id == completion.completion_id
        )

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        raise NotImplementedError

    async def score_completion_batch(
        self,
        episode: EpisodeRecord,
        trajectories: Sequence[Trajectory],
    ) -> tuple[CompletionReward, ...]:
        calls = [
            self.score_completion_role(episode, trajectory, character)
            for trajectory in trajectories
            for character in dict.fromkeys(
                completion.character for completion in trajectory.completions
            )
        ]
        batches = tuple(await asyncio.gather(*calls)) if calls else ()
        return tuple(reward for batch in batches for reward in batch)

    async def score_trajectory_batch(
        self,
        episode: EpisodeRecord,
        trajectories: Sequence[Trajectory],
    ) -> tuple[TrajectoryReward, ...]:
        calls = [
            self.score_trajectory(episode, trajectory, character)
            for trajectory in trajectories
            for character in dict.fromkeys(
                completion.character for completion in trajectory.completions
            )
        ]
        return tuple(await asyncio.gather(*calls)) if calls else ()

    async def score_group(
        self,
        episode: EpisodeRecord,
        trajectories: Sequence[Trajectory],
    ) -> RewardBatch:
        local, trajectory_role_rewards = await asyncio.gather(
            self.score_completion_batch(episode, trajectories),
            self.score_trajectory_batch(episode, trajectories),
        )
        return RewardBatch(local, trajectory_role_rewards)


def _placeholder_audit(
    *,
    scope: Literal["completion_local", "trajectory_role"],
    identifiers: Mapping[str, str],
) -> str:
    return json.dumps(
        {
            "audit_schema_version": "rlff.reward.audit.v1",
            "provider": PLACEHOLDER_PROVIDER,
            "development": True,
            "placeholder": True,
            "scope": scope,
            "reason": "explicitly_enabled_zero_reward_development_provider",
            "response_schema": REWARD_RESPONSE_SCHEMA_VERSION,
            "identifiers": dict(identifiers),
            "response": {"reward": 0.0, "scores": []},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class PlaceholderRewardProvider(_RewardProviderBase):
    """Explicit development-only zero reward provider."""

    provider_name = PLACEHOLDER_PROVIDER

    def __init__(
        self,
        *,
        allow_placeholder: bool = False,
        config: Any = None,
    ) -> None:
        if config is not None:
            if getattr(config, "provider", None) != "placeholder":
                raise ValueError("placeholder provider requires config.provider='placeholder'")
            allow_placeholder = bool(getattr(config, "allow_placeholder", False))
        if not allow_placeholder:
            raise ValueError(
                "placeholder rewards require provider=placeholder and allow_placeholder=true"
            )
        self.development = True

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]:
        _character(episode, character)
        completions = _trajectory_character_completions(trajectory, character)
        raw = _placeholder_audit(
            scope="completion_local",
            identifiers={
                "episode_id": episode.episode_id or "",
                "group_id": trajectory.group_id,
                "trajectory_id": trajectory.trajectory_id,
                "character": character,
            },
        )
        return tuple(
            CompletionReward(
                completion_id=completion.completion_id,
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                character=completion.character,
                reward=0.0,
                provider=self.provider_name,
                raw_response=raw,
            )
            for completion in completions
        )

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        _character(episode, character)
        raw = _placeholder_audit(
            scope="trajectory_role",
            identifiers={
                "episode_id": episode.episode_id or "",
                "group_id": trajectory.group_id,
                "trajectory_id": trajectory.trajectory_id,
                "character": character,
            },
        )
        return TrajectoryReward(
            trajectory_id=trajectory.trajectory_id,
            group_id=trajectory.group_id,
            character=character,
            reward=0.0,
            provider=self.provider_name,
            raw_response=raw,
        )

    async def score_proxy_completion_role(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, float]:
        """Return explicit zeros for a text-only proxy role reward view."""

        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise RewardResponseError("proxy completion payload is missing ids")
        values = identifiers.get("completion_ids")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RewardResponseError("proxy completion_ids must be a sequence")
        return {str(completion_id): 0.0 for completion_id in values}

    async def score_proxy_trajectory_role(self, payload: Mapping[str, Any]) -> float:
        """Return explicit zero for a text-only proxy reward view."""

        _ = payload
        return 0.0

    async def score_trajectory_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        return await self.score_trajectory(episode, trajectory, character)

    async def score_local(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        completion: CompletionTrace,
    ) -> CompletionReward:
        return await self.score_completion(episode, trajectory, completion)
