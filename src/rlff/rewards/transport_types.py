"""Narrow reward transport and provider interfaces.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from ..contracts import CompletionReward, EpisodeRecord, Trajectory, TrajectoryReward


@dataclass(frozen=True, slots=True)
class RewardHTTPResponse:
    """Raw response returned by the reward-local transport boundary."""

    status_code: int
    text: str
    headers: Mapping[str, str] = field(default_factory=dict)


RewardTransportResult: TypeAlias = (
    RewardHTTPResponse | Mapping[str, Any] | tuple[int, str] | str | bytes
)


RewardTransportCallable: TypeAlias = Callable[..., Awaitable[RewardTransportResult]]


class RewardTransport(Protocol):
    """Small injectable async HTTP boundary used by remote reward providers."""

    async def __call__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> RewardTransportResult: ...


@dataclass(frozen=True, slots=True)
class RewardBatch:
    """Ordered completion and trajectory-role records for one or more trajectories."""

    completion_rewards: tuple[CompletionReward, ...]
    trajectory_rewards: tuple[TrajectoryReward, ...]

    @property
    def local(self) -> tuple[CompletionReward, ...]:
        return self.completion_rewards

    @property
    def trajectory_role_rewards(self) -> tuple[TrajectoryReward, ...]:
        return self.trajectory_rewards


@runtime_checkable
class RewardProvider(Protocol):
    """Internal provider boundary for exactly the two Phase C providers."""

    provider_name: str

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]: ...

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward: ...
