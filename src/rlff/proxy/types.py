"""Proxy-facing views, provider protocol, and audit provider.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class ProxyGroupError(RuntimeError):
    """Raised when one fixed-size proxy group cannot complete atomically."""


@dataclass(frozen=True, slots=True)
class ProxyCompletionView:
    """Text-only completion metadata available before official proxy export."""

    completion_id: str
    group_id: str
    trajectory_id: str
    character: str
    turn_index: int
    text: str
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyTrajectoryView:
    """Text-only trajectory view used by reward prompts and normalization."""

    group_id: str
    trajectory_id: str
    episode_id: str
    episode: Mapping[str, Any]
    completions: tuple[ProxyCompletionView, ...]
    turns: tuple[Mapping[str, Any], ...]
    planned_rounds: int
    completed_rounds: int
    termination_reason: str = "max_rounds"
    valid: bool = True
    truncated: bool = False
    invalid_reason: str | None = None


class ProxyRewardProvider(Protocol):
    """Narrow runtime reward interface; no CompletionTrace is fabricated."""

    async def score_proxy_completion_role(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, float]: ...

    async def score_proxy_trajectory_role(self, payload: Mapping[str, Any]) -> float: ...


DETERMINISTIC_AUDIT_REWARD_PROVIDER = "deterministic_audit"


class DeterministicAuditRewardProvider:
    """Assign deterministic trajectory-slot scores for the cloud rollout audit.

    AReaL serializes workflow keyword arguments before sending them to rollout
    workers, so the audit passes only ``reward_provider_name`` over RPC.  The
    worker constructs this provider locally through ``_get_reward_provider``.
    """

    @staticmethod
    def _slot(payload: Mapping[str, Any]) -> int:
        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise ValueError("audit reward payload is missing ids")
        trajectory_id = str(identifiers.get("trajectory_id", ""))
        prefix, separator, slot = trajectory_id.rpartition(":trajectory:")
        if not separator or not prefix or not slot.isdigit():
            raise ValueError(f"cannot extract trajectory slot from {trajectory_id!r}")
        return int(slot)

    async def score_proxy_completion_role(self, payload: Mapping[str, Any]) -> dict[str, float]:
        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise ValueError("audit completion payload is missing ids")
        completion_ids = identifiers.get("completion_ids")
        if not isinstance(completion_ids, Sequence) or isinstance(
            completion_ids, (str, bytes, bytearray)
        ):
            raise ValueError("audit completion payload has invalid completion_ids")
        value = float(self._slot(payload))
        return {str(completion_id): value for completion_id in completion_ids}

    async def score_proxy_trajectory_role(self, payload: Mapping[str, Any]) -> float:
        return float(self._slot(payload))


ProxyTrajectoryRunner = Callable[..., ProxyTrajectoryView | Awaitable[ProxyTrajectoryView]]
