"""Compatibility facade for the split :mod:`rlff.proxy` implementation."""

# ruff: noqa: F401 -- this module intentionally preserves the former module namespace.

from __future__ import annotations

from .agent import RLFFGroupAwareAgent, _GroupState
from .grouping import _group_character_order, logger, reward_weights_for_step
from .normalization import (
    _finite,
    _mapping_value,
    decode_proxy_episode_payload,
    normalize_proxy_role_advantages,
)
from .types import (
    DETERMINISTIC_AUDIT_REWARD_PROVIDER,
    DeterministicAuditRewardProvider,
    ProxyCompletionView,
    ProxyGroupError,
    ProxyRewardProvider,
    ProxyTrajectoryRunner,
    ProxyTrajectoryView,
)

__all__ = [
    "DETERMINISTIC_AUDIT_REWARD_PROVIDER",
    "DeterministicAuditRewardProvider",
    "ProxyCompletionView",
    "ProxyGroupError",
    "ProxyRewardProvider",
    "ProxyTrajectoryRunner",
    "ProxyTrajectoryView",
    "RLFFGroupAwareAgent",
    "decode_proxy_episode_payload",
    "normalize_proxy_role_advantages",
    "reward_weights_for_step",
]
