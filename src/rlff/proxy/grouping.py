"""Stable group character ordering and reward-weight scheduling.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Sequence

from .normalization import _finite

logger = logging.getLogger(__name__)


def _group_character_order(characters: Sequence[str], group_key: str) -> tuple[str, ...]:
    """Return one reproducible shuffled order shared by a native GRPO group."""

    ordered = list(characters)
    if len(ordered) < 2:
        return tuple(ordered)
    seed = int.from_bytes(
        hashlib.sha256(f"rlff-character-order:{group_key}".encode()).digest()[:8],
        "big",
    )
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def reward_weights_for_step(
    step: int,
    *,
    completion_start_weight: float,
    global_start_weight: float,
    schedule_start_step: int | None = None,
    schedule_end_step: int | None = None,
    completion_end_weight: float | None = None,
    global_end_weight: float | None = None,
) -> tuple[float, float]:
    """Return constant or linearly interpolated reward weights for one update step."""

    if type(step) is not int or step < 0:
        raise ValueError("reward schedule step must be a non-negative integer")
    start_weights = (
        _finite(completion_start_weight, field="completion_start_weight"),
        _finite(global_start_weight, field="global_start_weight"),
    )
    schedule = (
        schedule_start_step,
        schedule_end_step,
        completion_end_weight,
        global_end_weight,
    )
    if all(value is None for value in schedule):
        return start_weights
    if any(value is None for value in schedule):
        raise ValueError("reward weight schedule fields must be supplied together")
    assert schedule_start_step is not None
    assert schedule_end_step is not None
    assert completion_end_weight is not None
    assert global_end_weight is not None
    if schedule_start_step < 0 or schedule_end_step <= schedule_start_step:
        raise ValueError("reward schedule requires 0 <= start_step < end_step")
    end_weights = (
        _finite(completion_end_weight, field="completion_end_weight"),
        _finite(global_end_weight, field="global_end_weight"),
    )
    if any(value < 0 for value in (*start_weights, *end_weights)):
        raise ValueError("reward weights must be non-negative")
    if step <= schedule_start_step:
        return start_weights
    if step >= schedule_end_step:
        return end_weights
    progress = (step - schedule_start_step) / (schedule_end_step - schedule_start_step)
    completion_weight = start_weights[0] + (end_weights[0] - start_weights[0]) * progress
    global_weight = start_weights[1] + (end_weights[1] - start_weights[1]) * progress
    return completion_weight, global_weight
