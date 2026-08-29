"""Round policy, configuration resolution, and group input validation.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import math

from ..contracts import EpisodeSample, GroupedEpisodeSamples
from .generation import _config_value
from .types import RolloutConfigurationError, RolloutValidationError


def rounds_for_character_count(character_count: int) -> int:
    """Return the production round-robin horizon for a character count.

    One and two-character episodes use seven rounds, three-character episodes
    use six, and episodes with four or more characters use five.  This keeps
    the total dialogue depth comparable while ensuring every character gets
    the same number of turns.
    """

    if type(character_count) is not int or character_count <= 0:
        raise RolloutConfigurationError("character_count must be a positive integer")
    if character_count <= 2:
        return 7
    if character_count == 3:
        return 6
    return 5


def _resolve_setting(
    explicit: object | None,
    config: object | None,
    path: tuple[str, ...],
    default: object,
) -> object:
    if explicit is not None:
        return explicit
    configured = _config_value(config, path)
    return default if configured is None else configured


def _positive_int_setting(value: object, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RolloutConfigurationError(f"{field_name} must be a positive integer")
    return value


def _numeric_setting(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutConfigurationError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RolloutConfigurationError(f"{field_name} must be finite")
    return number


def _validate_group_inputs(group: GroupedEpisodeSamples) -> tuple[EpisodeSample, ...]:
    if not isinstance(group, GroupedEpisodeSamples):
        raise RolloutConfigurationError("rollout input must be GroupedEpisodeSamples")
    samples = tuple(sorted(group.samples, key=lambda sample: sample.sample_index))
    if not samples:
        raise RolloutValidationError("rollout groups require at least one sample")
    first_render = samples[0].render
    if any(sample.render != first_render for sample in samples[1:]):
        raise RolloutValidationError("all samples in one group must share the render spec")
    sample_ids = {sample.trajectory_id for sample in samples}
    if len(sample_ids) != len(samples):
        raise RolloutValidationError("group samples must have distinct trajectory IDs")
    if any(
        sample.group_id != group.group_id or sample.episode_id != group.episode_id
        for sample in samples
    ):
        raise RolloutValidationError("group sample identity does not match the enclosing group")
    if any(sample.episode.fingerprint != group.episode.fingerprint for sample in samples):
        raise RolloutValidationError("group samples must carry the canonical episode")
    characters = tuple(character.name for character in group.episode.characters)
    if not characters:
        raise RolloutValidationError("episodes require at least one registered character")
    if len(set(characters)) != len(characters):
        raise RolloutValidationError("episode character order must be unique")
    if any(character.casefold() == "environment" for character in characters):
        raise RolloutValidationError("Environment is not a valid rollout character")
    return samples
