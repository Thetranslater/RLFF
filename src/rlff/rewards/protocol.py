"""Reward protocol constants, errors, and response models.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

REWARD_PROMPT_VERSION: Final = "rlff.reward.prompts.v2"


REWARD_REQUEST_SCHEMA_VERSION: Final = "rlff.reward.request.v2"


REWARD_RESPONSE_SCHEMA_VERSION: Final = "rlff.reward.response.v2"


COMPLETION_REWARD_PROMPT_FILENAME: Final = "completion_reward_system_v2.txt"


TRAJECTORY_REWARD_PROMPT_FILENAME: Final = "trajectory_reward_system.txt"


REPAIR_REWARD_PROMPT_FILENAME: Final = "error_system.txt"


DEFAULT_REPAIR_PROMPT: Final = (
    "Current result:\n{json_result}\nError:\n{error_message}\n"
    "Character: {character}\nIndexed replies:\n{utters}\n"
    "Return only the repaired JSON result."
)


PLACEHOLDER_MARKER: Final = "[PLACEHOLDER]"


DEEPSEEK_V4_FLASH_PROVIDER: Final = "deepseek-v4-flash"


QWEN3_7_FLASH_PROVIDER: Final = "qwen3.7-flash"


QWEN_DASHSCOPE_PROVIDER: Final = "qwen_dashscope"


QWEN_DASHSCOPE_GENERATION_URL: Final = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)


PLACEHOLDER_PROVIDER: Final = "placeholder-null-v1"


CHAT_COMPLETIONS_PATH: Final = "/chat/completions"


COMPLETION_DIMENSIONS: Final = (
    "一致性",
    "流畅",
    "知识边界",
    "行为",
)


_PROMPT_VARIABLE_PATTERN: Final = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


_ACTION_MARKUP_PATTERN: Final = re.compile(r"\([^()]+\)|（[^（）]+）")


_ACTION_DIMENSION_INDEX: Final = 3


_NO_ACTION_SCORE: Final = 3


class RewardError(ValueError):
    """Base class for explicit reward-boundary failures."""


class RewardPromptError(RewardError):
    """Raised when a configured reward prompt cannot be used safely."""


class RewardTransportError(RewardError):
    """Raised for an HTTP/transport response that needs a bounded retry."""


class RewardResponseError(RewardError):
    """Raised when a model response violates the scope-specific reward protocol."""


class RewardAggregationError(RewardError):
    """Raised when local/global reward records cannot form role rewards."""


class _StrictModel(BaseModel):
    # Keep collection inputs JSON-friendly (the model response naturally
    # contains a list for ``dimensions``), while scalar fields below remain
    # strict through their ``Strict*`` annotations.
    model_config = ConfigDict(extra="forbid")


def _score_value(value: Any) -> int:
    """Accept only an integer or a digit string in the rubric's closed 1-5 range."""

    if isinstance(value, bool):
        raise ValueError("reward score must be an integer from 1 to 5")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped not in {"1", "2", "3", "4", "5"}:
            raise ValueError("reward score string must be one of 1, 2, 3, 4, 5")
        return int(stripped)
    if type(value) is not int or value not in {1, 2, 3, 4, 5}:
        raise ValueError("reward score must be an integer from 1 to 5")
    return value


class CompletionScore(_StrictModel):
    """Four ordered dimensions for one target-character reply."""

    values: tuple[int, ...]

    @field_validator("values", mode="before")
    @classmethod
    def scores_are_valid(cls, value: Any) -> tuple[int, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError("completion values must be a sequence")
        return tuple(_score_value(item) for item in value)

    @field_validator("values")
    @classmethod
    def has_exact_dimensions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != len(COMPLETION_DIMENSIONS):
            raise ValueError(
                f"completion values must contain exactly {len(COMPLETION_DIMENSIONS)} scores"
            )
        return value


class CompletionRewardResponse(_StrictModel):
    """Ordered reply scores for one character in one complete trajectory."""

    scores: tuple[CompletionScore, ...]


class TrajectoryTaskScore(_StrictModel):
    """One target-character task score returned for a full trajectory."""

    task: StrictStr
    value: int

    @field_validator("task")
    @classmethod
    def task_is_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("trajectory task must be non-empty")
        return normalized

    @field_validator("value", mode="before")
    @classmethod
    def score_is_valid(cls, value: Any) -> int:
        return _score_value(value)


class TrajectoryRewardResponse(_StrictModel):
    """Strict target-character task response for one full trajectory."""

    score: tuple[TrajectoryTaskScore, ...]


@dataclass(frozen=True, slots=True)
class RewardPrompts:
    """Loaded completion/global prompts and their provisional version."""

    completion: str
    trajectory: str
    version: str = REWARD_PROMPT_VERSION
