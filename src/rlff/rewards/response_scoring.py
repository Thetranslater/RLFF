"""Deterministic conversion of verifier responses into scalar rewards.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

from collections.abc import Sequence

from .protocol import (
    _ACTION_DIMENSION_INDEX,
    _ACTION_MARKUP_PATTERN,
    _NO_ACTION_SCORE,
    CompletionRewardResponse,
    CompletionScore,
    RewardResponseError,
    TrajectoryRewardResponse,
)


def completion_response_reward(parsed: CompletionRewardResponse) -> float:
    """Compatibility scalar for a response containing exactly one reply."""

    rewards = completion_response_rewards(parsed)
    if len(rewards) != 1:
        raise RewardResponseError("a scalar completion reward requires exactly one scored reply")
    return rewards[0]


def completion_effective_values(score: CompletionScore, reply_text: str) -> tuple[int, ...]:
    """Neutralize behavior when the corresponding reply has no action markup."""

    values = list(score.values)
    if not _ACTION_MARKUP_PATTERN.search(reply_text):
        values[_ACTION_DIMENSION_INDEX] = _NO_ACTION_SCORE
    return tuple(values)


def completion_response_rewards(
    parsed: CompletionRewardResponse,
    *,
    reply_texts: Sequence[str] | None = None,
) -> tuple[float, ...]:
    """Average effective dimensions independently for each ordered reply."""

    if reply_texts is None:
        values = tuple(item.values for item in parsed.scores)
    else:
        texts = tuple(reply_texts)
        if len(texts) != len(parsed.scores):
            raise RewardResponseError(
                "completion reply texts must exactly cover parsed completion scores"
            )
        values = tuple(
            completion_effective_values(score, text)
            for score, text in zip(parsed.scores, texts, strict=True)
        )
    return tuple(sum(item) / len(item) for item in values)


def trajectory_response_reward(parsed: TrajectoryRewardResponse) -> float:
    if not parsed.score:
        return 0.0
    return sum(item.value for item in parsed.score) / len(parsed.score)
