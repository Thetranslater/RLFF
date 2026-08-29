"""Framework-neutral rollout errors and tokenized prompt record.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass


class RolloutError(RuntimeError):
    """Base class for explicit rollout boundary and execution errors."""


class RolloutConfigurationError(ValueError, RolloutError):
    """Raised when a rollout cannot be started with a complete identity/config."""


class RolloutValidationError(ValueError, RolloutError):
    """Raised when group, tokenizer, or backend output violates the protocol."""


class BackendFailure(RolloutError):
    """Raised by a backend for an infrastructure failure."""


class ContextLimitExceeded(RolloutError):
    """Raised internally when a prompt cannot fit the configured context."""


class GenerationProtocolError(RolloutValidationError):
    """Raised when a backend response has no usable generated token sequence."""


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    """Exact token IDs (and optional rendered text) returned by a tokenizer."""

    token_ids: tuple[int, ...]
    text: str | None = None

    def __post_init__(self) -> None:
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise RolloutValidationError("prompt token IDs must be non-negative integers")
