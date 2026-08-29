"""Generation request/result validation and prompt projection helpers.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts import DialogueTurn, EpisodeSample, PromptProjection
from ..episodes import project_target_prompt
from .types import (
    BackendFailure,
    GenerationProtocolError,
    RolloutConfigurationError,
    RolloutValidationError,
    TokenizedPrompt,
)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One generation request made after prompt projection/tokenization."""

    projection: PromptProjection
    prompt_token_ids: tuple[int, ...]
    character: str
    trajectory_id: str
    sample_index: int
    seed: int
    turn_index: int
    policy_version: str
    tokenizer_fingerprint: str
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Exact backend output used to construct a :class:`CompletionTrace`.

    ``completion_token_ids`` and ``rollout_logprobs`` are already aligned by
    the backend.  The rollout never tokenizes ``text`` and never reconstructs
    IDs from it.
    """

    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    text: str
    finish_reason: str
    completion_start: int | None = None
    completion_end: int | None = None
    policy_version: str | None = None
    tokenizer_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if any(type(token_id) is not int or token_id < 0 for token_id in self.prompt_token_ids):
            raise GenerationProtocolError("prompt token IDs must be non-negative integers")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.completion_token_ids):
            raise GenerationProtocolError("completion token IDs must be non-negative integers")
        if not self.completion_token_ids:
            raise GenerationProtocolError("backend returned an empty generated token sequence")
        if len(self.rollout_logprobs) != len(self.completion_token_ids):
            raise GenerationProtocolError(
                "backend must return one rollout log probability per generated token"
            )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in self.rollout_logprobs
        ):
            raise GenerationProtocolError("rollout log probabilities must be finite numbers")
        if not isinstance(self.text, str):
            raise GenerationProtocolError("backend completion text must be a string")
        if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
            raise GenerationProtocolError("backend finish_reason must be a non-empty string")
        start = (
            len(self.prompt_token_ids) if self.completion_start is None else self.completion_start
        )
        end = (
            start + len(self.completion_token_ids)
            if self.completion_end is None
            else self.completion_end
        )
        if type(start) is not int or start != len(self.prompt_token_ids):
            raise GenerationProtocolError(
                "completion_start must equal the exact prompt token count"
            )
        if type(end) is not int or end != start + len(self.completion_token_ids):
            raise GenerationProtocolError("completion_end does not match generated token IDs")
        object.__setattr__(self, "completion_start", start)
        object.__setattr__(self, "completion_end", end)
        for field_name in ("policy_version", "tokenizer_fingerprint"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise GenerationProtocolError(f"{field_name} must be a non-empty string")


def _strict_identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RolloutConfigurationError(f"{field_name} must be a non-empty string")
    return value


def _optional_attribute(owner: object, name: str, default: object = None) -> object:
    """Read an optional protocol attribute without import-time assumptions."""

    return getattr(owner, name, default)


def _identity_from_object(
    owner: object,
    *,
    explicit: str | None,
    field_name: str,
    aliases: tuple[str, ...],
) -> str:
    if explicit is not None:
        return _strict_identity(explicit, field_name=field_name)
    for alias in aliases:
        value = getattr(owner, alias, None)
        if callable(value):
            value = value()
        if value is not None:
            return _strict_identity(value, field_name=field_name)
    raise RolloutConfigurationError(
        f"{field_name} must be supplied explicitly or exposed by the boundary"
    )


def _token_id_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    """Validate one flat tokenizer output without importing tensor libraries."""

    # Hugging Face BatchEncoding and simple fake tokenizers commonly return a
    # mapping containing input_ids.  This is still a tokenizer boundary; it is
    # not a completion-text fallback.
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise RolloutValidationError(f"{field_name} mapping must contain input_ids")
        value = value["input_ids"]

    # Tensor-like CPU values can be converted without importing torch.  A
    # nested single-item batch is accepted because it preserves exact IDs.
    tolist = _optional_attribute(value, "tolist")
    if callable(tolist):
        value = tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RolloutValidationError(f"{field_name} must be a sequence of token IDs")
    if (
        value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes, bytearray))
    ):
        if len(value) != 1:
            raise RolloutValidationError(f"{field_name} must contain one prompt, not a batch")
        value = value[0]
    result: list[int] = []
    for token_id in value:
        if type(token_id) is not int or token_id < 0:
            raise RolloutValidationError(f"{field_name} must contain non-negative integers")
        result.append(token_id)
    return tuple(result)


def _tokenize_projection(tokenizer: object, projection: PromptProjection) -> TokenizedPrompt:
    """Call the configured tokenizer/chat-template boundary exactly once."""

    direct = _optional_attribute(tokenizer, "tokenize_prompt")
    if callable(direct):
        try:
            value = direct(projection)
        except Exception as exc:  # pragma: no cover - backend-specific tokenizer errors
            raise BackendFailure(f"prompt tokenization failed: {exc}") from exc
        if isinstance(value, TokenizedPrompt):
            return value
        if isinstance(value, Mapping) and "token_ids" in value:
            token_ids = _token_id_tuple(value["token_ids"], field_name="prompt token IDs")
            text = value.get("text")
            if text is not None and not isinstance(text, str):
                raise RolloutValidationError("tokenizer prompt text must be a string or None")
            return TokenizedPrompt(token_ids=token_ids, text=text)
        return TokenizedPrompt(token_ids=_token_id_tuple(value, field_name="prompt token IDs"))

    apply_template = _optional_attribute(tokenizer, "apply_chat_template")
    if not callable(apply_template):
        raise BackendFailure("prompt tokenizer must expose tokenize_prompt or apply_chat_template")
    messages = tuple(
        message.model_dump(mode="python", exclude_none=True) for message in projection.messages
    )
    try:
        value = apply_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception as exc:  # pragma: no cover - backend-specific tokenizer errors
        raise BackendFailure(f"prompt chat-template tokenization failed: {exc}") from exc
    return TokenizedPrompt(token_ids=_token_id_tuple(value, field_name="prompt token IDs"))


def _projection_with_history(
    sample: EpisodeSample,
    history: Sequence[DialogueTurn],
    character: str,
) -> PromptProjection:
    """Use Phase A's projection on the current character-only history."""

    # model_copy intentionally keeps the canonical sample identity untouched;
    # project_target_prompt only consumes dialogue/character/prompt fields.
    evolving_episode = sample.episode.model_copy(update={"dialogue": tuple(history)})
    return project_target_prompt(evolving_episode, character, render=sample.render)


def _config_value(config: object | None, path: tuple[str, ...]) -> object | None:
    value = config
    for part in path:
        if value is None:
            return None
        value = getattr(value, part, None)
    return value
