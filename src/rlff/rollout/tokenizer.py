"""Hugging Face prompt-tokenizer adapter and generation protocols.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..contracts import PromptProjection, stable_fingerprint
from .generation import (
    GenerationRequest,
    GenerationResult,
    _optional_attribute,
    _strict_identity,
    _token_id_tuple,
)
from .types import BackendFailure, RolloutConfigurationError, TokenizedPrompt


def _stable_tokenizer_value(value: object) -> object:
    """Keep tokenizer fingerprint inputs JSON-compatible and deterministic."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_tokenizer_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_tokenizer_value(item) for item in value]
    return str(value)


def huggingface_tokenizer_fingerprint(tokenizer: object) -> str:
    """Fingerprint the configured HF tokenizer and its exact chat template.

    The function intentionally reads metadata only; it never imports
    Transformers and never hashes an object repr that could contain a memory
    address.  ``chat_template`` and special-token settings are included so a
    changed prompt renderer cannot silently reuse the old tokenizer identity.
    """

    tokenizer_type = type(tokenizer)
    metadata = {
        "class": f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}",
        "name_or_path": _stable_tokenizer_value(_optional_attribute(tokenizer, "name_or_path")),
        "init_kwargs": _stable_tokenizer_value(_optional_attribute(tokenizer, "init_kwargs")),
        "special_tokens_map": _stable_tokenizer_value(
            _optional_attribute(tokenizer, "special_tokens_map")
        ),
        "chat_template": _stable_tokenizer_value(_optional_attribute(tokenizer, "chat_template")),
        "vocab_size": _stable_tokenizer_value(_optional_attribute(tokenizer, "vocab_size")),
    }
    return stable_fingerprint(metadata)


class HuggingFacePromptTokenizer:
    """Lazy, exact chat-template adapter around a configured HF tokenizer."""

    def __init__(self, tokenizer: object, *, tokenizer_fingerprint: str | None = None) -> None:
        self.tokenizer = tokenizer
        fingerprint = (
            huggingface_tokenizer_fingerprint(tokenizer)
            if tokenizer_fingerprint is None
            else tokenizer_fingerprint
        )
        self.tokenizer_fingerprint = _strict_identity(
            fingerprint,
            field_name="tokenizer_fingerprint",
        )

    @property
    def fingerprint(self) -> str:
        return self.tokenizer_fingerprint

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: object,
    ) -> HuggingFacePromptTokenizer:
        """Load Transformers only when a cloud/model path explicitly asks for it."""

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - local CPU tests do not install it
            raise RolloutConfigurationError(
                "HuggingFacePromptTokenizer requires the optional transformers package"
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        except Exception as exc:  # pragma: no cover - model/network-specific
            raise RolloutConfigurationError(
                f"cannot load HuggingFace tokenizer {model_name_or_path!r}: {exc}"
            ) from exc
        return cls(tokenizer)

    def tokenize_prompt(self, projection: PromptProjection) -> TokenizedPrompt:
        apply_template = _optional_attribute(self.tokenizer, "apply_chat_template")
        if not callable(apply_template):
            raise BackendFailure("HuggingFace tokenizer has no apply_chat_template method")
        messages = [
            message.model_dump(mode="python", exclude_none=True) for message in projection.messages
        ]
        try:
            value = apply_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:  # pragma: no cover - tokenizer-specific
            raise BackendFailure(f"HuggingFace chat-template tokenization failed: {exc}") from exc
        return TokenizedPrompt(_token_id_tuple(value, field_name="prompt token IDs"))


@runtime_checkable
class PromptTokenizer(Protocol):
    """Minimal configured-tokenizer boundary used by the rollout core.

    Implementations must apply the model's exact chat template.  A tokenizer
    may expose ``tokenize_prompt`` directly, or the common
    ``apply_chat_template(messages, tokenize=True, add_generation_prompt=True)``
    method.  No completion text is passed through this boundary.
    """

    def tokenize_prompt(self, projection: PromptProjection) -> TokenizedPrompt: ...


@runtime_checkable
class GenerationBackend(Protocol):
    """Minimal asynchronous generation boundary for a fake or real backend."""

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
