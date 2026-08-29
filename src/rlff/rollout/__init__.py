"""Compatibility facade for the split :mod:`rlff.rollout` implementation."""

# ruff: noqa: F401 -- this module intentionally preserves the former module namespace.

from __future__ import annotations

from .engine import (
    _backend_result,
    _completion_trace,
    _failed_trajectory,
    _rollout_one,
    rollout_group,
    validate_rollout_group,
)
from .generation import (
    GenerationRequest,
    GenerationResult,
    _config_value,
    _identity_from_object,
    _optional_attribute,
    _projection_with_history,
    _strict_identity,
    _token_id_tuple,
    _tokenize_projection,
)
from .policy import (
    _numeric_setting,
    _positive_int_setting,
    _resolve_setting,
    _validate_group_inputs,
    rounds_for_character_count,
)
from .tokenizer import (
    GenerationBackend,
    HuggingFacePromptTokenizer,
    PromptTokenizer,
    _stable_tokenizer_value,
    huggingface_tokenizer_fingerprint,
)
from .types import (
    BackendFailure,
    ContextLimitExceeded,
    GenerationProtocolError,
    RolloutConfigurationError,
    RolloutError,
    RolloutValidationError,
    TokenizedPrompt,
)

__all__ = [
    "BackendFailure",
    "ContextLimitExceeded",
    "GenerationBackend",
    "GenerationProtocolError",
    "GenerationRequest",
    "GenerationResult",
    "HuggingFacePromptTokenizer",
    "PromptTokenizer",
    "RolloutConfigurationError",
    "RolloutError",
    "RolloutValidationError",
    "TokenizedPrompt",
    "huggingface_tokenizer_fingerprint",
    "rollout_group",
    "rounds_for_character_count",
    "validate_rollout_group",
]
