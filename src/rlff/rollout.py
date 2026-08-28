"""Framework-neutral, character-only round-robin rollout state machine.

The SGLang transport is deliberately not part of this checkpoint.  The core
accepts two small boundaries that are usable by a CPU fake backend:

* :class:`PromptTokenizer` applies the configured chat template and returns
  the exact prompt token IDs used by generation.
* :class:`GenerationBackend` receives those IDs and returns exact generated
  IDs/log-probabilities and finish metadata.

The state machine owns no scheduler, environment, natural-stop decision, or
reward logic.  It advances every registered character in source order for
every complete round.  An EOS/stop finish reason therefore ends one
completion, not the trajectory.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import (
    CompletionTrace,
    DialogueTurn,
    EpisodeSample,
    GroupedEpisodeSamples,
    PromptProjection,
    RolloutGroup,
    TerminationReason,
    Trajectory,
    stable_fingerprint,
    stable_id,
)
from .episodes import project_target_prompt


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


def _backend_result(raw: object, *, prompt_token_ids: tuple[int, ...]) -> GenerationResult:
    if isinstance(raw, GenerationResult):
        if raw.prompt_token_ids != prompt_token_ids:
            raise GenerationProtocolError("backend prompt token IDs differ from tokenizer output")
        return raw
    if not isinstance(raw, Mapping):
        raise GenerationProtocolError(
            "generation backend must return GenerationResult, not an untyped object"
        )
    # This direct mapping is the framework-neutral fake-backend wire shape.
    required = {
        "prompt_token_ids",
        "completion_token_ids",
        "rollout_logprobs",
        "text",
        "finish_reason",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise GenerationProtocolError(
            "generation result is missing required fields: " + ", ".join(missing)
        )
    prompt_ids = _token_id_tuple(raw["prompt_token_ids"], field_name="prompt token IDs")
    if prompt_ids != prompt_token_ids:
        raise GenerationProtocolError("backend prompt token IDs differ from tokenizer output")
    completion_ids = _token_id_tuple(raw["completion_token_ids"], field_name="completion token IDs")
    logprobs_raw = raw["rollout_logprobs"]
    if not isinstance(logprobs_raw, Sequence) or isinstance(logprobs_raw, (str, bytes, bytearray)):
        raise GenerationProtocolError("rollout_logprobs must be a sequence")
    logprobs: list[float] = []
    for value in logprobs_raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GenerationProtocolError("rollout_logprobs must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise GenerationProtocolError("rollout_logprobs must contain finite numbers")
        logprobs.append(number)
    text = raw["text"]
    finish_reason = raw["finish_reason"]
    if not isinstance(text, str) or not isinstance(finish_reason, str):
        raise GenerationProtocolError("generation text and finish_reason must be strings")
    start = raw.get("completion_start")
    end = raw.get("completion_end")
    if start is not None and type(start) is not int:
        raise GenerationProtocolError("completion_start must be an integer")
    if end is not None and type(end) is not int:
        raise GenerationProtocolError("completion_end must be an integer")
    policy = raw.get("policy_version")
    tokenizer = raw.get("tokenizer_fingerprint")
    if policy is not None and not isinstance(policy, str):
        raise GenerationProtocolError("policy_version must be a string or None")
    if tokenizer is not None and not isinstance(tokenizer, str):
        raise GenerationProtocolError("tokenizer_fingerprint must be a string or None")
    return GenerationResult(
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        rollout_logprobs=tuple(logprobs),
        text=text,
        finish_reason=finish_reason,
        completion_start=start,
        completion_end=end,
        policy_version=policy,
        tokenizer_fingerprint=tokenizer,
    )


def _completion_trace(
    *,
    sample: EpisodeSample,
    result: GenerationResult,
    character: str,
    turn_index: int,
    policy_version: str,
    tokenizer_fingerprint: str,
) -> CompletionTrace:
    if result.policy_version is not None and result.policy_version != policy_version:
        raise GenerationProtocolError(
            f"backend policy_version {result.policy_version!r} differs from fixed group policy"
        )
    if (
        result.tokenizer_fingerprint is not None
        and result.tokenizer_fingerprint != tokenizer_fingerprint
    ):
        raise GenerationProtocolError(
            "backend tokenizer_fingerprint differs from fixed group tokenizer"
        )
    if not result.completion_token_ids:
        raise GenerationProtocolError("backend returned an empty generated token sequence")
    if result.completion_start is None or result.completion_end is None:
        raise GenerationProtocolError("backend completion boundaries were not populated")
    completion_id = stable_id(
        "completion",
        sample.group_id,
        sample.trajectory_id,
        turn_index,
        character,
    )
    return CompletionTrace(
        completion_id=completion_id,
        episode_id=sample.episode_id,
        group_id=sample.group_id,
        trajectory_id=sample.trajectory_id,
        character=character,
        turn_index=turn_index,
        prompt_token_ids=result.prompt_token_ids,
        completion_token_ids=result.completion_token_ids,
        completion_start=result.completion_start,
        completion_end=result.completion_end,
        rollout_logprobs=result.rollout_logprobs,
        text=result.text,
        finish_reason=result.finish_reason,
        policy_version=policy_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


def _failed_trajectory(
    *,
    sample: EpisodeSample,
    history: Sequence[DialogueTurn],
    completions: Sequence[CompletionTrace],
    max_rounds: int,
    character_count: int,
    policy_version: str,
    tokenizer_fingerprint: str,
    reason: TerminationReason,
    truncated: bool,
    error: str,
) -> Trajectory:
    return Trajectory(
        trajectory_id=sample.trajectory_id,
        episode_id=sample.episode_id,
        group_id=sample.group_id,
        completions=tuple(completions),
        turns=tuple(history),
        termination_reason=reason,
        planned_rounds=max_rounds,
        completed_rounds=len(completions) // character_count,
        completion_count=len(completions),
        valid=False,
        truncated=truncated,
        invalid_reason=error,
        policy_version=policy_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


async def _rollout_one(
    sample: EpisodeSample,
    *,
    backend: GenerationBackend,
    tokenizer: object,
    characters: tuple[str, ...],
    max_rounds: int,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    policy_version: str,
    tokenizer_fingerprint: str,
) -> Trajectory:
    history = list(sample.episode.dialogue)
    completions: list[CompletionTrace] = []
    for round_index in range(max_rounds):
        for character_index, character in enumerate(characters):
            turn_index = round_index * len(characters) + character_index
            try:
                projection = _projection_with_history(sample, history, character)
                tokenized = _tokenize_projection(tokenizer, projection)
                if len(tokenized.token_ids) + max_new_tokens > context_length:
                    raise ContextLimitExceeded(
                        f"prompt length {len(tokenized.token_ids)} plus max_new_tokens "
                        f"{max_new_tokens} exceeds context_length {context_length}"
                    )
                request = GenerationRequest(
                    projection=projection,
                    prompt_token_ids=tokenized.token_ids,
                    character=character,
                    trajectory_id=sample.trajectory_id,
                    sample_index=sample.sample_index,
                    seed=sample.seed,
                    turn_index=turn_index,
                    policy_version=policy_version,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                raw = await backend.generate(request)
                result = _backend_result(raw, prompt_token_ids=tokenized.token_ids)
                trace = _completion_trace(
                    sample=sample,
                    result=result,
                    character=character,
                    turn_index=turn_index,
                    policy_version=policy_version,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                )
            except ContextLimitExceeded as exc:
                return _failed_trajectory(
                    sample=sample,
                    history=history,
                    completions=completions,
                    max_rounds=max_rounds,
                    character_count=len(characters),
                    policy_version=policy_version,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    reason=TerminationReason.CONTEXT_LIMIT,
                    truncated=True,
                    error=str(exc),
                )
            except GenerationProtocolError as exc:
                return _failed_trajectory(
                    sample=sample,
                    history=history,
                    completions=completions,
                    max_rounds=max_rounds,
                    character_count=len(characters),
                    policy_version=policy_version,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    reason=TerminationReason.INVALID,
                    truncated=False,
                    error=str(exc),
                )
            except Exception as exc:  # backend/tokenizer infrastructure failure
                return _failed_trajectory(
                    sample=sample,
                    history=history,
                    completions=completions,
                    max_rounds=max_rounds,
                    character_count=len(characters),
                    policy_version=policy_version,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    reason=TerminationReason.INFRASTRUCTURE_FAILURE,
                    truncated=True,
                    error=f"generation failed for {character}: {exc}",
                )
            completions.append(trace)
            # EOS/stop is represented on this trace only.  A non-empty text is
            # appended as the next character-only dialogue turn; an EOS-only
            # token sequence remains exact without inventing empty content.
            if trace.text:
                history.append(
                    DialogueTurn(character=character, content=trace.text, turn_id=turn_index)
                )

    return Trajectory(
        trajectory_id=sample.trajectory_id,
        episode_id=sample.episode_id,
        group_id=sample.group_id,
        completions=tuple(completions),
        turns=tuple(history),
        termination_reason=TerminationReason.MAX_ROUNDS,
        planned_rounds=max_rounds,
        completed_rounds=max_rounds,
        completion_count=len(completions),
        valid=True,
        truncated=False,
        policy_version=policy_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


def validate_rollout_group(
    group: GroupedEpisodeSamples,
    rollout: RolloutGroup,
) -> RolloutGroup:
    """Validate group completeness and fixed identity after rollout assembly."""

    samples = _validate_group_inputs(group)
    expected_ids = tuple(sample.trajectory_id for sample in samples)
    actual_ids = tuple(trajectory.trajectory_id for trajectory in rollout.trajectories)
    if actual_ids != expected_ids:
        raise RolloutValidationError(
            "rollout group must contain exactly one trajectory per ordered sample"
        )
    if rollout.group_id != group.group_id or rollout.episode_id != group.episode_id:
        raise RolloutValidationError("rollout group identity does not match episode group")
    for trajectory in rollout.trajectories:
        if trajectory.group_id != rollout.group_id or trajectory.episode_id != rollout.episode_id:
            raise RolloutValidationError("trajectory identity does not match rollout group")
        if trajectory.policy_version != rollout.policy_version:
            raise RolloutValidationError("trajectory policy identity differs from rollout group")
        if trajectory.tokenizer_fingerprint != rollout.tokenizer_fingerprint:
            raise RolloutValidationError("trajectory tokenizer identity differs from rollout group")
    return rollout


async def rollout_group(
    group: GroupedEpisodeSamples,
    backend: GenerationBackend | None = None,
    tokenizer: object | None = None,
    *,
    engine: GenerationBackend | None = None,
    config: object | None = None,
    max_rounds: int | None = None,
    context_length: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    policy_version: str | None = None,
    tokenizer_fingerprint: str | None = None,
) -> RolloutGroup:
    """Run one grouped episode using direct ordered round-robin generation.

    Samples are processed in ``sample_index`` order and trajectories are
    returned in that same order.  This is intentionally a direct loop rather
    than a scheduler abstraction.  A failed sample becomes an explicit
    invalid/truncated trajectory while other samples still produce their
    deterministic result.
    """

    if backend is None:
        backend = engine
    if backend is None:
        raise RolloutConfigurationError("a generation backend is required")
    if tokenizer is None:
        raise RolloutConfigurationError("a configured prompt tokenizer is required")
    samples = _validate_group_inputs(group)
    resolved_max_rounds = _resolve_setting(max_rounds, config, ("rollout", "max_rounds"), 7)
    resolved_context = _resolve_setting(context_length, config, ("sglang", "context_length"), 8192)
    resolved_max_new = _resolve_setting(max_new_tokens, config, ("sglang", "max_new_tokens"), 256)
    resolved_temperature = _resolve_setting(temperature, config, ("sglang", "temperature"), 0.9)
    resolved_top_p = _resolve_setting(top_p, config, ("sglang", "top_p"), 1.0)
    rounds_cap = _positive_int_setting(resolved_max_rounds, field_name="max_rounds")
    context_value = _positive_int_setting(resolved_context, field_name="context_length")
    max_new_value = _positive_int_setting(resolved_max_new, field_name="max_new_tokens")
    temperature_value = _numeric_setting(resolved_temperature, field_name="temperature")
    top_p_value = _numeric_setting(resolved_top_p, field_name="top_p")
    if not 0 <= temperature_value <= 2:
        raise RolloutConfigurationError("temperature must be between 0 and 2")
    if not 0 < top_p_value <= 1:
        raise RolloutConfigurationError("top_p must be in (0, 1]")
    fixed_policy = _identity_from_object(
        backend,
        explicit=policy_version,
        field_name="policy_version",
        aliases=("policy_version", "version"),
    )
    fixed_tokenizer = _identity_from_object(
        tokenizer,
        explicit=tokenizer_fingerprint,
        field_name="tokenizer_fingerprint",
        aliases=("tokenizer_fingerprint", "fingerprint"),
    )
    characters = tuple(character.name for character in group.episode.characters)
    rounds_value = min(rounds_cap, rounds_for_character_count(len(characters)))
    trajectories = tuple(
        await asyncio.gather(
            *(
                _rollout_one(
                    sample,
                    backend=backend,
                    tokenizer=tokenizer,
                    characters=characters,
                    max_rounds=rounds_value,
                    context_length=context_value,
                    max_new_tokens=max_new_value,
                    temperature=temperature_value,
                    top_p=top_p_value,
                    policy_version=fixed_policy,
                    tokenizer_fingerprint=fixed_tokenizer,
                )
                for sample in samples
            )
        )
    )
    result = RolloutGroup(
        group_id=group.group_id,
        episode_id=group.episode_id,
        trajectories=trajectories,
        policy_version=fixed_policy,
        tokenizer_fingerprint=fixed_tokenizer,
    )
    return validate_rollout_group(group, result)


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
