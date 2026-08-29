"""Trajectory generation state machine and rollout-group validation.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence

from ..contracts import (
    CompletionTrace,
    DialogueTurn,
    EpisodeSample,
    GroupedEpisodeSamples,
    RolloutGroup,
    TerminationReason,
    Trajectory,
    stable_id,
)
from .generation import (
    GenerationRequest,
    GenerationResult,
    _identity_from_object,
    _projection_with_history,
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
from .tokenizer import GenerationBackend
from .types import (
    ContextLimitExceeded,
    GenerationProtocolError,
    RolloutConfigurationError,
    RolloutValidationError,
)


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
