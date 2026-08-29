"""Reward prompt loading, rendering, and response parsing.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from .protocol import (
    _PROMPT_VARIABLE_PATTERN,
    PLACEHOLDER_MARKER,
    CompletionRewardResponse,
    RewardPromptError,
    RewardPrompts,
    RewardResponseError,
    TrajectoryRewardResponse,
    _StrictModel,
)


def _validate_prompt_text(
    text: str,
    *,
    source: str,
    provider: Literal["qwen_dashscope", "deepseek", "placeholder"],
    development: bool,
) -> str:
    content = text.strip()
    if not content:
        raise RewardPromptError(f"reward prompt is empty: {source}")
    if provider != "placeholder" and PLACEHOLDER_MARKER in content and not development:
        raise RewardPromptError(
            "production reward mode refuses a [PLACEHOLDER] prompt; "
            "use finalized prompts or an explicit development=True path"
        )
    return content


def load_reward_prompt(
    path: str | Path,
    *,
    provider: Literal["qwen_dashscope", "deepseek", "placeholder"] = "qwen_dashscope",
    development: bool = False,
    allow_placeholder: bool | None = None,
) -> str:
    """Load one non-empty UTF-8 reward prompt with placeholder safeguards."""

    if allow_placeholder is not None:
        development = development or allow_placeholder
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RewardPromptError(f"cannot read reward prompt {source}: {exc}") from exc
    return _validate_prompt_text(
        text,
        source=str(source),
        provider=provider,
        development=development,
    )


def load_reward_prompts(
    completion_path: str | Path,
    trajectory_path: str | Path,
    *,
    provider: Literal["qwen_dashscope", "deepseek", "placeholder"] = "qwen_dashscope",
    development: bool = False,
    allow_placeholder: bool | None = None,
) -> RewardPrompts:
    """Load both configured reward prompts, refusing missing/empty files."""

    return RewardPrompts(
        completion=load_reward_prompt(
            completion_path,
            provider=provider,
            development=development,
            allow_placeholder=allow_placeholder,
        ),
        trajectory=load_reward_prompt(
            trajectory_path,
            provider=provider,
            development=development,
            allow_placeholder=allow_placeholder,
        ),
    )


def render_reward_prompt(
    template: str,
    variables: Mapping[str, str],
    *,
    required: Sequence[str],
) -> str:
    """Replace only named RLFF variables, leaving JSON example braces untouched."""

    names = set(_PROMPT_VARIABLE_PATTERN.findall(template))
    required_names = set(required)
    missing_in_template = required_names - names
    if missing_in_template:
        raise RewardPromptError(
            "reward prompt is missing variables: " + ", ".join(sorted(missing_in_template))
        )
    unknown = names - set(variables)
    if unknown:
        raise RewardPromptError(
            "reward prompt contains unsupported variables: " + ", ".join(sorted(unknown))
        )
    rendered = template
    for name in names:
        rendered = rendered.replace("{" + name + "}", variables[name])
    return rendered


def _parse_model_response(
    raw: str | bytes | Mapping[str, Any], model: type[_StrictModel]
) -> _StrictModel:
    try:
        if isinstance(raw, Mapping):
            return model.model_validate(dict(raw))
        return model.model_validate_json(raw)
    except Exception as exc:  # Pydantic's concrete error is retained for audit/retry.
        raise RewardResponseError(f"invalid reward response: {exc}") from exc


def _reward_response_debug(raw_response: str) -> dict[str, Any]:
    """Extract final content and model reasoning from supported HTTP envelopes."""

    try:
        outer = json.loads(raw_response)
    except json.JSONDecodeError:
        return {"content": raw_response, "thinking": None, "usage": {}, "finish_reason": None}
    if not isinstance(outer, Mapping):
        return {"content": raw_response, "thinking": None, "usage": {}, "finish_reason": None}

    usage = outer.get("usage")
    usage_data = dict(usage) if isinstance(usage, Mapping) else {}
    choices = outer.get("choices")
    if not isinstance(choices, list):
        output = outer.get("output")
        choices = output.get("choices") if isinstance(output, Mapping) else None

    message: Mapping[str, Any] = {}
    finish_reason: Any = None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        candidate = choices[0].get("message")
        if isinstance(candidate, Mapping):
            message = candidate
        finish_reason = choices[0].get("finish_reason")

    content = message.get("content")
    if isinstance(content, list):
        parts = [
            str(part["text"])
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        content = "".join(parts) if parts else content
    if not message:
        content = raw_response

    thinking: Any = None
    for key in ("reasoning_content", "reasoning", "thinking"):
        if message.get(key) is not None:
            thinking = message[key]
            break
    return {
        "content": content,
        "thinking": thinking,
        "usage": usage_data,
        "finish_reason": finish_reason,
    }


def parse_completion_reward_response(
    raw: str | bytes | Mapping[str, Any],
    *,
    expected_count: int | None = None,
) -> CompletionRewardResponse:
    """Parse ordered reply scores and optionally require exact reply coverage."""

    parsed = cast(
        CompletionRewardResponse,
        _parse_model_response(raw, CompletionRewardResponse),
    )
    if expected_count is not None:
        if type(expected_count) is not int or expected_count <= 0:
            raise RewardResponseError("expected completion count must be a positive integer")
        if len(parsed.scores) != expected_count:
            raise RewardResponseError(
                "completion scores must exactly cover the target character replies: "
                f"expected {expected_count}, received {len(parsed.scores)}"
            )
    return parsed


def parse_trajectory_reward_response(
    raw: str | bytes | Mapping[str, Any],
    *,
    expected_tasks: Sequence[str],
) -> TrajectoryRewardResponse:
    """Parse and require an exact one-to-one cover of the target role's tasks."""

    tasks = tuple(expected_tasks)
    if not tasks:
        raise RewardResponseError("trajectory reward cannot be parsed without target tasks")
    if len(set(tasks)) != len(tasks):
        raise RewardResponseError("expected trajectory tasks must be distinct")
    parsed = cast(
        TrajectoryRewardResponse,
        _parse_model_response(raw, TrajectoryRewardResponse),
    )
    returned = tuple(item.task for item in parsed.score)
    if len(returned) != len(tasks) or set(returned) != set(tasks):
        raise RewardResponseError(
            "trajectory score tasks must exactly cover the target character task list"
        )
    if len(set(returned)) != len(returned):
        raise RewardResponseError("trajectory score contains duplicate tasks")
    return parsed


def parse_reward_response(
    raw: str | bytes | Mapping[str, Any],
) -> CompletionRewardResponse:
    """Compatibility spelling for the completion-scope response parser."""

    return parse_completion_reward_response(raw)
