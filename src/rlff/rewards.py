"""Phase C reward providers and deterministic reward aggregation.

The reward boundary is deliberately small.  A provider scores all replies by
one character in a trajectory or scores that character's trajectory tasks,
while the helpers in this module preserve rollout order
and aggregate the resulting frozen :mod:`rlff.contracts` records.  The
The production implementations use a reward-local HTTP transport because the shared
``src/LLM`` client intentionally exposes decoded JSON rather than raw HTTP
responses; tests can inject the same narrow transport without making network
requests.

The prompt and response formats in this first implementation are explicitly
provisional.  They are not the final reward rubric and contain no state
verifier/checklist fields.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, TypeAlias, cast, runtime_checkable

import aiohttp
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    field_validator,
)

from .contracts import (
    CharacterSpec,
    CompletionReward,
    CompletionTrace,
    EpisodeRecord,
    RewardDimension,
    RewardStatus,
    RoleEffectiveReward,
    Trajectory,
    TrajectoryReward,
)

REWARD_PROMPT_VERSION: Final = "rlff.reward.prompts.v2"
REWARD_REQUEST_SCHEMA_VERSION: Final = "rlff.reward.request.v2"
REWARD_RESPONSE_SCHEMA_VERSION: Final = "rlff.reward.response.v2"
COMPLETION_REWARD_PROMPT_FILENAME: Final = "completion_reward_system_v2.txt"
TRAJECTORY_REWARD_PROMPT_FILENAME: Final = "trajectory_reward_system.txt"
PLACEHOLDER_MARKER: Final = "[PLACEHOLDER]"
DEEPSEEK_V4_FLASH_PROVIDER: Final = "deepseek-v4-flash"
QWEN3_7_FLASH_PROVIDER: Final = "qwen3.7-flash"
QWEN_DASHSCOPE_PROVIDER: Final = "qwen_dashscope"
QWEN_DASHSCOPE_GENERATION_URL: Final = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
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


def completion_response_reward(parsed: CompletionRewardResponse) -> float:
    """Compatibility scalar for a response containing exactly one reply."""

    rewards = completion_response_rewards(parsed)
    if len(rewards) != 1:
        raise RewardResponseError(
            "a scalar completion reward requires exactly one scored reply"
        )
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


@dataclass(frozen=True, slots=True)
class RewardHTTPResponse:
    """Raw response returned by the reward-local transport boundary."""

    status_code: int
    text: str
    headers: Mapping[str, str] = field(default_factory=dict)


RewardTransportResult: TypeAlias = (
    RewardHTTPResponse | Mapping[str, Any] | tuple[int, str] | str | bytes
)
RewardTransportCallable: TypeAlias = Callable[..., Awaitable[RewardTransportResult]]


class RewardTransport(Protocol):
    """Small injectable async HTTP boundary used by remote reward providers."""

    async def __call__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> RewardTransportResult: ...


@dataclass(frozen=True, slots=True)
class RewardBatch:
    """Ordered completion and trajectory-role records for one or more trajectories."""

    completion_rewards: tuple[CompletionReward, ...]
    trajectory_rewards: tuple[TrajectoryReward, ...]

    @property
    def local(self) -> tuple[CompletionReward, ...]:
        return self.completion_rewards

    @property
    def trajectory_role_rewards(self) -> tuple[TrajectoryReward, ...]:
        return self.trajectory_rewards


@runtime_checkable
class RewardProvider(Protocol):
    """Internal provider boundary for exactly the two Phase C providers."""

    provider_name: str

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]: ...

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward: ...


def _character(episode: EpisodeRecord, name: str) -> CharacterSpec:
    for character in episode.characters:
        if character.name == name:
            return character
    raise ValueError(f"character {name!r} is not registered in episode {episode.episode_id}")


def _turn_payload(turn: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "character": turn.character,
        "content": turn.content,
    }
    if getattr(turn, "turn_id", None) is not None:
        result["turn_id"] = turn.turn_id
    return result


def _trajectory_character_completions(
    trajectory: Trajectory,
    character: str,
) -> tuple[CompletionTrace, ...]:
    """Return one character's completions in their stable trajectory order."""

    result = tuple(
        sorted(
            (
                completion
                for completion in trajectory.completions
                if completion.character == character
            ),
            key=lambda completion: completion.turn_index,
        )
    )
    if not result:
        raise ValueError(
            f"trajectory {trajectory.trajectory_id!r} has no completion for {character!r}"
        )
    return result


def _full_trajectory_history(
    episode: EpisodeRecord,
    trajectory: Trajectory,
) -> list[dict[str, Any]]:
    """Return initial dialogue plus every generated turn in trajectory order."""

    history = [_turn_payload(turn) for turn in episode.dialogue]
    history.extend(_turn_payload(turn) for turn in trajectory.turns)
    return history


def _history_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('character', turn.get('speaker', ''))!s}:{turn.get('content', '')!s}"
        for turn in turns
    )


def _merged_tasks(shared_tasks: Sequence[Any], private_tasks: Sequence[Any]) -> tuple[str, ...]:
    """Merge shared and target-private tasks once, preserving their source order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in (*shared_tasks, *private_tasks):
        task = str(value).strip()
        if task and task not in seen:
            result.append(task)
            seen.add(task)
    return tuple(result)


def build_completion_reward_payload(
    episode: EpisodeRecord,
    trajectory: Trajectory,
    character: str,
) -> dict[str, Any]:
    """Build one request for all replies by a character in a trajectory."""

    target = _character(episode, character)
    completions = _trajectory_character_completions(trajectory, character)
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "completion_local",
        "ids": {
            "episode_id": episode.episode_id,
            "group_id": trajectory.group_id,
            "trajectory_id": trajectory.trajectory_id,
            "character": target.name,
            "completion_ids": [completion.completion_id for completion in completions],
        },
        "template_args": {
            "character": target.name,
            "profile": target.profile,
            "plot": episode.plot,
        },
        "completion_texts": [completion.text for completion in completions],
        "input": {
            "utterances": _history_text(_full_trajectory_history(episode, trajectory)),
        },
    }


def build_trajectory_reward_payload(
    episode: EpisodeRecord,
    trajectory: Trajectory,
    character: str,
) -> dict[str, Any]:
    """Build one target-character task payload over the complete trajectory."""

    target = _character(episode, character)
    tasks = _merged_tasks(episode.shared_tasks, target.private_tasks)
    full_history = [*(_turn_payload(turn) for turn in episode.dialogue)]
    full_history.extend(_turn_payload(turn) for turn in trajectory.turns)
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "trajectory_role",
        "ids": {
            "episode_id": episode.episode_id,
            "group_id": trajectory.group_id,
            "trajectory_id": trajectory.trajectory_id,
            "character": target.name,
        },
        "template_args": {
            "character": target.name,
            "plot": episode.plot,
            "tasks": json.dumps(tasks, ensure_ascii=False),
        },
        "tasks": list(tasks),
        "input": {
            "utterances": _history_text(full_history),
        },
    }


def _proxy_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _proxy_episode_value(episode: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(episode, Mapping):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _proxy_characters(episode: Mapping[str, Any] | Any) -> tuple[Mapping[str, Any], ...]:
    values = _proxy_episode_value(episode, "characters", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RewardResponseError("proxy episode must contain a character sequence")
    result: list[Mapping[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            result.append(value)
        else:
            dumped = getattr(value, "model_dump", None)
            if not callable(dumped):
                raise RewardResponseError("proxy character must be a mapping")
            raw = dumped(mode="json")
            if not isinstance(raw, Mapping):
                raise RewardResponseError("proxy character model_dump must be a mapping")
            result.append(raw)
    return tuple(result)


def build_proxy_completion_reward_payload(
    episode: Mapping[str, Any] | Any,
    trajectory: Any,
    character: str,
) -> dict[str, Any]:
    """Build one role-completion request from a text-only proxy view.

    This boundary intentionally accepts no token fields and never constructs a
    :class:`CompletionTrace`; exact token metadata remains the responsibility
    of AReaL's official export path.
    """

    characters = _proxy_characters(episode)
    character_name = str(character)
    target = next(
        (
            item
            for item in characters
            if str(item.get("name", item.get("character", ""))) == character_name
        ),
        None,
    )
    if target is None or character_name.casefold() == "environment":
        raise RewardResponseError(f"proxy completion character {character_name!r} is invalid")
    completions = tuple(_proxy_field(trajectory, "completions", ()))
    target_completions = tuple(
        item
        for item in sorted(
            completions,
            key=lambda value: int(_proxy_field(value, "turn_index", 0)),
        )
        if str(_proxy_field(item, "character", "")) == character_name
    )
    if not target_completions:
        raise RewardResponseError(
            f"proxy trajectory has no completion for character {character_name!r}"
        )
    history: list[dict[str, Any]] = []
    base_history = (
        _proxy_episode_value(episode, "dialogue", ())
        or _proxy_episode_value(episode, "history", ())
        or ()
    )
    for turn in base_history:
        history.append(
            {
                "character": str(
                    _proxy_field(
                        turn,
                        "character",
                        _proxy_field(turn, "speaker", ""),
                    )
                ),
                "content": str(_proxy_field(turn, "content", "")),
            }
        )
    for item in sorted(completions, key=lambda value: int(_proxy_field(value, "turn_index", 0))):
        history.append(
            {
                "character": str(_proxy_field(item, "character", "")),
                "content": str(_proxy_field(item, "text", "")),
                "completion_id": str(_proxy_field(item, "completion_id", "")),
            }
        )
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "completion_local",
        "ids": {
            "episode_id": str(_proxy_episode_value(episode, "episode_id", "")),
            "group_id": str(_proxy_field(trajectory, "group_id", "")),
            "trajectory_id": str(_proxy_field(trajectory, "trajectory_id", "")),
            "character": character_name,
            "completion_ids": [
                str(_proxy_field(item, "completion_id", ""))
                for item in target_completions
            ],
        },
        "template_args": {
            "character": character_name,
            "profile": str(target.get("profile", "")),
            "plot": str(_proxy_episode_value(episode, "plot", "")),
        },
        "completion_texts": [
            str(_proxy_field(item, "text", "")) for item in target_completions
        ],
        "input": {
            "utterances": _history_text(history),
        },
    }


def build_proxy_trajectory_reward_payload(
    episode: Mapping[str, Any] | Any,
    trajectory: Any,
    character: str,
) -> dict[str, Any]:
    """Build one role-task payload from a complete text-only proxy trajectory."""

    characters = _proxy_characters(episode)
    target = next(
        (
            item
            for item in characters
            if str(item.get("name", item.get("character", ""))) == character
        ),
        None,
    )
    if target is None or character.casefold() == "environment":
        raise RewardResponseError(f"proxy trajectory character {character!r} is invalid")
    shared_tasks = tuple(
        _proxy_episode_value(
            episode,
            "shared_tasks",
            _proxy_episode_value(episode, "tasks", ()),
        )
        or ()
    )
    tasks = _merged_tasks(shared_tasks, tuple(target.get("private_tasks", ())))
    history: list[dict[str, Any]] = []
    base_history = (
        _proxy_episode_value(episode, "dialogue", ())
        or _proxy_episode_value(episode, "history", ())
        or ()
    )
    for turn in (*base_history, *_proxy_field(trajectory, "turns", ())):
        history.append(
            {
                "character": str(
                    _proxy_field(
                        turn,
                        "character",
                        _proxy_field(turn, "speaker", ""),
                    )
                ),
                "content": str(_proxy_field(turn, "content", "")),
            }
        )
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "trajectory_role",
        "ids": {
            "episode_id": str(_proxy_episode_value(episode, "episode_id", "")),
            "group_id": str(_proxy_field(trajectory, "group_id", "")),
            "trajectory_id": str(_proxy_field(trajectory, "trajectory_id", "")),
            "character": character,
        },
        "template_args": {
            "character": character,
            "plot": str(_proxy_episode_value(episode, "plot", "")),
            "tasks": json.dumps(tasks, ensure_ascii=False),
        },
        "tasks": list(tasks),
        "input": {
            "utterances": _history_text(history),
        },
    }


def _json_payload(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(message) for message in messages]


def _redact(value: str, secrets: Sequence[str]) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _response_from_transport(value: RewardTransportResult) -> RewardHTTPResponse:
    if isinstance(value, RewardHTTPResponse):
        return value
    if isinstance(value, bytes):
        return RewardHTTPResponse(status_code=200, text=value.decode("utf-8", "replace"))
    if isinstance(value, str):
        return RewardHTTPResponse(status_code=200, text=value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], str)
    ):
        return RewardHTTPResponse(status_code=value[0], text=value[1])
    if isinstance(value, Mapping):
        # A mapping with status/text is the convenient fake-transport shape;
        # any other mapping is treated as an already-decoded HTTP JSON body.
        if "status_code" in value or "status" in value or "text" in value:
            status = value.get("status_code", value.get("status", 200))
            text = value.get("text", "")
            if not isinstance(status, int) or not isinstance(text, str):
                raise TypeError("fake reward transport status_code/text have invalid types")
            return RewardHTTPResponse(status_code=status, text=text)
        return RewardHTTPResponse(status_code=200, text=json.dumps(value, ensure_ascii=False))
    status = getattr(value, "status_code", getattr(value, "status", None))
    text = getattr(value, "text", None)
    if isinstance(status, int) and isinstance(text, str):
        return RewardHTTPResponse(status_code=status, text=text)
    raise TypeError(f"unsupported reward transport result: {type(value).__name__}")


async def _default_reward_transport(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> RewardHTTPResponse:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(url, headers=dict(headers), json=dict(payload)) as response,
    ):
        return RewardHTTPResponse(
            status_code=response.status,
            text=await response.text(),
            headers=dict(response.headers),
        )


async def _call_transport(
    transport: RewardTransportCallable | RewardTransport,
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> RewardTransportResult:
    """Invoke the documented keyword transport and a few test-double shapes."""

    target: Any = transport
    if not callable(target):
        target = getattr(target, "request", None) or getattr(transport, "post", None)
    if target is None or not callable(target):
        raise TypeError("reward transport must be callable or expose request()")
    try:
        result = target(
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
    except TypeError:
        try:
            parameters = inspect.signature(target).parameters
        except (TypeError, ValueError):
            raise
        kwargs: dict[str, Any] = {}
        if "url" in parameters:
            kwargs["url"] = url
        if "headers" in parameters:
            kwargs["headers"] = headers
        if "payload" in parameters:
            kwargs["payload"] = payload
        elif "json" in parameters:
            kwargs["json"] = payload
        if "timeout_seconds" in parameters:
            kwargs["timeout_seconds"] = timeout_seconds
        elif "timeout" in parameters:
            kwargs["timeout"] = timeout_seconds
        result = target(**kwargs) if kwargs else target(url, headers, payload, timeout_seconds)
    if inspect.isawaitable(result):
        return await cast(Awaitable[RewardTransportResult], result)
    return cast(RewardTransportResult, result)


class LangSmithTracer:
    """Small direct-LangSmith SDK adapter; no LangChain/LangGraph involved."""

    def __init__(self, *, project: str, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            try:
                from langsmith import Client
            except ImportError as exc:  # pragma: no cover - optional local install
                raise RuntimeError("LangSmith SDK is unavailable") from exc
            client = Client(api_key=api_key) if api_key else Client()
        self.client = client
        self.project = project

    def start(self, *, name: str, inputs: Mapping[str, Any]) -> str | None:
        try:
            run = self.client.create_run(
                name=name,
                run_type="llm",
                inputs=dict(inputs),
                project_name=self.project,
            )
            return str(getattr(run, "id", run)) if run is not None else None
        except Exception:
            return None

    def finish(
        self,
        run_id: str | None,
        *,
        outputs: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if run_id is None:
            return
        try:
            kwargs: dict[str, Any] = {}
            if outputs is not None:
                kwargs["outputs"] = dict(outputs)
            if error is not None:
                kwargs["error"] = error
            self.client.update_run(run_id, **kwargs)
        except Exception:
            return


class _RewardProviderBase:
    """Shared ordered batch helpers for the two concrete providers."""

    provider_name: str

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]:
        raise NotImplementedError

    async def score_completion(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        completion: CompletionTrace,
    ) -> CompletionReward:
        """Compatibility helper; batch paths call ``score_completion_role`` directly."""

        rewards = await self.score_completion_role(
            episode,
            trajectory,
            completion.character,
        )
        return next(
            reward for reward in rewards if reward.completion_id == completion.completion_id
        )

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        raise NotImplementedError

    async def score_completion_batch(
        self,
        episode: EpisodeRecord,
        trajectories: Sequence[Trajectory],
    ) -> tuple[CompletionReward, ...]:
        calls = [
            self.score_completion_role(episode, trajectory, character)
            for trajectory in trajectories
            for character in dict.fromkeys(
                completion.character for completion in trajectory.completions
            )
        ]
        batches = tuple(await asyncio.gather(*calls)) if calls else ()
        return tuple(reward for batch in batches for reward in batch)

    async def score_trajectory_batch(
        self,
        episode: EpisodeRecord,
        trajectories: Sequence[Trajectory],
    ) -> tuple[TrajectoryReward, ...]:
        calls = [
            self.score_trajectory(episode, trajectory, character)
            for trajectory in trajectories
            for character in dict.fromkeys(
                completion.character for completion in trajectory.completions
            )
        ]
        return tuple(await asyncio.gather(*calls)) if calls else ()

    async def score_group(
        self,
        episode: EpisodeRecord,
        trajectories: Sequence[Trajectory],
    ) -> RewardBatch:
        local, trajectory_role_rewards = await asyncio.gather(
            self.score_completion_batch(episode, trajectories),
            self.score_trajectory_batch(episode, trajectories),
        )
        return RewardBatch(local, trajectory_role_rewards)


def _placeholder_audit(
    *,
    scope: Literal["completion_local", "trajectory_role"],
    identifiers: Mapping[str, str],
) -> str:
    return json.dumps(
        {
            "audit_schema_version": "rlff.reward.audit.v1",
            "provider": PLACEHOLDER_PROVIDER,
            "development": True,
            "placeholder": True,
            "scope": scope,
            "reason": "explicitly_enabled_zero_reward_development_provider",
            "response_schema": REWARD_RESPONSE_SCHEMA_VERSION,
            "identifiers": dict(identifiers),
            "response": {"reward": 0.0, "scores": []},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class PlaceholderRewardProvider(_RewardProviderBase):
    """Explicit development-only zero reward provider."""

    provider_name = PLACEHOLDER_PROVIDER

    def __init__(
        self,
        *,
        allow_placeholder: bool = False,
        config: Any = None,
    ) -> None:
        if config is not None:
            if getattr(config, "provider", None) != "placeholder":
                raise ValueError("placeholder provider requires config.provider='placeholder'")
            allow_placeholder = bool(getattr(config, "allow_placeholder", False))
        if not allow_placeholder:
            raise ValueError(
                "placeholder rewards require provider=placeholder and allow_placeholder=true"
            )
        self.development = True

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]:
        _character(episode, character)
        completions = _trajectory_character_completions(trajectory, character)
        raw = _placeholder_audit(
            scope="completion_local",
            identifiers={
                "episode_id": episode.episode_id or "",
                "group_id": trajectory.group_id,
                "trajectory_id": trajectory.trajectory_id,
                "character": character,
            },
        )
        return tuple(
            CompletionReward(
                completion_id=completion.completion_id,
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                character=completion.character,
                reward=0.0,
                provider=self.provider_name,
                raw_response=raw,
            )
            for completion in completions
        )

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        _character(episode, character)
        raw = _placeholder_audit(
            scope="trajectory_role",
            identifiers={
                "episode_id": episode.episode_id or "",
                "group_id": trajectory.group_id,
                "trajectory_id": trajectory.trajectory_id,
                "character": character,
            },
        )
        return TrajectoryReward(
            trajectory_id=trajectory.trajectory_id,
            group_id=trajectory.group_id,
            character=character,
            reward=0.0,
            provider=self.provider_name,
            raw_response=raw,
        )

    async def score_proxy_completion_role(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, float]:
        """Return explicit zeros for a text-only proxy role reward view."""

        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise RewardResponseError("proxy completion payload is missing ids")
        values = identifiers.get("completion_ids")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RewardResponseError("proxy completion_ids must be a sequence")
        return {str(completion_id): 0.0 for completion_id in values}

    async def score_proxy_trajectory_role(self, payload: Mapping[str, Any]) -> float:
        """Return explicit zero for a text-only proxy reward view."""

        _ = payload
        return 0.0

    async def score_trajectory_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        return await self.score_trajectory(episode, trajectory, character)

    async def score_local(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        completion: CompletionTrace,
    ) -> CompletionReward:
        return await self.score_completion(episode, trajectory, completion)


class DeepSeekRewardProvider(_RewardProviderBase):
    """DeepSeek v4 flash HTTP reward provider with bounded retries."""

    provider_name = DEEPSEEK_V4_FLASH_PROVIDER
    config_provider_name = "deepseek"
    default_api_key_env = "DEEPSEEK_API_KEY"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = DEEPSEEK_V4_FLASH_PROVIDER,
        completion_prompt: str | None = None,
        trajectory_prompt: str | None = None,
        completion_prompt_path: str | Path | None = None,
        trajectory_prompt_path: str | Path | None = None,
        trajectory_reward_model: str | None = None,
        completion_timeout_seconds: float = 120.0,
        trajectory_timeout_seconds: float | None = None,
        completion_retries: int = 2,
        trajectory_retries: int | None = None,
        completion_concurrency: int = 4,
        trajectory_concurrency: int | None = None,
        completion_temperature: float = 1.0,
        trajectory_temperature: float | None = None,
        completion_reasoning_effort: str = "low",
        trajectory_reasoning_effort: str | None = None,
        completion_max_tokens: int = 25000,
        trajectory_max_tokens: int | None = None,
        transport: RewardTransportCallable | RewardTransport | None = None,
        tracer: Any = None,
        langsmith_project: str = "rlff",
        langsmith_api_key: str | None = None,
        config: Any = None,
        development: bool = False,
        allow_placeholder_prompt: bool | None = None,
    ) -> None:
        if allow_placeholder_prompt is not None:
            development = development or allow_placeholder_prompt
        if config is not None:
            if getattr(config, "provider", self.config_provider_name) != self.config_provider_name:
                raise ValueError(
                    f"{self.provider_name} provider requires "
                    f"config.provider={self.config_provider_name!r}"
                )
            completion_scope = config.completion
            trajectory_scope = config.global_reward
            base_url = str(config.base_url)
            completion_prompt = load_reward_prompt(
                completion_scope.prompt_path,
                provider=cast(Any, self.config_provider_name),
                development=development,
            )
            trajectory_prompt = load_reward_prompt(
                trajectory_scope.prompt_path,
                provider=cast(Any, self.config_provider_name),
                development=development,
            )
            completion_timeout_seconds = float(completion_scope.timeout_seconds)
            trajectory_timeout_seconds = float(trajectory_scope.timeout_seconds)
            completion_retries = int(completion_scope.retries)
            trajectory_retries = int(trajectory_scope.retries)
            completion_concurrency = int(completion_scope.concurrency)
            trajectory_concurrency = int(trajectory_scope.concurrency)
            completion_temperature = float(completion_scope.temperature)
            trajectory_temperature = float(trajectory_scope.temperature)
            completion_reasoning_effort = str(completion_scope.reasoning_effort)
            trajectory_reasoning_effort = str(trajectory_scope.reasoning_effort)
            completion_max_tokens = int(completion_scope.max_tokens)
            trajectory_max_tokens = int(trajectory_scope.max_tokens)
            model = str(completion_scope.model)
            self._trajectory_model = str(trajectory_scope.model)
        else:
            self._trajectory_model = trajectory_reward_model or model

        if completion_prompt is None or trajectory_prompt is None:
            if completion_prompt is None and completion_prompt_path is not None:
                completion_prompt = load_reward_prompt(
                    completion_prompt_path,
                    provider=cast(Any, self.config_provider_name),
                    development=development,
                )
            if trajectory_prompt is None and trajectory_prompt_path is not None:
                trajectory_prompt = load_reward_prompt(
                    trajectory_prompt_path,
                    provider=cast(Any, self.config_provider_name),
                    development=development,
                )
        if completion_prompt is None or trajectory_prompt is None:
            raise RewardPromptError(
                f"{self.provider_name} provider requires non-empty completion_prompt "
                "and trajectory_prompt"
            )
        self._completion_prompt = _validate_prompt_text(
            completion_prompt,
            source="completion_prompt",
            provider=cast(Any, self.config_provider_name),
            development=development,
        )
        self._trajectory_prompt = _validate_prompt_text(
            trajectory_prompt,
            source="trajectory_prompt",
            provider=cast(Any, self.config_provider_name),
            development=development,
        )
        # Fail before any rollout/reward traffic if a configured prompt does
        # not satisfy the finalized RLFF interpolation contract.  Named-only
        # rendering deliberately leaves the JSON examples in each prompt alone.
        render_reward_prompt(
            self._completion_prompt,
            {"character": "character", "profile": "profile", "plot": "plot"},
            required=("character", "profile", "plot"),
        )
        render_reward_prompt(
            self._trajectory_prompt,
            {"character": "character", "plot": "plot", "tasks": "[]"},
            required=("character", "plot", "tasks"),
        )
        api_key = api_key or os.getenv(self.default_api_key_env)
        if transport is None and not api_key:
            raise ValueError(f"{self.provider_name} reward provider requires an API key")
        if completion_timeout_seconds <= 0 or (
            trajectory_timeout_seconds is not None and trajectory_timeout_seconds <= 0
        ):
            raise ValueError("reward timeout_seconds must be positive")
        if completion_retries < 0 or (trajectory_retries is not None and trajectory_retries < 0):
            raise ValueError("reward retries must be non-negative")
        if completion_concurrency <= 0 or (
            trajectory_concurrency is not None and trajectory_concurrency <= 0
        ):
            raise ValueError("reward concurrency must be positive")
        if not 0 <= completion_temperature <= 2 or (
            trajectory_temperature is not None and not 0 <= trajectory_temperature <= 2
        ):
            raise ValueError("reward temperature must be between 0 and 2")
        if completion_max_tokens <= 0 or (
            trajectory_max_tokens is not None and trajectory_max_tokens <= 0
        ):
            raise ValueError("reward max_tokens must be positive")
        supported_reasoning_efforts = {"low", "medium", "high", "max"}
        if completion_reasoning_effort not in supported_reasoning_efforts or (
            trajectory_reasoning_effort is not None
            and trajectory_reasoning_effort not in supported_reasoning_efforts
        ):
            raise ValueError("reward reasoning_effort must be low, medium, high, or max")

        self._api_key = api_key or ""
        self._url = self._normalize_url(base_url)
        self._model = model
        self._completion_timeout = float(completion_timeout_seconds)
        self._trajectory_timeout = float(
            trajectory_timeout_seconds
            if trajectory_timeout_seconds is not None
            else completion_timeout_seconds
        )
        self._completion_retries = int(completion_retries)
        self._trajectory_retries = int(
            trajectory_retries if trajectory_retries is not None else completion_retries
        )
        self._completion_temperature = float(completion_temperature)
        self._trajectory_temperature = float(
            trajectory_temperature
            if trajectory_temperature is not None
            else completion_temperature
        )
        self._completion_reasoning_effort = completion_reasoning_effort
        self._trajectory_reasoning_effort = (
            trajectory_reasoning_effort
            if trajectory_reasoning_effort is not None
            else completion_reasoning_effort
        )
        self._completion_max_tokens = int(completion_max_tokens)
        self._trajectory_max_tokens = int(
            trajectory_max_tokens if trajectory_max_tokens is not None else completion_max_tokens
        )
        self._completion_semaphore = asyncio.Semaphore(int(completion_concurrency))
        self._trajectory_semaphore = asyncio.Semaphore(
            int(
                trajectory_concurrency
                if trajectory_concurrency is not None
                else completion_concurrency
            )
        )
        self._transport: RewardTransportCallable | RewardTransport = (
            transport or _default_reward_transport
        )
        self._tracer = tracer
        if self._tracer is None and langsmith_api_key:
            self._tracer = LangSmithTracer(
                project=langsmith_project,
                api_key=langsmith_api_key,
            )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        api_key: str | None = None,
        transport: RewardTransportCallable | RewardTransport | None = None,
        tracer: Any = None,
        langsmith_api_key: str | None = None,
        development: bool = False,
    ) -> DeepSeekRewardProvider:
        return cls(
            config=config,
            api_key=api_key,
            transport=transport,
            tracer=tracer,
            langsmith_api_key=langsmith_api_key,
            development=development,
        )

    async def aclose(self) -> None:
        """Close a custom transport if it provides an async close hook."""

        close = getattr(self._transport, "aclose", None)
        close = close or getattr(self._transport, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def score_completion_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> tuple[CompletionReward, ...]:
        completions = _trajectory_character_completions(trajectory, character)
        payload = build_completion_reward_payload(episode, trajectory, character)
        raw, parsed, error = await self._request_reward(
            scope="completion_local",
            prompt=self._completion_prompt,
            model=self._model,
            payload=payload,
            timeout_seconds=self._completion_timeout,
            retries=self._completion_retries,
            semaphore=self._completion_semaphore,
            identifiers={
                "episode_id": episode.episode_id or "",
                "group_id": trajectory.group_id,
                "trajectory_id": trajectory.trajectory_id,
                "character": character,
            },
        )
        if parsed is None:
            return tuple(
                CompletionReward(
                    completion_id=completion.completion_id,
                    trajectory_id=trajectory.trajectory_id,
                    group_id=trajectory.group_id,
                    character=completion.character,
                    status=RewardStatus.INVALID,
                    provider=self.provider_name,
                    raw_response=raw,
                    error=error or "reward response was invalid after retries",
                )
                for completion in completions
            )
        completion_parsed = cast(CompletionRewardResponse, parsed)
        effective_values = tuple(
            completion_effective_values(score, completion.text)
            for completion, score in zip(
                completions,
                completion_parsed.scores,
                strict=True,
            )
        )
        scalar_rewards = tuple(sum(values) / len(values) for values in effective_values)
        return tuple(
            CompletionReward(
                completion_id=completion.completion_id,
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                character=completion.character,
                reward=scalar,
                dimensions=tuple(
                    RewardDimension(name=name, value=float(value))
                    for name, value in zip(
                        COMPLETION_DIMENSIONS,
                        values,
                        strict=True,
                    )
                ),
                provider=self.provider_name,
                raw_response=raw,
            )
            for completion, values, scalar in zip(
                completions,
                effective_values,
                scalar_rewards,
                strict=True,
            )
        )

    async def score_trajectory(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        payload = build_trajectory_reward_payload(episode, trajectory, character)
        tasks = tuple(cast(Sequence[str], payload["tasks"]))
        if not tasks:
            return TrajectoryReward(
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                character=character,
                reward=0.0,
                provider=self.provider_name,
                raw_response=json.dumps(
                    {"skipped": True, "reason": "target character has no tasks"},
                    ensure_ascii=False,
                ),
            )
        raw, parsed, error = await self._request_reward(
            scope="trajectory_role",
            prompt=self._trajectory_prompt,
            model=self._trajectory_model,
            payload=payload,
            timeout_seconds=self._trajectory_timeout,
            retries=self._trajectory_retries,
            semaphore=self._trajectory_semaphore,
            identifiers={
                "episode_id": episode.episode_id or "",
                "group_id": trajectory.group_id,
                "trajectory_id": trajectory.trajectory_id,
                "character": character,
            },
        )
        if parsed is None:
            return TrajectoryReward(
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                character=character,
                status=RewardStatus.INVALID,
                provider=self.provider_name,
                raw_response=raw,
                error=error or "reward response was invalid after retries",
            )
        trajectory_parsed = cast(TrajectoryRewardResponse, parsed)
        return TrajectoryReward(
            trajectory_id=trajectory.trajectory_id,
            group_id=trajectory.group_id,
            character=character,
            reward=trajectory_response_reward(trajectory_parsed),
            dimensions=tuple(
                RewardDimension(name=task.task, value=float(task.value))
                for task in trajectory_parsed.score
            ),
            provider=self.provider_name,
            raw_response=raw,
        )

    async def score_proxy_completion_role(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, float]:
        """Score all target-role replies before official token export."""

        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise RewardResponseError("proxy completion payload is missing ids")
        completion_ids_value = identifiers.get("completion_ids")
        if not isinstance(completion_ids_value, Sequence) or isinstance(
            completion_ids_value, (str, bytes)
        ):
            raise RewardResponseError("proxy completion_ids must be a sequence")
        completion_ids = tuple(str(value) for value in completion_ids_value)
        if not completion_ids or len(set(completion_ids)) != len(completion_ids):
            raise RewardResponseError("proxy completion_ids must be non-empty and unique")
        completion_texts_value = payload.get("completion_texts")
        if not isinstance(completion_texts_value, Sequence) or isinstance(
            completion_texts_value, (str, bytes)
        ):
            raise RewardResponseError("proxy completion_texts must be a sequence")
        completion_texts = tuple(str(value) for value in completion_texts_value)
        if len(completion_texts) != len(completion_ids):
            raise RewardResponseError(
                "proxy completion_texts must exactly cover completion_ids"
            )
        _raw, parsed, error = await self._request_reward(
            scope="completion_local",
            prompt=self._completion_prompt,
            model=self._model,
            payload=payload,
            timeout_seconds=self._completion_timeout,
            retries=self._completion_retries,
            semaphore=self._completion_semaphore,
            identifiers={str(key): str(value) for key, value in identifiers.items()},
        )
        if parsed is None:
            raise RewardResponseError(error or "proxy completion reward response was invalid")
        rewards = completion_response_rewards(
            cast(CompletionRewardResponse, parsed),
            reply_texts=completion_texts,
        )
        return dict(zip(completion_ids, rewards, strict=True))

    async def score_proxy_trajectory_role(self, payload: Mapping[str, Any]) -> float:
        """Score one target role over a complete proxy trajectory."""

        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise RewardResponseError("proxy trajectory payload is missing ids")
        tasks_value = payload.get("tasks", ())
        if not isinstance(tasks_value, Sequence) or isinstance(tasks_value, (str, bytes)):
            raise RewardResponseError("proxy trajectory payload tasks must be a sequence")
        tasks = tuple(str(task) for task in tasks_value)
        if not tasks:
            return 0.0
        _raw, parsed, error = await self._request_reward(
            scope="trajectory_role",
            prompt=self._trajectory_prompt,
            model=self._trajectory_model,
            payload=payload,
            timeout_seconds=self._trajectory_timeout,
            retries=self._trajectory_retries,
            semaphore=self._trajectory_semaphore,
            identifiers={str(key): str(value) for key, value in identifiers.items()},
        )
        if parsed is None:
            raise RewardResponseError(error or "proxy trajectory reward response was invalid")
        return trajectory_response_reward(cast(TrajectoryRewardResponse, parsed))

    async def score_trajectory_role(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        character: str,
    ) -> TrajectoryReward:
        return await self.score_trajectory(episode, trajectory, character)

    async def score_local(
        self,
        episode: EpisodeRecord,
        trajectory: Trajectory,
        completion: CompletionTrace,
    ) -> CompletionReward:
        return await self.score_completion(episode, trajectory, completion)

    def _normalize_url(self, base_url: str) -> str:
        root = base_url.rstrip("/")
        return root if root.endswith(CHAT_COMPLETIONS_PATH) else root + CHAT_COMPLETIONS_PATH

    def _build_request_payload(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        reasoning_effort: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the provider-specific wire payload for one reward request."""

        return {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }

    async def _request_reward(
        self,
        *,
        scope: Literal["completion_local", "trajectory_role"],
        prompt: str,
        model: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        retries: int,
        semaphore: asyncio.Semaphore,
        identifiers: Mapping[str, str],
    ) -> tuple[
        str | None,
        CompletionRewardResponse | TrajectoryRewardResponse | None,
        str | None,
    ]:
        template_args_value = payload.get("template_args")
        if not isinstance(template_args_value, Mapping):
            raise RewardPromptError("reward payload is missing template_args")
        if any(not isinstance(value, str) for value in template_args_value.values()):
            raise RewardPromptError("reward template arguments must all be strings")
        template_args = {str(key): cast(str, value) for key, value in template_args_value.items()}
        required_variables = (
            ("character", "profile", "plot")
            if scope == "completion_local"
            else ("character", "plot", "tasks")
        )
        rendered_prompt = render_reward_prompt(
            prompt,
            template_args,
            required=required_variables,
        )
        user_input = payload.get("input")
        if not isinstance(user_input, Mapping):
            raise RewardResponseError("reward payload is missing input JSON")
        expected_tasks_value = payload.get("tasks", ())
        if not isinstance(expected_tasks_value, Sequence) or isinstance(
            expected_tasks_value, (str, bytes)
        ):
            raise RewardResponseError("reward payload tasks must be a sequence")
        expected_tasks = tuple(str(task) for task in expected_tasks_value)
        expected_completion_count: int | None = None
        if scope == "completion_local":
            identifiers_value = payload.get("ids")
            if not isinstance(identifiers_value, Mapping):
                raise RewardResponseError("completion reward payload is missing ids")
            completion_ids_value = identifiers_value.get("completion_ids")
            if not isinstance(completion_ids_value, Sequence) or isinstance(
                completion_ids_value, (str, bytes)
            ):
                raise RewardResponseError("completion reward payload completion_ids is invalid")
            completion_ids = tuple(str(value) for value in completion_ids_value)
            if not completion_ids or len(set(completion_ids)) != len(completion_ids):
                raise RewardResponseError(
                    "completion reward payload completion_ids must be non-empty and unique"
                )
            expected_completion_count = len(completion_ids)
        messages = _json_payload(
            [
                {"role": "system", "content": rendered_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_input, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        request_payload = self._build_request_payload(
            model=model,
            messages=messages,
            temperature=(
                self._completion_temperature
                if scope == "completion_local"
                else self._trajectory_temperature
            ),
            reasoning_effort=(
                self._completion_reasoning_effort
                if scope == "completion_local"
                else self._trajectory_reasoning_effort
            ),
            max_tokens=int(
                self._completion_max_tokens
                if scope == "completion_local"
                else self._trajectory_max_tokens
            ),
        )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        raw_response: str | None = None
        last_error: str | None = None
        run_id: str | None = None
        trace_finished = False
        if self._tracer is not None:
            run_id = self._trace_start(
                scope,
                {
                    "protocol_payload": dict(payload),
                    "request": request_payload,
                },
            )
        try:
            async with semaphore:
                for attempt in range(retries + 1):
                    try:
                        result = await asyncio.wait_for(
                            _call_transport(
                                self._transport,
                                url=self._url,
                                headers=headers,
                                payload=request_payload,
                                timeout_seconds=timeout_seconds,
                            ),
                            timeout=timeout_seconds,
                        )
                        response = _response_from_transport(result)
                        raw_response = _redact(response.text, [self._api_key])
                        if not 200 <= response.status_code < 300:
                            raise RewardTransportError(
                                f"{self.provider_name} HTTP status {response.status_code}: "
                                f"{raw_response}"
                            )
                        model_text = self._extract_model_text(raw_response)
                        if scope == "completion_local":
                            completion_parsed = parse_completion_reward_response(
                                model_text,
                                expected_count=expected_completion_count,
                            )
                            parsed: CompletionRewardResponse | TrajectoryRewardResponse = (
                                completion_parsed
                            )
                            completion_texts_value = payload.get("completion_texts")
                            reply_texts = (
                                tuple(str(value) for value in completion_texts_value)
                                if isinstance(completion_texts_value, Sequence)
                                and not isinstance(completion_texts_value, (str, bytes))
                                else None
                            )
                            reply_rewards = completion_response_rewards(
                                completion_parsed,
                                reply_texts=reply_texts,
                            )
                            scalar_reward = sum(reply_rewards) / len(reply_rewards)
                        else:
                            parsed = parse_trajectory_reward_response(
                                model_text,
                                expected_tasks=expected_tasks,
                            )
                            scalar_reward = trajectory_response_reward(parsed)
                        if self._tracer is not None:
                            self._trace_finish(
                                run_id,
                                outputs={"raw_response": raw_response, "reward": scalar_reward},
                            )
                            trace_finished = True
                        return raw_response, parsed, None
                    except TimeoutError:
                        last_error = (
                            f"reward attempt {attempt + 1} timed out after {timeout_seconds}s"
                        )
                    except (
                        RewardTransportError,
                        RewardResponseError,
                        ValueError,
                        TypeError,
                    ) as exc:
                        last_error = _redact(str(exc), [self._api_key])
                    except Exception as exc:  # network/client errors are retryable too
                        last_error = _redact(
                            f"{type(exc).__name__}: {exc}",
                            [self._api_key],
                        )
                    if attempt < retries:
                        await asyncio.sleep(0)
        finally:
            if self._tracer is not None and run_id is not None and not trace_finished:
                self._trace_finish(run_id, error=last_error)
        return (
            raw_response,
            None,
            (
                f"{self.provider_name} {scope} reward exhausted {retries + 1} attempts: "
                f"{last_error or 'unknown reward error'}"
            ),
        )

    @staticmethod
    def _extract_model_text(raw_response: str) -> str:
        try:
            outer = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response
        if isinstance(outer, Mapping) and "reward" in outer:
            return raw_response
        if isinstance(outer, Mapping):
            choices = outer.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, Mapping):
                    message = first.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, Mapping):
                            return json.dumps(content, ensure_ascii=False)
                    text = first.get("text")
                    if isinstance(text, str):
                        return text
        raise RewardResponseError("DeepSeek response has no choices[0].message.content envelope")

    def _trace_start(self, scope: str, payload: Mapping[str, Any]) -> str | None:
        start = getattr(self._tracer, "start", None)
        if callable(start):
            try:
                return cast(str | None, start(name=f"rlff-{scope}", inputs=payload))
            except Exception:
                return None
        return None

    def _trace_finish(
        self,
        run_id: str | None,
        *,
        outputs: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        finish = getattr(self._tracer, "finish", None)
        if callable(finish):
            try:
                finish(run_id, outputs=outputs, error=error)
            except Exception:
                return


class QwenDashScopeRewardProvider(DeepSeekRewardProvider):
    """Qwen3.7-Flash provider using DashScope's native HTTP message protocol."""

    provider_name = QWEN3_7_FLASH_PROVIDER
    config_provider_name = QWEN_DASHSCOPE_PROVIDER
    default_api_key_env = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        *,
        base_url: str = QWEN_DASHSCOPE_GENERATION_URL,
        model: str = QWEN3_7_FLASH_PROVIDER,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, model=model, **kwargs)

    def _normalize_url(self, base_url: str) -> str:
        """DashScope receives requests at the configured native generation endpoint."""

        return base_url.rstrip("/")

    def _build_request_payload(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        reasoning_effort: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        dashscope_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise RewardResponseError("reward request message role must be non-empty")
            if not isinstance(content, str):
                raise RewardResponseError("reward request message content must be text")
            dashscope_messages.append(
                {
                    "role": role,
                    "content": [{"text": content}],
                }
            )
        return {
            "model": model,
            "input": {"messages": dashscope_messages},
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
                "enable_thinking": True,
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
        }

    @staticmethod
    def _extract_model_text(raw_response: str) -> str:
        try:
            outer = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response
        if not isinstance(outer, Mapping):
            return raw_response
        if "reward" in outer:
            return raw_response
        output = outer.get("output")
        if isinstance(output, Mapping):
            choices = output.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, Mapping):
                    message = first.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            parts = [
                                str(part["text"])
                                for part in content
                                if isinstance(part, Mapping)
                                and isinstance(part.get("text"), str)
                            ]
                            if parts:
                                return "".join(parts)
            output_text = output.get("text")
            if isinstance(output_text, str):
                return output_text
        raise RewardResponseError(
            "Qwen DashScope response has no output.choices[0].message.content"
        )


def _ensure_sequence(value: Trajectory | Sequence[Trajectory]) -> tuple[Trajectory, ...]:
    return (value,) if isinstance(value, Trajectory) else tuple(value)


async def score_completions(
    provider: RewardProvider,
    episode: EpisodeRecord,
    trajectories: Trajectory | Sequence[Trajectory],
) -> tuple[CompletionReward, ...]:
    """Score each trajectory/character once and return all reply rewards."""

    active = _ensure_sequence(trajectories)
    calls = [
        provider.score_completion_role(episode, trajectory, character)
        for trajectory in active
        for character in dict.fromkeys(
            completion.character for completion in trajectory.completions
        )
    ]
    batches = tuple(await asyncio.gather(*calls)) if calls else ()
    return tuple(reward for batch in batches for reward in batch)


async def score_trajectories(
    provider: RewardProvider,
    episode: EpisodeRecord,
    trajectories: Trajectory | Sequence[Trajectory],
) -> tuple[TrajectoryReward, ...]:
    """Score every trajectory/character pair in stable source order."""

    active = _ensure_sequence(trajectories)
    calls = [
        provider.score_trajectory(episode, trajectory, character)
        for trajectory in active
        for character in dict.fromkeys(
            completion.character for completion in trajectory.completions
        )
    ]
    return tuple(await asyncio.gather(*calls)) if calls else ()


async def score_rollout_group(
    provider: RewardProvider,
    episode: EpisodeRecord,
    trajectories: Trajectory | Sequence[Trajectory],
) -> RewardBatch:
    """Score completion-local and trajectory-role rewards in stable order."""

    local, trajectory_role_rewards = await asyncio.gather(
        score_completions(provider, episode, trajectories),
        score_trajectories(provider, episode, trajectories),
    )
    return RewardBatch(local, trajectory_role_rewards)


def _invalid_reward_message(reward: Any, scope: str) -> str:
    return (
        f"invalid {scope} reward for "
        f"{getattr(reward, 'trajectory_id', '?')}/{getattr(reward, 'character', '')}: "
        f"{getattr(reward, 'error', None) or 'unspecified error'}"
    )


def aggregate_role_rewards(
    completion_rewards: Sequence[CompletionReward],
    trajectory_rewards: Sequence[TrajectoryReward],
    *,
    completion_weight: float = 0.6,
    global_weight: float = 0.4,
) -> tuple[RoleEffectiveReward, ...]:
    """Aggregate local means and weighted trajectory-role task rewards."""

    if not math.isfinite(completion_weight) or completion_weight < 0:
        raise RewardAggregationError("completion_weight must be finite and non-negative")
    if not math.isfinite(global_weight) or global_weight < 0:
        raise RewardAggregationError("global_weight must be finite and non-negative")

    role_rewards_by_key: dict[tuple[str, str], TrajectoryReward] = {}
    for trajectory_reward in trajectory_rewards:
        if trajectory_reward.status is not RewardStatus.VALID:
            raise RewardAggregationError(_invalid_reward_message(trajectory_reward, "trajectory"))
        if trajectory_reward.reward is None or not math.isfinite(trajectory_reward.reward):
            raise RewardAggregationError("valid trajectory reward has no finite scalar")
        key = (trajectory_reward.trajectory_id, trajectory_reward.character)
        if key in role_rewards_by_key:
            raise RewardAggregationError(
                "duplicate trajectory role reward "
                f"{trajectory_reward.trajectory_id!r}/{trajectory_reward.character!r}"
            )
        role_rewards_by_key[key] = trajectory_reward

    grouped: dict[tuple[str, str], list[CompletionReward]] = {}
    key_order: list[tuple[str, str]] = []
    completion_ids: set[str] = set()
    for completion_reward in completion_rewards:
        if completion_reward.status is not RewardStatus.VALID:
            raise RewardAggregationError(_invalid_reward_message(completion_reward, "completion"))
        if completion_reward.reward is None or not math.isfinite(completion_reward.reward):
            raise RewardAggregationError("valid completion reward has no finite scalar")
        if completion_reward.completion_id in completion_ids:
            raise RewardAggregationError(
                f"duplicate completion reward {completion_reward.completion_id!r}"
            )
        completion_ids.add(completion_reward.completion_id)
        key = (completion_reward.trajectory_id, completion_reward.character)
        if key not in grouped:
            grouped[key] = []
            key_order.append(key)
        grouped[key].append(completion_reward)

    records: list[RoleEffectiveReward] = []
    for trajectory_id, character in key_order:
        local = grouped[(trajectory_id, character)]
        trajectory_role_reward = role_rewards_by_key.get((trajectory_id, character))
        if trajectory_role_reward is None:
            raise RewardAggregationError(
                f"missing valid trajectory role reward for {trajectory_id!r}/{character!r}"
            )
        mean_local = sum(cast(float, item.reward) for item in local) / len(local)
        weighted_local = mean_local * completion_weight
        weighted_trajectory_role = cast(float, trajectory_role_reward.reward) * global_weight
        effective = weighted_local + weighted_trajectory_role
        if not all(
            math.isfinite(value)
            for value in (mean_local, weighted_local, weighted_trajectory_role, effective)
        ):
            raise RewardAggregationError("reward aggregation produced a non-finite value")
        records.append(
            RoleEffectiveReward(
                group_id=local[0].group_id,
                trajectory_id=trajectory_id,
                character=character,
                completion_ids=tuple(item.completion_id for item in local),
                aggregated_local_reward=weighted_local,
                trajectory_role_contribution=weighted_trajectory_role,
                effective_reward=effective,
            )
        )
    return tuple(records)


def create_reward_provider(
    config: Any,
    *,
    api_key: str | None = None,
    transport: RewardTransportCallable | RewardTransport | None = None,
    tracer: Any = None,
    langsmith_api_key: str | None = None,
    development: bool = False,
) -> RewardProvider:
    """Instantiate the explicitly configured reward provider."""

    provider = getattr(config, "provider", None)
    if provider == "placeholder":
        # Prompt paths are still a required, auditable boundary in a
        # development run.  The files must exist and be non-empty; their
        # marker remains visible rather than being silently replaced.
        load_reward_prompts(
            config.completion.prompt_path,
            config.global_reward.prompt_path,
            provider="placeholder",
            development=True,
        )
        return PlaceholderRewardProvider(config=config)
    if provider == "deepseek":
        return DeepSeekRewardProvider(
            config=config,
            api_key=api_key,
            transport=transport,
            tracer=tracer,
            langsmith_api_key=langsmith_api_key,
            development=development,
        )
    if provider == QWEN_DASHSCOPE_PROVIDER:
        return QwenDashScopeRewardProvider(
            config=config,
            api_key=api_key,
            transport=transport,
            tracer=tracer,
            langsmith_api_key=langsmith_api_key,
            development=development,
        )
    raise ValueError(f"unsupported reward provider {provider!r}")


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "COMPLETION_DIMENSIONS",
    "COMPLETION_REWARD_PROMPT_FILENAME",
    "DEEPSEEK_V4_FLASH_PROVIDER",
    "PLACEHOLDER_MARKER",
    "PLACEHOLDER_PROVIDER",
    "QWEN3_7_FLASH_PROVIDER",
    "QWEN_DASHSCOPE_GENERATION_URL",
    "QWEN_DASHSCOPE_PROVIDER",
    "REWARD_PROMPT_VERSION",
    "REWARD_REQUEST_SCHEMA_VERSION",
    "REWARD_RESPONSE_SCHEMA_VERSION",
    "TRAJECTORY_REWARD_PROMPT_FILENAME",
    "CompletionRewardResponse",
    "CompletionScore",
    "DeepSeekRewardProvider",
    "LangSmithTracer",
    "PlaceholderRewardProvider",
    "QwenDashScopeRewardProvider",
    "RewardAggregationError",
    "RewardBatch",
    "RewardError",
    "RewardHTTPResponse",
    "RewardPromptError",
    "RewardPrompts",
    "RewardProvider",
    "RewardResponseError",
    "RewardTransport",
    "RewardTransportError",
    "TrajectoryRewardResponse",
    "TrajectoryTaskScore",
    "aggregate_role_rewards",
    "build_completion_reward_payload",
    "build_proxy_completion_reward_payload",
    "build_proxy_trajectory_reward_payload",
    "build_trajectory_reward_payload",
    "completion_effective_values",
    "create_reward_provider",
    "load_reward_prompt",
    "load_reward_prompts",
    "parse_completion_reward_response",
    "parse_reward_response",
    "parse_trajectory_reward_response",
    "render_reward_prompt",
    "score_completions",
    "score_rollout_group",
    "score_trajectories",
]
