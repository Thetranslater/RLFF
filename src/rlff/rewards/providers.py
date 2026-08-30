"""Production DeepSeek and Qwen reward providers.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from ..contracts import (
    CompletionReward,
    CompletionTrace,
    EpisodeRecord,
    RewardDimension,
    RewardStatus,
    Trajectory,
    TrajectoryReward,
)
from .payloads import (
    _trajectory_character_completions,
    build_completion_reward_payload,
    build_trajectory_reward_payload,
)
from .prompts import (
    _reward_response_debug,
    _validate_prompt_text,
    load_reward_prompt,
    parse_completion_reward_response,
    parse_trajectory_reward_response,
    render_reward_prompt,
)
from .protocol import (
    CHAT_COMPLETIONS_PATH,
    COMPLETION_DIMENSIONS,
    DEEPSEEK_V4_FLASH_PROVIDER,
    DEFAULT_REPAIR_PROMPT,
    QWEN3_7_FLASH_PROVIDER,
    QWEN_DASHSCOPE_GENERATION_URL,
    QWEN_DASHSCOPE_PROVIDER,
    CompletionRewardResponse,
    RewardPromptError,
    RewardResponseError,
    RewardTransportError,
    TrajectoryRewardResponse,
)
from .provider_base import _RewardProviderBase
from .response_scoring import (
    completion_effective_values,
    completion_response_rewards,
    trajectory_response_reward,
)
from .transport import (
    LangSmithTracer,
    _call_transport,
    _default_reward_transport,
    _json_payload,
    _redact,
    _response_from_transport,
)
from .transport_types import RewardTransport, RewardTransportCallable


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
        repair_prompt: str | None = None,
        completion_prompt_path: str | Path | None = None,
        trajectory_prompt_path: str | Path | None = None,
        repair_prompt_path: str | Path | None = None,
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
        audit_jsonl: str | Path | None = None,
        detail_jsonl: str | Path | None = None,
        detail_sample_rate: float = 0.0,
        failure_jsonl: str | Path | None = None,
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
            repair_prompt = load_reward_prompt(
                config.repair_prompt_path,
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

        if completion_prompt is None or trajectory_prompt is None or repair_prompt is None:
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
            if repair_prompt is None and repair_prompt_path is not None:
                repair_prompt = load_reward_prompt(
                    repair_prompt_path,
                    provider=cast(Any, self.config_provider_name),
                    development=development,
                )
        if repair_prompt is None and config is None:
            repair_prompt = DEFAULT_REPAIR_PROMPT
        if completion_prompt is None or trajectory_prompt is None or repair_prompt is None:
            raise RewardPromptError(
                f"{self.provider_name} provider requires non-empty completion_prompt "
                "trajectory_prompt, and repair_prompt"
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
        self._repair_prompt = _validate_prompt_text(
            repair_prompt,
            source="repair_prompt",
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
        render_reward_prompt(
            self._repair_prompt,
            {
                "json_result": "{}",
                "error_message": "error",
                "character": "character",
                "utters": "[]",
            },
            required=("json_result", "error_message", "character", "utters"),
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
            trajectory_temperature if trajectory_temperature is not None else completion_temperature
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
        from ..observability import RewardAuditWriter

        self._audit_writer = RewardAuditWriter(
            audit_jsonl=audit_jsonl,
            detail_jsonl=detail_jsonl,
            detail_sample_rate=detail_sample_rate,
            failure_jsonl=failure_jsonl,
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
            raise RewardResponseError("proxy completion_texts must exactly cover completion_ids")
        raw, parsed, error = await self._request_reward(
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
            self._write_proxy_reward_audit(
                scope="completion_local",
                payload=payload,
                raw_response=raw,
                parsed=None,
                rewards=None,
                error=error,
            )
            raise RewardResponseError(error or "proxy completion reward response was invalid")
        completion_parsed = cast(CompletionRewardResponse, parsed)
        rewards = completion_response_rewards(
            completion_parsed,
            reply_texts=completion_texts,
        )
        self._write_proxy_reward_audit(
            scope="completion_local",
            payload=payload,
            raw_response=raw,
            parsed=completion_parsed,
            rewards=rewards,
            error=None,
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
            self._write_proxy_reward_audit(
                scope="trajectory_role",
                payload=payload,
                raw_response=None,
                parsed=None,
                rewards=(0.0,),
                error=None,
                status="skipped",
            )
            return 0.0
        raw, parsed, error = await self._request_reward(
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
            self._write_proxy_reward_audit(
                scope="trajectory_role",
                payload=payload,
                raw_response=raw,
                parsed=None,
                rewards=None,
                error=error,
            )
            raise RewardResponseError(error or "proxy trajectory reward response was invalid")
        trajectory_parsed = cast(TrajectoryRewardResponse, parsed)
        reward = trajectory_response_reward(trajectory_parsed)
        self._write_proxy_reward_audit(
            scope="trajectory_role",
            payload=payload,
            raw_response=raw,
            parsed=trajectory_parsed,
            rewards=(reward,),
            error=None,
        )
        return reward

    def _write_proxy_reward_audit(
        self,
        *,
        scope: Literal["completion_local", "trajectory_role"],
        payload: Mapping[str, Any],
        raw_response: str | None,
        parsed: CompletionRewardResponse | TrajectoryRewardResponse | None,
        rewards: Sequence[float] | None,
        error: str | None,
        status: str | None = None,
    ) -> None:
        """Record the exact post-parse reward used by the proxy training path."""

        identifiers_value = payload.get("ids", {})
        identifiers = (
            {str(key): value for key, value in identifiers_value.items()}
            if isinstance(identifiers_value, Mapping)
            else {}
        )
        trajectory_id = str(identifiers.get("trajectory_id", ""))
        resolved_status = status or ("ok" if parsed is not None else "invalid")
        record: dict[str, Any] = {
            "provider": self.provider_name,
            "scope": scope,
            "status": resolved_status,
            "ids": identifiers,
            "model": self._model if scope == "completion_local" else self._trajectory_model,
        }
        if error:
            record["error"] = error
        if scope == "completion_local" and isinstance(parsed, CompletionRewardResponse):
            completion_ids = tuple(str(value) for value in identifiers.get("completion_ids", ()))
            reply_texts_value = payload.get("completion_texts", ())
            reply_texts = tuple(str(value) for value in reply_texts_value)
            effective_values = tuple(
                completion_effective_values(score, text)
                for score, text in zip(parsed.scores, reply_texts, strict=True)
            )
            record["completion_rewards"] = [
                {
                    "completion_id": completion_id,
                    "model_values": list(score.values),
                    "effective_values": list(values),
                    "reward": reward,
                }
                for completion_id, score, values, reward in zip(
                    completion_ids,
                    parsed.scores,
                    effective_values,
                    rewards or (),
                    strict=True,
                )
            ]
        elif scope == "trajectory_role":
            record["reward"] = float(rewards[0]) if rewards else None
            record["task_rewards"] = (
                [item.model_dump(mode="json") for item in parsed.score]
                if isinstance(parsed, TrajectoryRewardResponse)
                else []
            )
        self._audit_writer.write_reward(record)
        if raw_response is not None:
            self._audit_writer.write_detail(
                trajectory_id=trajectory_id,
                record={
                    **record,
                    "protocol_payload": dict(payload),
                    "parsed_response": (
                        parsed.model_dump(mode="json") if parsed is not None else None
                    ),
                    "raw_response": raw_response,
                },
            )

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

    def _completion_repair_message(
        self,
        *,
        payload: Mapping[str, Any],
        json_result: str,
        error_message: str,
    ) -> dict[str, str]:
        identifiers = payload.get("ids")
        if not isinstance(identifiers, Mapping):
            raise RewardResponseError("completion repair payload is missing ids")
        character = str(identifiers.get("character", "")).strip()
        utters = payload.get("utters")
        if (
            not character
            or not isinstance(utters, Sequence)
            or isinstance(utters, (str, bytes, bytearray))
        ):
            raise RewardResponseError("completion repair payload has invalid indexed utters")
        rendered = render_reward_prompt(
            self._repair_prompt,
            {
                "json_result": json_result,
                "error_message": error_message,
                "character": character,
                "utters": json.dumps(list(utters), ensure_ascii=False, indent=2),
            },
            required=("json_result", "error_message", "character", "utters"),
        )
        return {"role": "user", "content": rendered}

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
        attempt_records: list[dict[str, Any]] = []
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
                    repair_result: str | None = None
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt + 1,
                        "request": request_payload,
                    }
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
                        attempt_record["status_code"] = response.status_code
                        attempt_record["raw_response"] = raw_response
                        attempt_record.update(_reward_response_debug(raw_response))
                        if not 200 <= response.status_code < 300:
                            raise RewardTransportError(
                                f"{self.provider_name} HTTP status {response.status_code}: "
                                f"{raw_response}"
                            )
                        model_text = self._extract_model_text(raw_response)
                        repair_result = model_text
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
                    attempt_record["error"] = last_error
                    attempt_records.append(attempt_record)
                    if attempt < retries:
                        if scope == "completion_local" and repair_result is not None:
                            messages.extend(
                                [
                                    {"role": "assistant", "content": repair_result},
                                    self._completion_repair_message(
                                        payload=payload,
                                        json_result=repair_result,
                                        error_message=(
                                            last_error
                                            or "invalid completion reward response"
                                        ),
                                    ),
                                ]
                            )
                            request_payload = self._build_request_payload(
                                model=model,
                                messages=messages,
                                temperature=self._completion_temperature,
                                reasoning_effort=self._completion_reasoning_effort,
                                max_tokens=self._completion_max_tokens,
                            )
                        await asyncio.sleep(0)
        finally:
            if self._tracer is not None and run_id is not None and not trace_finished:
                self._trace_finish(run_id, error=last_error)
        terminal_error = (
            f"{self.provider_name} {scope} reward exhausted {retries + 1} attempts: "
            f"{last_error or 'unknown reward error'}"
        )
        payload_ids = payload.get("ids")
        self._audit_writer.write_failure(
            {
                "provider": self.provider_name,
                "scope": scope,
                "status": "invalid",
                "model": model,
                "ids": dict(payload_ids) if isinstance(payload_ids, Mapping) else dict(identifiers),
                "error": terminal_error,
                "protocol_payload": dict(payload),
                "attempts": attempt_records,
            }
        )
        return (
            raw_response,
            None,
            terminal_error,
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
                                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                            ]
                            if parts:
                                return "".join(parts)
            output_text = output.get("text")
            if isinstance(output_text, str):
                return output_text
        raise RewardResponseError(
            "Qwen DashScope response has no output.choices[0].message.content"
        )
