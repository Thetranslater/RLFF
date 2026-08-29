"""AReaL OpenAI-proxy group-aware rollout agent.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ..rollout import rounds_for_character_count
from .grouping import _group_character_order, logger, reward_weights_for_step
from .normalization import (
    _mapping_value,
    decode_proxy_episode_payload,
    normalize_proxy_role_advantages,
)
from .types import (
    DETERMINISTIC_AUDIT_REWARD_PROVIDER,
    DeterministicAuditRewardProvider,
    ProxyCompletionView,
    ProxyGroupError,
    ProxyRewardProvider,
    ProxyTrajectoryRunner,
    ProxyTrajectoryView,
)


@dataclass(slots=True)
class _GroupState:
    expected: int
    future: asyncio.Future[dict[str, float]]
    reward_step: int
    trajectories: dict[int, ProxyTrajectoryView] = field(default_factory=dict)
    next_slot: int = 0
    finished_runs: int = 0
    failed: bool = False


class RLFFGroupAwareAgent:
    """Inline AReaL agent that synchronizes one native rollout group."""

    # RLFFPPOTrainer uses this marker as an explicit v1.0.4 startup guard;
    # the official OpenAIProxyWorkflow still owns the actual proxy sessions.
    requires_rlff_proxy_start = True

    def __init__(
        self,
        *,
        group_size: int = 4,
        max_rounds: int = 7,
        model: str = "",
        temperature: float = 0.9,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        max_new_tokens: int = 256,
        group_timeout_seconds: float = 600.0,
        rollout_request_timeout_seconds: float = 120.0,
        completion_weight: float = 0.6,
        global_weight: float = 0.4,
        reward_schedule_start_step: int | None = None,
        reward_schedule_end_step: int | None = None,
        reward_schedule_completion_end_weight: float | None = None,
        reward_schedule_global_end_weight: float | None = None,
        reward_schedule_initial_step: int = 0,
        reward_schedule_use_proxy_version: bool = False,
        min_group_size: int = 2,
        reward_std_epsilon: float = 1e-8,
        system_prompt: str = "",
        reward_provider: ProxyRewardProvider | None = None,
        reward_provider_name: str = "qwen_dashscope",
        reward_api_key_env: str = "DASHSCOPE_API_KEY",
        reward_base_url: str = (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        ),
        completion_prompt_path: str | None = None,
        trajectory_prompt_path: str | None = None,
        reward_repair_prompt_path: str | None = None,
        reward_model: str = "qwen3.7-flash",
        trajectory_reward_model: str | None = None,
        completion_reward_timeout_seconds: float = 120.0,
        trajectory_reward_timeout_seconds: float | None = None,
        completion_reward_retries: int = 2,
        trajectory_reward_retries: int | None = None,
        completion_reward_concurrency: int = 4,
        trajectory_reward_concurrency: int | None = None,
        completion_reward_temperature: float = 1.0,
        trajectory_reward_temperature: float | None = None,
        completion_reward_reasoning_effort: str = "low",
        trajectory_reward_reasoning_effort: str | None = None,
        completion_reward_max_tokens: int = 25000,
        trajectory_reward_max_tokens: int | None = None,
        langsmith_tracing: bool = False,
        langsmith_project: str = "rlff",
        langsmith_api_key_env: str = "LANGSMITH_API_KEY",
        reward_audit_jsonl: str | None = None,
        reward_detail_jsonl: str | None = None,
        reward_detail_sample_rate: float = 0.0,
        reward_failure_jsonl: str | None = None,
        trajectory_runner: ProxyTrajectoryRunner | None = None,
    ) -> None:
        if type(group_size) is not int or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if group_timeout_seconds <= 0:
            raise ValueError("group_timeout_seconds must be positive")
        if rollout_request_timeout_seconds <= 0:
            raise ValueError("rollout_request_timeout_seconds must be positive")
        if not -2 <= frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        self.group_size = group_size
        self.max_rounds = max_rounds
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.max_new_tokens = max_new_tokens
        self.group_timeout_seconds = group_timeout_seconds
        self.rollout_request_timeout_seconds = rollout_request_timeout_seconds
        self.completion_weight = completion_weight
        self.global_weight = global_weight
        self._reward_schedule_start_step = reward_schedule_start_step
        self._reward_schedule_end_step = reward_schedule_end_step
        self._reward_schedule_completion_end_weight = reward_schedule_completion_end_weight
        self._reward_schedule_global_end_weight = reward_schedule_global_end_weight
        self._reward_schedule_step = reward_schedule_initial_step
        self._reward_schedule_use_proxy_version = reward_schedule_use_proxy_version
        reward_weights_for_step(
            reward_schedule_initial_step,
            completion_start_weight=completion_weight,
            global_start_weight=global_weight,
            schedule_start_step=reward_schedule_start_step,
            schedule_end_step=reward_schedule_end_step,
            completion_end_weight=reward_schedule_completion_end_weight,
            global_end_weight=reward_schedule_global_end_weight,
        )
        self.min_group_size = min_group_size
        self.reward_std_epsilon = reward_std_epsilon
        self.system_prompt = system_prompt
        self._reward_provider = reward_provider
        self._reward_provider_name = reward_provider_name
        self._reward_api_key_env = reward_api_key_env
        self._reward_base_url = reward_base_url
        self._completion_prompt_path = completion_prompt_path
        self._trajectory_prompt_path = trajectory_prompt_path
        self._reward_repair_prompt_path = reward_repair_prompt_path
        self._reward_model = reward_model
        self._trajectory_reward_model = trajectory_reward_model
        self._completion_reward_timeout_seconds = completion_reward_timeout_seconds
        self._trajectory_reward_timeout_seconds = trajectory_reward_timeout_seconds
        self._completion_reward_retries = completion_reward_retries
        self._trajectory_reward_retries = trajectory_reward_retries
        self._completion_reward_concurrency = completion_reward_concurrency
        self._trajectory_reward_concurrency = trajectory_reward_concurrency
        self._completion_reward_temperature = completion_reward_temperature
        self._trajectory_reward_temperature = trajectory_reward_temperature
        self._completion_reward_reasoning_effort = completion_reward_reasoning_effort
        self._trajectory_reward_reasoning_effort = trajectory_reward_reasoning_effort
        self._completion_reward_max_tokens = completion_reward_max_tokens
        self._trajectory_reward_max_tokens = trajectory_reward_max_tokens
        self._langsmith_tracing = langsmith_tracing
        self._langsmith_project = langsmith_project
        self._langsmith_api_key_env = langsmith_api_key_env
        self._reward_audit_jsonl = reward_audit_jsonl
        self._reward_detail_jsonl = reward_detail_jsonl
        self._reward_detail_sample_rate = reward_detail_sample_rate
        self._reward_failure_jsonl = reward_failure_jsonl
        self._trajectory_runner = trajectory_runner
        self._states: dict[str, _GroupState] = {}
        self._states_lock = asyncio.Lock()
        self._reward_schedule_lock = asyncio.Lock()

    async def run(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
    ) -> dict[str, float]:
        """Run one trajectory and wait for all group peers to be scored."""

        key = self._group_key(data)
        reward_step = await self._resolve_reward_step(base_url, http_client)
        state = await self._get_state(key, reward_step)
        async with self._states_lock:
            if state.next_slot >= state.expected:
                error = ProxyGroupError(f"too many proxy runs for group {key!r}")
                self._fail_state(state, error)
                raise error
            slot = state.next_slot
            state.next_slot += 1
        trajectory_id = f"{key}:trajectory:{slot}"
        try:
            if state.failed:
                raise ProxyGroupError(f"proxy group {key!r} already failed")
            trajectory = await self._run_trajectory(
                data,
                base_url=base_url,
                http_client=http_client,
                api_key=api_key,
                group_id=self._group_id(data),
                trajectory_id=trajectory_id,
            )
            async with self._states_lock:
                if state.failed:
                    raise ProxyGroupError("proxy group already failed")
                state.trajectories[slot] = trajectory
                if len(state.trajectories) == state.expected:
                    task = asyncio.create_task(self._score_group(state))
                    task.add_done_callback(self._consume_task_exception)
            try:
                rewards = await asyncio.wait_for(
                    asyncio.shield(state.future), timeout=self.group_timeout_seconds
                )
            except TimeoutError as exc:
                error = ProxyGroupError(
                    f"proxy group {key!r} did not reach {state.expected} trajectories "
                    f"within {self.group_timeout_seconds}s"
                )
                async with self._states_lock:
                    self._fail_state(state, error)
                raise error from exc
            return {
                completion.completion_id: rewards[completion.completion_id]
                for completion in trajectory.completions
            }
        except asyncio.CancelledError:
            async with self._states_lock:
                self._fail_state(state, ProxyGroupError("proxy group run was cancelled"))
            raise
        except Exception as exc:
            error = exc if isinstance(exc, ProxyGroupError) else ProxyGroupError(str(exc))
            async with self._states_lock:
                self._fail_state(state, error)
            if error is exc:
                raise
            raise error from exc
        finally:
            async with self._states_lock:
                state.finished_runs += 1
                if (
                    state.next_slot >= state.expected
                    and state.finished_runs >= state.expected
                    and self._states.get(key) is state
                ):
                    self._states.pop(key, None)

    async def generate_trajectory(
        self,
        data: Mapping[str, Any],
        *,
        base_url: str,
        http_client: Any = None,
        api_key: str = "EMPTY",
        group_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> ProxyTrajectoryView:
        """Generate one trajectory through the exact training rollout path.

        This public boundary intentionally skips only the native group barrier,
        reward-model scoring, and advantage normalization.  Prompt rendering,
        character round-robin order, history accumulation, sampling arguments,
        and the OpenAI-compatible transport are shared with :meth:`run`.
        """

        payload = dict(data)
        resolved_group_id = group_id or self._group_id(payload)
        resolved_trajectory_id = trajectory_id or f"{resolved_group_id}:trajectory:smoke"
        return await self._run_trajectory(
            payload,
            base_url=base_url,
            http_client=http_client,
            api_key=api_key,
            group_id=resolved_group_id,
            trajectory_id=resolved_trajectory_id,
        )

    def _group_id(self, data: Mapping[str, Any]) -> str:
        value = data.get("group_id")
        if value is None:
            episode = decode_proxy_episode_payload(data)
            value = _mapping_value(episode, "episode_id", "episode")
        if not isinstance(value, str) or not value.strip():
            raise ProxyGroupError("rollout data must contain a non-empty group/episode ID")
        return value

    def _group_key(self, data: Mapping[str, Any]) -> str:
        try:
            from areal.infra import workflow_context

            task_id = workflow_context.get().task_id
        except Exception:  # CPU fake/unit-test path
            task_id = None
        return f"{task_id if task_id is not None else 'local'}:{self._group_id(data)}"

    async def _resolve_reward_step(self, base_url: str, http_client: Any) -> int:
        if not self._reward_schedule_use_proxy_version:
            return self._reward_schedule_step
        try:
            response = await http_client.post(
                f"{base_url.rstrip('/')}/call",
                json={"method": "get_version", "args": [], "kwargs": {}},
            )
            response.raise_for_status()
            payload = response.json()
            version = payload.get("result") if isinstance(payload, Mapping) else None
        except Exception as exc:
            raise ProxyGroupError("cannot read AReaL rollout model version") from exc
        if type(version) is not int or version < 0:
            raise ProxyGroupError("AReaL rollout model version must be a non-negative integer")
        return version

    async def _get_state(self, key: str, reward_step: int) -> _GroupState:
        async with self._states_lock:
            state = self._states.get(key)
            if state is None:
                future: asyncio.Future[dict[str, float]] = (
                    asyncio.get_running_loop().create_future()
                )
                state = _GroupState(
                    expected=self.group_size,
                    future=future,
                    reward_step=reward_step,
                )
                future.add_done_callback(self._consume_future_exception)
                self._states[key] = state
            elif state.reward_step != reward_step:
                error = ProxyGroupError("one proxy group observed multiple model versions")
                self._fail_state(state, error)
                raise error
            return state

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[dict[str, float]]) -> None:
        if future.cancelled():
            return
        if future.exception() is not None:
            _ = future.exception()

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            _ = task.exception()

    @staticmethod
    def _fail_state(state: _GroupState, error: BaseException) -> None:
        state.failed = True
        if not state.future.done():
            state.future.set_exception(error)

    async def _score_group(self, state: _GroupState) -> None:
        async with self._reward_schedule_lock:
            await self._score_group_at_step(state)

    async def _score_group_at_step(self, state: _GroupState) -> None:
        try:
            reward_step = state.reward_step
            completion_weight, global_weight = reward_weights_for_step(
                reward_step,
                completion_start_weight=self.completion_weight,
                global_start_weight=self.global_weight,
                schedule_start_step=self._reward_schedule_start_step,
                schedule_end_step=self._reward_schedule_end_step,
                completion_end_weight=self._reward_schedule_completion_end_weight,
                global_end_weight=self._reward_schedule_global_end_weight,
            )
            trajectories = tuple(state.trajectories[index] for index in range(state.expected))
            provider = self._get_reward_provider()
            completion_payloads = [
                self._proxy_completion_payload(trajectory, character)
                for trajectory in trajectories
                for character in dict.fromkeys(
                    completion.character for completion in trajectory.completions
                )
            ]
            trajectory_role_payloads = [
                self._proxy_trajectory_payload(trajectory, character)
                for trajectory in trajectories
                for character in dict.fromkeys(
                    completion.character for completion in trajectory.completions
                )
            ]
            completion_values, trajectory_values = await asyncio.gather(
                asyncio.gather(
                    *(
                        provider.score_proxy_completion_role(payload)
                        for payload in completion_payloads
                    )
                ),
                asyncio.gather(
                    *(
                        provider.score_proxy_trajectory_role(payload)
                        for payload in trajectory_role_payloads
                    )
                ),
            )
            completion_rewards: dict[str, float] = {}
            for values in completion_values:
                for completion_id, value in values.items():
                    key = str(completion_id)
                    if key in completion_rewards:
                        raise ProxyGroupError(f"duplicate completion reward {key!r}")
                    completion_rewards[key] = float(value)
            trajectory_role_rewards = {
                (
                    str(payload["ids"]["trajectory_id"]),
                    str(payload["ids"]["character"]),
                ): float(value)
                for payload, value in zip(trajectory_role_payloads, trajectory_values, strict=True)
            }
            result = normalize_proxy_role_advantages(
                trajectories,
                completion_rewards,
                trajectory_role_rewards,
                completion_weight=completion_weight,
                global_weight=global_weight,
                min_group_size=self.min_group_size,
                reward_std_epsilon=self.reward_std_epsilon,
            )
            if not state.future.done():
                logger.info(
                    "RLFF reward weights global_step=%d completion=%.6f trajectory=%.6f",
                    reward_step,
                    completion_weight,
                    global_weight,
                )
                if not self._reward_schedule_use_proxy_version:
                    self._reward_schedule_step += 1
                state.future.set_result(result)
        except BaseException as exc:
            self._fail_state(
                state,
                exc if isinstance(exc, ProxyGroupError) else ProxyGroupError(str(exc)),
            )

    def _get_reward_provider(self) -> ProxyRewardProvider:
        if self._reward_provider is not None:
            return self._reward_provider
        if self._reward_provider_name == DETERMINISTIC_AUDIT_REWARD_PROVIDER:
            self._reward_provider = DeterministicAuditRewardProvider()
            return self._reward_provider
        if self._reward_provider_name == "placeholder":
            from ..rewards import PlaceholderRewardProvider

            self._reward_provider = cast(
                ProxyRewardProvider,
                PlaceholderRewardProvider(allow_placeholder=True),
            )
            return self._reward_provider
        if self._reward_provider_name not in {"deepseek", "qwen_dashscope"}:
            raise ProxyGroupError(
                f"unsupported proxy reward provider {self._reward_provider_name!r}"
            )
        from ..rewards import DeepSeekRewardProvider, QwenDashScopeRewardProvider

        api_key = os.getenv(self._reward_api_key_env)
        langsmith_api_key = (
            os.getenv(self._langsmith_api_key_env) if self._langsmith_tracing else None
        )
        if self._langsmith_tracing and not langsmith_api_key:
            raise ProxyGroupError(
                f"LangSmith tracing requires environment variable {self._langsmith_api_key_env}"
            )
        provider_class = (
            QwenDashScopeRewardProvider
            if self._reward_provider_name == "qwen_dashscope"
            else DeepSeekRewardProvider
        )
        self._reward_provider = cast(
            ProxyRewardProvider,
            provider_class(
                api_key=api_key,
                base_url=self._reward_base_url,
                model=self._reward_model,
                trajectory_reward_model=self._trajectory_reward_model,
                completion_prompt_path=self._completion_prompt_path,
                trajectory_prompt_path=self._trajectory_prompt_path,
                repair_prompt_path=self._reward_repair_prompt_path,
                completion_timeout_seconds=self._completion_reward_timeout_seconds,
                trajectory_timeout_seconds=self._trajectory_reward_timeout_seconds,
                completion_retries=self._completion_reward_retries,
                trajectory_retries=self._trajectory_reward_retries,
                completion_concurrency=self._completion_reward_concurrency,
                trajectory_concurrency=self._trajectory_reward_concurrency,
                completion_temperature=self._completion_reward_temperature,
                trajectory_temperature=self._trajectory_reward_temperature,
                completion_reasoning_effort=self._completion_reward_reasoning_effort,
                trajectory_reasoning_effort=self._trajectory_reward_reasoning_effort,
                completion_max_tokens=self._completion_reward_max_tokens,
                trajectory_max_tokens=self._trajectory_reward_max_tokens,
                langsmith_project=self._langsmith_project,
                langsmith_api_key=langsmith_api_key,
                audit_jsonl=self._reward_audit_jsonl,
                detail_jsonl=self._reward_detail_jsonl,
                detail_sample_rate=self._reward_detail_sample_rate,
                failure_jsonl=self._reward_failure_jsonl,
            ),
        )
        return self._reward_provider

    @staticmethod
    def _proxy_completion_payload(
        trajectory: ProxyTrajectoryView, character: str
    ) -> Mapping[str, Any]:
        from ..rewards import build_proxy_completion_reward_payload

        return build_proxy_completion_reward_payload(
            trajectory.episode,
            trajectory,
            character,
        )

    @staticmethod
    def _proxy_trajectory_payload(
        trajectory: ProxyTrajectoryView, character: str
    ) -> Mapping[str, Any]:
        from ..rewards import build_proxy_trajectory_reward_payload

        return build_proxy_trajectory_reward_payload(
            trajectory.episode,
            trajectory,
            character,
        )

    async def _run_trajectory(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        group_id: str,
        trajectory_id: str,
    ) -> ProxyTrajectoryView:
        if self._trajectory_runner is not None:
            result: ProxyTrajectoryView | Awaitable[ProxyTrajectoryView] = self._trajectory_runner(
                data,
                base_url=base_url,
                http_client=http_client,
                api_key=api_key,
                group_id=group_id,
                trajectory_id=trajectory_id,
                max_rounds=self.max_rounds,
            )
            if inspect.isawaitable(result):
                result = await result
            return result
        return await self._run_openai_trajectory(
            data,
            base_url=base_url,
            http_client=http_client,
            api_key=api_key,
            group_id=group_id,
            trajectory_id=trajectory_id,
        )

    async def _run_openai_trajectory(
        self,
        data: Mapping[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        group_id: str,
        trajectory_id: str,
    ) -> ProxyTrajectoryView:
        from openai import AsyncOpenAI

        from ..config import DEFAULT_PROMPT_TEMPLATE
        from ..contracts import DialogueTurn, EpisodeRecord, PromptRenderSpec

        episode = self._episode_payload(data)
        try:
            episode_record = EpisodeRecord.model_validate(episode)
        except Exception as exc:
            raise ProxyGroupError("proxy episode is not a valid EpisodeRecord") from exc
        episode_id = str(episode.get("episode_id") or group_id)
        characters = _group_character_order(
            tuple(character.name for character in episode_record.characters),
            self._group_key(data),
        )
        if not characters:
            raise ProxyGroupError("episode must declare ordered characters")
        render = self._render_spec(data, PromptRenderSpec, DEFAULT_PROMPT_TEMPLATE)
        model = str(data.get("model") or self.model).strip()
        if not model:
            raise ProxyGroupError("proxy agent requires a model name")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=self.rollout_request_timeout_seconds,
            # Let RLFF's group failure path handle retries and wake peer runs.
            max_retries=0,
        )
        history: list[DialogueTurn] = list(episode_record.dialogue)
        completions: list[ProxyCompletionView] = []
        generated_turns: list[Mapping[str, Any]] = []
        turn_index = 0
        planned_rounds = min(
            self.max_rounds,
            rounds_for_character_count(len(characters)),
        )
        for _round_index in range(planned_rounds):
            for character in characters:
                messages = self._messages(episode_record, character, history, render)
                response = await client.chat.completions.create(
                    model=model,
                    messages=cast(Any, messages),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=self.max_new_tokens,
                )
                response_id = getattr(response, "id", None)
                choices = getattr(response, "choices", None)
                if not isinstance(response_id, str) or not response_id.strip() or not choices:
                    raise ProxyGroupError("OpenAI proxy response lacks an interaction ID/choice")
                first = choices[0]
                message = getattr(first, "message", None)
                text = getattr(message, "content", None)
                if not isinstance(text, str) or not text.strip():
                    raise ProxyGroupError("OpenAI proxy returned an empty character completion")
                finish_reason = getattr(first, "finish_reason", None)
                completion = ProxyCompletionView(
                    completion_id=response_id,
                    group_id=group_id,
                    trajectory_id=trajectory_id,
                    character=character,
                    turn_index=turn_index,
                    text=text,
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                )
                completions.append(completion)
                turn = DialogueTurn(character=character, content=text, turn_id=turn_index)
                generated_turns.append(turn.model_dump(mode="json", exclude_none=True))
                history.append(turn)
                turn_index += 1
        return ProxyTrajectoryView(
            group_id=group_id,
            trajectory_id=trajectory_id,
            episode_id=episode_id,
            episode=episode,
            completions=tuple(completions),
            turns=tuple(generated_turns),
            planned_rounds=planned_rounds,
            completed_rounds=planned_rounds,
        )

    @staticmethod
    def _episode_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
        return decode_proxy_episode_payload(data)

    @staticmethod
    def _characters(episode: Mapping[str, Any]) -> tuple[str, ...]:
        values = episode.get("characters")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ProxyGroupError("episode must declare ordered characters")
        result: list[str] = []
        for item in values:
            name = _mapping_value(item, "name", _mapping_value(item, "character"))
            if not isinstance(name, str) or not name.strip() or name.casefold() == "environment":
                raise ProxyGroupError("Environment/narrator is not a valid character")
            result.append(name)
        if not result or len(set(result)) != len(result):
            raise ProxyGroupError("episode characters must be non-empty and distinct")
        return tuple(result)

    @staticmethod
    def _render_spec(
        data: Mapping[str, Any],
        render_model: Any,
        default_template: str,
    ) -> Any:
        raw_render = data.get("render")
        if isinstance(raw_render, Mapping):
            try:
                return render_model.model_validate(raw_render)
            except Exception as exc:
                raise ProxyGroupError("rollout render metadata is invalid") from exc
        return render_model(
            template_id="default",
            template=default_template,
            render_seed=0,
        )

    def _messages(
        self,
        episode: Any,
        character: str,
        history: Sequence[Any],
        render: Any,
    ) -> list[dict[str, str]]:
        from ..contracts import EpisodeRecord
        from ..episodes import project_target_prompt

        if not isinstance(episode, EpisodeRecord):
            raise ProxyGroupError("prompt projection requires EpisodeRecord")
        try:
            evolving = episode.model_copy(update={"dialogue": tuple(history)})
            projection = project_target_prompt(evolving, character, render=render)
        except Exception as exc:
            raise ProxyGroupError(
                f"cannot project target prompt for character {character!r}"
            ) from exc
        return [
            cast(dict[str, str], message.model_dump(mode="json", exclude_none=True))
            for message in projection.messages
        ]
