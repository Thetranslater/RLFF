"""AReaL OpenAI-proxy agent and group-local reward normalization.

The official AReaL wrapper owns proxy sessions, reward assignment, session
shutdown, and exact interaction export.  This module only supplies the
agent-like ``run`` implementation used by that wrapper.  It deliberately
keeps the reward views text-only: token IDs and log-probabilities are never
reconstructed here and arrive only from AReaL's proxy export.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast


class ProxyGroupError(RuntimeError):
    """Raised when one fixed-size proxy group cannot complete atomically."""


@dataclass(frozen=True, slots=True)
class ProxyCompletionView:
    """Text-only completion metadata available before official proxy export."""

    completion_id: str
    group_id: str
    trajectory_id: str
    character: str
    turn_index: int
    text: str
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyTrajectoryView:
    """Text-only trajectory view used by reward prompts and normalization."""

    group_id: str
    trajectory_id: str
    episode_id: str
    episode: Mapping[str, Any]
    completions: tuple[ProxyCompletionView, ...]
    turns: tuple[Mapping[str, Any], ...]
    planned_rounds: int
    completed_rounds: int
    termination_reason: str = "max_rounds"
    valid: bool = True
    truncated: bool = False
    invalid_reason: str | None = None


class ProxyRewardProvider(Protocol):
    """Narrow runtime reward interface; no CompletionTrace is fabricated."""

    async def score_proxy_completion_role(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, float]: ...

    async def score_proxy_trajectory_role(self, payload: Mapping[str, Any]) -> float: ...


ProxyTrajectoryRunner = Callable[..., ProxyTrajectoryView | Awaitable[ProxyTrajectoryView]]


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProxyGroupError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProxyGroupError(f"{field} must be finite")
    return result


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def normalize_proxy_role_advantages(
    trajectories: Sequence[ProxyTrajectoryView],
    completion_rewards: Mapping[str, float],
    trajectory_role_rewards: Mapping[tuple[str, str], float],
    *,
    completion_weight: float = 0.6,
    global_weight: float = 0.4,
    min_group_size: int = 2,
    reward_std_epsilon: float = 1e-8,
) -> dict[str, float]:
    """Aggregate then normalize role rewards for one complete proxy group.

    Local completion scores are first averaged by ``(trajectory, character)``;
    the same character's full-trajectory task score is then combined.  Only after that step are
    values normalized independently by ``(group_id, character)``.  Every
    completion belonging to one trajectory/character receives exactly the
    same resulting scalar.
    """

    if not trajectories:
        raise ProxyGroupError("proxy group must contain at least one trajectory")
    if type(min_group_size) is not int or min_group_size <= 0:
        raise ProxyGroupError("min_group_size must be a positive integer")
    completion_weight = _finite(completion_weight, field="completion_weight")
    global_weight = _finite(global_weight, field="global_weight")
    epsilon = _finite(reward_std_epsilon, field="reward_std_epsilon")
    if completion_weight < 0 or global_weight < 0:
        raise ProxyGroupError("reward weights must be non-negative")
    if epsilon < 0:
        raise ProxyGroupError("reward_std_epsilon must be non-negative")

    first_group = trajectories[0].group_id
    if any(item.group_id != first_group for item in trajectories):
        raise ProxyGroupError("all proxy trajectories must belong to one group")
    role_values: dict[tuple[str, str], float] = {}
    role_completion_ids: dict[tuple[str, str], tuple[str, ...]] = {}
    role_order: list[str] = []
    seen_trajectory_ids: set[str] = set()
    seen_completion_ids: set[str] = set()
    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise ProxyGroupError("proxy group contains duplicate trajectory IDs")
        seen_trajectory_ids.add(trajectory.trajectory_id)
        if not trajectory.valid or trajectory.truncated:
            raise ProxyGroupError(
                f"trajectory {trajectory.trajectory_id!r} is invalid or truncated"
            )
        if (
            trajectory.termination_reason != "max_rounds"
            or trajectory.completed_rounds != trajectory.planned_rounds
        ):
            raise ProxyGroupError(
                f"trajectory {trajectory.trajectory_id!r} did not complete max_rounds"
            )
        by_character: dict[str, list[ProxyCompletionView]] = {}
        for completion in trajectory.completions:
            if completion.group_id != first_group:
                raise ProxyGroupError("completion group differs from trajectory group")
            if completion.trajectory_id != trajectory.trajectory_id:
                raise ProxyGroupError("completion trajectory differs from trajectory")
            if not completion.character or completion.character.casefold() == "environment":
                raise ProxyGroupError("Environment/narrator completions are forbidden")
            if completion.completion_id in seen_completion_ids:
                raise ProxyGroupError("proxy group contains duplicate completion IDs")
            seen_completion_ids.add(completion.completion_id)
            if completion.completion_id in completion_rewards:
                by_character.setdefault(completion.character, []).append(completion)
            else:
                raise ProxyGroupError(f"missing completion reward {completion.completion_id!r}")
        if not by_character:
            raise ProxyGroupError(f"trajectory {trajectory.trajectory_id!r} has no completions")
        for character, completions in by_character.items():
            if character not in role_order:
                role_order.append(character)
            local_values = [
                _finite(
                    completion_rewards[completion.completion_id],
                    field=f"completion reward {completion.completion_id}",
                )
                for completion in completions
            ]
            local_mean = sum(local_values) / len(local_values)
            key = (trajectory.trajectory_id, character)
            trajectory_role_reward = _finite(
                trajectory_role_rewards.get(key),
                field=f"trajectory role reward {trajectory.trajectory_id}/{character}",
            )
            role_values[key] = (
                completion_weight * local_mean + global_weight * trajectory_role_reward
            )
            role_completion_ids[key] = tuple(item.completion_id for item in completions)

    expected_roles = set(role_order)
    if len(trajectories) < min_group_size:
        raise ProxyGroupError(
            f"proxy group has {len(trajectories)} trajectories; minimum is {min_group_size}"
        )
    for trajectory in trajectories:
        actual = {
            character
            for (trajectory_id, character) in role_values
            if trajectory_id == trajectory.trajectory_id
        }
        if actual != expected_roles:
            raise ProxyGroupError(
                f"trajectory {trajectory.trajectory_id!r} role set differs from group"
            )

    result: dict[str, float] = {}
    for character in role_order:
        records = [
            (trajectory.trajectory_id, role_values[(trajectory.trajectory_id, character)])
            for trajectory in trajectories
        ]
        mean = sum(value for _, value in records) / len(records)
        variance = sum((value - mean) ** 2 for _, value in records) / len(records)
        std = math.sqrt(variance)
        for trajectory_id, value in records:
            advantage = 0.0 if std <= epsilon else (value - mean) / std
            for completion_id in role_completion_ids[(trajectory_id, character)]:
                result[completion_id] = advantage
    expected_ids = {
        completion.completion_id
        for trajectory in trajectories
        for completion in trajectory.completions
    }
    if set(result) != expected_ids:
        raise ProxyGroupError("normalized reward mapping does not cover exactly the group")
    return result


@dataclass(slots=True)
class _GroupState:
    expected: int
    future: asyncio.Future[dict[str, float]]
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
        max_rounds: int = 1,
        model: str = "",
        temperature: float = 0.9,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        group_timeout_seconds: float = 3600.0,
        completion_weight: float = 0.6,
        global_weight: float = 0.4,
        min_group_size: int = 2,
        reward_std_epsilon: float = 1e-8,
        system_prompt: str = "",
        reward_provider: ProxyRewardProvider | None = None,
        reward_provider_name: str = "deepseek",
        reward_api_key_env: str = "DEEPSEEK_API_KEY",
        reward_base_url: str = "https://api.deepseek.com",
        completion_prompt_path: str | None = None,
        trajectory_prompt_path: str | None = None,
        reward_model: str = "deepseek-v4-flash",
        trajectory_reward_model: str | None = None,
        completion_reward_timeout_seconds: float = 120.0,
        trajectory_reward_timeout_seconds: float | None = None,
        completion_reward_retries: int = 2,
        trajectory_reward_retries: int | None = None,
        completion_reward_concurrency: int = 4,
        trajectory_reward_concurrency: int | None = None,
        completion_reward_max_tokens: int = 1024,
        trajectory_reward_max_tokens: int | None = None,
        langsmith_tracing: bool = False,
        langsmith_project: str = "rlff",
        langsmith_api_key_env: str = "LANGSMITH_API_KEY",
        trajectory_runner: ProxyTrajectoryRunner | None = None,
    ) -> None:
        if type(group_size) is not int or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if group_timeout_seconds <= 0:
            raise ValueError("group_timeout_seconds must be positive")
        self.group_size = group_size
        self.max_rounds = max_rounds
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.group_timeout_seconds = group_timeout_seconds
        self.completion_weight = completion_weight
        self.global_weight = global_weight
        self.min_group_size = min_group_size
        self.reward_std_epsilon = reward_std_epsilon
        self.system_prompt = system_prompt
        self._reward_provider = reward_provider
        self._reward_provider_name = reward_provider_name
        self._reward_api_key_env = reward_api_key_env
        self._reward_base_url = reward_base_url
        self._completion_prompt_path = completion_prompt_path
        self._trajectory_prompt_path = trajectory_prompt_path
        self._reward_model = reward_model
        self._trajectory_reward_model = trajectory_reward_model
        self._completion_reward_timeout_seconds = completion_reward_timeout_seconds
        self._trajectory_reward_timeout_seconds = trajectory_reward_timeout_seconds
        self._completion_reward_retries = completion_reward_retries
        self._trajectory_reward_retries = trajectory_reward_retries
        self._completion_reward_concurrency = completion_reward_concurrency
        self._trajectory_reward_concurrency = trajectory_reward_concurrency
        self._completion_reward_max_tokens = completion_reward_max_tokens
        self._trajectory_reward_max_tokens = trajectory_reward_max_tokens
        self._langsmith_tracing = langsmith_tracing
        self._langsmith_project = langsmith_project
        self._langsmith_api_key_env = langsmith_api_key_env
        self._trajectory_runner = trajectory_runner
        self._states: dict[str, _GroupState] = {}
        self._states_lock = asyncio.Lock()

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
        state = await self._get_state(key)
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

    def _group_id(self, data: Mapping[str, Any]) -> str:
        value = data.get("group_id")
        if value is None:
            episode = data.get("episode")
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

    async def _get_state(self, key: str) -> _GroupState:
        async with self._states_lock:
            state = self._states.get(key)
            if state is None:
                future: asyncio.Future[dict[str, float]] = (
                    asyncio.get_running_loop().create_future()
                )
                state = _GroupState(expected=self.group_size, future=future)
                future.add_done_callback(self._consume_future_exception)
                self._states[key] = state
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
        try:
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
                completion_weight=self.completion_weight,
                global_weight=self.global_weight,
                min_group_size=self.min_group_size,
                reward_std_epsilon=self.reward_std_epsilon,
            )
            if not state.future.done():
                state.future.set_result(result)
        except BaseException as exc:
            self._fail_state(
                state,
                exc if isinstance(exc, ProxyGroupError) else ProxyGroupError(str(exc)),
            )

    def _get_reward_provider(self) -> ProxyRewardProvider:
        if self._reward_provider is not None:
            return self._reward_provider
        if self._reward_provider_name == "placeholder":
            from .rewards import PlaceholderRewardProvider

            self._reward_provider = cast(
                ProxyRewardProvider,
                PlaceholderRewardProvider(allow_placeholder=True),
            )
            return self._reward_provider
        if self._reward_provider_name != "deepseek":
            raise ProxyGroupError(
                f"unsupported proxy reward provider {self._reward_provider_name!r}"
            )
        from .rewards import DeepSeekRewardProvider

        api_key = os.getenv(self._reward_api_key_env)
        langsmith_api_key = (
            os.getenv(self._langsmith_api_key_env) if self._langsmith_tracing else None
        )
        if self._langsmith_tracing and not langsmith_api_key:
            raise ProxyGroupError(
                f"LangSmith tracing requires environment variable {self._langsmith_api_key_env}"
            )
        self._reward_provider = cast(
            ProxyRewardProvider,
            DeepSeekRewardProvider(
                api_key=api_key,
                base_url=self._reward_base_url,
                model=self._reward_model,
                trajectory_reward_model=self._trajectory_reward_model,
                completion_prompt_path=self._completion_prompt_path,
                trajectory_prompt_path=self._trajectory_prompt_path,
                completion_timeout_seconds=self._completion_reward_timeout_seconds,
                trajectory_timeout_seconds=self._trajectory_reward_timeout_seconds,
                completion_retries=self._completion_reward_retries,
                trajectory_retries=self._trajectory_reward_retries,
                completion_concurrency=self._completion_reward_concurrency,
                trajectory_concurrency=self._trajectory_reward_concurrency,
                completion_max_tokens=self._completion_reward_max_tokens,
                trajectory_max_tokens=self._trajectory_reward_max_tokens,
                langsmith_project=self._langsmith_project,
                langsmith_api_key=langsmith_api_key,
            ),
        )
        return self._reward_provider

    @staticmethod
    def _proxy_completion_payload(
        trajectory: ProxyTrajectoryView, character: str
    ) -> Mapping[str, Any]:
        from .rewards import build_proxy_completion_reward_payload

        return build_proxy_completion_reward_payload(
            trajectory.episode,
            trajectory,
            character,
        )

    @staticmethod
    def _proxy_trajectory_payload(
        trajectory: ProxyTrajectoryView, character: str
    ) -> Mapping[str, Any]:
        from .rewards import build_proxy_trajectory_reward_payload

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

        from .config import DEFAULT_PROMPT_TEMPLATE
        from .contracts import DialogueTurn, EpisodeRecord, PromptRenderSpec

        episode = self._episode_payload(data)
        try:
            episode_record = EpisodeRecord.model_validate(episode)
        except Exception as exc:
            raise ProxyGroupError("proxy episode is not a valid EpisodeRecord") from exc
        episode_id = str(episode.get("episode_id") or group_id)
        characters = tuple(character.name for character in episode_record.characters)
        if not characters:
            raise ProxyGroupError("episode must declare ordered characters")
        render = self._render_spec(data, PromptRenderSpec, DEFAULT_PROMPT_TEMPLATE)
        model = str(data.get("model") or self.model).strip()
        if not model:
            raise ProxyGroupError("proxy agent requires a model name")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        history: list[DialogueTurn] = list(episode_record.dialogue)
        completions: list[ProxyCompletionView] = []
        generated_turns: list[Mapping[str, Any]] = []
        turn_index = 0
        for _round_index in range(self.max_rounds):
            for character in characters:
                messages = self._messages(episode_record, character, history, render)
                response = await client.chat.completions.create(
                    model=model,
                    messages=cast(Any, messages),
                    temperature=self.temperature,
                    top_p=self.top_p,
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
            planned_rounds=self.max_rounds,
            completed_rounds=self.max_rounds,
        )

    @staticmethod
    def _episode_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
        raw: Any = data.get("episode", data)
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        if not isinstance(raw, Mapping):
            raise ProxyGroupError("rollout data episode must be a mapping")
        return cast(Mapping[str, Any], dict(raw))

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
        from .contracts import EpisodeRecord
        from .episodes import project_target_prompt

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


__all__ = [
    "ProxyCompletionView",
    "ProxyGroupError",
    "ProxyRewardProvider",
    "ProxyTrajectoryRunner",
    "ProxyTrajectoryView",
    "RLFFGroupAwareAgent",
    "normalize_proxy_role_advantages",
]
