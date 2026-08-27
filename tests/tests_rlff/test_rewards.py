from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rlff.contracts import (
    CompletionReward,
    CompletionTrace,
    EpisodeRecord,
    RewardStatus,
    TerminationReason,
    Trajectory,
    TrajectoryReward,
)
from rlff.rewards import (
    DeepSeekRewardProvider,
    PlaceholderRewardProvider,
    RewardAggregationError,
    RewardHTTPResponse,
    RewardPromptError,
    RewardResponseError,
    aggregate_role_rewards,
    build_completion_reward_payload,
    build_trajectory_reward_payload,
    completion_response_rewards,
    load_reward_prompt,
    parse_completion_reward_response,
    parse_reward_response,
    parse_trajectory_reward_response,
    render_reward_prompt,
    score_completions,
)

COMPLETION_PROMPT = "角色={character}\n档案={profile}\n剧情={plot}"
TRAJECTORY_PROMPT = "角色={character}\n剧情={plot}\n任务={tasks}"


def episode() -> EpisodeRecord:
    return EpisodeRecord(
        title="reward demo",
        plot="A quiet room.",
        shared_tasks=["keep the scene coherent"],
        characters=[
            {"name": "Alice", "profile": "curious", "private_tasks": ["alice-secret"]},
            {"name": "Bob", "profile": "careful", "private_tasks": ["bob-secret"]},
        ],
        dialogue=[{"speaker": "Alice", "content": "Opening."}],
    )


def completion(
    episode_id: str,
    trajectory_id: str,
    completion_id: str,
    character: str,
    turn_index: int,
) -> CompletionTrace:
    return CompletionTrace(
        completion_id=completion_id,
        episode_id=episode_id,
        group_id="group-1",
        trajectory_id=trajectory_id,
        character=character,
        turn_index=turn_index,
        prompt_token_ids=[1],
        completion_token_ids=[2],
        completion_start=1,
        completion_end=2,
        rollout_logprobs=[-0.1],
        text=f"reply-{completion_id}",
        finish_reason="stop",
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )


def trajectory(source: EpisodeRecord, trajectory_id: str = "trajectory-1") -> Trajectory:
    first = completion(source.episode_id or "", trajectory_id, f"{trajectory_id}-a", "Alice", 0)
    second = completion(source.episode_id or "", trajectory_id, f"{trajectory_id}-b", "Bob", 1)
    return Trajectory(
        trajectory_id=trajectory_id,
        episode_id=source.episode_id or "",
        group_id="group-1",
        completions=[first, second],
        turns=[
            {"character": "Alice", "content": first.text, "turn_id": 0},
            {"character": "Bob", "content": second.text, "turn_id": 1},
        ],
        termination_reason=TerminationReason.MAX_ROUNDS,
        planned_rounds=1,
        completed_rounds=1,
        completion_count=2,
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )


def completion_model_response(value: int, *, count: int = 1) -> str:
    scores = [{"values": [str(value)] * 4} for _ in range(count)]
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps({"scores": scores})}}]}
    )


def trajectory_model_response(tasks: list[str], values: list[int]) -> str:
    score = [{"task": task, "value": str(value)} for task, value in zip(tasks, values, strict=True)]
    return json.dumps({"choices": [{"message": {"content": json.dumps({"score": score})}}]})


class FakeTransport:
    def __init__(self, responses: list[str | RewardHTTPResponse], delay: float = 0.0) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.active = 0
        self.max_active = 0
        self.delay = delay

    async def __call__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> str | RewardHTTPResponse:
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.responses.pop(0)
        finally:
            self.active -= 1


def test_response_schema_is_strict_and_has_no_state_fields() -> None:
    response = {"scores": [{"values": [1, "2", 3, "5"]}]}
    parsed = parse_reward_response(response)
    assert parsed.scores[0].values == (1, 2, 3, 5)
    with pytest.raises(RewardResponseError):
        parse_completion_reward_response({"scores": [{"values": [1, 2, 3]}]})
    with pytest.raises(RewardResponseError):
        parse_completion_reward_response(response, expected_count=2)
    with pytest.raises(RewardResponseError):
        parse_completion_reward_response({"scores": [{"values": [1, 2, 3, 4.5]}]})


def test_behavior_score_is_neutral_without_parenthesized_action() -> None:
    parsed = parse_completion_reward_response(
        {"scores": [{"values": [5, 5, 5, 5]}, {"values": [5, 5, 5, 5]}]},
        expected_count=2,
    )
    assert completion_response_rewards(
        parsed,
        reply_texts=("plain dialogue", "(takes a step) dialogue"),
    ) == (4.5, 5.0)
    assert completion_response_rewards(
        parsed,
        reply_texts=("（点头）对白", "() empty action"),
    ) == (5.0, 4.5)


def test_trajectory_response_requires_exact_task_coverage() -> None:
    tasks = ["shared", "private"]
    parsed = parse_trajectory_reward_response(
        {"score": [{"task": "private", "value": 5}, {"task": "shared", "value": "3"}]},
        expected_tasks=tasks,
    )
    assert [item.value for item in parsed.score] == [5, 3]
    with pytest.raises(RewardResponseError, match="exactly cover"):
        parse_trajectory_reward_response(
            {"score": [{"task": "shared", "value": 3}]},
            expected_tasks=tasks,
        )


def test_prompt_renderer_only_replaces_named_variables() -> None:
    template = '角色={character}\n示例={"history":"..."}\n{plot}'
    assert (
        render_reward_prompt(
            template,
            {"character": "Alice", "plot": "scene"},
            required=("character", "plot"),
        )
        == '角色=Alice\n示例={"history":"..."}\nscene'
    )


def test_prompt_loader_rejects_missing_empty_and_production_placeholder(tmp_path: Path) -> None:
    with pytest.raises(RewardPromptError, match="cannot read"):
        load_reward_prompt(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n", encoding="utf-8")
    with pytest.raises(RewardPromptError, match="empty"):
        load_reward_prompt(empty)
    placeholder = tmp_path / "placeholder.txt"
    placeholder.write_text("[PLACEHOLDER]", encoding="utf-8")
    with pytest.raises(RewardPromptError, match="production"):
        load_reward_prompt(placeholder)
    assert load_reward_prompt(placeholder, provider="placeholder", development=True)


@pytest.mark.asyncio
async def test_placeholder_provider_requires_opt_in_and_emits_audit() -> None:
    source = episode()
    sample = trajectory(source)
    with pytest.raises(ValueError, match="allow_placeholder"):
        PlaceholderRewardProvider()
    provider = PlaceholderRewardProvider(allow_placeholder=True)
    local = await provider.score_completion(source, sample, sample.completions[0])
    trajectory_role_reward = await provider.score_trajectory(source, sample, "Alice")
    assert local.reward == 0.0 and trajectory_role_reward.reward == 0.0
    assert local.provider.startswith("placeholder")
    assert '"development": true' in (local.raw_response or "")
    assert '"placeholder": true' in (trajectory_role_reward.raw_response or "")


def test_completion_and_trajectory_payloads_isolate_private_tasks() -> None:
    source = episode()
    sample = trajectory(source)
    local = build_completion_reward_payload(source, sample, "Alice")
    trajectory_payload = build_trajectory_reward_payload(source, sample, "Alice")
    local_text = json.dumps(local)
    assert "alice-secret" not in local_text
    assert "bob-secret" not in local_text
    assert trajectory_payload["tasks"] == ["keep the scene coherent", "alice-secret"]
    assert "bob-secret" not in json.dumps(trajectory_payload)
    assert sample.completions[1].text in trajectory_payload["input"]["history"]
    assert sample.completions[0].text in local["input"]["history"]
    assert sample.completions[1].text in local["input"]["history"]
    assert local["ids"]["completion_ids"] == [sample.completions[0].completion_id]
    assert set(local["input"]) == {"history"}


@pytest.mark.asyncio
async def test_deepseek_transport_parses_raw_response_and_orders_batch() -> None:
    source = episode()
    first = trajectory(source, "trajectory-1")
    second = trajectory(source, "trajectory-2")
    transport = FakeTransport(
        [completion_model_response(value) for value in (1, 2, 3, 4)],
        delay=0.001,
    )
    provider = DeepSeekRewardProvider(
        api_key="secret-key",
        completion_prompt=COMPLETION_PROMPT,
        trajectory_prompt=TRAJECTORY_PROMPT,
        completion_concurrency=1,
        transport=transport,
    )
    rewards = await score_completions(provider, source, [first, second])
    assert [item.completion_id for item in rewards] == [
        item.completion_id for sample in (first, second) for item in sample.completions
    ]
    assert [item.reward for item in rewards] == [1.5, 2.25, 3.0, 3.75]
    assert transport.max_active == 1
    assert all("secret-key" not in json.dumps(call["payload"]) for call in transport.calls)
    assert all(item.raw_response for item in rewards)


@pytest.mark.asyncio
async def test_deepseek_invalid_json_retries_then_returns_invalid() -> None:
    source = episode()
    sample = trajectory(source)
    transport = FakeTransport(
        [
            "not-json",
            '{"scores": []}',
            '{"scores": [], "extra": 2}',
        ]
    )
    provider = DeepSeekRewardProvider(
        completion_prompt=COMPLETION_PROMPT,
        trajectory_prompt=TRAJECTORY_PROMPT,
        completion_retries=2,
        transport=transport,
    )
    result = await provider.score_completion(source, sample, sample.completions[0])
    assert result.status is RewardStatus.INVALID
    assert len(transport.calls) == 3
    assert result.reward is None
    assert result.error and "exhausted" in result.error
    assert result.raw_response is not None


@pytest.mark.asyncio
async def test_deepseek_timeout_is_bounded_and_retried() -> None:
    source = episode()
    sample = trajectory(source)

    async def slow_transport(**_: object) -> RewardHTTPResponse:
        await asyncio.sleep(0.02)
        return RewardHTTPResponse(200, completion_model_response(1))

    provider = DeepSeekRewardProvider(
        completion_prompt=COMPLETION_PROMPT,
        trajectory_prompt=TRAJECTORY_PROMPT,
        completion_timeout_seconds=0.001,
        completion_retries=1,
        transport=slow_transport,
    )
    result = await provider.score_completion(source, sample, sample.completions[0])
    assert result.status is RewardStatus.INVALID
    assert result.error and "timed out" in result.error


@pytest.mark.asyncio
async def test_deepseek_http_retry_and_scope_specific_payload_messages() -> None:
    source = episode()
    sample = trajectory(source)
    transport = FakeTransport(
        [
            RewardHTTPResponse(503, "busy"),
            completion_model_response(4),
            trajectory_model_response(["keep the scene coherent", "alice-secret"], [5, 3]),
        ]
    )
    provider = DeepSeekRewardProvider(
        api_key="key",
        completion_prompt=COMPLETION_PROMPT,
        trajectory_prompt=TRAJECTORY_PROMPT,
        completion_retries=1,
        transport=transport,
    )
    local = await provider.score_completion(source, sample, sample.completions[0])
    trajectory_role_reward = await provider.score_trajectory(source, sample, "Alice")
    assert local.status is RewardStatus.VALID and local.reward == 3.75
    assert trajectory_role_reward.status is RewardStatus.VALID
    assert trajectory_role_reward.reward == 4.0
    local_body = json.loads(transport.calls[1]["payload"]["messages"][1]["content"])
    trajectory_body = json.loads(transport.calls[2]["payload"]["messages"][1]["content"])
    assert set(local_body) == {"history"}
    assert set(trajectory_body) == {"history"}
    assert "角色=Alice" in transport.calls[1]["payload"]["messages"][0]["content"]
    assert "alice-secret" in transport.calls[2]["payload"]["messages"][0]["content"]
    assert transport.calls[1]["payload"]["temperature"] == 1.0
    assert transport.calls[2]["payload"]["temperature"] == 1.0
    assert transport.calls[1]["payload"]["reasoning_effort"] == "high"
    assert transport.calls[2]["payload"]["reasoning_effort"] == "high"
    assert transport.calls[1]["payload"]["max_tokens"] == 25000
    assert transport.calls[2]["payload"]["max_tokens"] == 25000


@pytest.mark.asyncio
async def test_completion_role_request_maps_multiple_replies_in_stable_order() -> None:
    source = episode()
    base = trajectory(source)
    third = completion(
        source.episode_id or "",
        base.trajectory_id,
        f"{base.trajectory_id}-c",
        "Alice",
        2,
    )
    sample = base.model_copy(
        update={
            "completions": (*base.completions, third),
            "turns": (
                *base.turns,
                type(base.turns[0])(
                    character="Alice",
                    content=third.text,
                    turn_id=2,
                ),
            ),
            "completion_count": 3,
        }
    )
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"scores": [{"values": [1, 2, 3, 4]}, {"values": [5, 4, 3, 2]}]}
                    )
                }
            }
        ]
    }
    transport = FakeTransport([json.dumps(response)])
    provider = DeepSeekRewardProvider(
        completion_prompt=COMPLETION_PROMPT,
        trajectory_prompt=TRAJECTORY_PROMPT,
        transport=transport,
    )
    rewards = await provider.score_completion_role(source, sample, "Alice")
    assert [reward.completion_id for reward in rewards] == [
        base.completions[0].completion_id,
        third.completion_id,
    ]
    assert [reward.reward for reward in rewards] == [2.25, 3.75]
    assert len(transport.calls) == 1


def _local_reward(
    trajectory_id: str,
    completion_id: str,
    character: str,
    value: float,
) -> CompletionReward:
    return CompletionReward(
        completion_id=completion_id,
        trajectory_id=trajectory_id,
        group_id="group-1",
        character=character,
        reward=value,
        provider="test",
    )


def _trajectory_role_reward(trajectory_id: str, character: str, value: float) -> TrajectoryReward:
    return TrajectoryReward(
        trajectory_id=trajectory_id,
        group_id="group-1",
        character=character,
        reward=value,
        provider="test",
    )


def test_role_aggregation_uses_means_weights_and_explicit_order() -> None:
    local = [
        _local_reward("trajectory-2", "b-1", "Bob", 0.4),
        _local_reward("trajectory-1", "a-1", "Alice", 0.2),
        _local_reward("trajectory-1", "a-2", "Alice", 0.8),
    ]
    records = aggregate_role_rewards(
        local,
        [
            _trajectory_role_reward("trajectory-1", "Alice", 0.5),
            _trajectory_role_reward("trajectory-2", "Bob", 0.25),
        ],
        completion_weight=0.6,
        global_weight=0.4,
    )
    assert [(record.trajectory_id, record.character) for record in records] == [
        ("trajectory-2", "Bob"),
        ("trajectory-1", "Alice"),
    ]
    assert records[1].aggregated_local_reward == pytest.approx(0.3)
    assert records[1].trajectory_role_contribution == pytest.approx(0.2)
    assert records[1].effective_reward == pytest.approx(0.5)


def test_invalid_required_reward_propagates_from_aggregation() -> None:
    invalid = CompletionReward(
        completion_id="bad",
        trajectory_id="trajectory-1",
        group_id="group-1",
        character="Alice",
        status=RewardStatus.INVALID,
        provider="deepseek-v4-flash",
        error="schema failed",
    )
    with pytest.raises(RewardAggregationError, match="schema failed"):
        aggregate_role_rewards(
            [invalid],
            [_trajectory_role_reward("trajectory-1", "Alice", 0.2)],
        )
