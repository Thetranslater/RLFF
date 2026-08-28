from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from rlff.contracts import (
    EpisodeRecord,
    PromptProjection,
    RolloutGroup,
    TerminationReason,
)
from rlff.episodes import build_episode_group, project_target_prompt
from rlff.rollout import (
    GenerationRequest,
    GenerationResult,
    HuggingFacePromptTokenizer,
    RolloutValidationError,
    huggingface_tokenizer_fingerprint,
    rollout_group,
    rounds_for_character_count,
    validate_rollout_group,
)


@pytest.mark.parametrize(
    ("character_count", "expected_rounds"),
    ((1, 7), (2, 7), (3, 6), (4, 5), (5, 5), (6, 5)),
)
def test_rounds_are_derived_from_character_count(
    character_count: int,
    expected_rounds: int,
) -> None:
    assert rounds_for_character_count(character_count) == expected_rounds


def episode() -> EpisodeRecord:
    return EpisodeRecord(
        title="rollout test",
        plot="Two people solve a puzzle.",
        shared_tasks=["stay coherent"],
        characters=[
            {"name": "Alice", "profile": "curious", "private_tasks": ["ask"]},
            {"name": "Bob", "profile": "careful", "private_tasks": ["check"]},
        ],
        dialogue=[{"speaker": "Alice", "content": "Opening."}],
    )


class FakeTokenizer:
    fingerprint = "tokenizer-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def tokenize_prompt(self, projection: PromptProjection) -> list[int]:
        messages = projection.messages
        self.calls.append(tuple(messages))
        # A deterministic fake for the exact tokenizer boundary.  The IDs are
        # intentionally unrelated to completion text, so retokenization would
        # be observable in the assertions below.
        return [17, len(messages), sum(len(message.content) for message in messages)]


@dataclass
class FakeBackend:
    policy_version: str = "policy-v1"
    requests: list[GenerationRequest] = field(default_factory=list)
    fail_on_turn: int | None = None
    mismatch_policy: str | None = None
    active_requests: int = 0
    max_active_requests: int = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        await asyncio.sleep(0)
        if self.fail_on_turn is not None and request.turn_index == self.fail_on_turn:
            self.active_requests -= 1
            raise RuntimeError("fake backend unavailable")
        policy = self.mismatch_policy
        try:
            return GenerationResult(
                prompt_token_ids=request.prompt_token_ids,
                completion_token_ids=(700 + request.turn_index, 900 + request.turn_index),
                rollout_logprobs=(-0.125 - request.turn_index, -0.25 - request.turn_index),
                text=f"reply-{request.character}-{request.turn_index}",
                finish_reason="stop" if request.turn_index % 2 == 0 else "length",
                policy_version=policy,
                tokenizer_fingerprint=request.tokenizer_fingerprint,
            )
        finally:
            self.active_requests -= 1


async def run(
    *,
    rounds: int = 2,
    backend: FakeBackend | None = None,
    tokenizer: FakeTokenizer | None = None,
    context_length: int = 4096,
) -> tuple[RolloutGroup, FakeBackend, FakeTokenizer]:
    actual_backend = backend or FakeBackend()
    actual_tokenizer = tokenizer or FakeTokenizer()
    group = build_episode_group(episode(), group_size=2, base_seed=12)
    result = await rollout_group(
        group,
        actual_backend,
        actual_tokenizer,
        max_rounds=rounds,
        max_new_tokens=2,
        context_length=context_length,
        policy_version="policy-v1",
        tokenizer_fingerprint="tokenizer-v1",
    )
    return result, actual_backend, actual_tokenizer


@pytest.mark.asyncio
async def test_direct_round_robin_fixed_horizon_and_eos_continuation() -> None:
    result, backend, tokenizer = await run(rounds=2)

    assert [trajectory.trajectory_id for trajectory in result.trajectories] == [
        "trajectory_" + result.trajectories[0].trajectory_id.split("_", 1)[1],
        result.trajectories[1].trajectory_id,
    ]
    assert len(result.trajectories) == 2
    for trajectory in result.trajectories:
        assert trajectory.termination_reason is TerminationReason.MAX_ROUNDS
        assert trajectory.completed_rounds == 2
        assert trajectory.completion_count == 4
        assert [trace.character for trace in trajectory.completions] == [
            "Alice",
            "Bob",
            "Alice",
            "Bob",
        ]
        assert all(trace.finish_reason in {"stop", "length"} for trace in trajectory.completions)
        assert trajectory.valid is True
    assert len(backend.requests) == 8
    assert backend.max_active_requests >= 2
    assert tokenizer.calls.count(tokenizer.calls[0]) == 2
    requests_by_trajectory: dict[str, list[GenerationRequest]] = {}
    for request in backend.requests:
        requests_by_trajectory.setdefault(request.trajectory_id, []).append(request)
    assert len(requests_by_trajectory) == 2
    for requests in requests_by_trajectory.values():
        ordered = sorted(requests, key=lambda request: request.turn_index)
        assert [request.character for request in ordered] == [
            "Alice",
            "Bob",
            "Alice",
            "Bob",
        ]
    # The next target sees the previous completion in the evolving projected
    # history, while Bob sees Alice as a named user rather than an assistant.
    bob_projection = next(
        request.projection for request in backend.requests if request.turn_index == 1
    )
    assert any(
        message.role == "user" and "Alice:reply-Alice-0" in message.content
        for message in bob_projection.messages
    )
    assert "Environment" not in repr(result)


@pytest.mark.asyncio
async def test_exact_backend_tokens_logprobs_boundaries_and_identity_are_preserved() -> None:
    result, backend, _ = await run(rounds=1)
    first = result.trajectories[0].completions[0]

    first_request = next(request for request in backend.requests if request.turn_index == 0)
    assert first.prompt_token_ids == first_request.prompt_token_ids
    assert first.prompt_token_ids[0] == 17
    assert first.completion_token_ids == (700, 900)
    assert first.rollout_logprobs == (-0.125, -0.25)
    assert first.completion_start == len(first.prompt_token_ids)
    assert first.completion_end == first.completion_start + 2
    assert first.policy_version == "policy-v1"
    assert first.tokenizer_fingerprint == "tokenizer-v1"


@pytest.mark.asyncio
async def test_context_limit_is_checked_before_backend_and_does_not_fabricate_completion() -> None:
    backend = FakeBackend()
    result, backend, _ = await run(
        rounds=2,
        backend=backend,
        context_length=4,
    )

    trajectory = result.trajectories[0]
    assert trajectory.valid is False
    assert trajectory.truncated is True
    assert trajectory.termination_reason is TerminationReason.CONTEXT_LIMIT
    assert trajectory.completion_count == 0
    assert trajectory.completions == ()
    assert backend.requests == []


@pytest.mark.asyncio
async def test_backend_failure_is_explicit_and_keeps_only_real_prior_completions() -> None:
    result, backend, _ = await run(rounds=2, backend=FakeBackend(fail_on_turn=1))

    trajectory = result.trajectories[0]
    assert trajectory.valid is False
    assert trajectory.truncated is True
    assert trajectory.termination_reason is TerminationReason.INFRASTRUCTURE_FAILURE
    assert trajectory.completion_count == 1
    assert [trace.character for trace in trajectory.completions] == ["Alice"]
    assert "unavailable" in (trajectory.invalid_reason or "")
    assert len(backend.requests) == 4


@pytest.mark.asyncio
async def test_policy_mismatch_invalidates_sample_without_mixing_group_identity() -> None:
    result, _, _ = await run(rounds=1, backend=FakeBackend(mismatch_policy="other-policy"))

    for trajectory in result.trajectories:
        assert trajectory.valid is False
        assert trajectory.termination_reason is TerminationReason.INVALID
        assert trajectory.completions == ()
        assert "policy" in (trajectory.invalid_reason or "")
        assert trajectory.policy_version == "policy-v1"
    assert result.policy_version == "policy-v1"


@pytest.mark.asyncio
async def test_group_completeness_and_order_are_validated() -> None:
    result, _, _ = await run(rounds=1)
    group = build_episode_group(episode(), group_size=2, base_seed=12)
    incomplete = RolloutGroup(
        group_id=result.group_id,
        episode_id=result.episode_id,
        trajectories=[result.trajectories[0]],
        policy_version=result.policy_version,
        tokenizer_fingerprint=result.tokenizer_fingerprint,
    )
    with pytest.raises(RolloutValidationError, match="exactly one trajectory"):
        validate_rollout_group(group, incomplete)


class FakeHFTokenizer:
    def __init__(self) -> None:
        self.name_or_path = "fake/model"
        self.init_kwargs = {"revision": "main"}
        self.special_tokens_map = {"eos_token": "</s>"}
        self.chat_template = "{{ messages }}"
        self.vocab_size = 123
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return [11, len(messages)]


def test_huggingface_prompt_adapter_uses_exact_chat_template_and_stable_fingerprint() -> None:
    first_underlying = FakeHFTokenizer()
    second_underlying = FakeHFTokenizer()
    first = HuggingFacePromptTokenizer(first_underlying)
    second = HuggingFacePromptTokenizer(second_underlying)
    assert first.tokenizer_fingerprint == second.tokenizer_fingerprint
    assert first.tokenizer_fingerprint == huggingface_tokenizer_fingerprint(second_underlying)

    projection = project_target_prompt(episode(), "Alice")
    assert first.tokenize_prompt(projection).token_ids == (11, len(projection.messages))
    assert first_underlying.calls[0]["tokenize"] is True
    assert first_underlying.calls[0]["add_generation_prompt"] is True
