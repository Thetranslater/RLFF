from __future__ import annotations

import pytest
from pydantic import ValidationError

from rlff.contracts import (
    CompletionTrace,
    EpisodeRecord,
    EpisodeSample,
    GroupedEpisodeSamples,
    PromptRenderSpec,
    RoleAdvantage,
    RoleEffectiveReward,
    RoleReward,
    TerminationReason,
    TokenMask,
    Trajectory,
)


def episode() -> EpisodeRecord:
    return EpisodeRecord(
        title="demo",
        plot="A quiet room.",
        shared_tasks=["keep the scene coherent"],
        characters=[
            {"name": "Alice", "profile": "curious", "private_tasks": ["ask a question"]},
            {"name": "Bob", "profile": "careful"},
        ],
        dialogue=[
            {"speaker": "Alice", "content": "Hello."},
            {"speaker": "Bob", "content": "Hi."},
        ],
    )


def render_spec() -> PromptRenderSpec:
    return PromptRenderSpec(
        template_id="default",
        template="{character}\n{profile}\n{plot}\n{tasks}",
        render_seed=7,
    )


def completion(*, trajectory_id: str = "trajectory-1") -> CompletionTrace:
    return CompletionTrace(
        completion_id="completion-1",
        episode_id="episode-1",
        group_id="group-1",
        trajectory_id=trajectory_id,
        character="Alice",
        turn_index=0,
        prompt_token_ids=[10, 11],
        completion_token_ids=[12, 13],
        completion_start=2,
        completion_end=4,
        rollout_logprobs=[-0.1, -0.2],
        text="Hello.",
        finish_reason="stop",
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )


def test_episode_ids_and_fingerprint_are_deterministic() -> None:
    first = episode()
    second = episode()

    assert first.episode_id == second.episode_id
    assert first.fingerprint == second.fingerprint
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_episode_metadata_is_typed_and_part_of_fingerprint() -> None:
    first = episode()
    with_metadata = EpisodeRecord(
        **first.model_dump(exclude={"fingerprint", "episode_id", "metadata"}),
        metadata={"book": "demo", "plot": 1, "nested": [True, None]},
    )
    assert with_metadata.fingerprint != first.fingerprint
    with pytest.raises(ValidationError):
        EpisodeRecord(
            **first.model_dump(exclude={"fingerprint", "episode_id", "metadata"}),
            metadata={"not_json": object()},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"plot": "x", "characters": []},
        {
            "plot": "x",
            "characters": [{"name": "Alice"}, {"name": "Alice"}],
        },
        {"plot": "x", "characters": [{"name": "Environment"}]},
        {
            "plot": "x",
            "characters": [{"name": "Alice"}],
            "dialogue": [{"speaker": "Unknown", "content": "leak"}],
        },
        {
            "plot": "x",
            "characters": [
                {"name": "Alice", "aliases": ["A"]},
                {"name": "Bob", "aliases": ["a"]},
            ],
        },
        {
            "plot": "x",
            "characters": [{"name": "Alice", "aliases": ["Environment"]}],
        },
    ],
)
def test_episode_validation_rejects_invalid_character_data(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EpisodeRecord.model_validate(payload)


def test_contract_boundaries_reject_inexact_engine_output() -> None:
    invalid_boundary = completion().model_dump()
    invalid_boundary["completion_end"] = 999
    with pytest.raises(ValidationError, match="boundaries"):
        CompletionTrace(**invalid_boundary)

    invalid_start = completion().model_dump()
    invalid_start["completion_start"] = 1
    invalid_start["completion_end"] = 3
    with pytest.raises(ValidationError, match="prompt token count"):
        CompletionTrace(**invalid_start)

    with pytest.raises(ValidationError, match="one rollout"):
        CompletionTrace(
            completion_id="completion-1",
            episode_id="episode-1",
            group_id="group-1",
            trajectory_id="trajectory-1",
            character="Alice",
            turn_index=0,
            completion_token_ids=[12, 13],
            completion_start=0,
            completion_end=2,
            rollout_logprobs=[-0.1],
            policy_version="policy-1",
            tokenizer_fingerprint="tokenizer-1",
        )


def test_group_replicas_share_identity_and_have_distinct_seeds() -> None:
    source = episode()
    samples = [
        EpisodeSample(
            episode_id="episode-1",
            group_id="group-1",
            trajectory_id="trajectory-1",
            sample_index=0,
            seed=100,
            render=render_spec(),
            episode=source.model_copy(update={"episode_id": "episode-1"}),
        ),
        EpisodeSample(
            episode_id="episode-1",
            group_id="group-1",
            trajectory_id="trajectory-2",
            sample_index=1,
            seed=101,
            render=render_spec(),
            episode=source.model_copy(update={"episode_id": "episode-1"}),
        ),
    ]
    group = GroupedEpisodeSamples(
        group_id="group-1",
        episode_id="episode-1",
        episode=source.model_copy(update={"episode_id": "episode-1"}),
        samples=samples,
        group_size=2,
    )

    assert [sample.seed for sample in group.samples] == [100, 101]
    assert len({sample.trajectory_id for sample in group.samples}) == 2

    with pytest.raises(ValidationError, match="distinct"):
        GroupedEpisodeSamples(
            group_id="group-1",
            episode_id="episode-1",
            episode=source.model_copy(update={"episode_id": "episode-1"}),
            samples=[samples[0], samples[1].model_copy(update={"seed": 100})],
        )

    mismatched_episode = samples[1].episode.model_copy(update={"fingerprint": "wrong"})
    with pytest.raises(ValidationError, match="fingerprint"):
        GroupedEpisodeSamples(
            group_id="group-1",
            episode_id="episode-1",
            episode=source.model_copy(update={"episode_id": "episode-1"}),
            samples=[samples[0], samples[1].model_copy(update={"episode": mismatched_episode})],
        )


def test_trajectory_and_role_mask_contracts() -> None:
    trace = completion()
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        episode_id="episode-1",
        group_id="group-1",
        completions=[trace],
        termination_reason=TerminationReason.MAX_ROUNDS,
        planned_rounds=1,
        completed_rounds=1,
        completion_count=1,
        policy_version="policy-1",
        tokenizer_fingerprint="tokenizer-1",
    )
    assert trajectory.completions[0].completion_token_ids == (12, 13)

    advantage = RoleAdvantage(
        group_id="group-1",
        trajectory_id="trajectory-1",
        character="Alice",
        completion_ids=["completion-1"],
        effective_reward=1.0,
        advantage=0.5,
        valid_group_size=2,
        reward_mean=0.5,
        reward_std=1.0,
    )
    assert advantage.advantage == 0.5

    TokenMask(completion_id="completion-1", token_count=2, loss_mask=[1.0, 1.0])
    with pytest.raises(ValidationError, match="mask length"):
        TokenMask(completion_id="completion-1", token_count=2, loss_mask=[1.0])


def test_role_reward_contracts_require_ordered_unique_completion_ids() -> None:
    local = RoleReward(
        group_id="group-1",
        trajectory_id="trajectory-1",
        character="Alice",
        completion_ids=["completion-1", "completion-2"],
        aggregated_local_reward=1.0,
    )
    effective = RoleEffectiveReward(
        **local.model_dump(),
        trajectory_role_contribution=0.25,
        effective_reward=1.25,
    )
    assert effective.completion_ids == ("completion-1", "completion-2")
    with pytest.raises(ValidationError, match="distinct"):
        RoleReward(
            group_id="group-1",
            trajectory_id="trajectory-1",
            character="Alice",
            completion_ids=["completion-1", "completion-1"],
            aggregated_local_reward=1.0,
        )
    with pytest.raises(ValidationError):
        RoleAdvantage(
            group_id="group-1",
            trajectory_id="trajectory-1",
            character="Alice",
            completion_ids=[],
            effective_reward=1.0,
            advantage=0.5,
            valid_group_size=2,
            reward_mean=0.5,
            reward_std=1.0,
        )


def test_eos_is_not_a_trajectory_termination_reason() -> None:
    with pytest.raises(ValueError):
        TerminationReason("eos")


def test_max_rounds_requires_fixed_horizon_metadata() -> None:
    trace = completion()
    with pytest.raises(ValidationError, match="planned horizon"):
        Trajectory(
            trajectory_id="trajectory-1",
            episode_id="episode-1",
            group_id="group-1",
            completions=[trace],
            termination_reason=TerminationReason.MAX_ROUNDS,
            planned_rounds=2,
            completed_rounds=1,
            completion_count=1,
            policy_version="policy-1",
            tokenizer_fingerprint="tokenizer-1",
        )
    with pytest.raises(ValidationError, match="completion_count"):
        Trajectory(
            trajectory_id="trajectory-1",
            episode_id="episode-1",
            group_id="group-1",
            completions=[trace],
            termination_reason=TerminationReason.MAX_ROUNDS,
            planned_rounds=1,
            completed_rounds=1,
            completion_count=0,
            policy_version="policy-1",
            tokenizer_fingerprint="tokenizer-1",
        )
