"""Versioned, strict boundary contracts for RLFF.

The rollout and training phases exchange these models rather than ad-hoc
dictionaries.  The models deliberately contain only character messages and
exact engine output; there is no ``Environment`` role and no state-verifier
payload in the protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

CONTRACT_VERSION: Final = "rlff.contracts.v1"
EPISODE_SCHEMA_VERSION: Final = "rlff.episode.v1"
ROLLOUT_SCHEMA_VERSION: Final = "rlff.rollout.v1"
REWARD_SCHEMA_VERSION: Final = "rlff.reward.v1"
CHECKPOINT_SCHEMA_VERSION: Final = "rlff.checkpoint.v1"
RENDERER_VERSION: Final = "system-prompt-renderer.v1"

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_NON_EMPTY = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]
_TOKEN_ID = Annotated[StrictInt, Field(ge=0)]
_POSITIVE_INT = Annotated[StrictInt, Field(gt=0)]
_NON_NEGATIVE_INT = Annotated[StrictInt, Field(ge=0)]


class ContractModel(BaseModel):
    """Base model used at every serialized boundary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used for fingerprints."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_fingerprint(value: Any) -> str:
    """Hash a JSON-compatible value with a stable SHA-256 fingerprint."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    """Build a deterministic, human-searchable identifier."""

    if not _ID_RE.fullmatch(prefix):
        raise ValueError(f"Invalid ID prefix: {prefix!r}")
    return f"{prefix}_{stable_fingerprint(parts)[:length]}"


def _validate_id(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(
            "identifier must contain only ASCII letters, digits, '_', '.', ':', or '-'"
        )
    return value


class CharacterSpec(ContractModel):
    """One registered character in its source-order position."""

    name: _NON_EMPTY
    profile: str = Field(
        default="",
        validation_alias=AliasChoices("profile", "description"),
    )
    private_tasks: tuple[_NON_EMPTY, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("private_tasks", "tasks"),
    )
    aliases: tuple[_NON_EMPTY, ...] = ()

    @field_validator("name")
    @classmethod
    def reject_environment_name(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment is not a valid RLFF character")
        return value

    @field_validator("profile")
    @classmethod
    def normalize_profile(cls, value: str) -> str:
        return value.strip()

    @field_validator("aliases")
    @classmethod
    def aliases_are_distinct(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("character aliases must be distinct")
        return value


class DialogueTurn(ContractModel):
    """A character-only historical or generated dialogue turn."""

    character: _NON_EMPTY = Field(validation_alias=AliasChoices("character", "speaker"))
    content: _NON_EMPTY
    role: Literal["user", "assistant"] | None = None
    turn_id: _NON_NEGATIVE_INT | None = None

    @field_validator("character")
    @classmethod
    def reject_environment_character(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment turns are not part of RLFF")
        return value


class ChatMessage(ContractModel):
    """Prompt message after target-role projection."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: _NON_EMPTY | None = None


class PromptChoice(ContractModel):
    """One deterministic choice made while rendering a prompt template."""

    choice_id: _NON_EMPTY
    index: _NON_NEGATIVE_INT


class PromptRenderSpec(ContractModel):
    """Renderer/template choices shared by every sample in one group."""

    template_id: _NON_EMPTY
    template: _NON_EMPTY
    renderer_version: _NON_EMPTY = RENDERER_VERSION
    render_seed: StrictInt
    choices: tuple[PromptChoice, ...] = ()

    @field_validator("render_seed")
    @classmethod
    def render_seed_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("render_seed must be non-negative")
        return value


class PromptProjection(ContractModel):
    """One target-role view of an episode's dialogue history."""

    target_character: _NON_EMPTY
    messages: tuple[ChatMessage, ...]
    render: PromptRenderSpec

    @field_validator("target_character")
    @classmethod
    def no_environment_target(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment is not a valid target character")
        return value


class EpisodeRecord(ContractModel):
    """Canonical source record loaded from RLFF episode JSONL."""

    schema_version: Literal["rlff.episode.v1"] = EPISODE_SCHEMA_VERSION
    episode_id: str | None = None
    title: str = ""
    plot: _NON_EMPTY
    shared_tasks: tuple[_NON_EMPTY, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("shared_tasks", "tasks"),
    )
    characters: tuple[CharacterSpec, ...]
    dialogue: tuple[DialogueTurn, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("dialogue", "history", "turns"),
    )
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    fingerprint: str | None = None

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str | None) -> str | None:
        return None if value is None else _validate_id(value)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_and_identify(self) -> EpisodeRecord:
        names = [character.name for character in self.characters]
        if not names:
            raise ValueError("episodes require at least one character")
        if len(set(names)) != len(names):
            raise ValueError("episode character names must be unique and ordered")
        registered = set(names)
        labels: dict[str, str] = {}
        for character in self.characters:
            for label in (character.name, *character.aliases):
                normalized = label.casefold()
                if normalized == "environment":
                    raise ValueError("Environment cannot be a character name or alias")
                previous_owner = labels.get(normalized)
                if previous_owner is not None and previous_owner != character.name:
                    raise ValueError(
                        f"character name/alias {label!r} collides with {previous_owner!r}"
                    )
                labels[normalized] = character.name
        for turn in self.dialogue:
            if turn.character not in registered:
                raise ValueError(
                    f"dialogue character {turn.character!r} is not registered in the episode"
                )

        identity_payload = {
            "title": self.title,
            "plot": self.plot,
            "shared_tasks": self.shared_tasks,
            "characters": [character.model_dump(mode="json") for character in self.characters],
            "dialogue": [turn.model_dump(mode="json", exclude_none=True) for turn in self.dialogue],
            "metadata": self.metadata,
        }
        computed_fingerprint = stable_fingerprint(identity_payload)
        if self.fingerprint is not None and self.fingerprint != computed_fingerprint:
            raise ValueError("episode fingerprint does not match canonical episode content")
        object.__setattr__(self, "fingerprint", computed_fingerprint)
        if self.episode_id is None:
            object.__setattr__(self, "episode_id", stable_id("episode", computed_fingerprint))
        return self


class EpisodeSample(ContractModel):
    """One reproducible trajectory sample in a grouped episode."""

    schema_version: Literal["rlff.episode.v1"] = EPISODE_SCHEMA_VERSION
    episode_id: _NON_EMPTY
    group_id: _NON_EMPTY
    trajectory_id: _NON_EMPTY
    sample_index: _NON_NEGATIVE_INT = Field(
        validation_alias=AliasChoices("sample_index", "trajectory_index", "index"),
    )
    sampling_iteration: _NON_NEGATIVE_INT = 0
    seed: _NON_NEGATIVE_INT
    render: PromptRenderSpec
    episode: EpisodeRecord

    @model_validator(mode="after")
    def validate_identity(self) -> EpisodeSample:
        if self.episode_id != self.episode.episode_id:
            raise ValueError("sample episode_id does not match embedded episode")
        return self


class GroupedEpisodeSamples(ContractModel):
    """A GRPO group containing G replicas of one canonical episode."""

    schema_version: Literal["rlff.episode.v1"] = EPISODE_SCHEMA_VERSION
    group_id: _NON_EMPTY
    episode_id: _NON_EMPTY
    episode: EpisodeRecord
    samples: tuple[EpisodeSample, ...] = Field(min_length=1)
    group_size: _NON_NEGATIVE_INT | None = None
    sampling_iteration: _NON_NEGATIVE_INT = 0

    @model_validator(mode="after")
    def validate_group(self) -> GroupedEpisodeSamples:
        if self.episode_id != self.episode.episode_id:
            raise ValueError("group episode_id does not match embedded episode")
        if self.group_size is not None and self.group_size != len(self.samples):
            raise ValueError("group_size must match samples length")
        indexes = [sample.sample_index for sample in self.samples]
        if len(set(indexes)) != len(indexes):
            raise ValueError("sample indexes must be distinct in a group")
        seeds = [sample.seed for sample in self.samples]
        if len(set(seeds)) != len(seeds):
            raise ValueError("sampling seeds must be distinct in a group")
        for sample in self.samples:
            if sample.group_id != self.group_id or sample.episode_id != self.episode_id:
                raise ValueError("all samples must belong to the enclosing group")
            if sample.sampling_iteration != self.sampling_iteration:
                raise ValueError("all samples must share the enclosing sampling_iteration")
            if sample.episode.fingerprint != self.episode.fingerprint:
                raise ValueError("all samples must embed the enclosing episode fingerprint")
        return self


class CompletionTrace(ContractModel):
    """Exact SGLang completion output used for training."""

    schema_version: Literal["rlff.rollout.v1"] = ROLLOUT_SCHEMA_VERSION
    completion_id: _NON_EMPTY
    episode_id: _NON_EMPTY
    group_id: _NON_EMPTY
    trajectory_id: _NON_EMPTY
    character: _NON_EMPTY
    turn_index: _NON_NEGATIVE_INT
    prompt_token_ids: tuple[_TOKEN_ID, ...] = ()
    completion_token_ids: tuple[_TOKEN_ID, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("completion_token_ids", "token_ids"),
    )
    completion_start: _NON_NEGATIVE_INT
    completion_end: _NON_NEGATIVE_INT
    rollout_logprobs: tuple[StrictFloat, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("rollout_logprobs", "logprobs"),
    )
    text: str = ""
    finish_reason: _NON_EMPTY | None = None
    policy_version: _NON_EMPTY
    tokenizer_fingerprint: _NON_EMPTY

    @field_validator("character")
    @classmethod
    def no_environment_character(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment completions are not valid")
        return value

    @field_validator("rollout_logprobs")
    @classmethod
    def finite_logprobs(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("rollout log probabilities must be finite")
        return value

    @model_validator(mode="after")
    def validate_boundaries(self) -> CompletionTrace:
        if self.completion_start != len(self.prompt_token_ids):
            raise ValueError("completion_start must equal prompt token count")
        if self.completion_end != self.completion_start + len(self.completion_token_ids):
            raise ValueError("completion boundaries do not match exact completion token IDs")
        if len(self.rollout_logprobs) != len(self.completion_token_ids):
            raise ValueError("one rollout log probability is required per completion token")
        return self


class TerminationReason(StrEnum):
    MAX_ROUNDS = "max_rounds"
    CONTEXT_LIMIT = "context_limit"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INVALID = "invalid"
    TRUNCATED = "truncated"


class Trajectory(ContractModel):
    """One multi-turn rollout trajectory."""

    schema_version: Literal["rlff.rollout.v1"] = ROLLOUT_SCHEMA_VERSION
    trajectory_id: _NON_EMPTY
    episode_id: _NON_EMPTY
    group_id: _NON_EMPTY
    completions: tuple[CompletionTrace, ...] = ()
    turns: tuple[DialogueTurn, ...] = ()
    termination_reason: TerminationReason
    planned_rounds: _POSITIVE_INT
    completed_rounds: _NON_NEGATIVE_INT
    completion_count: _NON_NEGATIVE_INT
    valid: bool = True
    truncated: bool = False
    invalid_reason: str | None = None
    policy_version: _NON_EMPTY
    tokenizer_fingerprint: _NON_EMPTY

    @model_validator(mode="after")
    def validate_trajectory(self) -> Trajectory:
        if self.completion_count != len(self.completions):
            raise ValueError("completion_count must equal the number of completion traces")
        if self.completed_rounds > self.planned_rounds:
            raise ValueError("completed_rounds cannot exceed planned_rounds")
        if (
            self.valid
            and self.termination_reason is TerminationReason.MAX_ROUNDS
            and self.completed_rounds != self.planned_rounds
        ):
            raise ValueError("max_rounds trajectories must complete the planned horizon")
        if self.valid and self.invalid_reason is not None:
            raise ValueError("valid trajectories cannot carry invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid trajectories require invalid_reason")
        if self.truncated and self.termination_reason not in {
            TerminationReason.CONTEXT_LIMIT,
            TerminationReason.TRUNCATED,
            TerminationReason.INFRASTRUCTURE_FAILURE,
        }:
            raise ValueError("truncated trajectories require an explicit truncation reason")
        for completion in self.completions:
            if (
                completion.trajectory_id != self.trajectory_id
                or completion.episode_id != self.episode_id
                or completion.group_id != self.group_id
            ):
                raise ValueError("completion identity does not match enclosing trajectory")
            if completion.policy_version != self.policy_version:
                raise ValueError("completion policy version differs from trajectory")
            if completion.tokenizer_fingerprint != self.tokenizer_fingerprint:
                raise ValueError("completion tokenizer fingerprint differs from trajectory")
        return self


class RolloutGroup(ContractModel):
    """Rollout trajectories for one episode sampling group."""

    schema_version: Literal["rlff.rollout.v1"] = ROLLOUT_SCHEMA_VERSION
    group_id: _NON_EMPTY
    episode_id: _NON_EMPTY
    trajectories: tuple[Trajectory, ...] = Field(min_length=1)
    policy_version: _NON_EMPTY
    tokenizer_fingerprint: _NON_EMPTY

    @model_validator(mode="after")
    def validate_rollout_group(self) -> RolloutGroup:
        for trajectory in self.trajectories:
            if (
                trajectory.group_id != self.group_id
                or trajectory.episode_id != self.episode_id
                or trajectory.policy_version != self.policy_version
                or trajectory.tokenizer_fingerprint != self.tokenizer_fingerprint
            ):
                raise ValueError("trajectory identity or policy metadata differs from group")
        return self


class RewardDimension(ContractModel):
    name: _NON_EMPTY
    value: StrictFloat

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward dimension must be finite")
        return value


class RewardStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class CompletionReward(ContractModel):
    """Character-local reward for one completion."""

    schema_version: Literal["rlff.reward.v1"] = REWARD_SCHEMA_VERSION
    completion_id: _NON_EMPTY
    trajectory_id: _NON_EMPTY
    group_id: _NON_EMPTY
    character: _NON_EMPTY
    reward: StrictFloat | None = None
    dimensions: tuple[RewardDimension, ...] = ()
    status: RewardStatus = RewardStatus.VALID
    provider: _NON_EMPTY
    raw_response: str | None = None
    error: str | None = None

    @field_validator("character")
    @classmethod
    def no_environment_character(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment rewards are not valid")
        return value

    @model_validator(mode="after")
    def validate_reward(self) -> CompletionReward:
        if self.status is RewardStatus.VALID:
            if self.reward is None or not math.isfinite(self.reward):
                raise ValueError("valid completion rewards require a finite scalar")
            if self.error is not None:
                raise ValueError("valid completion rewards cannot carry an error")
        elif not self.error:
            raise ValueError("invalid completion rewards require an error")
        return self


class TrajectoryReward(ContractModel):
    """Full-trajectory task reward for one target character."""

    schema_version: Literal["rlff.reward.v1"] = REWARD_SCHEMA_VERSION
    trajectory_id: _NON_EMPTY
    group_id: _NON_EMPTY
    character: _NON_EMPTY
    reward: StrictFloat | None = None
    dimensions: tuple[RewardDimension, ...] = ()
    status: RewardStatus = RewardStatus.VALID
    provider: _NON_EMPTY
    raw_response: str | None = None
    error: str | None = None

    @field_validator("character")
    @classmethod
    def no_environment_character(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment trajectory rewards are not valid")
        return value

    @model_validator(mode="after")
    def validate_reward(self) -> TrajectoryReward:
        if self.status is RewardStatus.VALID:
            if self.reward is None or not math.isfinite(self.reward):
                raise ValueError("valid trajectory rewards require a finite scalar")
            if self.error is not None:
                raise ValueError("valid trajectory rewards cannot carry an error")
        elif not self.error:
            raise ValueError("invalid trajectory rewards require an error")
        return self


class RoleReward(ContractModel):
    """Aggregated completion-local reward for one trajectory and character."""

    schema_version: Literal["rlff.reward.v1"] = REWARD_SCHEMA_VERSION
    group_id: _NON_EMPTY
    trajectory_id: _NON_EMPTY
    character: _NON_EMPTY
    completion_ids: tuple[_NON_EMPTY, ...] = Field(min_length=1)
    aggregated_local_reward: StrictFloat

    @field_validator("character")
    @classmethod
    def no_environment_character(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment role rewards are not valid")
        return value

    @field_validator("completion_ids")
    @classmethod
    def distinct_ordered_completion_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("completion_ids must be non-empty and distinct in order")
        return value

    @field_validator("aggregated_local_reward")
    @classmethod
    def finite_local_reward(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("aggregated local reward must be finite")
        return value


class RoleEffectiveReward(RoleReward):
    """Role reward after adding its full-trajectory task contribution."""

    trajectory_role_contribution: StrictFloat
    effective_reward: StrictFloat

    @field_validator("trajectory_role_contribution", "effective_reward")
    @classmethod
    def finite_effective_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("role effective reward values must be finite")
        return value

    @model_validator(mode="after")
    def effective_reward_is_sum(self) -> RoleEffectiveReward:
        if not math.isclose(
            self.effective_reward,
            self.aggregated_local_reward + self.trajectory_role_contribution,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("effective_reward must equal local plus trajectory-role contribution")
        return self


class RoleAdvantage(ContractModel):
    """Normalized advantage broadcast only to one character completion."""

    schema_version: Literal["rlff.reward.v1"] = REWARD_SCHEMA_VERSION
    group_id: _NON_EMPTY
    trajectory_id: _NON_EMPTY
    character: _NON_EMPTY
    completion_ids: tuple[_NON_EMPTY, ...] = Field(min_length=1)
    effective_reward: StrictFloat
    advantage: StrictFloat
    valid_group_size: _NON_NEGATIVE_INT
    reward_mean: StrictFloat
    reward_std: StrictFloat

    @field_validator("character")
    @classmethod
    def no_environment_character(cls, value: str) -> str:
        if value.casefold() == "environment":
            raise ValueError("Environment advantages are not valid")
        return value

    @field_validator("completion_ids")
    @classmethod
    def distinct_ordered_completion_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("completion_ids must be non-empty and distinct in order")
        return value

    @field_validator("effective_reward", "advantage", "reward_mean", "reward_std")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("advantage values must be finite")
        return value


class TokenMask(ContractModel):
    """Completion-token-only mask carried with a role advantage."""

    completion_id: _NON_EMPTY
    token_count: _NON_NEGATIVE_INT
    loss_mask: tuple[StrictFloat | StrictInt, ...]

    @model_validator(mode="after")
    def validate_mask(self) -> TokenMask:
        if len(self.loss_mask) != self.token_count:
            raise ValueError("token mask length must equal token_count")
        if any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss masks must contain only 0.0 or 1.0")
        return self


class CheckpointBoundary(ContractModel):
    """RLFF metadata needed to validate checkpoint resume boundaries."""

    schema_version: Literal["rlff.checkpoint.v1"] = CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: _NON_EMPTY
    step: _NON_NEGATIVE_INT
    config_fingerprint: _NON_EMPTY
    dataset_fingerprint: _NON_EMPTY
    tokenizer_fingerprint: _NON_EMPTY
    prompt_fingerprint: _NON_EMPTY
    policy_version: _NON_EMPTY
    adapter_identity: _NON_EMPTY
    group_cursor: _NON_NEGATIVE_INT
    reward_schema_version: Literal["rlff.reward.v1"] = REWARD_SCHEMA_VERSION
    termination: Literal["complete", "resumable", "invalid"] = "resumable"


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "EPISODE_SCHEMA_VERSION",
    "RENDERER_VERSION",
    "REWARD_SCHEMA_VERSION",
    "ROLLOUT_SCHEMA_VERSION",
    "CharacterSpec",
    "ChatMessage",
    "CheckpointBoundary",
    "CompletionReward",
    "CompletionTrace",
    "DialogueTurn",
    "EpisodeRecord",
    "EpisodeSample",
    "GroupedEpisodeSamples",
    "PromptChoice",
    "PromptProjection",
    "PromptRenderSpec",
    "RewardDimension",
    "RewardStatus",
    "RoleAdvantage",
    "RoleEffectiveReward",
    "RoleReward",
    "RolloutGroup",
    "TerminationReason",
    "TokenMask",
    "Trajectory",
    "TrajectoryReward",
    "canonical_json",
    "stable_fingerprint",
    "stable_id",
]
