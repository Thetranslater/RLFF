"""Strict, serializable RLFF configuration.

This module intentionally models only the settings required by the RLFF
episode, rollout, reward, and training boundaries.  API keys are represented
by environment-variable *names* and are resolved only by
``RLFFConfig.resolve_runtime_secrets``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .contracts import RENDERER_VERSION, REWARD_SCHEMA_VERSION, stable_fingerprint

_NON_EMPTY = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]
_POSITIVE_INT = Annotated[StrictInt, Field(gt=0)]
_NON_NEGATIVE_INT = Annotated[StrictInt, Field(ge=0)]
_PROBABILITY = Annotated[StrictFloat, Field(ge=0, le=1)]

DEFAULT_REWARD_MODEL: Final = "qwen3.7-flash"
DEFAULT_SGLANG_MODEL: Final = ""
AREAL_VERSION: Final = "1.0.4"
AREAL_COMMIT: Final = "37d6c6400e99a05fa3409d6a067762a44df40d3b"
SGLANG_VERSION: Final = "0.5.10.post1"
DEFAULT_AREAL_YAML: Final = "configs/areal-v1.0.4-rlff.yaml"
DEFAULT_PROMPT_TEMPLATE: Final = (
    "You are {character}.\n"
    "Character profile:\n{profile}\n"
    "Plot:\n{plot}\n"
    "Shared and character tasks:\n{tasks}"
)


class ConfigModel(BaseModel):
    """Base for all config boundaries: no unknown settings are accepted."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


class EpisodeGroupingConfig(ConfigModel):
    """Canonical episode source and deterministic group sampling settings."""

    dataset_path: Path = Field(
        validation_alias=AliasChoices("dataset_path", "episode_jsonl", "episodes_path")
    )
    group_size: _POSITIVE_INT = 4
    base_seed: _NON_NEGATIVE_INT = 0
    limit: _POSITIVE_INT | None = None

    @field_validator("dataset_path")
    @classmethod
    def dataset_path_is_jsonl(cls, value: Path) -> Path:
        if value.suffix.lower() not in {".jsonl", ".ndjson"}:
            raise ValueError("dataset_path must point to a JSONL/NDJSON file")
        return value


class PromptTemplateConfig(ConfigModel):
    """Prompt renderer templates used by every replica in a group."""

    template_paths: tuple[Path, ...] = ()
    template_ids: tuple[_NON_EMPTY, ...] = ()
    renderer_version: _NON_EMPTY = RENDERER_VERSION
    default_template_id: _NON_EMPTY = "default"

    @model_validator(mode="after")
    def validate_template_identity(self) -> PromptTemplateConfig:
        if self.template_ids and len(self.template_ids) != len(self.template_paths):
            raise ValueError("template_ids must have one ID per template_paths entry")
        ids = self.template_ids or tuple(path.stem for path in self.template_paths)
        if len(set(ids)) != len(ids):
            raise ValueError("prompt template IDs must be distinct")
        if self.default_template_id != "default" and self.default_template_id not in ids:
            raise ValueError("default_template_id must name a configured template")
        return self

    def load_templates(self) -> tuple[tuple[str, str], ...]:
        """Load templates as UTF-8-SIG text without mutating configuration."""

        if not self.template_paths:
            return ((self.default_template_id, DEFAULT_PROMPT_TEMPLATE),)
        ids = self.template_ids or tuple(path.stem for path in self.template_paths)
        result: list[tuple[str, str]] = []
        for template_id, path in zip(ids, self.template_paths, strict=True):
            try:
                content = path.read_text(encoding="utf-8-sig")
            except OSError as exc:
                raise ValueError(f"cannot read prompt template {path}: {exc}") from exc
            if not content.strip():
                raise ValueError(f"prompt template is empty: {path}")
            result.append((template_id, content))
        return tuple(result)


class RolloutConfig(ConfigModel):
    """Direct ordered round-robin rollout settings."""

    # This is the configurable upper bound.  The actual horizon is derived
    # from the episode character count and is therefore 5, 6, or 7.
    max_rounds: _POSITIVE_INT = 7


class SGLangConfig(ConfigModel):
    """Lazy SGLang generation boundary (no SGLang import occurs here)."""

    model: _NON_EMPTY
    base_url: _NON_EMPTY = "http://127.0.0.1:30000"
    temperature: StrictFloat = 0.9
    top_p: StrictFloat = 1.0
    frequency_penalty: StrictFloat = 0.0
    max_new_tokens: _POSITIVE_INT = 256
    context_length: _POSITIVE_INT = 8192
    timeout_seconds: StrictFloat = 120.0
    api_key_env: _NON_EMPTY | None = None

    @field_validator("temperature")
    @classmethod
    def valid_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("top_p", "timeout_seconds")
    @classmethod
    def positive_generation_values(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("generation value must be positive")
        return value

    @field_validator("top_p")
    @classmethod
    def top_p_at_most_one(cls, value: float) -> float:
        if value > 1:
            raise ValueError("top_p must be no greater than 1")
        return value

    @field_validator("frequency_penalty")
    @classmethod
    def frequency_penalty_is_supported(cls, value: float) -> float:
        if not -2 <= value <= 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        return value

    @model_validator(mode="after")
    def context_can_fit_completion(self) -> SGLangConfig:
        if self.context_length <= self.max_new_tokens:
            raise ValueError("context_length must be greater than max_new_tokens")
        return self


class RewardScopeConfig(ConfigModel):
    """One reward-model scope: completion-local or trajectory-role."""

    model: _NON_EMPTY = DEFAULT_REWARD_MODEL
    prompt_path: Path
    timeout_seconds: StrictFloat = 120.0
    retries: _NON_NEGATIVE_INT = 2
    concurrency: _POSITIVE_INT = 4
    temperature: StrictFloat = 0.7
    reasoning_effort: Literal["low", "medium", "high", "max"] = "medium"
    max_tokens: _POSITIVE_INT = 18000

    @field_validator("timeout_seconds")
    @classmethod
    def timeout_is_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("reward timeout_seconds must be positive")
        return value

    @field_validator("temperature")
    @classmethod
    def temperature_is_supported(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("reward temperature must be between 0 and 2")
        return value


class RewardWeightScheduleConfig(ConfigModel):
    """Linear curriculum for completion/trajectory reward mixing."""

    start_step: _NON_NEGATIVE_INT
    end_step: _NON_NEGATIVE_INT
    completion_end_weight: StrictFloat
    global_end_weight: StrictFloat

    @field_validator("completion_end_weight", "global_end_weight")
    @classmethod
    def end_weights_are_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("scheduled reward weights must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_schedule(self) -> RewardWeightScheduleConfig:
        if self.end_step <= self.start_step:
            raise ValueError("reward weight schedule end_step must be greater than start_step")
        if self.completion_end_weight == 0 and self.global_end_weight == 0:
            raise ValueError("at least one scheduled end weight must be positive")
        return self


class RewardConfig(ConfigModel):
    """Completion/role-task provider settings and explicit placeholder opt-in.

    ``global_reward`` remains the configuration key for backward compatibility;
    its value is evaluated independently for every target character.
    """

    completion: RewardScopeConfig
    global_reward: RewardScopeConfig = Field(
        validation_alias=AliasChoices("global_reward", "global")
    )
    repair_prompt_path: Path = Path("prompts/error_user.txt")
    completion_weight: StrictFloat = 0.6
    global_weight: StrictFloat = 0.4
    weight_schedule: RewardWeightScheduleConfig | None = None
    provider: Literal["qwen_dashscope", "deepseek", "placeholder"] = "qwen_dashscope"
    allow_placeholder: StrictBool = Field(
        default=False,
        validation_alias=AliasChoices("allow_placeholder", "allow_placeholder_reward"),
    )
    api_key_env: _NON_EMPTY = "DASHSCOPE_API_KEY"
    base_url: _NON_EMPTY = (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
        "multimodal-generation/generation"
    )
    schema_version: Literal["rlff.reward.v1"] = REWARD_SCHEMA_VERSION

    @field_validator("completion_weight", "global_weight")
    @classmethod
    def weights_are_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("reward weights must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_reward_provider(self) -> RewardConfig:
        if self.provider == "placeholder" and not self.allow_placeholder:
            raise ValueError("placeholder rewards require allow_placeholder=true explicitly")
        if self.completion_weight == 0 and self.global_weight == 0:
            raise ValueError("at least one reward weight must be positive")
        return self


class RoleGRPOConfig(ConfigModel):
    """Numerical role-level GRPO post-processing settings."""

    reward_std_epsilon: StrictFloat = 1e-8
    min_group_size: _POSITIVE_INT = 2
    drop_incomplete_trajectory: Literal[True] = True
    normalize_by_role: Literal[True] = True
    # When enabled, a complete group is accepted only if at least one
    # character's raw trajectory reward differs across trajectories. Rejected
    # groups are regenerated from the same episode without an attempt limit.
    dynamic_trajectory_resampling: StrictBool = False

    @field_validator("reward_std_epsilon")
    @classmethod
    def epsilon_is_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("reward_std_epsilon must be non-negative")
        return value


class BF16LoRAConfig(ConfigModel):
    """Continue training the exact existing SFT adapter in BF16 LoRA mode."""

    base_model: _NON_EMPTY
    sft_adapter_path: _NON_EMPTY = Field(
        validation_alias=AliasChoices("sft_adapter_path", "adapter_path")
    )
    dtype: Literal["bfloat16"] = "bfloat16"
    # These fields are optional because older RLFF configs did not duplicate
    # the native AReaL LoRA section.  When supplied, runtime preflight checks
    # adapter_config.json against them instead of silently changing adapter
    # topology or creating a fresh adapter.
    rank: _POSITIVE_INT | None = Field(
        default=None,
        validation_alias=AliasChoices("rank", "lora_rank"),
    )
    alpha: StrictFloat | StrictInt | None = Field(
        default=None,
        validation_alias=AliasChoices("alpha", "lora_alpha"),
    )
    target_modules: tuple[_NON_EMPTY, ...] | None = None
    peft_type: Literal["lora"] = "lora"

class AReaLConfig(ConfigModel):
    """AReaL v1.0.4 integration references and RLFF-owned invariants.

    The native AReaL YAML remains the owner of optimizer, scheduler,
    micro-batch, allocation, and checkpoint details.  RLFF stores only the
    immutable upstream/version references and the handful of values that must
    not drift from the role-level reward semantics.
    """

    backend: Literal["areal"] = "areal"
    version: Literal["1.0.4"] = AREAL_VERSION
    commit: Literal["37d6c6400e99a05fa3409d6a067762a44df40d3b"] = AREAL_COMMIT
    sglang_version: Literal["0.5.10.post1"] = SGLANG_VERSION
    official_yaml: Path = Field(
        default=Path(DEFAULT_AREAL_YAML),
        validation_alias=AliasChoices(
            "official_yaml", "official_yaml_path", "yaml_path", "config_path"
        ),
    )
    world_size: _POSITIVE_INT = 1
    rank: _NON_NEGATIVE_INT = 0
    rollout_workers: _POSITIVE_INT = 1
    use_bf16: Literal[True] = True
    # AReaL's native grouping is the transport for one RLFF role-normalization
    # group.  The root validator below requires this to equal
    # episode_grouping.group_size; keeping it here makes the referenced native
    # YAML constraint explicit without copying the full AReaL config.
    native_n_samples: _POSITIVE_INT = 4
    native_reward_norm: Literal["none"] = "none"
    native_adv_norm: Literal["none"] = "none"
    native_reward_scaling: StrictFloat = 1.0
    native_reward_bias: StrictFloat = 0.0
    native_discount: StrictFloat = 1.0
    native_gae_lambda: StrictFloat = 1.0
    proxy_base_url: _NON_EMPTY = "http://127.0.0.1:8000"
    proxy_admin_api_key_env: _NON_EMPTY = "AREAL_PROXY_ADMIN_API_KEY"
    proxy_mode: Literal["inline"] = "inline"
    proxy_turn_discount: StrictFloat = 0.0
    proxy_export_style: Literal["individual"] = "individual"
    proxy_group_timeout_seconds: StrictFloat = 600.0
    # The default is intentionally absent: cloud launch must provide an agent
    # implementation (or a workflow path) rather than inventing a narrator or
    # environment actor.
    proxy_workflow_path: _NON_EMPTY | None = None

    @field_validator("official_yaml")
    @classmethod
    def official_yaml_is_yaml(cls, value: Path) -> Path:
        if value.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("official_yaml must point to an AReaL YAML file")
        return value

    @field_validator(
        "native_reward_scaling",
        "native_reward_bias",
        "native_discount",
        "native_gae_lambda",
    )
    @classmethod
    def native_numeric_constraints(cls, value: float, info: ValidationInfo) -> float:
        if info.field_name is None:  # defensive: this validator is field-bound
            raise ValueError("native numeric constraint has no field name")
        expected = {
            "native_reward_scaling": 1.0,
            "native_reward_bias": 0.0,
            "native_discount": 1.0,
            "native_gae_lambda": 1.0,
        }[info.field_name]
        if value != expected:
            raise ValueError(f"{info.field_name} is locked to {expected}")
        return value

    @field_validator("proxy_turn_discount")
    @classmethod
    def proxy_turn_discount_is_zero(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("proxy_turn_discount is locked to 0.0")
        return value

    @field_validator("proxy_group_timeout_seconds")
    @classmethod
    def proxy_timeout_is_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("proxy_group_timeout_seconds must be positive")
        return value

    @model_validator(mode="after")
    def rank_in_world(self) -> AReaLConfig:
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        return self


class CheckpointConfig(ConfigModel):
    """Checkpoint/output metadata settings."""

    output_dir: Path
    every_steps: _POSITIVE_INT = 100
    keep_last: _POSITIVE_INT = 3
    resume_from: Path | None = None


class ObservabilityConfig(ConfigModel):
    """Focused reward/training audit settings with env-only secrets."""

    langsmith_tracing: StrictBool = False
    langsmith_project: _NON_EMPTY = "rlff"
    langsmith_api_key_env: _NON_EMPTY = "LANGSMITH_API_KEY"
    # Compact rows for every parsed completion and trajectory reward.
    audit_jsonl: Path | None = None
    # Full request/response rows for a stable subset of trajectories.
    reward_detail_jsonl: Path | None = None
    reward_detail_sample_rate: _PROBABILITY = 0.015625
    # Full request/response history for every reward operation that exhausts retries.
    reward_failure_jsonl: Path | None = None
    # AReaL-native actor loss and clip ratio only; no loss is recomputed.
    training_metrics_jsonl: Path | None = None


@dataclass(frozen=True)
class RuntimeSecrets:
    """Process-local secret values; this type is never part of config dumps."""

    reward_api_key: str | None
    langsmith_api_key: str | None
    sglang_api_key: str | None

    @property
    def deepseek_api_key(self) -> str | None:
        """Backward-compatible alias for older reward-model test tooling."""

        return self.reward_api_key

    def redacted(self) -> dict[str, bool]:
        return {
            "reward_api_key": self.reward_api_key is not None,
            "langsmith_api_key": self.langsmith_api_key is not None,
            "sglang_api_key": self.sglang_api_key is not None,
        }


class RLFFConfig(ConfigModel):
    """Single root configuration consumed by the RLFF runtime."""

    episode_grouping: EpisodeGroupingConfig = Field(
        validation_alias=AliasChoices("episode_grouping", "episode", "grouping")
    )
    prompt: PromptTemplateConfig = Field(
        default_factory=PromptTemplateConfig,
        validation_alias=AliasChoices("prompt", "prompt_templates"),
    )
    rollout: RolloutConfig = RolloutConfig()
    sglang: SGLangConfig
    rewards: RewardConfig
    grpo: RoleGRPOConfig = Field(
        default_factory=RoleGRPOConfig,
        validation_alias=AliasChoices("grpo", "role_grpo"),
    )
    lora: BF16LoRAConfig = Field(validation_alias=AliasChoices("lora", "model"))
    areal: AReaLConfig = Field(
        default_factory=AReaLConfig,
        validation_alias=AliasChoices("areal", "runtime"),
    )
    checkpoint: CheckpointConfig = Field(
        validation_alias=AliasChoices("checkpoint", "checkpointing")
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        validation_alias=AliasChoices("observability", "logging"),
    )

    @model_validator(mode="after")
    def native_group_matches_rlff_group(self) -> RLFFConfig:
        if self.areal.native_n_samples != self.episode_grouping.group_size:
            raise ValueError(
                "areal.native_n_samples must equal episode_grouping.group_size; "
                "AReaL native grouping is the RLFF normalization group"
            )
        return self

    @property
    def episode(self) -> EpisodeGroupingConfig:
        return self.episode_grouping

    @property
    def model(self) -> BF16LoRAConfig:
        return self.lora

    @property
    def runtime(self) -> AReaLConfig:
        return self.areal

    def resolved_snapshot(self) -> dict[str, object]:
        """Return a JSON-ready non-secret configuration snapshot."""

        return self.model_dump(mode="json", exclude_none=True)

    def config_fingerprint(self) -> str:
        return stable_fingerprint(self.resolved_snapshot())

    def resolve_runtime_secrets(self, *, require_reward_key: bool | None = None) -> RuntimeSecrets:
        """Resolve env-referenced credentials without adding them to config state."""

        require_reward = (
            self.rewards.provider != "placeholder"
            if require_reward_key is None
            else require_reward_key
        )
        reward_key = os.getenv(self.rewards.api_key_env)
        if require_reward and not reward_key:
            raise RuntimeError(
                f"reward provider requires environment variable {self.rewards.api_key_env}"
            )
        langsmith_key = os.getenv(self.observability.langsmith_api_key_env)
        if self.observability.langsmith_tracing and not langsmith_key:
            raise RuntimeError(
                "LangSmith tracing requires environment variable "
                f"{self.observability.langsmith_api_key_env}"
            )
        sglang_key = (
            os.getenv(self.sglang.api_key_env) if self.sglang.api_key_env is not None else None
        )
        return RuntimeSecrets(reward_key, langsmith_key, sglang_key)


def load_config(path: str | Path) -> RLFFConfig:
    """Load a UTF-8/UTF-8-SIG YAML or JSON configuration file."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"cannot read config {config_path}: {exc}") from exc
    try:
        raw = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid config syntax in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"config root in {config_path} must be an object")
    try:
        return RLFFConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"invalid config in {config_path}: {exc}") from exc


__all__ = [
    "AREAL_COMMIT",
    "AREAL_VERSION",
    "DEFAULT_AREAL_YAML",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_REWARD_MODEL",
    "SGLANG_VERSION",
    "AReaLConfig",
    "BF16LoRAConfig",
    "CheckpointConfig",
    "EpisodeGroupingConfig",
    "ObservabilityConfig",
    "PromptTemplateConfig",
    "RLFFConfig",
    "RewardConfig",
    "RewardScopeConfig",
    "RewardWeightScheduleConfig",
    "RoleGRPOConfig",
    "RolloutConfig",
    "RuntimeSecrets",
    "SGLangConfig",
    "load_config",
]
