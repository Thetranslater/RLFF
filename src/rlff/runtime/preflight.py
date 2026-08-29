"""AReaL YAML validation and local runtime planning.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from ..config import AREAL_COMMIT, AREAL_VERSION, SGLANG_VERSION, RLFFConfig
from .adapter import (
    _coerce_float,
    _coerce_int,
    _normalise_target_modules,
    _same_model_path,
    inspect_sft_adapter,
)
from .types import (
    AREAL_PROXY_LORA_NAME,
    AdapterPreflightError,
    AReaLYamlConstraints,
    RuntimeCompatibilityError,
    RuntimePlan,
    _canonical_path,
)


def _nested(raw: Mapping[str, Any], *keys: str) -> Any:
    current: Any = raw
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, field=field)


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, field=field)


def _optional_targets(value: Any, *, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _normalise_target_modules(value, field=field)


def validate_areal_yaml(
    config: RLFFConfig,
    *,
    yaml_path: str | Path | None = None,
) -> AReaLYamlConstraints:
    """Validate only RLFF-owned invariants in the official native YAML."""

    path = _canonical_path(yaml_path or config.areal.official_yaml)
    if not path.is_file():
        raise RuntimeCompatibilityError(
            f"official AReaL v{AREAL_VERSION} YAML does not exist: {path}"
        )
    try:
        raw_value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeCompatibilityError(f"cannot read official AReaL YAML {path}: {exc}") from exc
    if not isinstance(raw_value, Mapping):
        raise RuntimeCompatibilityError("official AReaL YAML root must be a mapping")
    raw = cast(Mapping[str, Any], raw_value)

    def require_exact(keys: tuple[str, ...], expected: Any, label: str) -> None:
        value = _nested(raw, *keys)
        if value != expected:
            raise RuntimeCompatibilityError(
                f"official AReaL YAML {label} must be {expected!r}, found {value!r}"
            )

    scheduler = _nested(raw, "scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("type") != "local":
        raise RuntimeCompatibilityError(
            "official AReaL scheduler.type must be 'local' for the single-node "
            "single-controller RLFF runtime"
        )

    require_exact(("gconfig", "n_samples"), config.areal.native_n_samples, "gconfig.n_samples")
    require_exact(("gconfig", "temperature"), config.sglang.temperature, "gconfig.temperature")
    require_exact(("gconfig", "top_p"), config.sglang.top_p, "gconfig.top_p")
    require_exact(
        ("gconfig", "lora_name"),
        AREAL_PROXY_LORA_NAME,
        "gconfig.lora_name for the AReaL OpenAI agent proxy",
    )
    require_exact(
        ("gconfig", "max_new_tokens"),
        config.sglang.max_new_tokens,
        "gconfig.max_new_tokens",
    )
    for field, expected in (
        ("reward_scaling", config.areal.native_reward_scaling),
        ("reward_bias", config.areal.native_reward_bias),
        ("discount", config.areal.native_discount),
        ("gae_lambda", config.areal.native_gae_lambda),
    ):
        require_exact(("actor", field), expected, f"actor.{field}")

    actor_reward_norm = _nested(raw, "actor", "reward_norm")
    actor_adv_norm = _nested(raw, "actor", "adv_norm")
    if actor_reward_norm not in (None, "none"):
        raise RuntimeCompatibilityError("official AReaL actor.reward_norm must be null/none")
    if actor_adv_norm not in (None, "none"):
        raise RuntimeCompatibilityError("official AReaL actor.adv_norm must be null/none")

    actor = _nested(raw, "actor")
    if not isinstance(actor, Mapping):
        raise RuntimeCompatibilityError("official AReaL YAML must define actor")
    if actor.get("use_lora") is not True:
        raise RuntimeCompatibilityError("official AReaL actor.use_lora must be true")
    if actor.get("disable_dropout") is not True:
        raise RuntimeCompatibilityError("official AReaL actor.disable_dropout must be true")
    if actor.get("temperature") != config.sglang.temperature:
        raise RuntimeCompatibilityError(
            "official AReaL actor.temperature must match rollout temperature"
        )
    if actor.get("attn_impl") != "sdpa":
        raise RuntimeCompatibilityError(
            "official AReaL actor.attn_impl must be 'sdpa' to bypass external FlashAttention"
        )
    if actor.get("kl_ctl") != 0.0:
        raise RuntimeCompatibilityError(
            "official AReaL actor.kl_ctl must be 0.0; RLFF disables KL shaping"
        )
    if actor.get("recompute_logprob") is not False:
        raise RuntimeCompatibilityError(
            "official AReaL actor.recompute_logprob must be false for exact rollout logprobs"
        )
    if actor.get("use_decoupled_loss") is not False:
        raise RuntimeCompatibilityError(
            "official AReaL actor.use_decoupled_loss must be false for fixed RLFF advantages"
        )
    if _nested(raw, "teacher") is not None or _nested(raw, "ref") is not None:
        raise RuntimeCompatibilityError(
            "official AReaL teacher/ref must be absent because RLFF disables reference KL"
        )
    actor_path = actor.get("path") if isinstance(actor.get("path"), str) else None
    actor_backend = actor.get("backend") if isinstance(actor.get("backend"), str) else None
    if actor_backend is not None and not actor_backend.startswith("fsdp:"):
        raise RuntimeCompatibilityError(
            "existing PEFT adapter continuation is locked to the FSDP AReaL actor backend"
        )
    if isinstance(actor.get("peft_type"), str) and actor["peft_type"].casefold() != "lora":
        raise RuntimeCompatibilityError("official AReaL actor.peft_type must be lora")

    sglang = _nested(raw, "sglang")
    if not isinstance(sglang, Mapping):
        raise RuntimeCompatibilityError("official AReaL YAML must define sglang")
    if sglang.get("attention_backend") != "triton":
        raise RuntimeCompatibilityError(
            "official AReaL sglang.attention_backend must be 'triton' to bypass FA3/FA4"
        )
    if sglang.get("sampling_backend") != "pytorch":
        raise RuntimeCompatibilityError(
            "official AReaL sglang.sampling_backend must be 'pytorch' to bypass FlashInfer"
        )
    if sglang.get("context_length") != config.sglang.context_length:
        raise RuntimeCompatibilityError(
            "official AReaL sglang.context_length must match RLFF context_length"
        )

    train_dataset = _nested(raw, "train_dataset")
    if not isinstance(train_dataset, Mapping):
        raise RuntimeCompatibilityError("official AReaL YAML must define train_dataset")
    if train_dataset.get("max_length") != config.sglang.context_length:
        raise RuntimeCompatibilityError(
            "official AReaL train_dataset.max_length must match RLFF context_length"
        )

    rollout = _nested(raw, "rollout")
    if not isinstance(rollout, Mapping):
        raise RuntimeCompatibilityError("official AReaL YAML must define rollout")
    if "lora_name" in rollout:
        raise RuntimeCompatibilityError(
            "official AReaL rollout must not define lora_name; AReaL v1.0.4 "
            "accepts it only under gconfig"
        )
    if rollout.get("use_lora") is not True:
        raise RuntimeCompatibilityError("official AReaL rollout.use_lora must be true")
    rollout_specs = rollout.get("scheduling_spec")
    if not isinstance(rollout_specs, list) or not rollout_specs:
        raise RuntimeCompatibilityError(
            "official AReaL rollout.scheduling_spec must define the rollout RPC worker"
        )
    for index, spec in enumerate(rollout_specs):
        if not isinstance(spec, Mapping):
            raise RuntimeCompatibilityError(
                f"official AReaL rollout.scheduling_spec[{index}] must be a mapping"
            )
        env_vars = spec.get("env_vars")
        if not isinstance(env_vars, Mapping) or str(env_vars.get("TMS_INIT_ENABLE")) != "0":
            raise RuntimeCompatibilityError(
                "official AReaL rollout scheduling workers must set "
                "TMS_INIT_ENABLE='0' to avoid nested torch-memory-saver regions"
            )

    agent = rollout.get("agent")
    if not isinstance(agent, Mapping):
        raise RuntimeCompatibilityError("official AReaL rollout.agent is required")
    if agent.get("mode") != config.areal.proxy_mode:
        raise RuntimeCompatibilityError(
            f"official AReaL rollout.agent.mode must be {config.areal.proxy_mode!r}"
        )
    if agent.get("turn_discount") != config.areal.proxy_turn_discount:
        raise RuntimeCompatibilityError("official AReaL rollout.agent.turn_discount must be 0.0")
    if agent.get("export_style") != config.areal.proxy_export_style:
        raise RuntimeCompatibilityError(
            "official AReaL rollout.agent.export_style must be 'individual'"
        )
    agent_cls_path = agent.get("agent_cls_path")
    if not isinstance(agent_cls_path, str) or not agent_cls_path.strip():
        raise RuntimeCompatibilityError("official AReaL rollout.agent.agent_cls_path is required")

    return AReaLYamlConstraints(
        path=path,
        native_n_samples=int(raw["gconfig"]["n_samples"]),
        actor_backend=actor_backend,
        actor_path=actor_path,
        actor_lora_rank=_optional_int(actor.get("lora_rank"), field="actor.lora_rank"),
        actor_lora_alpha=_optional_float(actor.get("lora_alpha"), field="actor.lora_alpha"),
        actor_target_modules=_optional_targets(
            actor.get("target_modules"), field="actor.target_modules"
        ),
    )


def build_runtime_plan(config: RLFFConfig) -> RuntimePlan:
    """Run all local preflight checks needed before a cloud launch."""

    if config.areal.version != AREAL_VERSION or config.areal.commit != AREAL_COMMIT:
        raise RuntimeCompatibilityError(
            f"RLFF is locked to AReaL v{AREAL_VERSION} ({AREAL_COMMIT}); "
            f"received v{config.areal.version} ({config.areal.commit})"
        )
    if config.areal.sglang_version != SGLANG_VERSION:
        raise RuntimeCompatibilityError(
            f"RLFF is locked to sglang=={SGLANG_VERSION}; received {config.areal.sglang_version}"
        )
    native = validate_areal_yaml(config)
    adapter = inspect_sft_adapter(config, yaml_constraints=native)
    if (
        native.actor_path
        and not native.actor_path.startswith("${")
        and not _same_model_path(config.lora.base_model, native.actor_path)
    ):
        raise AdapterPreflightError("official AReaL actor.path does not match RLFF lora.base_model")
    return RuntimePlan(
        config_fingerprint=config.config_fingerprint(),
        adapter=adapter,
        areal_yaml=native,
    )
