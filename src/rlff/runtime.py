"""Lazy AReaL v1.0.4 runtime integration.

The CPU-facing part of RLFF is deliberately independent of AReaL, PEFT,
Transformers, CUDA, and SGLang.  This module therefore performs all local
preflight using ordinary files and YAML, and imports the pinned training stack
only from the cloud execution helpers.  The most important boundary is the
existing LLaMA-Factory adapter: AReaL's native ``_apply_peft_wrapper`` creates
a new adapter, so the custom FSDP actor below loads the existing adapter with
``PeftModel.from_pretrained`` for the trainable actor.

The proxy integration uses AReaL v1.0.4's official agent-like workflow path.
``GroupedRolloutWorkflow`` runs the inline :class:`RLFFGroupAwareAgent` four
times against one shared event loop; the agent's barrier performs RLFF role
normalization before returning per-session reward IDs.  AReaL then owns
session lifecycle and exact token/log-probability export.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, cast

import yaml

from .config import AREAL_COMMIT, AREAL_VERSION, SGLANG_VERSION, RLFFConfig
from .contracts import canonical_json
from .proxy import RLFFGroupAwareAgent

AREAL_ADAPTER_ENV: Final[str] = "RLFF_SFT_ADAPTER_PATH"
"""Environment variable propagated to every actor AReaL worker."""

AREAL_VERSION_TAG: Final[str] = "areal-v1.0.4"

AREAL_PROXY_LORA_NAME: Final[str] = "default_lora"
"""LoRA selector required by AReaL v1.0.4's OpenAI agent proxy.

The proxy converts OpenAI requests back into AReaL generation parameters.  The
OpenAI request schema cannot carry ``lora_name``, so that conversion uses
``GenerationHyperparameters``' default selector.  The rollout server must load
the trainable adapter under the same name (with AReaL's ``-vN`` suffix).
"""

HF_CHECKPOINTS_TO_KEEP: Final[int] = 1
"""Number of versioned Hugging Face/LoRA exports retained per model."""

_HF_CHECKPOINT_DIR = re.compile(
    r"^epoch(?P<epoch>\d+)epochstep(?P<epoch_step>\d+)globalstep(?P<global_step>\d+)$"
)

logger = logging.getLogger(__name__)


class RuntimeCompatibilityError(RuntimeError):
    """Raised when a pinned cloud boundary cannot be proven safe."""


class AdapterPreflightError(RuntimeCompatibilityError):
    """Raised when an existing SFT PEFT adapter is not compatible."""


class AReaLUnavailableError(RuntimeCompatibilityError):
    """Raised when a cloud-only operation is requested without AReaL."""


class ProxyWorkflowUnavailableError(RuntimeCompatibilityError):
    """Raised when the pinned official agent-like proxy path is unavailable."""


def prune_old_hf_checkpoints(
    model_save_root: str | Path,
    *,
    keep: int = HF_CHECKPOINTS_TO_KEEP,
) -> tuple[Path, ...]:
    """Delete old versioned HF exports while preserving recovery state.

    AReaL writes ordinary evaluation exports to directories named
    ``epoch{n}epochstep{n}globalstep{n}``, but writes resumable DCP state to
    the fixed ``recover_checkpoint`` directory. Matching the complete HF
    directory name keeps cleanup deliberately narrow: recovery state, logs,
    metadata, and unrelated user files are never candidates.

    The caller invokes this only after AReaL's synchronous save and recovery
    barriers complete, so the newest export is fully written before an older
    one is removed.
    """

    if keep < 1:
        raise ValueError("keep must be at least 1")
    root = Path(model_save_root)
    if not root.is_dir():
        return ()

    checkpoints: list[tuple[tuple[int, int, int], Path]] = []
    for child in root.iterdir():
        match = _HF_CHECKPOINT_DIR.fullmatch(child.name)
        if match is None or not child.is_dir():
            continue
        checkpoints.append(
            (
                (
                    int(match.group("global_step")),
                    int(match.group("epoch")),
                    int(match.group("epoch_step")),
                ),
                child,
            )
        )

    checkpoints.sort(key=lambda item: item[0])
    removed: list[Path] = []
    for _order, checkpoint in checkpoints[:-keep]:
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return tuple(removed)


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    """The PEFT fields needed to prove adapter identity and topology."""

    path: Path
    config_path: Path
    base_model_name_or_path: str
    peft_type: str
    task_type: str
    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    adapter_fingerprint: str


@dataclass(frozen=True, slots=True)
class AReaLYamlConstraints:
    """Validated subset of the official native AReaL YAML."""

    path: Path
    native_n_samples: int
    actor_backend: str | None
    actor_path: str | None
    actor_lora_rank: int | None
    actor_lora_alpha: float | None
    actor_target_modules: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Non-secret launch plan produced by local preflight."""

    config_fingerprint: str
    adapter: AdapterMetadata
    areal_yaml: AReaLYamlConstraints
    adapter_env_var: str = AREAL_ADAPTER_ENV

    @property
    def environment(self) -> dict[str, str]:
        return {self.adapter_env_var: str(self.adapter.path)}

    def apply_environment(self) -> None:
        """Set the process env after checking any pre-existing value."""

        current = os.getenv(self.adapter_env_var)
        if current and _canonical_path(current) != self.adapter.path:
            raise AdapterPreflightError(
                f"{self.adapter_env_var} points to {current!r}, but RLFF config selects "
                f"the existing adapter at {str(self.adapter.path)!r}"
            )
        os.environ[self.adapter_env_var] = str(self.adapter.path)


def role_advantage_tensor_data(data: dict[str, Any]) -> dict[str, Any]:
    """Preserve proxy role rewards as fixed completion-token advantages.

    AReaL's native interaction export supplies one scalar ``rewards`` value per
    completion.  RLFF has already normalized that scalar by role, so this
    adapter broadcasts it over the aligned completion loss mask and never runs
    native reward/advantage normalization, GAE, or KL reward shaping.
    """

    import torch

    loss_mask = torch.roll(data["loss_mask"].float(), shifts=-1, dims=-1)
    reward_score = data["rewards"].float()
    old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
    old_logp = old_logp * loss_mask
    role_advantages = reward_score.unsqueeze(-1) * loss_mask
    data["advantages"] = role_advantages
    data["returns"] = role_advantages
    data["kl_rewards"] = torch.zeros_like(role_advantages)
    data["tot_rewards"] = role_advantages
    data["loss_mask"] = loss_mask
    data["logprobs"] = old_logp
    if not data.get("use_decoupled_loss", False):
        data["prox_logp"] = old_logp
    return data


def _canonical_path(value: str | Path) -> Path:
    """Resolve a path without requiring it to exist (useful for diagnostics)."""

    return Path(value).expanduser().resolve(strict=False)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise AdapterPreflightError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AdapterPreflightError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AdapterPreflightError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], raw)


def _normalise_target_modules(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or not value:
        raise AdapterPreflightError(f"{field} must be a non-empty list or string")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise AdapterPreflightError(f"{field} contains an empty target module")
    if len(set(result)) != len(result):
        raise AdapterPreflightError(f"{field} contains duplicate target modules")
    return result


def _same_target_modules(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Compare PEFT target modules as an unordered topology."""

    return set(left) == set(right)


def _disable_dropout_modules(model: Any, torch_module: Any) -> int:
    """Disable every dropout module, including those added by PEFT."""

    count = 0
    for module in model.modules():
        if isinstance(module, torch_module.nn.Dropout):
            module.p = 0.0
            count += 1
    return count


def _coerce_int(raw: Any, *, field: str) -> int:
    # bool is an int subclass but is never a valid PEFT topology value.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise AdapterPreflightError(f"adapter {field} must be a positive integer")
    return raw


def _coerce_float(raw: Any, *, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise AdapterPreflightError(f"adapter {field} must be numeric")
    value = float(raw)
    if value <= 0 or not value == value or value in {float("inf"), float("-inf")}:
        raise AdapterPreflightError(f"adapter {field} must be finite and positive")
    return value


def _same_model_path(expected: str, actual: str) -> bool:
    """Compare local paths canonically while preserving hub ID semantics."""

    expected_path = Path(expected).expanduser()
    actual_path = Path(actual).expanduser()
    if (
        expected_path.exists()
        or actual_path.exists()
        or expected_path.is_absolute()
        or actual_path.is_absolute()
    ):
        return _canonical_path(expected) == _canonical_path(actual)
    return expected == actual


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AdapterPreflightError(f"cannot fingerprint {path}: {exc}") from exc
    return digest.hexdigest()


def inspect_sft_adapter(
    config: RLFFConfig,
    *,
    yaml_constraints: AReaLYamlConstraints | None = None,
) -> AdapterMetadata:
    """Read and validate the exact existing PEFT adapter.

    No adapter is written, merged, or recreated.  ``adapter_model.safetensors``
    (or the PEFT bin equivalent) must already exist next to
    ``adapter_config.json``.  Native YAML LoRA fields are checked when present;
    omitted fields remain owned by the adapter's own config.
    """

    adapter_path = _canonical_path(config.lora.sft_adapter_path)
    if not adapter_path.is_dir():
        raise AdapterPreflightError(
            f"existing SFT adapter directory does not exist: {adapter_path}"
        )
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise AdapterPreflightError(
            f"existing SFT adapter is missing required {config_path.name}: {config_path}"
        )
    adapter_config = _read_json_object(config_path)

    peft_type = str(adapter_config.get("peft_type", "")).strip().casefold()
    if peft_type != "lora":
        raise AdapterPreflightError(
            f"existing adapter must be PEFT LoRA, found peft_type={peft_type!r}"
        )
    task_type = str(adapter_config.get("task_type", "")).strip().upper()
    if task_type != "CAUSAL_LM":
        raise AdapterPreflightError(
            f"existing adapter must use task_type=CAUSAL_LM, found {task_type!r}"
        )
    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise AdapterPreflightError(
            "existing adapter adapter_config.json must declare base_model_name_or_path"
        )
    if not _same_model_path(config.lora.base_model, base_model):
        raise AdapterPreflightError(
            "existing adapter base_model_name_or_path does not match RLFF lora.base_model: "
            f"{base_model!r} != {config.lora.base_model!r}"
        )

    rank = _coerce_int(adapter_config.get("r"), field="r")
    alpha = _coerce_float(adapter_config.get("lora_alpha"), field="lora_alpha")
    targets = _normalise_target_modules(
        adapter_config.get("target_modules"), field="adapter target_modules"
    )

    if config.lora.rank is not None and rank != config.lora.rank:
        raise AdapterPreflightError(
            f"existing adapter rank {rank} does not match RLFF lora.rank {config.lora.rank}"
        )
    if config.lora.alpha is not None and alpha != float(config.lora.alpha):
        raise AdapterPreflightError(
            f"existing adapter alpha {alpha} does not match RLFF lora.alpha {config.lora.alpha}"
        )
    if config.lora.target_modules is not None:
        expected_targets = tuple(config.lora.target_modules)
        if not _same_target_modules(targets, expected_targets):
            raise AdapterPreflightError(
                "existing adapter target_modules do not match RLFF configuration: "
                f"{targets!r} != {expected_targets!r}"
            )
    if yaml_constraints is not None:
        if (
            yaml_constraints.actor_lora_rank is not None
            and rank != yaml_constraints.actor_lora_rank
        ):
            raise AdapterPreflightError(
                "existing adapter rank does not match official AReaL actor.lora_rank"
            )
        if (
            yaml_constraints.actor_lora_alpha is not None
            and alpha != yaml_constraints.actor_lora_alpha
        ):
            raise AdapterPreflightError(
                "existing adapter alpha does not match official AReaL actor.lora_alpha"
            )
        if yaml_constraints.actor_target_modules is not None and not _same_target_modules(
            targets, yaml_constraints.actor_target_modules
        ):
            raise AdapterPreflightError(
                "existing adapter target_modules do not match official AReaL actor.target_modules"
            )

    weight_files = (
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
        adapter_path / "pytorch_model.bin",
    )
    if not any(item.is_file() for item in weight_files):
        raise AdapterPreflightError(
            "existing SFT adapter has no adapter_model.safetensors or adapter_model.bin"
        )

    return AdapterMetadata(
        path=adapter_path,
        config_path=config_path,
        base_model_name_or_path=base_model,
        peft_type=peft_type,
        task_type=task_type,
        rank=rank,
        alpha=alpha,
        target_modules=targets,
        adapter_fingerprint=_fingerprint_file(config_path),
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


def inject_adapter_env(native_config: Any, adapter_path: str | Path) -> Any:
    """Inject the adapter path into actor scheduling specs in-place.

    AReaL's ``PPOActorConfig`` has no adapter path field.  Scheduling env vars
    are the explicit v1.0.4 propagation mechanism, and this helper also sets
    the controller process environment for local/single-controller launches.
    """

    path = str(_canonical_path(adapter_path))
    current = os.getenv(AREAL_ADAPTER_ENV)
    if current and _canonical_path(current) != _canonical_path(path):
        raise AdapterPreflightError(
            f"{AREAL_ADAPTER_ENV} already points to a different adapter: {current!r}"
        )
    os.environ[AREAL_ADAPTER_ENV] = path
    for section_name in ("actor",):
        section = getattr(native_config, section_name, None)
        if section is None:
            continue
        specs = getattr(section, "scheduling_spec", None)
        if specs is None:
            continue
        for spec in specs:
            env_vars = getattr(spec, "env_vars", None)
            if env_vars is None:
                env_vars = {}
                spec.env_vars = env_vars
            env_vars[AREAL_ADAPTER_ENV] = path
    return native_config


def _load_peft_model() -> Any:
    try:
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover - exercised in cloud smoke tests
        raise AReaLUnavailableError("PEFT is required for existing adapter continuation") from exc
    return PeftModel


def apply_existing_sft_adapter(
    model: Any,
    *,
    adapter_path: str | Path | None = None,
    is_trainable: bool,
) -> Any:
    """Attach the existing adapter without constructing a new LoRA config."""

    path = adapter_path or os.getenv(AREAL_ADAPTER_ENV)
    if not path:
        raise AdapterPreflightError(
            f"{AREAL_ADAPTER_ENV} must be set before an AReaL actor worker starts"
        )
    canonical = _canonical_path(path)
    if not canonical.is_dir() or not (canonical / "adapter_config.json").is_file():
        raise AdapterPreflightError(f"existing adapter path is not loadable: {canonical}")
    peft_model = _load_peft_model()
    return peft_model.from_pretrained(
        model,
        str(canonical),
        is_trainable=is_trainable,
        autocast_adapter_dtype=False,
    )


def _load_areal_symbols() -> tuple[Any, Any, Any]:
    try:
        installed_areal = metadata.version("areal")
        installed_sglang = metadata.version("sglang")
    except metadata.PackageNotFoundError as exc:
        raise AReaLUnavailableError(
            "the pinned AReaL and SGLang distributions must be installed for cloud training"
        ) from exc
    if installed_areal != AREAL_VERSION:
        raise RuntimeCompatibilityError(
            f"installed areal=={installed_areal}, expected exactly {AREAL_VERSION}"
        )
    if installed_sglang != SGLANG_VERSION:
        raise RuntimeCompatibilityError(
            f"installed sglang=={installed_sglang}, expected exactly {SGLANG_VERSION}"
        )
    try:
        engine_module = importlib.import_module("areal.engine.fsdp_engine")
        trainer_module = importlib.import_module("areal.trainer.rl_trainer")
        cli_module = importlib.import_module("areal.api.cli_args")
    except Exception as exc:  # pragma: no cover - CPU environment has no AReaL
        raise AReaLUnavailableError(
            "AReaL v1.0.4 is required only for cloud training; install the pinned runtime"
        ) from exc
    return engine_module, trainer_module, cli_module


_AREAL_RUNTIME_CLASSES: tuple[Any, Any] | None = None


def _build_areal_runtime_classes() -> tuple[Any, Any]:
    """Create importable pinned actor/trainer classes only on the cloud path."""

    global _AREAL_RUNTIME_CLASSES
    if _AREAL_RUNTIME_CLASSES is not None:
        return _AREAL_RUNTIME_CLASSES

    engine_module, trainer_module, _cli_module = _load_areal_symbols()
    torch = importlib.import_module("torch")
    base_actor = engine_module.FSDPPPOActor
    base_trainer = trainer_module.PPOTrainer

    def actor_apply_peft_wrapper(self: Any) -> None:
        self.model.enable_input_require_grads()
        self.model = apply_existing_sft_adapter(
            self.model,
            is_trainable=True,
        )
        # AReaL disables dropout before this wrapper runs. PEFT creates its
        # LoRA dropout modules here, so disable them again after attachment.
        _disable_dropout_modules(self.model, torch)
        if self.rank == 0:
            self.model.print_trainable_parameters()

    def actor_compute_advantages(self: Any, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _ = self
        with torch.no_grad():
            return [role_advantage_tensor_data(item) for item in data]

    actor_class = type(
        "RLFFFSDPPPOActor",
        (base_actor,),
        {
            "__module__": __name__,
            "__doc__": (
                "Pinned AReaL FSDP actor loading the existing SFT adapter and "
                "preserving fixed RLFF role advantages."
            ),
            "_apply_peft_wrapper": actor_apply_peft_wrapper,
            "compute_advantages": actor_compute_advantages,
        },
    )
    actor_runtime_class = cast(Any, actor_class)

    def trainer_init(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
        adapter_path = os.getenv(AREAL_ADAPTER_ENV)
        if not adapter_path:
            raise AdapterPreflightError(
                f"{AREAL_ADAPTER_ENV} must be set before constructing RLFFPPOTrainer"
            )
        inject_adapter_env(config, adapter_path)
        base_trainer.__init__(self, config, *args, **kwargs)

    def trainer_create_train_engine(self: Any, actor_config: Any, alloc: Any) -> Any:
        if alloc.backend != "fsdp":
            raise RuntimeCompatibilityError(
                "continuing an existing PEFT adapter is supported only by "
                "the pinned AReaL FSDP actor"
            )
        environ_module = importlib.import_module("areal.utils.environ")
        if environ_module.is_single_controller():
            actor = actor_runtime_class.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_runtime_class(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def trainer_train(self: Any, workflow: Any = None, *args: Any, **kwargs: Any) -> Any:
        if getattr(workflow, "requires_rlff_proxy_start", False) and not getattr(
            self, "_proxy_started", False
        ):
            self._ensure_proxy_started()
        return base_trainer.train(self, *args, workflow=workflow, **kwargs)

    def trainer_save_recover_checkpoint(
        self: Any,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ) -> None:
        """Save recoverable state, then retain only the newest LoRA export."""

        base_trainer._save_recover_checkpoint(
            self,
            epoch=epoch,
            epoch_step=epoch_step,
            global_step=global_step,
        )
        saver_config = self.saver.config
        saver_class = type(self.saver)
        model_names = ("default", "critic") if self.critic is not None else ("default",)
        for model_name in model_names:
            model_root = saver_class.get_model_save_root(
                saver_config.experiment_name,
                saver_config.trial_name,
                saver_config.fileroot,
                name=model_name,
            )
            removed = prune_old_hf_checkpoints(model_root)
            for checkpoint in removed:
                logger.info("Removed superseded HF checkpoint: %s", checkpoint)

    trainer_class = type(
        "RLFFPPOTrainer",
        (base_trainer,),
        {
            "__module__": __name__,
            "__doc__": "Pinned AReaL trainer selecting the RLFF FSDP actor.",
            "__init__": trainer_init,
            "_create_train_engine": trainer_create_train_engine,
            "_save_recover_checkpoint": trainer_save_recover_checkpoint,
            "train": trainer_train,
        },
    )
    globals()["RLFFFSDPPPOActor"] = actor_class
    globals()["RLFFPPOTrainer"] = trainer_class
    _AREAL_RUNTIME_CLASSES = (actor_class, trainer_class)
    return _AREAL_RUNTIME_CLASSES


def __getattr__(name: str) -> Any:
    """Let AReaL workers resolve the lazy classes by stable import path."""

    if name in {"RLFFFSDPPPOActor", "RLFFPPOTrainer"}:
        actor_class, trainer_class = _build_areal_runtime_classes()
        return actor_class if name == "RLFFFSDPPPOActor" else trainer_class
    raise AttributeError(name)


def load_native_areal_config(config: RLFFConfig) -> Any:
    """Load the referenced official AReaL PPO YAML on the cloud only."""

    plan = build_runtime_plan(config)
    plan.apply_environment()
    _engine, _trainer, cli = _load_areal_symbols()
    ppo_config = cli.PPOConfig
    loaded, _ = cli.load_expr_config(["--config", str(plan.areal_yaml.path)], ppo_config)
    inject_adapter_env(loaded, plan.adapter.path)
    return loaded


def build_areal_training_dataset(config: RLFFConfig) -> Any:
    """Build one dataset row per canonical RLFF group for native grouping.

    AReaL's ``GroupedRolloutWorkflow`` repeats each row
    ``gconfig.n_samples`` times.  The row therefore carries one deterministic
    episode and its shared render specification; the inline agent uses the
    native group ID as its barrier key.  No tokenization happens here.
    """

    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover - cloud-only dependency
        raise AReaLUnavailableError(
            "datasets is required to build the AReaL training dataset"
        ) from exc
    from .episodes import build_episode_group, load_episode_jsonl, project_target_prompt

    loaded = load_episode_jsonl(
        config.episode_grouping.dataset_path,
        limit=config.episode_grouping.limit,
    )
    templates = config.prompt.load_templates()
    rows: list[dict[str, Any]] = []
    for record in loaded.records:
        group = build_episode_group(
            record,
            group_size=config.episode_grouping.group_size,
            base_seed=config.episode_grouping.base_seed,
            templates=templates,
        )
        render = group.samples[0].render
        # Exercise the exact worker-side renderer before allocating actor or
        # rollout models. This catches missing packaged renderer code, invalid
        # template variables, and character projection failures during preflight.
        for character in group.episode.characters:
            project_target_prompt(
                group.episode,
                character.name,
                render=render,
            )
        rows.append(
            {
                "episode_id": group.episode_id,
                "group_id": group.group_id,
                "render": render.model_dump(mode="json"),
                # Keep the canonical episode opaque to Arrow.  Dataset.from_list
                # otherwise unions arbitrary metadata keys across rows and adds
                # null fields, invalidating the embedded content fingerprint.
                "episode": canonical_json(group.episode),
            }
        )
    if not rows:
        raise RuntimeCompatibilityError("RLFF episode dataset contains no training records")
    return Dataset.from_list(rows)


def build_agent_workflow_kwargs(config: RLFFConfig) -> dict[str, Any]:
    """Translate RLFF-owned rollout/reward settings to the agent constructor."""

    return {
        "group_size": config.episode_grouping.group_size,
        "max_rounds": config.rollout.max_rounds,
        "model": config.sglang.model,
        "temperature": config.sglang.temperature,
        "top_p": config.sglang.top_p,
        "frequency_penalty": config.sglang.frequency_penalty,
        "max_new_tokens": config.sglang.max_new_tokens,
        "rollout_request_timeout_seconds": config.sglang.timeout_seconds,
        "group_timeout_seconds": config.areal.proxy_group_timeout_seconds,
        "completion_weight": config.rewards.completion_weight,
        "global_weight": config.rewards.global_weight,
        "min_group_size": config.grpo.min_group_size,
        "reward_std_epsilon": config.grpo.reward_std_epsilon,
        "reward_provider_name": config.rewards.provider,
        "reward_api_key_env": config.rewards.api_key_env,
        "reward_base_url": config.rewards.base_url,
        "completion_prompt_path": str(config.rewards.completion.prompt_path),
        "trajectory_prompt_path": str(config.rewards.global_reward.prompt_path),
        "reward_model": config.rewards.completion.model,
        "trajectory_reward_model": config.rewards.global_reward.model,
        "completion_reward_timeout_seconds": config.rewards.completion.timeout_seconds,
        "trajectory_reward_timeout_seconds": config.rewards.global_reward.timeout_seconds,
        "completion_reward_retries": config.rewards.completion.retries,
        "trajectory_reward_retries": config.rewards.global_reward.retries,
        "completion_reward_concurrency": config.rewards.completion.concurrency,
        "trajectory_reward_concurrency": config.rewards.global_reward.concurrency,
        "completion_reward_temperature": config.rewards.completion.temperature,
        "trajectory_reward_temperature": config.rewards.global_reward.temperature,
        "completion_reward_reasoning_effort": config.rewards.completion.reasoning_effort,
        "trajectory_reward_reasoning_effort": config.rewards.global_reward.reasoning_effort,
        "completion_reward_max_tokens": config.rewards.completion.max_tokens,
        "trajectory_reward_max_tokens": config.rewards.global_reward.max_tokens,
        "langsmith_tracing": config.observability.langsmith_tracing,
        "langsmith_project": config.observability.langsmith_project,
        "langsmith_api_key_env": config.observability.langsmith_api_key_env,
    }


def run_training(
    config: RLFFConfig,
    *,
    workflow: Any | None = None,
    train_dataset: Any = None,
    valid_dataset: Any = None,
    workflow_kwargs: dict[str, Any] | None = None,
    eval_workflow: Any = None,
    eval_workflow_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Run pinned AReaL training after all local checks pass."""

    plan = build_runtime_plan(config)
    plan.apply_environment()
    if workflow is None:
        workflow = RLFFGroupAwareAgent
    if train_dataset is None:
        train_dataset = build_areal_training_dataset(config)
    if workflow is RLFFGroupAwareAgent:
        merged_workflow_kwargs = build_agent_workflow_kwargs(config)
        if workflow_kwargs:
            merged_workflow_kwargs.update(workflow_kwargs)
        workflow_kwargs = merged_workflow_kwargs
    native_config = load_native_areal_config(config)
    _actor_class, trainer_class = _build_areal_runtime_classes()
    with trainer_class(
        native_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        return trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


def describe_runtime_plan(config: RLFFConfig) -> dict[str, Any]:
    """Return a JSON-friendly, non-secret dry-run report."""

    plan = build_runtime_plan(config)
    return {
        "config_fingerprint": plan.config_fingerprint,
        "areal": {
            "version": AREAL_VERSION,
            "commit": AREAL_COMMIT,
            "sglang_version": SGLANG_VERSION,
            "official_yaml": str(plan.areal_yaml.path),
            "native_n_samples": plan.areal_yaml.native_n_samples,
        },
        "adapter": {
            "path": str(plan.adapter.path),
            "base_model": plan.adapter.base_model_name_or_path,
            "peft_type": plan.adapter.peft_type,
            "task_type": plan.adapter.task_type,
            "rank": plan.adapter.rank,
            "alpha": plan.adapter.alpha,
            "target_modules": list(plan.adapter.target_modules),
            "adapter_fingerprint": plan.adapter.adapter_fingerprint,
        },
        "constraints": {
            "reward_norm": "none",
            "adv_norm": "none",
            "reward_scaling": 1.0,
            "reward_bias": 0.0,
            "discount": 1.0,
            "gae_lambda": 1.0,
            "reference": "disabled; no teacher/ref model and no KL term",
        },
    }


__all__ = [
    "AREAL_ADAPTER_ENV",
    "AREAL_VERSION_TAG",
    "AReaLUnavailableError",
    "AReaLYamlConstraints",
    "AdapterMetadata",
    "AdapterPreflightError",
    "ProxyWorkflowUnavailableError",
    "RLFFGroupAwareAgent",
    "RuntimeCompatibilityError",
    "RuntimePlan",
    "apply_existing_sft_adapter",
    "build_agent_workflow_kwargs",
    "build_areal_training_dataset",
    "build_runtime_plan",
    "describe_runtime_plan",
    "inject_adapter_env",
    "inspect_sft_adapter",
    "load_native_areal_config",
    "role_advantage_tensor_data",
    "run_training",
    "validate_areal_yaml",
]
