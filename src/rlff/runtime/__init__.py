"""Compatibility facade for the split :mod:`rlff.runtime` implementation."""

# ruff: noqa: F401 -- this module intentionally preserves the former module namespace.

from __future__ import annotations

from typing import Any

from ..proxy import RLFFGroupAwareAgent
from .adapter import (
    _coerce_float,
    _coerce_int,
    _disable_dropout_modules,
    _fingerprint_file,
    _normalise_target_modules,
    _read_json_object,
    _same_model_path,
    _same_target_modules,
    inspect_sft_adapter,
)
from .integration import (
    _AREAL_RUNTIME_CLASSES,
    _load_areal_symbols,
    _load_peft_model,
    apply_existing_sft_adapter,
    build_agent_workflow_kwargs,
    build_areal_training_dataset,
    inject_adapter_env,
    load_native_areal_config,
)
from .integration import (
    _build_areal_runtime_classes as _build_areal_runtime_classes,
)
from .preflight import (
    _nested,
    _optional_float,
    _optional_int,
    _optional_targets,
    build_runtime_plan,
    validate_areal_yaml,
)
from .training import describe_runtime_plan, run_training
from .types import (
    _HF_CHECKPOINT_DIR,
    AREAL_ADAPTER_ENV,
    AREAL_PROXY_LORA_NAME,
    AREAL_VERSION_TAG,
    HF_CHECKPOINTS_TO_KEEP,
    TRAINING_METRICS_ENV,
    AdapterMetadata,
    AdapterPreflightError,
    AReaLUnavailableError,
    AReaLYamlConstraints,
    ProxyWorkflowUnavailableError,
    RuntimeCompatibilityError,
    RuntimePlan,
    _canonical_path,
    logger,
    prune_old_hf_checkpoints,
    role_advantage_tensor_data,
)


def __getattr__(name: str) -> Any:
    from . import integration

    return getattr(integration, name)


__all__ = [
    "AREAL_ADAPTER_ENV",
    "AREAL_VERSION_TAG",
    "TRAINING_METRICS_ENV",
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
