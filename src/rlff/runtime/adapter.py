"""SFT LoRA metadata validation and adapter application.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from ..config import RLFFConfig
from .types import AdapterMetadata, AdapterPreflightError, AReaLYamlConstraints, _canonical_path


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
