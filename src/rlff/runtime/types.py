"""Pinned runtime constants, errors, plans, and tensor adaptation.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

AREAL_ADAPTER_ENV: Final[str] = "RLFF_SFT_ADAPTER_PATH"


TRAINING_METRICS_ENV: Final[str] = "RLFF_TRAINING_METRICS_JSONL"


AREAL_VERSION_TAG: Final[str] = "areal-v1.0.4"


AREAL_PROXY_LORA_NAME: Final[str] = "default_lora"


HF_CHECKPOINTS_TO_KEEP: Final[int] = 1


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
