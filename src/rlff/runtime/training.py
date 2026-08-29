"""Training entrypoint and dry-run runtime description.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import AREAL_COMMIT, AREAL_VERSION, SGLANG_VERSION, RLFFConfig
from ..proxy import RLFFGroupAwareAgent
from .integration import (
    _build_areal_runtime_classes,
    build_agent_workflow_kwargs,
    build_areal_training_dataset,
    load_native_areal_config,
)
from .preflight import build_runtime_plan
from .types import TRAINING_METRICS_ENV


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
    if config.observability.training_metrics_jsonl is not None:
        os.environ[TRAINING_METRICS_ENV] = str(
            config.observability.training_metrics_jsonl.resolve()
        )
    else:
        os.environ.pop(TRAINING_METRICS_ENV, None)
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
