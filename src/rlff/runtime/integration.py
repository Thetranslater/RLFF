"""Lazy AReaL class integration, dataset, and workflow wiring.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import importlib
import os
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from ..config import AREAL_VERSION, SGLANG_VERSION, RLFFConfig
from ..contracts import canonical_json
from .adapter import _disable_dropout_modules
from .preflight import build_runtime_plan
from .types import (
    AREAL_ADAPTER_ENV,
    TRAINING_METRICS_ENV,
    AdapterPreflightError,
    AReaLUnavailableError,
    RuntimeCompatibilityError,
    _canonical_path,
    logger,
    prune_old_hf_checkpoints,
    role_advantage_tensor_data,
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
        metrics_path = os.getenv(TRAINING_METRICS_ENV)
        if metrics_path:
            from ..observability import TrainingMetricsStatsLoggerProxy

            self.stats_logger = TrainingMetricsStatsLoggerProxy(
                self.stats_logger,
                metrics_path,
            )

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
        if getattr(workflow, "requires_rlff_proxy_start", False):
            workflow_kwargs = dict(kwargs.get("workflow_kwargs") or {})
            start_step = (
                self.recover_info.last_step_info.next().global_step
                if self.recover_info is not None
                else 0
            )
            workflow_kwargs["reward_schedule_initial_step"] = int(start_step)
            workflow_kwargs["reward_schedule_use_proxy_version"] = (
                workflow_kwargs.get("reward_schedule_start_step") is not None
            )
            kwargs["workflow_kwargs"] = workflow_kwargs
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
    from ..episodes import build_episode_group, load_episode_jsonl, project_target_prompt

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

    schedule = config.rewards.weight_schedule
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
        "reward_schedule_start_step": schedule.start_step if schedule is not None else None,
        "reward_schedule_end_step": schedule.end_step if schedule is not None else None,
        "reward_schedule_completion_end_weight": (
            schedule.completion_end_weight if schedule is not None else None
        ),
        "reward_schedule_global_end_weight": (
            schedule.global_end_weight if schedule is not None else None
        ),
        "reward_schedule_use_proxy_version": False,
        "min_group_size": config.grpo.min_group_size,
        "reward_std_epsilon": config.grpo.reward_std_epsilon,
        "dynamic_trajectory_resampling": config.grpo.dynamic_trajectory_resampling,
        "reward_provider_name": config.rewards.provider,
        "reward_api_key_env": config.rewards.api_key_env,
        "reward_base_url": config.rewards.base_url,
        "completion_prompt_path": str(config.rewards.completion.prompt_path),
        "trajectory_prompt_path": str(config.rewards.global_reward.prompt_path),
        "reward_repair_prompt_path": str(config.rewards.repair_prompt_path),
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
        "reward_audit_jsonl": (
            str(config.observability.audit_jsonl.resolve())
            if config.observability.audit_jsonl is not None
            else None
        ),
        "reward_detail_jsonl": (
            str(config.observability.reward_detail_jsonl.resolve())
            if config.observability.reward_detail_jsonl is not None
            else None
        ),
        "reward_detail_sample_rate": config.observability.reward_detail_sample_rate,
        "reward_failure_jsonl": (
            str(config.observability.reward_failure_jsonl.resolve())
            if config.observability.reward_failure_jsonl is not None
            else None
        ),
    }
