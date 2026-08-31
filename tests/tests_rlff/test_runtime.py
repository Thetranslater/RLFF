from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

import rlff.runtime as runtime
from rlff.cli import main
from rlff.config import RLFFConfig
from rlff.contracts import EpisodeRecord
from rlff.proxy import (
    ProxyCompletionView,
    ProxyGroupError,
    ProxyTrajectoryView,
    RLFFGroupAwareAgent,
    _group_character_order,
    decode_proxy_episode_payload,
    normalize_proxy_role_advantages,
    reward_weights_for_step,
)


def test_character_order_is_shuffled_once_per_group_key() -> None:
    characters = ("Alice", "Bob", "Carol")
    first = _group_character_order(characters, "task-1:group-1")
    assert first == _group_character_order(characters, "task-1:group-1")
    assert set(first) == set(characters)
    assert _group_character_order(("Alice",), "task-1:group-1") == ("Alice",)
    observed = {
        _group_character_order(characters, f"task-{index}:group-1")
        for index in range(12)
    }
    assert len(observed) > 1


def _write_runtime_files(tmp_path: Path) -> tuple[RLFFConfig, Path]:
    adapter = tmp_path / "sft-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fake")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "base-model",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    areal_yaml = tmp_path / "areal.yaml"
    areal_yaml.write_text(
        """
scheduler:
  type: local
gconfig:
  n_samples: 4
  temperature: 0.9
  top_p: 1.0
  max_new_tokens: 8
  lora_name: default_lora
actor:
  backend: fsdp:d1
  path: base-model
  attn_impl: sdpa
  use_lora: true
  peft_type: lora
  lora_rank: 8
  lora_alpha: 16
  target_modules: [q_proj, v_proj]
  disable_dropout: true
  temperature: 0.9
  kl_ctl: 0.0
  recompute_logprob: false
  use_decoupled_loss: false
  reward_scaling: 1.0
  reward_bias: 0.0
  reward_norm: null
  adv_norm: null
  discount: 1.0
  gae_lambda: 1.0
rollout:
  use_lora: true
  scheduling_spec:
    - env_vars:
        TMS_INIT_ENABLE: "0"
  agent:
    agent_cls_path: rlff.runtime:RLFFGroupAwareAgent
    mode: inline
    turn_discount: 0.0
    export_style: individual
sglang:
  attention_backend: triton
  sampling_backend: pytorch
  context_length: 32
train_dataset:
  max_length: 32
""",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "episode_grouping": {
            "dataset_path": str(tmp_path / "episodes.jsonl"),
            "group_size": 4,
        },
        "sglang": {
            "model": "base-model",
            "max_new_tokens": 8,
            "context_length": 32,
        },
        "rewards": {
            "completion": {"prompt_path": str(tmp_path / "completion.txt")},
            "global_reward": {"prompt_path": str(tmp_path / "global.txt")},
            "completion_weight": 0.7,
            "global_weight": 0.3,
            "weight_schedule": {
                "start_step": 150,
                "end_step": 250,
                "completion_end_weight": 0.4,
                "global_end_weight": 0.6,
            },
        },
        "lora": {
            "base_model": "base-model",
            "sft_adapter_path": str(adapter),
            "rank": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
        "areal": {"official_yaml": str(areal_yaml)},
        "checkpoint": {"output_dir": str(tmp_path / "out")},
    }
    (tmp_path / "completion.txt").write_text("completion", encoding="utf-8")
    (tmp_path / "global.txt").write_text("global", encoding="utf-8")
    return RLFFConfig.model_validate(payload), areal_yaml


def test_runtime_plan_checks_adapter_and_native_constraints(tmp_path: Path) -> None:
    config, _ = _write_runtime_files(tmp_path)
    plan = runtime.build_runtime_plan(config)
    assert plan.adapter.rank == 8
    assert plan.areal_yaml.native_n_samples == 4
    assert plan.environment[runtime.AREAL_ADAPTER_ENV].endswith("sft-adapter")


def test_agent_workflow_kwargs_include_reward_sampling_settings(tmp_path: Path) -> None:
    config, _ = _write_runtime_files(tmp_path)

    kwargs = runtime.build_agent_workflow_kwargs(config)

    assert kwargs["completion_reward_temperature"] == 0.7
    assert kwargs["trajectory_reward_temperature"] == 0.7
    assert kwargs["completion_reward_reasoning_effort"] == "medium"
    assert kwargs["trajectory_reward_reasoning_effort"] == "medium"
    assert kwargs["completion_reward_max_tokens"] == 18000
    assert kwargs["trajectory_reward_max_tokens"] == 18000
    assert kwargs["completion_weight"] == 0.7
    assert kwargs["global_weight"] == 0.3
    assert kwargs["reward_schedule_start_step"] == 150
    assert kwargs["reward_schedule_end_step"] == 250
    assert kwargs["reward_schedule_completion_end_weight"] == 0.4
    assert kwargs["reward_schedule_global_end_weight"] == 0.6
    assert kwargs["reward_schedule_use_proxy_version"] is False
    assert kwargs["dynamic_trajectory_resampling"] is False
    assert kwargs["frequency_penalty"] == 0.0
    assert kwargs["rollout_request_timeout_seconds"] == 120.0


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, (0.7, 0.3)),
        (149, (0.7, 0.3)),
        (150, (0.7, 0.3)),
        (200, (0.55, 0.45)),
        (250, (0.4, 0.6)),
        (399, (0.4, 0.6)),
    ],
)
def test_reward_weight_schedule_boundaries(
    step: int,
    expected: tuple[float, float],
) -> None:
    actual = reward_weights_for_step(
        step,
        completion_start_weight=0.7,
        global_start_weight=0.3,
        schedule_start_step=150,
        schedule_end_step=250,
        completion_end_weight=0.4,
        global_end_weight=0.6,
    )
    assert actual == pytest.approx(expected)


def test_reward_schedule_reads_areal_rollout_model_version() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"status": "success", "result": 173}

    class Client:
        async def post(self, url: str, **kwargs: object) -> Response:
            assert url == "http://proxy/call"
            assert kwargs["json"] == {"method": "get_version", "args": [], "kwargs": {}}
            return Response()

    agent = RLFFGroupAwareAgent(reward_schedule_use_proxy_version=True)
    step = asyncio.run(agent._resolve_reward_step("http://proxy", Client()))
    assert step == 173


def test_runtime_plan_rejects_nondefault_proxy_lora_name(
    tmp_path: Path,
) -> None:
    config, areal_yaml = _write_runtime_files(tmp_path)
    payload = yaml.safe_load(areal_yaml.read_text(encoding="utf-8"))
    payload["gconfig"]["lora_name"] = "named-adapter"
    areal_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(runtime.RuntimeCompatibilityError, match="lora_name"):
        runtime.build_runtime_plan(config)


def test_runtime_plan_rejects_missing_local_scheduler(tmp_path: Path) -> None:
    config, areal_yaml = _write_runtime_files(tmp_path)
    payload = yaml.safe_load(areal_yaml.read_text(encoding="utf-8"))
    payload["scheduler"]["type"] = None
    areal_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(runtime.RuntimeCompatibilityError, match=r"scheduler\.type"):
        runtime.build_runtime_plan(config)


def test_runtime_plan_rejects_lora_name_under_rollout(tmp_path: Path) -> None:
    config, areal_yaml = _write_runtime_files(tmp_path)
    payload = yaml.safe_load(areal_yaml.read_text(encoding="utf-8"))
    payload["rollout"]["lora_name"] = "default_lora"
    areal_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(runtime.RuntimeCompatibilityError, match="only under gconfig"):
        runtime.build_runtime_plan(config)


def test_runtime_plan_requires_rollout_tms_parent_disabled(tmp_path: Path) -> None:
    config, areal_yaml = _write_runtime_files(tmp_path)
    payload = yaml.safe_load(areal_yaml.read_text(encoding="utf-8"))
    payload["rollout"]["scheduling_spec"][0]["env_vars"] = {}
    areal_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(runtime.RuntimeCompatibilityError, match="TMS_INIT_ENABLE"):
        runtime.build_runtime_plan(config)


def test_runtime_plan_treats_target_modules_as_unordered(tmp_path: Path) -> None:
    config, _ = _write_runtime_files(tmp_path)
    path = Path(config.lora.sft_adapter_path) / "adapter_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_modules"] = ["v_proj", "q_proj"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    plan = runtime.build_runtime_plan(config)

    assert set(plan.adapter.target_modules) == {"q_proj", "v_proj"}


def test_disable_dropout_modules_includes_lora_modules() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(2, 2),
        torch.nn.Dropout(0.15),
    )

    count = runtime._disable_dropout_modules(model, torch)

    assert count == 2
    assert all(
        module.p == 0.0 for module in model.modules() if isinstance(module, torch.nn.Dropout)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("task_type", "SEQ_CLS"), ("peft_type", "IA3")],
)
def test_runtime_plan_rejects_non_lora_adapter(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    config, _ = _write_runtime_files(tmp_path)
    path = Path(config.lora.sft_adapter_path) / "adapter_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runtime.AdapterPreflightError):
        runtime.build_runtime_plan(config)


def test_existing_adapter_wrapper_passes_trainable_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlff.runtime import integration

    config, _ = _write_runtime_files(tmp_path)
    calls: list[dict[str, object]] = []

    class FakePeft:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            calls.append({"model": model, "path": path, **kwargs})
            return (model, kwargs["is_trainable"])

    monkeypatch.setattr(integration, "_load_peft_model", lambda: FakePeft)
    runtime.apply_existing_sft_adapter(
        "actor", adapter_path=config.lora.sft_adapter_path, is_trainable=True
    )
    assert [call["is_trainable"] for call in calls] == [True]
    assert all(call["autocast_adapter_dtype"] is False for call in calls)


def test_runtime_import_is_cpu_only_and_lazy() -> None:
    probe = (
        "import json, sys; import rlff.runtime; "
        "print(json.dumps({'areal': any(k == 'areal' or k.startswith('areal.') "
        "for k in sys.modules), 'torch': 'torch' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"areal": False, "torch": False}


def test_prune_old_hf_checkpoints_keeps_latest_and_recovery_state(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "default"
    model_root.mkdir()
    oldest = model_root / "epoch0epochstep19globalstep19"
    middle = model_root / "epoch0epochstep39globalstep39"
    newest = model_root / "epoch1epochstep9globalstep209"
    recover = model_root / "recover_checkpoint"
    unrelated = model_root / "manual-export"
    for directory in (oldest, middle, newest, recover, unrelated):
        directory.mkdir()
        (directory / "marker").write_text(directory.name, encoding="utf-8")

    removed = runtime.prune_old_hf_checkpoints(model_root)

    assert removed == (oldest, middle)
    assert newest.is_dir()
    assert recover.is_dir()
    assert unrelated.is_dir()


def test_prune_old_hf_checkpoints_rejects_deleting_every_export(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        runtime.prune_old_hf_checkpoints(tmp_path, keep=0)


def test_areal_dataset_preserves_episode_fingerprint_across_arrow(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    config, _ = _write_runtime_files(tmp_path)
    first = EpisodeRecord(
        title="first",
        plot="First plot.",
        characters=[{"name": "Alice"}, {"name": "Bob"}],
        metadata={"book": "demo", "source": "first"},
    )
    second = EpisodeRecord(
        title="second",
        plot="Second plot.",
        characters=[{"name": "Alice"}, {"name": "Bob"}],
        metadata={"book": "demo", "plot_index": 2},
    )
    config.episode_grouping.dataset_path.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for record in (first, second)
        ),
        encoding="utf-8",
    )

    dataset = runtime.build_areal_training_dataset(config)
    assert isinstance(dataset[0]["episode"], str)
    round_tripped = EpisodeRecord.model_validate(decode_proxy_episode_payload(dataset[0]))

    assert round_tripped.fingerprint == first.fingerprint
    assert round_tripped.metadata == first.metadata


def test_areal_dataset_preflights_every_character_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_runtime_files(tmp_path)
    record = EpisodeRecord(
        title="prompt preflight",
        plot="A plot.",
        characters=[{"name": "Alice"}, {"name": "Bob"}],
    )
    config.episode_grouping.dataset_path.write_text(
        json.dumps(record.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    seen: list[str] = []

    def project(record: EpisodeRecord, character: str, **kwargs: Any) -> object:
        _ = record, kwargs
        seen.append(character)
        return object()

    import rlff.episodes as episodes

    monkeypatch.setattr(episodes, "project_target_prompt", project)
    runtime.build_areal_training_dataset(config)

    assert seen == ["Alice", "Bob"]


def test_role_advantage_tensor_data_broadcasts_only_completion_tokens() -> None:
    torch = pytest.importorskip("torch")
    data: dict[str, Any] = {
        "loss_mask": torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]]),
        "rewards": torch.tensor([2.0, -1.0]),
        "logprobs": torch.arange(10, dtype=torch.float32).reshape(2, 5),
    }
    result = runtime.role_advantage_tensor_data(data)
    expected_mask = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(result["loss_mask"], expected_mask)
    assert torch.equal(result["advantages"], expected_mask * torch.tensor([[2.0], [-1.0]]))
    assert torch.equal(result["advantages"][:, 0], torch.zeros(2))
    assert torch.equal(result["kl_rewards"], torch.zeros_like(result["advantages"]))
    assert torch.equal(result["advantages"], expected_mask * torch.tensor([[2.0], [-1.0]]))


def _proxy_group() -> tuple[
    list[ProxyTrajectoryView],
    dict[str, float],
    dict[tuple[str, str], float],
]:
    trajectories: list[ProxyTrajectoryView] = []
    completion_rewards: dict[str, float] = {}
    trajectory_role_rewards: dict[tuple[str, str], float] = {}
    episode = {
        "episode_id": "episode-1",
        "plot": "A test plot.",
        "shared_tasks": ["stay coherent"],
        "characters": [
            {"name": "Alice", "profile": "curious", "private_tasks": ["ask"]},
            {"name": "Bob", "profile": "careful", "private_tasks": ["check"]},
        ],
    }
    for index in range(4):
        trajectory_id = f"trajectory-{index}"
        completions = tuple(
            ProxyCompletionView(
                completion_id=f"completion-{index}-{character}",
                group_id="group-1",
                trajectory_id=trajectory_id,
                character=character,
                turn_index=turn_index,
                text=f"{character}-{index}",
            )
            for turn_index, character in enumerate(("Alice", "Bob"))
        )
        trajectories.append(
            ProxyTrajectoryView(
                group_id="group-1",
                trajectory_id=trajectory_id,
                episode_id="episode-1",
                episode=episode,
                completions=completions,
                turns=tuple(
                    {"character": completion.character, "content": completion.text}
                    for completion in completions
                ),
                planned_rounds=1,
                completed_rounds=1,
            )
        )
        completion_rewards[f"completion-{index}-Alice"] = float(index * 2)
        completion_rewards[f"completion-{index}-Bob"] = float(index * 4)
        trajectory_role_rewards[(trajectory_id, "Alice")] = float(index)
        trajectory_role_rewards[(trajectory_id, "Bob")] = float(index * 2)
    return trajectories, completion_rewards, trajectory_role_rewards


def test_proxy_role_normalization_aggregates_then_normalizes_per_role() -> None:
    trajectories, completion_rewards, trajectory_role_rewards = _proxy_group()
    result = normalize_proxy_role_advantages(
        trajectories,
        completion_rewards,
        trajectory_role_rewards,
        completion_weight=0.5,
        global_weight=0.5,
        min_group_size=4,
    )
    for character in ("Alice", "Bob"):
        values = [result[f"completion-{index}-{character}"] for index in range(4)]
        assert sum(values) == pytest.approx(0.0)
        assert sum(value * value for value in values) / 4 == pytest.approx(1.0)
    assert result["completion-0-Alice"] == result["completion-0-Alice"]
    assert set(result) == set(completion_rewards)


def test_group_agent_barrier_returns_only_own_session_rewards() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.completion_calls: list[str] = []
            self.trajectory_calls: list[str] = []

        async def score_proxy_completion_role(self, payload: dict[str, Any]) -> dict[str, float]:
            completion_ids = [str(value) for value in payload["ids"]["completion_ids"]]
            self.completion_calls.extend(completion_ids)
            return {
                completion_id: float(completion_id.split("-")[1])
                for completion_id in completion_ids
            }

        async def score_proxy_trajectory_role(self, payload: dict[str, Any]) -> float:
            trajectory_id = str(payload["ids"]["trajectory_id"])
            character = str(payload["ids"]["character"])
            self.trajectory_calls.append(f"{trajectory_id}/{character}")
            return float(trajectory_id.rsplit(":", 1)[1])

    provider = FakeProvider()

    async def run_trajectory(data: dict[str, Any], **kwargs: Any) -> ProxyTrajectoryView:
        trajectory_id = str(kwargs["trajectory_id"])
        slot = int(trajectory_id.rsplit(":", 1)[1])
        completions = tuple(
            ProxyCompletionView(
                completion_id=f"completion-{slot}-{character}",
                group_id="group-1",
                trajectory_id=trajectory_id,
                character=character,
                turn_index=turn,
                text="ok",
            )
            for turn, character in enumerate(("Alice", "Bob"))
        )
        return ProxyTrajectoryView(
            group_id="group-1",
            trajectory_id=trajectory_id,
            episode_id="episode-1",
            episode={
                "episode_id": "episode-1",
                "plot": "test",
                "characters": [{"name": "Alice"}, {"name": "Bob"}],
            },
            completions=completions,
            turns=(),
            planned_rounds=1,
            completed_rounds=1,
        )

    agent = RLFFGroupAwareAgent(
        group_size=4,
        reward_provider=provider,
        trajectory_runner=run_trajectory,
        completion_weight=0.5,
        global_weight=0.5,
        reward_schedule_start_step=150,
        reward_schedule_end_step=250,
        reward_schedule_completion_end_weight=0.4,
        reward_schedule_global_end_weight=0.6,
        reward_schedule_initial_step=149,
        min_group_size=4,
    )

    async def execute() -> list[dict[str, float]]:
        return await asyncio.gather(
            *(
                agent.run(
                    {"group_id": "group-1"},
                    base_url="http://proxy",
                    http_client=object(),
                    api_key="session",
                )
                for _ in range(4)
            )
        )

    results = asyncio.run(execute())
    assert all(len(result) == 2 for result in results)
    assert len(set().union(*(set(result) for result in results))) == 8
    assert len(provider.completion_calls) == 8
    assert len(provider.trajectory_calls) == 8
    assert agent._reward_schedule_step == 150


def test_dynamic_group_scores_trajectory_first_and_skips_completion_when_rejected() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.completion_calls: list[str] = []
            self.trajectory_calls: list[str] = []

        async def score_proxy_completion_role(self, payload: dict[str, Any]) -> dict[str, float]:
            completion_ids = tuple(str(value) for value in payload["ids"]["completion_ids"])
            self.completion_calls.extend(completion_ids)
            return {completion_id: 1.0 for completion_id in completion_ids}

        async def score_proxy_trajectory_role(self, payload: dict[str, Any]) -> float:
            trajectory_id = str(payload["ids"]["trajectory_id"])
            self.trajectory_calls.append(trajectory_id)
            attempt_text = trajectory_id.rsplit(":attempt:", 1)[1]
            attempt = int(attempt_text.split(":", 1)[0])
            slot = int(trajectory_id.rsplit(":", 1)[1])
            character = str(payload["ids"]["character"])
            # Attempt one is accepted because Bob varies even though Alice is
            # still constant across all four trajectories.
            return 0.0 if attempt == 0 or character == "Alice" else float(slot)

    provider = FakeProvider()

    async def run_trajectory(data: dict[str, Any], **kwargs: Any) -> ProxyTrajectoryView:
        trajectory_id = str(kwargs["trajectory_id"])
        completions = tuple(
            ProxyCompletionView(
                completion_id=f"{trajectory_id}:{character}",
                group_id="group-1",
                trajectory_id=trajectory_id,
                character=character,
                turn_index=turn,
                text="ok",
            )
            for turn, character in enumerate(("Alice", "Bob"))
        )
        return ProxyTrajectoryView(
            group_id="group-1",
            trajectory_id=trajectory_id,
            episode_id="episode-1",
            episode={
                "episode_id": "episode-1",
                "plot": "test",
                "characters": [{"name": "Alice"}, {"name": "Bob"}],
            },
            completions=completions,
            turns=(),
            planned_rounds=1,
            completed_rounds=1,
        )

    agent = RLFFGroupAwareAgent(
        group_size=4,
        reward_provider=provider,
        trajectory_runner=run_trajectory,
        dynamic_trajectory_resampling=True,
        min_group_size=4,
    )

    async def run_attempt(attempt: int) -> tuple[list[dict[str, float]], bool]:
        data = {"group_id": "group-1", "_rlff_resample_attempt": attempt}
        results = await asyncio.gather(
            *(
                agent.run(
                    data,
                    base_url="http://proxy",
                    http_client=object(),
                    api_key="session",
                )
                for _ in range(4)
            )
        )
        return results, await agent.consume_group_sampling_decision(data)

    async def execute() -> tuple[
        tuple[list[dict[str, float]], bool, int, int, int],
        tuple[list[dict[str, float]], bool],
    ]:
        rejected_results, rejected = await run_attempt(0)
        rejected_attempt = (
            rejected_results,
            rejected,
            len(provider.completion_calls),
            len(provider.trajectory_calls),
            agent._reward_schedule_step,
        )
        accepted_attempt = await run_attempt(1)
        return rejected_attempt, accepted_attempt

    (
        rejected_results,
        rejected,
        completion_calls_after_rejection,
        trajectory_calls_after_rejection,
        schedule_step_after_rejection,
    ), (accepted_results, accepted) = asyncio.run(execute())
    assert rejected is False
    assert completion_calls_after_rejection == 0
    assert trajectory_calls_after_rejection == 8
    assert all(set(result.values()) == {0.0} for result in rejected_results)
    assert schedule_step_after_rejection == 0

    assert accepted is True
    assert len(provider.completion_calls) == 8
    assert len(provider.trajectory_calls) == 16
    assert all(len(result) == 2 for result in accepted_results)
    assert agent._reward_schedule_step == 1


def test_dynamic_group_wrapper_retries_same_episode_with_fresh_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlff.proxy.dynamic_sampling import install_same_episode_group_wrapper

    class FakeRolloutWorkflow:
        pass

    class FakeInteraction:
        pass

    class OriginalGroupedWorkflow:
        pass

    fake_areal = ModuleType("areal")
    fake_api = ModuleType("areal.api")
    fake_api.RolloutWorkflow = FakeRolloutWorkflow  # type: ignore[attr-defined]
    fake_experimental = ModuleType("areal.experimental")
    fake_openai = ModuleType("areal.experimental.openai")
    fake_openai.InteractionWithTokenLogpReward = FakeInteraction  # type: ignore[attr-defined]
    fake_infra = ModuleType("areal.infra")
    fake_remote = ModuleType("areal.infra.remote_inf_engine")
    fake_remote.GroupedRolloutWorkflow = OriginalGroupedWorkflow  # type: ignore[attr-defined]
    fake_infra.remote_inf_engine = fake_remote  # type: ignore[attr-defined]
    fake_utils = ModuleType("areal.utils")
    fake_data = ModuleType("areal.utils.data")
    fake_data.concat_padded_tensors = lambda values: {"values": values}  # type: ignore[attr-defined]

    for name, module in {
        "areal": fake_areal,
        "areal.api": fake_api,
        "areal.experimental": fake_experimental,
        "areal.experimental.openai": fake_openai,
        "areal.infra": fake_infra,
        "areal.infra.remote_inf_engine": fake_remote,
        "areal.utils": fake_utils,
        "areal.utils.data": fake_data,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    class FakeAgent:
        dynamic_trajectory_resampling = True

        def __init__(self) -> None:
            self.decisions = [False, True]

        async def consume_group_sampling_decision(self, data: dict[str, Any]) -> bool:
            assert data["episode"] == "same-episode"
            return self.decisions.pop(0)

    class FakeInnerWorkflow:
        def __init__(self) -> None:
            self.agent = FakeAgent()
            self.calls: list[int] = []

        async def arun_episode(self, _engine: Any, data: dict[str, Any]) -> dict[str, Any]:
            attempt = int(data["_rlff_resample_attempt"])
            slot = sum(value == attempt for value in self.calls)
            self.calls.append(attempt)
            return {f"attempt-{attempt}-slot-{slot}": FakeInteraction()}

    assert install_same_episode_group_wrapper() is True
    wrapper_class = fake_remote.GroupedRolloutWorkflow  # type: ignore[attr-defined]
    inner = FakeInnerWorkflow()
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_a: None)
    wrapper = wrapper_class(inner, 4, logger)
    result = asyncio.run(
        wrapper.arun_episode(object(), {"episode": "same-episode"})
    )

    assert inner.calls == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result is not None
    assert set(result) == {
        "attempt-1-slot-0",
        "attempt-1-slot-1",
        "attempt-1-slot-2",
        "attempt-1-slot-3",
    }


def test_group_agent_barrier_failure_fails_every_run() -> None:
    async def run_trajectory(data: dict[str, Any], **kwargs: Any) -> ProxyTrajectoryView:
        trajectory_id = str(kwargs["trajectory_id"])
        if trajectory_id.endswith(":2"):
            raise RuntimeError("synthetic trajectory failure")
        return ProxyTrajectoryView(
            group_id="group-1",
            trajectory_id=trajectory_id,
            episode_id="episode-1",
            episode={
                "episode_id": "episode-1",
                "plot": "test",
                "characters": [{"name": "Alice"}],
            },
            completions=(
                ProxyCompletionView(
                    completion_id=trajectory_id,
                    group_id="group-1",
                    trajectory_id=trajectory_id,
                    character="Alice",
                    turn_index=0,
                    text="ok",
                ),
            ),
            turns=(),
            planned_rounds=1,
            completed_rounds=1,
        )

    class ZeroProvider:
        async def score_proxy_completion_role(self, payload: dict[str, Any]) -> dict[str, float]:
            return {str(completion_id): 0.0 for completion_id in payload["ids"]["completion_ids"]}

        async def score_proxy_trajectory_role(self, payload: dict[str, Any]) -> float:
            return 0.0

    agent = RLFFGroupAwareAgent(
        group_size=4,
        reward_provider=ZeroProvider(),
        trajectory_runner=run_trajectory,
        min_group_size=4,
    )

    async def execute() -> list[BaseException]:
        return await asyncio.gather(
            *(
                agent.run(
                    {"group_id": "group-1"},
                    base_url="http://proxy",
                    http_client=object(),
                    api_key="session",
                )
                for _ in range(4)
            ),
            return_exceptions=True,
        )

    results = asyncio.run(execute())
    assert all(isinstance(result, ProxyGroupError) for result in results)
    assert agent._reward_schedule_step == 0


def test_public_generate_trajectory_uses_training_runner_without_reward_barrier() -> None:
    calls: list[dict[str, Any]] = []

    async def run_trajectory(data: dict[str, Any], **kwargs: Any) -> ProxyTrajectoryView:
        calls.append({"data": data, **kwargs})
        return ProxyTrajectoryView(
            group_id=str(kwargs["group_id"]),
            trajectory_id=str(kwargs["trajectory_id"]),
            episode_id="episode-1",
            episode={"episode_id": "episode-1", "plot": "test", "characters": []},
            completions=(),
            turns=(),
            planned_rounds=1,
            completed_rounds=1,
        )

    agent = RLFFGroupAwareAgent(group_size=4, trajectory_runner=run_trajectory)
    result = asyncio.run(
        agent.generate_trajectory(
            {"group_id": "group-1", "episode": {"episode_id": "episode-1"}},
            base_url="http://sglang/v1",
            http_client=object(),
            api_key="EMPTY",
            trajectory_id="trajectory-smoke",
        )
    )

    assert result.trajectory_id == "trajectory-smoke"
    assert len(calls) == 1
    assert calls[0]["group_id"] == "group-1"
    assert calls[0]["max_rounds"] == 7


def test_openai_rollout_uses_explicit_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeCompletions:
        async def create(self, **_kwargs: Any) -> Any:
            captured["request"] = _kwargs
            return SimpleNamespace(
                id="completion-1",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    agent = RLFFGroupAwareAgent(
        group_size=4,
        max_rounds=1,
        model="test-model",
        frequency_penalty=1.5,
        rollout_request_timeout_seconds=7.5,
    )
    result = asyncio.run(
        agent.generate_trajectory(
            {
                "group_id": "group-1",
                "episode": {
                    "episode_id": "episode-1",
                    "title": "test",
                    "plot": "test plot",
                    "characters": [{"name": "Alice", "profile": "profile"}],
                },
            },
            base_url="http://sglang/v1",
            http_client=object(),
            api_key="EMPTY",
        )
    )

    assert len(result.completions) == 1
    assert captured["timeout"] == 7.5
    assert captured["max_retries"] == 0
    assert captured["request"]["frequency_penalty"] == 1.5


def test_proxy_prompt_uses_canonical_target_projection() -> None:
    from rlff.episodes import build_episode_group

    record = EpisodeRecord(
        title="prompt",
        plot="The plot must be present.",
        shared_tasks=["shared task"],
        characters=[
            {"name": "Alice", "profile": "Alice profile", "private_tasks": ["Alice task"]},
            {"name": "Bob", "profile": "Bob profile", "private_tasks": ["Bob task"]},
        ],
        dialogue=[{"speaker": "Bob", "content": "Opening."}],
    )
    render = build_episode_group(record, group_size=4).samples[0].render
    agent = RLFFGroupAwareAgent(group_size=4)
    messages = agent._messages(record, "Alice", list(record.dialogue), render)
    assert messages[0]["role"] == "system"
    assert "The plot must be present." in messages[0]["content"]
    assert "Alice profile" in messages[0]["content"]
    assert "Alice task" in messages[0]["content"]
    assert "shared task" in messages[0]["content"]
    assert any(message["content"] == "Bob:Opening." for message in messages)


def test_cli_dry_run_uses_cpu_preflight(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config, areal_yaml = _write_runtime_files(tmp_path)
    config_path = tmp_path / "rlff.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    assert areal_yaml.is_file()
    assert main(["dry-run", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert '"native_n_samples": 4' in output
    assert "adapter_model" not in output
