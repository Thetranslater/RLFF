from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
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
    decode_proxy_episode_payload,
    normalize_proxy_role_advantages,
)


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

    assert kwargs["completion_reward_temperature"] == 0.2
    assert kwargs["trajectory_reward_temperature"] == 0.2
    assert kwargs["completion_reward_max_tokens"] == 15000
    assert kwargs["trajectory_reward_max_tokens"] == 15000


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
    config, _ = _write_runtime_files(tmp_path)
    calls: list[dict[str, object]] = []

    class FakePeft:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            calls.append({"model": model, "path": path, **kwargs})
            return (model, kwargs["is_trainable"])

    monkeypatch.setattr(runtime, "_load_peft_model", lambda: FakePeft)
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
    assert calls[0]["max_rounds"] == 1


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
