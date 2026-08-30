from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlff.config import (
    BF16LoRAConfig,
    PromptTemplateConfig,
    RewardConfig,
    RLFFConfig,
    RolloutConfig,
    SGLangConfig,
    load_config,
)


def config_payload(tmp_path: Path) -> dict[str, object]:
    completion_prompt = tmp_path / "completion.txt"
    global_prompt = tmp_path / "global.txt"
    completion_prompt.write_text("completion", encoding="utf-8")
    global_prompt.write_text("global", encoding="utf-8")
    return {
        "episode_grouping": {
            "dataset_path": str(tmp_path / "episodes.jsonl"),
            "group_size": 4,
            "base_seed": 42,
        },
        "prompt": {"template_paths": [], "template_ids": []},
        "rollout": {"max_rounds": 3},
        "sglang": {
            "model": "local-model",
            "max_new_tokens": 64,
            "context_length": 2048,
        },
        "rewards": {
            "completion": {"prompt_path": str(completion_prompt)},
            "global_reward": {"prompt_path": str(global_prompt)},
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
            "sft_adapter_path": str(tmp_path / "adapter"),
        },
        "checkpoint": {"output_dir": str(tmp_path / "output")},
    }


def test_config_hierarchy_defaults_and_non_secret_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RLFFConfig.model_validate(config_payload(tmp_path))

    assert config.episode.group_size == 4
    assert config.rollout.max_rounds == 3
    assert config.rewards.completion.model == "qwen3.7-flash"
    assert config.rewards.global_reward.model == "qwen3.7-flash"
    assert config.rewards.completion.temperature == 0.7
    assert config.rewards.global_reward.temperature == 0.7
    assert config.rewards.completion.reasoning_effort == "medium"
    assert config.rewards.global_reward.reasoning_effort == "medium"
    assert config.rewards.completion.max_tokens == 18000
    assert config.rewards.global_reward.max_tokens == 18000
    assert config.rewards.completion_weight == 0.7
    assert config.rewards.global_weight == 0.3
    assert config.rewards.weight_schedule is not None
    assert config.rewards.weight_schedule.start_step == 150
    assert config.rewards.weight_schedule.end_step == 250
    assert config.lora.dtype == "bfloat16"
    assert config.areal.use_bf16 is True
    assert config.grpo.normalize_by_role is True
    assert config.grpo.drop_incomplete_trajectory is True

    monkeypatch.setenv("DASHSCOPE_API_KEY", "do-not-serialize")
    secrets = config.resolve_runtime_secrets()
    assert secrets.reward_api_key == "do-not-serialize"
    snapshot = json.dumps(config.resolved_snapshot(), ensure_ascii=False)
    assert "do-not-serialize" not in snapshot
    assert config.rewards.api_key_env in snapshot


def test_default_rollout_output_limit_is_256_tokens() -> None:
    assert SGLangConfig(model="local-model").max_new_tokens == 256


def test_yaml_and_json_loaders_accept_utf8_sig(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    yaml_path = tmp_path / "config.yaml"
    json_path = tmp_path / "config.json"
    import yaml

    yaml_path.write_bytes(b"\xef\xbb\xbf" + yaml.safe_dump(payload, allow_unicode=True).encode())
    json_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode())

    assert (
        load_config(yaml_path).config_fingerprint() == load_config(json_path).config_fingerprint()
    )


def test_placeholder_rewards_require_explicit_opt_in(tmp_path: Path) -> None:
    scope = {"prompt_path": str(tmp_path / "reward.txt")}
    with pytest.raises(ValidationError, match="placeholder"):
        RewardConfig(
            completion=scope,
            global_reward=scope,
            provider="placeholder",
        )
    allowed = RewardConfig(
        completion=scope,
        global_reward=scope,
        provider="placeholder",
        allow_placeholder=True,
    )
    assert allowed.allow_placeholder is True


def test_reward_weight_schedule_rejects_invalid_range(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    rewards = dict(payload["rewards"])  # type: ignore[arg-type]
    rewards["weight_schedule"] = {
        "start_step": 250,
        "end_step": 150,
        "completion_end_weight": 0.4,
        "global_end_weight": 0.6,
    }
    payload["rewards"] = rewards
    with pytest.raises(ValidationError, match="end_step"):
        RLFFConfig.model_validate(payload)


@pytest.mark.parametrize(
    "field, value",
    [
        ("scheduler", {}),
        ("tools", {}),
        ("states", {}),
        ("environment", {}),
        ("quantization", "4bit"),
        ("qlora", True),
        ("natural_stop", True),
    ],
)
def test_forbidden_legacy_settings_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = config_payload(tmp_path)
    payload[field] = value
    with pytest.raises(ValidationError):
        RLFFConfig.model_validate(payload)


def test_unsafe_numeric_and_adapter_combinations_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RolloutConfig(max_rounds=0)
    payload = config_payload(tmp_path)
    payload["sglang"] = {
        "model": "local-model",
        "top_p": 1.1,
        "max_new_tokens": 64,
        "context_length": 64,
    }
    with pytest.raises(ValidationError):
        RLFFConfig.model_validate(payload)
    payload["sglang"] = {
        "model": "local-model",
        "max_new_tokens": 64,
        "context_length": 64,
    }
    with pytest.raises(ValidationError):
        RLFFConfig.model_validate(payload)
    with pytest.raises(ValidationError):
        RLFFConfig.model_validate({**config_payload(tmp_path), "areal": {"use_bf16": False}})
    with pytest.raises(ValidationError):
        RLFFConfig.model_validate(
            {**config_payload(tmp_path), "grpo": {"normalize_by_role": False}}
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BF16LoRAConfig(
            base_model="base",
            sft_adapter_path="adapter-a",
            reference_adapter_path="adapter-b",
        )
    with pytest.raises(ValidationError):
        PromptTemplateConfig(
            template_paths=[tmp_path / "a.txt"],
            template_ids=["one", "two"],
        )


def test_missing_reward_secret_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = config_payload(tmp_path)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = RLFFConfig.model_validate(payload)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        config.resolve_runtime_secrets()


def test_langsmith_tracing_requires_its_env_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = config_payload(tmp_path)
    payload["observability"] = {"langsmith_tracing": True}
    monkeypatch.setenv("DASHSCOPE_API_KEY", "reward-key")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    config = RLFFConfig.model_validate(payload)
    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY"):
        config.resolve_runtime_secrets()
