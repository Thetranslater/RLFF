from __future__ import annotations

import json
from pathlib import Path

import yaml

from rlff.config import load_config
from scripts.build_epoch2_dataset import reorder_records


def test_epoch2_reorder_is_stable_mixed_curriculum() -> None:
    records = []
    for index in range(10):
        records.extend(
            [
                {"id": f"hard-{index}", "metadata": {"difficulty": "hard"}},
                {"id": f"easy-{index}", "metadata": {"difficulty": "easy"}},
            ]
        )

    ordered = reorder_records(records)

    assert sum(record["metadata"]["difficulty"] == "easy" for record in ordered[:10]) == 7
    assert sum(record["metadata"]["difficulty"] == "hard" for record in ordered[:10]) == 3
    assert sum(record["metadata"]["difficulty"] == "easy" for record in ordered[10:]) == 3
    assert sum(record["metadata"]["difficulty"] == "hard" for record in ordered[10:]) == 7
    assert [record["id"] for record in ordered if record["id"].startswith("easy-")] == [
        f"easy-{index}" for index in range(10)
    ]
    assert [record["id"] for record in ordered if record["id"].startswith("hard-")] == [
        f"hard-{index}" for index in range(10)
    ]


def test_checked_in_epoch2_dataset_is_a_stable_mixed_curriculum() -> None:
    root = Path(__file__).resolve().parents[2]

    def read(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    source = read(root / "data" / "episodes.jsonl")
    epoch2 = read(root / "data" / "episodes_epoch2.jsonl")

    assert len(epoch2) == len(source) == 200
    assert {record["episode_id"] for record in epoch2} == {
        record["episode_id"] for record in source
    }
    assert sum(
        record.get("metadata", {}).get("difficulty") == "easy"
        for record in epoch2[:100]
    ) == 70
    assert sum(
        record.get("metadata", {}).get("difficulty") != "easy"
        for record in epoch2[:100]
    ) == 30
    assert sum(
        record.get("metadata", {}).get("difficulty") == "easy"
        for record in epoch2[100:]
    ) == 30
    assert sum(
        record.get("metadata", {}).get("difficulty") != "easy"
        for record in epoch2[100:]
    ) == 70
    assert any(
        record.get("metadata", {}).get("difficulty") != "easy"
        for record in epoch2[:20]
    )
    assert any(
        record.get("metadata", {}).get("difficulty") == "easy"
        for record in epoch2[-20:]
    )
    assert epoch2 == reorder_records(source)


def test_checked_in_epoch2_configs_enable_intended_second_epoch_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "rlff_epoch2.yaml")
    native = yaml.safe_load(
        (root / "configs" / "areal_epoch2.yaml").read_text(encoding="utf-8")
    )

    assert config.episode.dataset_path == Path("data/episodes_epoch2.jsonl")
    assert config.grpo.dynamic_trajectory_resampling is True
    assert native["total_train_epochs"] == 2
    assert native["train_dataset"]["path"] == "data/episodes_epoch2.jsonl"
    assert native["train_dataset"]["shuffle"] is False
    assert native["actor"]["eps_clip"] == 0.2
    assert native["actor"]["eps_clip_higher"] == 0.28
