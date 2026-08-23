from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlff.contracts import EpisodeRecord
from rlff.episodes import (
    EpisodeLoadError,
    build_episode_group,
    build_target_prompt,
    load_episode_jsonl,
    project_sample_prompt,
)


def record(*, episode_id: str | None = None) -> EpisodeRecord:
    payload: dict[str, object] = {
        "title": "golden",
        "plot": "A shared plot.",
        "shared_tasks": ["shared task"],
        "characters": [
            {
                "name": "Alice",
                "profile": "Alice profile",
                "private_tasks": ["Alice private"],
            },
            {
                "name": "Bob",
                "profile": "Bob profile",
                "private_tasks": ["Bob secret"],
            },
        ],
        "dialogue": [
            {"speaker": "Alice", "content": "a1"},
            {"speaker": "Alice", "content": "a2"},
            {"speaker": "Bob", "content": "b1"},
            {"speaker": "Bob", "content": "b2"},
            {"speaker": "Alice", "content": "a3"},
        ],
        "metadata": {"book": "golden-book", "plot_index": 3},
    }
    if episode_id is not None:
        payload["episode_id"] = episode_id
    return EpisodeRecord.model_validate(payload)


def write_jsonl(path: Path, rows: list[dict[str, object]], *, bom: bool = False) -> None:
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def test_bom_loader_and_dataset_fingerprint_are_deterministic(tmp_path: Path) -> None:
    first = record()
    second = record(episode_id="episode-explicit")
    path = tmp_path / "episodes.jsonl"
    write_jsonl(path, [first.model_dump(mode="json"), second.model_dump(mode="json")], bom=True)

    loaded = load_episode_jsonl(path)
    repeated = load_episode_jsonl(path)
    assert len(loaded) == 2
    assert loaded.fingerprint == repeated.fingerprint
    assert loaded.records == repeated.records
    assert load_episode_jsonl(path, limit=1).records == loaded.records[:1]

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(EpisodeLoadError, match="empty"):
        load_episode_jsonl(empty)


def test_invalid_jsonl_records_are_line_aware(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(EpisodeLoadError, match=r"bad\.jsonl:1"):
        load_episode_jsonl(path)

    valid_row = json.dumps(record().model_dump(mode="json"))
    path.write_text(f"{valid_row}\nnot-json\n", encoding="utf-8")
    with pytest.raises(EpisodeLoadError, match=r"bad\.jsonl:2"):
        load_episode_jsonl(path)

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(EpisodeLoadError, match=r"bad\.jsonl:1.*JSON object"):
        load_episode_jsonl(path)


def test_duplicate_ids_and_unknown_speakers_are_rejected(tmp_path: Path) -> None:
    source = record(episode_id="same")
    path = tmp_path / "duplicate.jsonl"
    row = source.model_dump(mode="json")
    write_jsonl(path, [row, row])
    with pytest.raises(EpisodeLoadError, match="duplicate episode_id"):
        load_episode_jsonl(path)

    with pytest.raises(ValidationError, match="not registered"):
        EpisodeRecord(
            plot="x",
            characters=[{"name": "Alice"}],
            dialogue=[{"speaker": "Unknown", "content": "x"}],
        )


def test_group_ids_seeds_and_render_choices_are_stable() -> None:
    source = record()
    templates = (
        ("first", "<text>{character}</text><random>one;two</random>"),
        ("second", "<text>{profile}</text>"),
    )
    first = build_episode_group(source, group_size=3, base_seed=17, templates=templates)
    second = build_episode_group(source, group_size=3, base_seed=17, templates=templates)
    assert first == second
    assert len({sample.seed for sample in first.samples}) == 3
    assert len({sample.trajectory_id for sample in first.samples}) == 3
    assert len({sample.render.template_id for sample in first.samples}) == 1
    assert len({sample.render.render_seed for sample in first.samples}) == 1
    later = build_episode_group(
        source,
        group_size=3,
        base_seed=17,
        sampling_iteration=1,
        templates=templates,
    )
    assert later.group_id != first.group_id
    assert {sample.trajectory_id for sample in later.samples}.isdisjoint(
        {sample.trajectory_id for sample in first.samples}
    )
    assert all(sample.sampling_iteration == 1 for sample in later.samples)


def test_target_projection_matches_sft_role_merge_and_hides_other_tasks() -> None:
    source = record()
    template = (
        "<text>Role={character}</text>"
        "<text>Profile={profile}</text>"
        "<text>Plot={plot}</text>"
        "<text>Tasks={tasks}</text>"
    )
    projection = build_target_prompt(source, "Alice", template=template, render_seed=9)
    assert [message.role for message in projection.messages] == [
        "system",
        "assistant",
        "user",
        "assistant",
    ]
    assert projection.messages[0].content == (
        "Role=Alice\nProfile=Alice profile\nPlot=A shared plot.\nTasks=shared task;Alice private"
    )
    assert projection.messages[1].content == "a1\na2"
    assert projection.messages[2].content == "Bob:b1\nBob:b2"
    assert projection.messages[3].content == "a3"
    prompt_text = "\n".join(message.content for message in projection.messages)
    assert "Bob profile" not in prompt_text
    assert "Bob secret" not in prompt_text


def test_renderer_does_not_pollute_global_random_state() -> None:
    source = record()
    template = "<random>alpha;beta</random><rearrange><item>one</item><item>two</item></rearrange>"
    random.seed(12345)
    expected_next = random.random()
    random.seed(12345)
    build_target_prompt(source, "Alice", template=template, render_seed=999)
    actual_next = random.random()
    assert actual_next == expected_next


def test_sampling_seed_does_not_change_group_prompt() -> None:
    group = build_episode_group(record(), group_size=2, base_seed=44)
    first = project_sample_prompt(group.samples[0], "Alice")
    second = project_sample_prompt(group.samples[1], "Alice")
    assert first.messages == second.messages
    assert group.samples[0].seed != group.samples[1].seed
