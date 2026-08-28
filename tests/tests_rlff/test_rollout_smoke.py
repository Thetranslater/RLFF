from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from rlff.config import RLFFConfig
from rlff.contracts import EpisodeRecord
from rlff.episodes import EpisodeDataset
from rlff.proxy import ProxyCompletionView, ProxyTrajectoryView
from rlff.rollout_smoke import (
    generate_smoke_rollouts,
    lora_request_model,
    normalize_openai_base_url,
    select_episode_records,
)


def _episode(index: int) -> EpisodeRecord:
    return EpisodeRecord(
        title=f"episode-{index}",
        plot=f"plot-{index}",
        characters=[
            {"name": "Alice", "profile": "A", "private_tasks": ["ask"]},
            {"name": "Bob", "profile": "B", "private_tasks": ["answer"]},
        ],
    )


def _dataset(size: int = 30) -> EpisodeDataset:
    records = tuple(_episode(index) for index in range(size))
    return EpisodeDataset(records=records, fingerprint="dataset-fingerprint")


def _config(tmp_path: Path) -> RLFFConfig:
    completion = tmp_path / "completion.txt"
    trajectory = tmp_path / "trajectory.txt"
    completion.write_text("completion", encoding="utf-8")
    trajectory.write_text("trajectory", encoding="utf-8")
    return RLFFConfig.model_validate(
        {
            "episode_grouping": {
                "dataset_path": str(tmp_path / "episodes.jsonl"),
                "group_size": 4,
                "base_seed": 7,
            },
            "prompt": {"template_paths": [], "template_ids": []},
            "rollout": {"max_rounds": 7},
            "sglang": {
                "model": "models/base/Qwen2.5-7B-Instruct",
                "base_url": "http://127.0.0.1:30000",
                "max_new_tokens": 32,
                "context_length": 2048,
            },
            "rewards": {
                "completion": {"prompt_path": str(completion)},
                "global_reward": {"prompt_path": str(trajectory)},
                "provider": "placeholder",
                "allow_placeholder": True,
            },
            "lora": {
                "base_model": "models/base/Qwen2.5-7B-Instruct",
                "sft_adapter_path": "models/adapters/test",
            },
            "checkpoint": {"output_dir": str(tmp_path / "outputs")},
        }
    )


def test_selection_is_deterministic_unique_and_seeded() -> None:
    dataset = _dataset()
    first = select_episode_records(dataset, limit=20, seed=42)
    second = select_episode_records(dataset, limit=20, seed=42)
    other = select_episode_records(dataset, limit=20, seed=43)

    assert [item.episode_id for item in first] == [item.episode_id for item in second]
    assert len({item.episode_id for item in first}) == 20
    assert [item.episode_id for item in first] != [item.episode_id for item in other]
    with pytest.raises(ValueError, match="cannot sample"):
        select_episode_records(dataset, limit=31)


def test_sglang_openai_address_and_lora_selector(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert normalize_openai_base_url("http://127.0.0.1:30000") == (
        "http://127.0.0.1:30000/v1"
    )
    assert normalize_openai_base_url("http://127.0.0.1:30000/v1/") == (
        "http://127.0.0.1:30000/v1"
    )
    assert lora_request_model(config) == "models/base/Qwen2.5-7B-Instruct:rlff-sft"


@pytest.mark.asyncio
async def test_smoke_rollout_delegates_to_public_production_agent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    records = (_episode(0), _episode(1))

    class FakeAgent:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate_trajectory(self, data: dict[str, Any], **kwargs: Any) -> Any:
            self.calls.append({"data": data, **kwargs})
            character = data["episode"]["characters"][0]["name"]
            completion = ProxyCompletionView(
                completion_id=f"completion-{len(self.calls)}",
                group_id=kwargs["group_id"],
                trajectory_id=kwargs["trajectory_id"],
                character=character,
                turn_index=0,
                text="hello",
                finish_reason="stop",
            )
            return ProxyTrajectoryView(
                group_id=kwargs["group_id"],
                trajectory_id=kwargs["trajectory_id"],
                episode_id=data["episode_id"],
                episode=data["episode"],
                completions=(completion,),
                turns=({"character": character, "content": "hello", "turn_id": 0},),
                planned_rounds=2,
                completed_rounds=2,
            )

    fake = FakeAgent()
    results = await generate_smoke_rollouts(
        config,
        records,
        base_url="http://sglang:30000",
        request_model="base:rlff-sft",
        concurrency=2,
        agent=fake,  # type: ignore[arg-type]
        http_client=object(),
    )

    assert [item["status"] for item in results] == ["ok", "ok"]
    assert len(fake.calls) == 2
    assert all(call["base_url"] == "http://sglang:30000/v1" for call in fake.calls)
    assert all(call["data"]["model"] == "base:rlff-sft" for call in fake.calls)
    assert all(call["data"]["render"]["template"] for call in fake.calls)
    assert json.dumps(results, ensure_ascii=False)


@pytest.mark.asyncio
async def test_real_agent_openai_transport_accumulates_round_robin_history(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    requests: list[dict[str, Any]] = []

    async def chat_completion(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(payload)
        index = len(requests) - 1
        character = "Alice" if index % 2 == 0 else "Bob"
        return web.json_response(
            {
                "id": f"chatcmpl-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"reply-{character}-{index}",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    application = web.Application()
    application.router.add_post("/v1/chat/completions", chat_completion)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        results = await generate_smoke_rollouts(
            config,
            (_episode(0),),
            base_url=f"http://127.0.0.1:{port}",
            request_model="base:rlff-sft",
            concurrency=1,
        )
    finally:
        await runner.cleanup()

    assert results[0]["status"] == "ok"
    trajectory = results[0]["trajectory"]
    assert [turn["character"] for turn in trajectory["turns"]] == [
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
        "Bob",
    ]
    assert len(requests) == 14
    assert all(request["model"] == "base:rlff-sft" for request in requests)
    assert any(
        message["role"] == "user" and "Alice:reply-Alice-0" in message["content"]
        for message in requests[1]["messages"]
    )
