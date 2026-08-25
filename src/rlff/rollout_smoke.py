"""Cloud smoke test for RLFF's production multi-character rollout path.

The command samples canonical episodes, then calls
``RLFFGroupAwareAgent.generate_trajectory``.  It does not reproduce prompt or
round-robin logic and deliberately does not invoke rewards or training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .config import RLFFConfig, load_config
from .contracts import EpisodeRecord
from .episodes import EpisodeDataset, build_episode_group, load_episode_jsonl
from .proxy import RLFFGroupAwareAgent
from .runtime import build_agent_workflow_kwargs

SMOKE_SCHEMA_VERSION = "rlff.rollout-smoke.v1"
DEFAULT_LIMIT = 20
DEFAULT_CONCURRENCY = 4
DEFAULT_SEED = 42
DEFAULT_ADAPTER_NAME = "rlff-sft"


def select_episode_records(
    dataset: EpisodeDataset,
    *,
    limit: int = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
) -> tuple[EpisodeRecord, ...]:
    """Select unique episodes deterministically without changing source order."""

    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if limit > len(dataset):
        raise ValueError(f"cannot sample {limit} episodes from dataset of size {len(dataset)}")
    indexes = random.Random(seed).sample(range(len(dataset)), limit)
    return tuple(dataset[index] for index in indexes)


def normalize_openai_base_url(value: str) -> str:
    """Normalize an SGLang server root to its OpenAI-compatible ``/v1`` root."""

    root = value.strip().rstrip("/")
    if not root:
        raise ValueError("base_url must be non-empty")
    return root if root.endswith("/v1") else root + "/v1"


def lora_request_model(config: RLFFConfig, adapter_name: str = DEFAULT_ADAPTER_NAME) -> str:
    """Return SGLang's OpenAI model selector for the configured LoRA."""

    name = adapter_name.strip()
    if not name or ":" in name:
        raise ValueError("adapter_name must be non-empty and cannot contain ':'")
    return f"{config.sglang.model}:{name}"


def _episode_payload(record: EpisodeRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _selected_payload(index: int, record: EpisodeRecord) -> dict[str, Any]:
    return {
        "selection_index": index,
        "episode_id": record.episode_id,
        "title": record.title,
        "characters": [character.name for character in record.characters],
        "episode": _episode_payload(record),
    }


async def _run_one(
    *,
    index: int,
    record: EpisodeRecord,
    config: RLFFConfig,
    agent: RLFFGroupAwareAgent,
    base_url: str,
    request_model: str,
    api_key: str,
    http_client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    templates = config.prompt.load_templates()
    group = build_episode_group(
        record,
        group_size=config.episode_grouping.group_size,
        base_seed=config.episode_grouping.base_seed,
        templates=templates,
    )
    sample = group.samples[0]
    data = {
        "episode_id": sample.episode_id,
        "group_id": sample.group_id,
        "render": sample.render.model_dump(mode="json"),
        "episode": _episode_payload(record),
        "model": request_model,
    }
    started = time.perf_counter()
    try:
        async with semaphore:
            trajectory = await agent.generate_trajectory(
                data,
                base_url=base_url,
                http_client=http_client,
                api_key=api_key,
                group_id=sample.group_id,
                trajectory_id=sample.trajectory_id,
            )
        return {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "selection_index": index,
            "status": "ok",
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "render": sample.render.model_dump(mode="json"),
            "trajectory": asdict(trajectory),
        }
    except Exception as exc:
        return {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "selection_index": index,
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "episode_id": record.episode_id,
            "title": record.title,
            "characters": [character.name for character in record.characters],
            "render": sample.render.model_dump(mode="json"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


async def generate_smoke_rollouts(
    config: RLFFConfig,
    records: Sequence[EpisodeRecord],
    *,
    base_url: str,
    request_model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    api_key: str = "EMPTY",
    agent: RLFFGroupAwareAgent | None = None,
    http_client: Any = None,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate one complete production-path trajectory per selected episode."""

    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if not records:
        raise ValueError("at least one episode is required")
    actual_agent = agent or RLFFGroupAwareAgent(**build_agent_workflow_kwargs(config))
    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    completed = 0

    async def run_and_report(index: int, record: EpisodeRecord, client: Any) -> dict[str, Any]:
        nonlocal completed
        result = await _run_one(
            index=index,
            record=record,
            config=config,
            agent=actual_agent,
            base_url=normalize_openai_base_url(base_url),
            request_model=request_model,
            api_key=api_key,
            http_client=client,
            semaphore=semaphore,
        )
        if progress is not None:
            async with progress_lock:
                completed += 1
                progress(completed, len(records), result)
        return result

    async def execute(client: Any) -> tuple[dict[str, Any], ...]:
        return tuple(
            await asyncio.gather(
                *(
                    run_and_report(index, record, client)
                    for index, record in enumerate(records)
                )
            )
        )

    if http_client is not None:
        return await execute(http_client)
    timeout = httpx.Timeout(config.sglang.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await execute(client)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in values
    )
    path.write_text(content, encoding="utf-8")


def _write_text_report(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for result in results:
        index = result["selection_index"]
        lines.append(f"=== SAMPLE {index:02d} | {result['status']} ===")
        if result["status"] != "ok":
            lines.extend(
                (
                    f"episode_id: {result.get('episode_id', '')}",
                    f"title: {result.get('title', '')}",
                    f"error: {result.get('error_type', '')}: {result.get('error', '')}",
                    "",
                )
            )
            continue
        trajectory = result["trajectory"]
        episode = trajectory["episode"]
        lines.extend(
            (
                f"episode_id: {trajectory['episode_id']}",
                f"trajectory_id: {trajectory['trajectory_id']}",
                f"title: {episode.get('title', '')}",
                "characters: "
                + ", ".join(item["name"] for item in episode.get("characters", [])),
                f"template: {result['render']['template_id']}",
                f"elapsed_seconds: {result['elapsed_seconds']}",
                "--- GENERATED DIALOGUE ---",
            )
        )
        for turn in trajectory["turns"]:
            lines.append(f"{turn['character']}: {turn['content']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("outputs") / "rollout-smoke" / timestamp


def _print_progress(done: int, total: int, result: Mapping[str, Any]) -> None:
    print(
        f"[{done}/{total}] sample={result['selection_index']} "
        f"status={result['status']} elapsed={result['elapsed_seconds']}s",
        file=sys.stderr,
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", help="OpenAI-compatible SGLang URL; defaults to config")
    parser.add_argument(
        "--request-model",
        help="OpenAI model selector; defaults to <configured-model>:<adapter-name>",
    )
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER_NAME)
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    dataset = load_episode_jsonl(config.episode_grouping.dataset_path)
    selected = select_episode_records(dataset, limit=args.limit, seed=args.seed)
    base_url = args.base_url or config.sglang.base_url
    request_model = args.request_model or lora_request_model(config, args.adapter_name)
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(
        output_dir / "selected_episodes.jsonl",
        [_selected_payload(index, record) for index, record in enumerate(selected)],
    )

    secrets = config.resolve_runtime_secrets(require_reward_key=False)
    started_at = datetime.now(UTC)
    results = asyncio.run(
        generate_smoke_rollouts(
            config,
            selected,
            base_url=base_url,
            request_model=request_model,
            concurrency=args.concurrency,
            api_key=secrets.sglang_api_key or "EMPTY",
            progress=_print_progress,
        )
    )
    finished_at = datetime.now(UTC)
    _write_jsonl(output_dir / "rollouts.jsonl", results)
    _write_text_report(output_dir / "rollouts.txt", results)
    success_count = sum(result["status"] == "ok" for result in results)
    summary = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round((finished_at - started_at).total_seconds(), 6),
        "config": str(args.config.resolve()),
        "config_fingerprint": config.config_fingerprint(),
        "dataset": str(config.episode_grouping.dataset_path),
        "dataset_fingerprint": dataset.fingerprint,
        "selection_seed": args.seed,
        "requested": args.limit,
        "succeeded": success_count,
        "failed": len(results) - success_count,
        "concurrency": args.concurrency,
        "max_rounds": config.rollout.max_rounds,
        "temperature": config.sglang.temperature,
        "top_p": config.sglang.top_p,
        "max_new_tokens": config.sglang.max_new_tokens,
        "base_url": normalize_openai_base_url(base_url),
        "request_model": request_model,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        summary = run_from_args(args)
    except Exception as exc:
        print(f"rlff-rollout-smoke: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ADAPTER_NAME",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_LIMIT",
    "DEFAULT_SEED",
    "SMOKE_SCHEMA_VERSION",
    "generate_smoke_rollouts",
    "lora_request_model",
    "main",
    "normalize_openai_base_url",
    "run_from_args",
    "select_episode_records",
]
