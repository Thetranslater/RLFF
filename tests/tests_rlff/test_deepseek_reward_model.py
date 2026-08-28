"""Live DeepSeek reward-model test for previously generated rollout samples.

This is an executable diagnostic script, not an automatic pytest test.  It
loads either the text-only trajectory records produced by ``rollout_smoke`` or
the token-level ``interactions.jsonl`` files produced by ``rollout_audit``.
For the latter, it restores the source episode from the configured local
episode JSONL and rebuilds only the text view required by the production proxy.
It uses the same proxy payload builders and ``DeepSeekRewardProvider`` methods
as the training proxy.  It intentionally does not start SGLang, AReaL, or
perform an optimizer update.

Typical usage from the deployment directory::

    python tests/tests_rlff/test_deepseek_reward_model.py \
        --config configs/rlff.yaml \
        --input ../../temp/10smaples/rollout-audit

The output directory contains one JSONL record per completion-local or
trajectory-role request, including the parsed result, raw DeepSeek response,
thinking/reasoning content when returned by the API, usage, and latency.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import re
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlff.config import RLFFConfig, load_config
from rlff.contracts import EpisodeRecord
from rlff.episodes import EpisodeDataset, load_episode_jsonl
from rlff.rewards import (
    DeepSeekRewardProvider,
    RewardHTTPResponse,
    _default_reward_transport,
    build_proxy_completion_reward_payload,
    build_proxy_trajectory_reward_payload,
    create_reward_provider,
)

_REQUEST_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "rlff_reward_debug_request",
    default=None,
)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RewardSample:
    """One generated proxy trajectory and its canonical source episode."""

    source_index: int
    source_group: int
    source_path: str
    trajectory_slot: int | None
    episode: EpisodeRecord
    trajectory: Mapping[str, Any]


class RecordingTransport:
    """Delegate to RLFF's real HTTP transport while retaining API responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> RewardHTTPResponse:
        context = dict(_REQUEST_CONTEXT.get() or {})
        started = time.perf_counter()
        try:
            response = await _default_reward_transport(
                url=url,
                headers=headers,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            self.calls.append(
                {
                    **context,
                    "url": url,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        self.calls.append(
            {
                **context,
                "url": url,
                "status_code": response.status_code,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "request": {
                    "model": payload.get("model"),
                    "temperature": payload.get("temperature"),
                    "reasoning_effort": payload.get("reasoning_effort"),
                    "max_tokens": payload.get("max_tokens"),
                },
                "raw_response": response.text,
            }
        )
        return response


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"input file is empty: {path}")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON input must contain a list")
        records = value
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(item, Mapping) for item in records):
        raise ValueError("every input record must be a JSON object")
    return [item for item in records if isinstance(item, Mapping)]


def _clean_audit_completion_text(value: Any) -> str:
    """Remove tokenizer stop markers emitted by audit's lossless decoder."""

    text = str(value)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    return text.strip()


def _rollout_samples(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[RewardSample, ...]:
    samples: list[RewardSample] = []
    for source_index, record in enumerate(records):
        if record.get("status", "ok") != "ok":
            raise ValueError(
                f"input record {source_index} is not successful: "
                f"{record.get('error_type', 'unknown error')} {record.get('error', '')}"
            )
        trajectory_value = record.get("trajectory", record)
        if not isinstance(trajectory_value, Mapping):
            raise ValueError(f"input record {source_index} has no trajectory object")
        episode_value = trajectory_value.get("episode", record.get("episode"))
        if not isinstance(episode_value, Mapping):
            raise ValueError(f"input record {source_index} has no embedded episode")
        completions = trajectory_value.get("completions")
        if not isinstance(completions, Sequence) or isinstance(completions, (str, bytes)):
            raise ValueError(f"input record {source_index} trajectory has no completions")
        samples.append(
            RewardSample(
                source_index=source_index,
                source_group=source_index,
                source_path=str(path),
                trajectory_slot=None,
                episode=EpisodeRecord.model_validate(episode_value),
                trajectory=trajectory_value,
            )
        )
        if len(samples) >= limit:
            break
    return tuple(samples)


def _audit_interaction_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,) if path.name == "interactions.jsonl" else ()
    direct = path / "interactions.jsonl"
    if direct.is_file():
        return (direct,)
    return tuple(
        sorted(
            child / "interactions.jsonl"
            for child in path.iterdir()
            if child.is_dir()
            and not child.name.startswith(".")
            and (child / "interactions.jsonl").is_file()
        )
    )


def _audit_samples(
    path: Path,
    interaction_paths: Sequence[Path],
    *,
    episode_dataset: EpisodeDataset,
    limit: int,
) -> tuple[RewardSample, ...]:
    """Rebuild proxy text views from rollout-audit interaction records."""

    samples: list[RewardSample] = []
    for source_group, interaction_path in enumerate(interaction_paths[:limit]):
        records = _read_records(interaction_path)
        if not records:
            raise ValueError(f"audit interaction file is empty: {interaction_path}")
        by_slot: dict[int, list[Mapping[str, Any]]] = {}
        for record in records:
            try:
                slot = int(record["trajectory_slot"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"audit record in {interaction_path} has no trajectory_slot"
                ) from exc
            by_slot.setdefault(slot, []).append(record)
        first = records[0]
        summary: Mapping[str, Any] = {}
        summary_path = interaction_path.parent / "summary.json"
        if summary_path.is_file():
            summary_value = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            if isinstance(summary_value, Mapping):
                summary = summary_value
        episode_index_value = first.get("episode_index")
        if episode_index_value is None:
            episode_indices = summary.get("episode_indices")
            if isinstance(episode_indices, Sequence) and not isinstance(
                episode_indices, (str, bytes)
            ) and len(episode_indices) == 1:
                episode_index_value = episode_indices[0]
        try:
            episode_index = int(episode_index_value)
            episode = episode_dataset[episode_index]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"cannot resolve episode_index in audit file {interaction_path}"
            ) from exc
        expected_episode_id = str(first.get("episode_id", ""))
        if expected_episode_id and expected_episode_id != episode.episode_id:
            raise ValueError(
                f"episode mismatch in {interaction_path}: "
                f"audit={expected_episode_id!r}, dataset={episode.episode_id!r}"
            )
        group_id = f"{episode.episode_id}:reward-audit:{interaction_path.parent.name}"
        for slot, slot_records in sorted(by_slot.items()):
            ordered = sorted(
                slot_records,
                key=lambda item: (
                    int(item.get("turn_index", 0)),
                    int(item.get("interaction_index", 0)),
                ),
            )
            trajectory_id = f"{group_id}:trajectory:{slot}"
            completions = tuple(
                {
                    "completion_id": (
                        f"{trajectory_id}:completion:"
                        f"{int(item.get('interaction_index', index))}"
                    ),
                    "group_id": group_id,
                    "trajectory_id": trajectory_id,
                    "character": str(item.get("character", "")),
                    "turn_index": int(item.get("turn_index", index)),
                    "text": _clean_audit_completion_text(item.get("completion_text", "")),
                }
                for index, item in enumerate(ordered)
            )
            trajectory: Mapping[str, Any] = {
                "group_id": group_id,
                "trajectory_id": trajectory_id,
                "episode_id": str(episode.episode_id),
                "completions": completions,
                "turns": tuple(
                    {
                        "character": item["character"],
                        "content": item["text"],
                        "turn_id": item["turn_index"],
                    }
                    for item in completions
                ),
            }
            samples.append(
                RewardSample(
                    source_index=len(samples),
                    source_group=source_group,
                    source_path=str(interaction_path),
                    trajectory_slot=slot,
                    episode=episode,
                    trajectory=trajectory,
                )
            )
    if not samples:
        raise ValueError(f"no audit trajectories found in {path}")
    return tuple(samples)


def load_reward_samples(
    path: Path,
    *,
    limit: int = 10,
    episode_dataset: EpisodeDataset | None = None,
) -> tuple[RewardSample, ...]:
    """Load smoke ``rollouts.jsonl`` or one/multiple audit ``interactions.jsonl`` files."""

    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    smoke_path = path / "rollouts.jsonl" if path.is_dir() else path
    if smoke_path.is_file() and smoke_path.name == "rollouts.jsonl":
        samples = _rollout_samples(smoke_path, _read_records(smoke_path), limit=limit)
        if not samples:
            raise ValueError(f"no successful rollout samples found in {smoke_path}")
        return samples
    interaction_paths = _audit_interaction_paths(path)
    if interaction_paths:
        if episode_dataset is None:
            raise ValueError("episode_dataset is required for rollout-audit interactions.jsonl")
        return _audit_samples(
            path,
            interaction_paths,
            episode_dataset=episode_dataset,
            limit=limit,
        )
    raise ValueError(
        f"unsupported input {path}; expected rollouts.jsonl or a directory containing "
        "audit interactions.jsonl files"
    )


def _response_debug(raw_response: str) -> dict[str, Any]:
    """Extract content, reasoning, and usage from chat, Responses, or DashScope JSON."""

    try:
        outer = json.loads(raw_response)
    except json.JSONDecodeError:
        return {"content": raw_response, "thinking": None, "usage": {}}
    if not isinstance(outer, Mapping):
        return {"content": raw_response, "thinking": None, "usage": {}}

    usage = outer.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    choices = outer.get("choices")
    message: Mapping[str, Any] = {}
    finish_reason: Any = None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        candidate = choices[0].get("message")
        if isinstance(candidate, Mapping):
            message = candidate
        finish_reason = choices[0].get("finish_reason")

    content = message.get("content")
    content_text = content if isinstance(content, str) else None
    thinking: Any = None
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if value is not None:
            thinking = value
            break

    # Qwen's native DashScope API returns choices inside ``output``.  Its
    # reasoning content remains on the assistant message.
    if not message:
        output = outer.get("output")
        if isinstance(output, Mapping):
            dashscope_choices = output.get("choices")
            if (
                isinstance(dashscope_choices, list)
                and dashscope_choices
                and isinstance(dashscope_choices[0], Mapping)
            ):
                candidate = dashscope_choices[0].get("message")
                if isinstance(candidate, Mapping):
                    message = candidate
                finish_reason = dashscope_choices[0].get("finish_reason")
            content = message.get("content")
            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                content_parts = [
                    str(part.get("text"))
                    for part in content
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                ]
                content_text = "\n".join(content_parts) or None
            else:
                content_text = None
            for key in ("reasoning_content", "reasoning", "thinking"):
                value = message.get(key)
                if value is not None:
                    thinking = value
                    break

        if isinstance(output, list):
            # Qwen's Responses API returns reasoning and the final answer as
            # separate items in ``output`` rather than choices[0].message.
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                item_type = item.get("type")
                if item_type == "reasoning":
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_parts.extend(
                            str(part.get("text"))
                            for part in summary
                            if isinstance(part, Mapping) and part.get("text") is not None
                        )
                    elif isinstance(summary, str):
                        reasoning_parts.append(summary)
                elif item_type == "message":
                    parts = item.get("content")
                    if isinstance(parts, list):
                        content_parts.extend(
                            str(part.get("text"))
                            for part in parts
                            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                        )
                    elif isinstance(parts, str):
                        content_parts.append(parts)
                if item.get("status") is not None:
                    finish_reason = item.get("status")
            content_text = "\n".join(content_parts) or None
            thinking = "\n".join(reasoning_parts) or None

    if thinking is None and content_text is not None:
        match = _THINK_PATTERN.search(content_text)
        if match is not None:
            thinking = match.group(1).strip()
    return {
        "content": content_text if content_text is not None else content,
        "thinking": thinking,
        "usage": dict(usage),
        "finish_reason": finish_reason,
    }


def _numeric_usage(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for call in calls:
        usage = _response_debug(str(call.get("raw_response", ""))).get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                totals[str(key)] = totals.get(str(key), 0) + value
    return dict(sorted(totals.items()))


async def _score_operation(
    *,
    provider: DeepSeekRewardProvider,
    transport: RecordingTransport,
    sample: RewardSample,
    scope: str,
    character: str,
) -> dict[str, Any]:
    operation_id = f"sample-{sample.source_index}:{scope}:{character}"
    if scope == "completion_local":
        payload = build_proxy_completion_reward_payload(
            sample.episode,
            sample.trajectory,
            character,
        )
    else:
        payload = build_proxy_trajectory_reward_payload(
            sample.episode,
            sample.trajectory,
            character,
        )
    token = _REQUEST_CONTEXT.set(
        {
            "operation_id": operation_id,
            "scope": scope,
            "source_index": str(sample.source_index),
            "episode_id": str(sample.episode.episode_id or ""),
            "trajectory_id": str(sample.trajectory.get("trajectory_id", "")),
            "character": character,
        }
    )
    started = time.perf_counter()
    try:
        if scope == "completion_local":
            result: Any = await provider.score_proxy_completion_role(payload)
        else:
            result = await provider.score_proxy_trajectory_role(payload)
        error: str | None = None
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _REQUEST_CONTEXT.reset(token)

    calls = [call for call in transport.calls if call.get("operation_id") == operation_id]
    attempts = []
    for call in calls:
        debug = _response_debug(str(call.get("raw_response", "")))
        attempts.append({**call, **debug})
    return {
        "operation_id": operation_id,
        "source_index": sample.source_index,
        "source_group": sample.source_group,
        "source_path": sample.source_path,
        "trajectory_slot": sample.trajectory_slot,
        "episode_id": sample.episode.episode_id,
        "trajectory_id": sample.trajectory.get("trajectory_id"),
        "group_id": sample.trajectory.get("group_id"),
        "scope": scope,
        "character": character,
        "status": "ok" if error is None else "error",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "result": result,
        "error": error,
        "attempts": attempts,
        "usage": _numeric_usage(calls),
        "payload": payload,
    }


async def run_reward_model_test(
    config: RLFFConfig,
    samples: Sequence[RewardSample],
    *,
    sample_concurrency: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Score each sample with production-equivalent proxy calls."""

    if type(sample_concurrency) is not int or sample_concurrency <= 0:
        raise ValueError("sample_concurrency must be a positive integer")
    transport = RecordingTransport()
    secrets = config.resolve_runtime_secrets(require_reward_key=True)
    provider = create_reward_provider(
        config.rewards,
        api_key=secrets.deepseek_api_key,
        transport=transport,
        langsmith_api_key=None,
    )
    if not isinstance(provider, DeepSeekRewardProvider):
        raise ValueError("configs/rlff.yaml must use rewards.provider=deepseek")

    semaphore = asyncio.Semaphore(sample_concurrency)

    async def score_sample(sample: RewardSample) -> list[dict[str, Any]]:
        async with semaphore:
            characters = tuple(
                str(character.name) for character in sample.episode.characters
            )
            return list(
                await asyncio.gather(
                    *(
                        _score_operation(
                            provider=provider,
                            transport=transport,
                            sample=sample,
                            scope=scope,
                            character=character,
                        )
                        for scope in ("completion_local", "trajectory_role")
                        for character in characters
                    )
                )
            )

    try:
        nested = await asyncio.gather(*(score_sample(sample) for sample in samples))
        return tuple(item for group in nested for item in group)
    finally:
        await provider.aclose()


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("outputs") / "deepseek-reward-test" / timestamp


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _print_debug(record: Mapping[str, Any], *, provider_label: str = "DeepSeek") -> None:
    print(
        f"\n=== sample={record['source_index']} scope={record['scope']} "
        f"character={record['character']} status={record['status']} "
        f"http_attempts={len(record.get('attempts', ())) } "
        f"elapsed={record['elapsed_seconds']}s ==="
    )
    if record.get("error"):
        print(f"error: {record['error']}")
    print("parsed result:")
    print(json.dumps(record.get("result"), ensure_ascii=False, indent=2, sort_keys=True))
    for attempt_index, attempt in enumerate(record.get("attempts", ()), start=1):
        print(f"--- {provider_label} attempt {attempt_index} ---")
        print("usage:")
        print(json.dumps(attempt.get("usage", {}), ensure_ascii=False, indent=2))
        print("thinking/reasoning:")
        print(attempt.get("thinking") or "<none returned>")
        print("content:")
        print(attempt.get("content") or "<none returned>")
        print("raw response:")
        print(attempt.get("raw_response", "<none>"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="rollouts.jsonl, interactions.jsonl, or a directory of audit result directories",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sample-concurrency", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        config = load_config(args.config)
        episode_dataset = load_episode_jsonl(config.episode_grouping.dataset_path)
        samples = load_reward_samples(
            args.input,
            limit=args.limit,
            episode_dataset=episode_dataset,
        )
        started = time.perf_counter()
        records = asyncio.run(
            run_reward_model_test(
                config,
                samples,
                sample_concurrency=args.sample_concurrency,
            )
        )
        for record in records:
            _print_debug(record)
        _write_jsonl(output_dir / "reward_results.jsonl", records)
        http_attempts = sum(len(record.get("attempts", ())) for record in records)
        skipped_operations = sum(
            record["status"] == "ok" and not record.get("attempts") for record in records
        )
        successful_http_attempts = sum(
            200 <= int(attempt.get("status_code", 0)) < 300
            for record in records
            for attempt in record.get("attempts", ())
        )
        usage: dict[str, int] = {}
        for record in records:
            for key, value in record["usage"].items():
                usage[key] = usage.get(key, 0) + int(value)
        summary = {
            "status": "passed" if all(item["status"] == "ok" for item in records) else "failed",
            "input": str(args.input.resolve()),
            "config": str(args.config.resolve()),
            "requested_samples": args.limit,
            "loaded_trajectories": len(samples),
            "loaded_episode_groups": len({item.source_group for item in samples}),
            "logical_reward_operations": len(records),
            "successful_logical_operations": sum(
                item["status"] == "ok" for item in records
            ),
            "failed_logical_operations": sum(item["status"] != "ok" for item in records),
            "skipped_no_task_operations": skipped_operations,
            "deepseek_http_attempts": http_attempts,
            "successful_http_attempts": successful_http_attempts,
            "sample_concurrency": args.sample_concurrency,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "usage_totals": dict(sorted(usage.items())),
            "provider": config.rewards.provider,
            "completion_model": config.rewards.completion.model,
            "trajectory_model": config.rewards.global_reward.model,
            "completion_concurrency": config.rewards.completion.concurrency,
            "trajectory_concurrency": config.rewards.global_reward.concurrency,
            "completion_retries": config.rewards.completion.retries,
            "trajectory_retries": config.rewards.global_reward.retries,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("\n=== summary ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"debug files: {output_dir}", file=sys.stderr)
        return 0 if summary["status"] == "passed" else 1
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"deepseek-reward-test: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"failure details: {output_dir / 'failure.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RewardSample", "load_reward_samples", "main", "run_reward_model_test"]
