"""Live Qwen3.7-Flash reward-model test using the native DashScope API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlff.config import RLFFConfig, load_config
from rlff.episodes import load_episode_jsonl
from rlff.rewards import (
    QwenDashScopeRewardProvider,
    RewardHTTPResponse,
    _default_reward_transport,
)

try:
    from test_deepseek_reward_model import (
        _REQUEST_CONTEXT,
        RewardSample,
        _numeric_usage,
        _print_debug,
        _score_operation,
        _write_jsonl,
        load_reward_samples,
    )
except ModuleNotFoundError:  # pragma: no cover - supports package imports too
    from tests.tests_rlff.test_deepseek_reward_model import (
        _REQUEST_CONTEXT,
        RewardSample,
        _numeric_usage,
        _print_debug,
        _score_operation,
        _write_jsonl,
        load_reward_samples,
    )


QWEN_MODEL = "qwen3.7-flash"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"
QWEN_TEMPERATURE = 0.7
QWEN_REASONING_EFFORT = "medium"
QWEN_MAX_COMPLETION_TOKENS = 16384


class QwenDashScopeTransport:
    """Call native DashScope while recording its response for debugging."""

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
        qwen_payload = dict(payload)
        qwen_url = url
        try:
            response = await _default_reward_transport(
                url=qwen_url,
                headers=headers,
                payload=qwen_payload,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            self.calls.append(
                {
                    **context,
                    "url": qwen_url,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "request": qwen_payload,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        self.calls.append(
            {
                **context,
                "url": qwen_url,
                "status_code": response.status_code,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "request": qwen_payload,
                "raw_response": response.text,
            }
        )
        return response

    async def aclose(self) -> None:
        return None


async def run_qwen_reward_model_test(
    config: RLFFConfig,
    samples: Sequence[RewardSample],
    *,
    sample_concurrency: int = 1,
    api_key_env: str = QWEN_API_KEY_ENV,
    base_url: str = QWEN_BASE_URL,
    model: str = QWEN_MODEL,
) -> tuple[dict[str, Any], ...]:
    """Score samples with Qwen while retaining production proxy boundaries."""

    if type(sample_concurrency) is not int or sample_concurrency <= 0:
        raise ValueError("sample_concurrency must be a positive integer")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Qwen reward test requires environment variable {api_key_env}")

    rewards = config.rewards
    transport = QwenDashScopeTransport()
    provider = QwenDashScopeRewardProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        trajectory_reward_model=model,
        completion_prompt_path=rewards.completion.prompt_path,
        trajectory_prompt_path=rewards.global_reward.prompt_path,
        completion_timeout_seconds=float(rewards.completion.timeout_seconds),
        trajectory_timeout_seconds=float(rewards.global_reward.timeout_seconds),
        completion_retries=int(rewards.completion.retries),
        trajectory_retries=int(rewards.global_reward.retries),
        completion_concurrency=int(rewards.completion.concurrency),
        trajectory_concurrency=int(rewards.global_reward.concurrency),
        completion_temperature=QWEN_TEMPERATURE,
        trajectory_temperature=QWEN_TEMPERATURE,
        completion_reasoning_effort=QWEN_REASONING_EFFORT,
        trajectory_reasoning_effort=QWEN_REASONING_EFFORT,
        completion_max_tokens=QWEN_MAX_COMPLETION_TOKENS,
        trajectory_max_tokens=QWEN_MAX_COMPLETION_TOKENS,
        transport=transport,
    )

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
    return Path("outputs") / "qwen3.7-flash-reward-test" / timestamp


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
    parser.add_argument("--api-key-env", default=QWEN_API_KEY_ENV)
    parser.add_argument("--base-url", default=QWEN_BASE_URL)
    parser.add_argument("--model", default=QWEN_MODEL)
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
            run_qwen_reward_model_test(
                config,
                samples,
                sample_concurrency=args.sample_concurrency,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
            )
        )
        for record in records:
            _print_debug(record, provider_label="Qwen3.7-Flash")
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
        usage = _numeric_usage(
            [attempt for record in records for attempt in record.get("attempts", ())]
        )
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
            "qwen_http_attempts": http_attempts,
            "successful_http_attempts": successful_http_attempts,
            "sample_concurrency": args.sample_concurrency,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "usage_totals": usage,
            "provider": QWEN_MODEL,
            "protocol": "dashscope-native-http",
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "temperature": QWEN_TEMPERATURE,
            "enable_thinking": True,
            "reasoning_effort": QWEN_REASONING_EFFORT,
            "max_completion_tokens": QWEN_MAX_COMPLETION_TOKENS,
            "completion_model": args.model,
            "trajectory_model": args.model,
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
        print(f"qwen3.7-flash-reward-test: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"failure details: {output_dir / 'failure.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["QwenDashScopeRewardProvider", "main", "run_qwen_reward_model_test"]
