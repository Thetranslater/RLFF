"""Small RLFF CLI for local preflight and pinned cloud launch."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from typing import Any

from .config import RLFFConfig, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlff")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("dry-run", "validate", "train"),
        default="dry-run",
        help="local CPU preflight (default), data validation, or cloud training",
    )
    parser.add_argument("--config", required=True, help="RLFF YAML/JSON configuration")
    parser.add_argument(
        "--workflow",
        help=(
            "cloud workflow import path module:object; defaults to the pinned RLFFGroupAwareAgent"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON only; dry-run and validate are JSON by default",
    )
    return parser


def _import_symbol(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("workflow must use module:object syntax")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def _run_local_validation(config: RLFFConfig, *, include_data: bool) -> dict[str, Any]:
    from .runtime import describe_runtime_plan

    report = describe_runtime_plan(config)
    report["config"] = config.resolved_snapshot()
    if include_data:
        from .episodes import load_episode_jsonl

        dataset = load_episode_jsonl(
            config.episode_grouping.dataset_path,
            limit=config.episode_grouping.limit,
        )
        report["dataset"] = {
            "path": str(config.episode_grouping.dataset_path),
            "records": len(dataset.records),
            "fingerprint": dataset.fingerprint,
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config)
        if args.command in {"dry-run", "validate"}:
            report = _run_local_validation(config, include_data=args.command == "validate")
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        from .runtime import run_training

        workflow = _import_symbol(args.workflow or "rlff.runtime:RLFFGroupAwareAgent")
        run_training(config, workflow=workflow)
        return 0
    except Exception as exc:
        print(f"rlff: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by console entry point
    raise SystemExit(main())


__all__ = ["main"]
