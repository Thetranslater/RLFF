"""Focused JSONL observability for RLFF rewards and policy updates.

Reward audit rows contain every parsed completion/trajectory reward without
prompt text.  A stable hash samples a small subset of trajectories into a
separate detail file that may contain the complete request and raw reward-model
response.  Training rows reuse AReaL's native statistics; no PPO/GRPO quantity
is recomputed here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REWARD_AUDIT_SCHEMA_VERSION = "rlff.reward-audit.v1"
REWARD_DETAIL_SCHEMA_VERSION = "rlff.reward-detail.v1"
REWARD_FAILURE_SCHEMA_VERSION = "rlff.reward-failure.v1"
TRAINING_METRICS_SCHEMA_VERSION = "rlff.training-metrics.v1"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common scalar/container values to strict JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                _json_safe(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


class RewardAuditWriter:
    """Write compact rewards, sampled details, and unsampled terminal failures."""

    def __init__(
        self,
        *,
        audit_jsonl: str | Path | None = None,
        detail_jsonl: str | Path | None = None,
        detail_sample_rate: float = 0.0,
        failure_jsonl: str | Path | None = None,
    ) -> None:
        if not 0.0 <= detail_sample_rate <= 1.0:
            raise ValueError("detail_sample_rate must be between 0 and 1")
        self.audit_jsonl = Path(audit_jsonl) if audit_jsonl is not None else None
        self.detail_jsonl = Path(detail_jsonl) if detail_jsonl is not None else None
        self.detail_sample_rate = float(detail_sample_rate)
        self.failure_jsonl = Path(failure_jsonl) if failure_jsonl is not None else None

    def detail_selected(self, trajectory_id: str) -> bool:
        """Select a trajectory deterministically, independently of concurrency."""

        if self.detail_jsonl is None or self.detail_sample_rate <= 0.0:
            return False
        if self.detail_sample_rate >= 1.0:
            return True
        digest = hashlib.sha256(trajectory_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return bucket < self.detail_sample_rate

    def write_reward(self, record: Mapping[str, Any]) -> None:
        if self.audit_jsonl is None:
            return
        _append_jsonl(
            self.audit_jsonl,
            {
                "schema_version": REWARD_AUDIT_SCHEMA_VERSION,
                "timestamp": _timestamp(),
                **record,
            },
        )

    def write_detail(
        self,
        *,
        trajectory_id: str,
        record: Mapping[str, Any],
    ) -> None:
        if not self.detail_selected(trajectory_id) or self.detail_jsonl is None:
            return
        _append_jsonl(
            self.detail_jsonl,
            {
                "schema_version": REWARD_DETAIL_SCHEMA_VERSION,
                "timestamp": _timestamp(),
                **record,
            },
        )

    def write_failure(self, record: Mapping[str, Any]) -> None:
        """Persist every terminal reward failure, without deterministic sampling."""

        if self.failure_jsonl is None:
            return
        _append_jsonl(
            self.failure_jsonl,
            {
                "schema_version": REWARD_FAILURE_SCHEMA_VERSION,
                "timestamp": _timestamp(),
                **record,
            },
        )


def _metric_value(item: Mapping[str, Any], suffix: str) -> tuple[str, Any] | None:
    average_suffix = f"{suffix}/avg"
    matches = [
        (str(key), value)
        for key, value in item.items()
        if str(key).endswith(average_suffix) or str(key).endswith(suffix)
    ]
    if not matches:
        return None
    matches.sort(
        key=lambda pair: (
            not pair[0].endswith(average_suffix),
            "ppo_actor/update/" not in pair[0],
            len(pair[0]),
        )
    )
    return matches[0]


class TrainingMetricsStatsLoggerProxy:
    """Delegate AReaL StatsLogger while persisting two native PPO statistics."""

    def __init__(self, delegate: Any, jsonl_path: str | Path) -> None:
        self._delegate = delegate
        self._jsonl_path = Path(jsonl_path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def commit(
        self,
        epoch: int,
        step: int,
        global_step: int,
        data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> Any:
        items = [data] if isinstance(data, Mapping) else list(data)
        for item_index, item in enumerate(items):
            policy = _metric_value(item, "/actor_loss")
            clip = _metric_value(item, "/clip_ratio")
            if policy is None and clip is None:
                continue
            source_metrics: dict[str, Any] = {}
            record: dict[str, Any] = {
                "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
                "timestamp": _timestamp(),
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "stats_item_index": item_index,
            }
            if policy is not None:
                source_metrics[policy[0]] = policy[1]
                record["policy_loss"] = policy[1]
            if clip is not None:
                source_metrics[clip[0]] = clip[1]
                record["clip_fraction"] = clip[1]
            record["source_metrics"] = source_metrics
            _append_jsonl(self._jsonl_path, record)
        return self._delegate.commit(epoch, step, global_step, data)


__all__ = [
    "REWARD_AUDIT_SCHEMA_VERSION",
    "REWARD_DETAIL_SCHEMA_VERSION",
    "REWARD_FAILURE_SCHEMA_VERSION",
    "TRAINING_METRICS_SCHEMA_VERSION",
    "RewardAuditWriter",
    "TrainingMetricsStatsLoggerProxy",
]
