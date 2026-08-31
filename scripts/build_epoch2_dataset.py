"""Build the ordered dataset used by RLFF's second training epoch.

The first epoch's dataset is intentionally left untouched. Epoch two uses the
same canonical episode records, but applies a stable two-stage mixed curriculum
so the curriculum is reproducible and does not change episode content or
fingerprints. The first half is 70% easy and 30% normal/difficult; the second
half reverses those densities. Both halves interleave their two classes rather
than storing either class as one contiguous run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _is_easy(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    return isinstance(metadata, dict) and metadata.get("difficulty") == "easy"


def _weave(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stably distribute ``secondary`` records throughout ``primary`` records."""

    total = len(primary) + len(secondary)
    if total == 0:
        return []
    output: list[dict[str, Any]] = []
    primary_index = 0
    secondary_index = 0
    for position in range(total):
        # Rounded cumulative quota spaces the minority class across the phase.
        secondary_target = (
            (position + 1) * len(secondary) + total // 2
        ) // total
        if secondary_index < secondary_target:
            output.append(secondary[secondary_index])
            secondary_index += 1
        else:
            output.append(primary[primary_index])
            primary_index += 1
    return output


def reorder_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable 70/30 then 30/70 mixed curriculum."""

    materialized = list(records)
    if len(materialized) % 2:
        raise ValueError("epoch-two curriculum requires an even record count")
    easy = [record for record in materialized if _is_easy(record)]
    difficult = [record for record in materialized if not _is_easy(record)]
    phase_size = len(materialized) // 2
    early_difficult_count = round(phase_size * 0.30)
    early_easy_count = phase_size - early_difficult_count
    if len(easy) < early_easy_count or len(difficult) < early_difficult_count:
        raise ValueError(
            "dataset cannot satisfy the first-half 70% easy / 30% difficult mix"
        )
    late_easy_count = len(easy) - early_easy_count
    late_difficult_count = len(difficult) - early_difficult_count
    if late_easy_count + late_difficult_count != phase_size:
        raise ValueError(
            "dataset cannot satisfy equal-size curriculum halves with available classes"
        )

    early = _weave(
        easy[:early_easy_count],
        difficult[:early_difficult_count],
    )
    late = _weave(
        difficult[early_difficult_count:],
        easy[early_easy_count:],
    )
    return early + late


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="first-epoch episode JSONL")
    parser.add_argument("output", type=Path, help="second-epoch episode JSONL")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    ordered = reorder_records(records)
    write_jsonl(args.output, ordered)
    easy_count = sum(_is_easy(record) for record in ordered)
    midpoint = len(ordered) // 2
    early_easy = sum(_is_easy(record) for record in ordered[:midpoint])
    late_easy = sum(_is_easy(record) for record in ordered[midpoint:])
    print(
        f"wrote {len(ordered)} records ({easy_count} easy, "
        f"{len(ordered) - easy_count} difficult); "
        f"first half={early_easy} easy/{midpoint - early_easy} difficult, "
        f"second half={late_easy} easy/{midpoint - late_easy} difficult"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
