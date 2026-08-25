"""Cloud audit for the exact AReaL/SGLang rollout-to-training boundary.

Unlike :mod:`rlff.rollout_smoke`, this command intentionally starts the pinned
AReaL runtime.  It captures the tensors exported by AReaL's
``OpenAIProxyWorkflow`` and then runs RLFF's real actor advantage adapter.  No
DeepSeek request and no optimizer update is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RLFFConfig, load_config
from .proxy import (
    DETERMINISTIC_AUDIT_REWARD_PROVIDER,
    DeterministicAuditRewardProvider,
    decode_proxy_episode_payload,
)

AUDIT_SCHEMA_VERSION = "rlff.rollout-audit.v1"
_FLOAT_TOLERANCE = 1e-5


def _normalised_slot_rewards(group_size: int) -> tuple[float, ...]:
    values = tuple(float(index) for index in range(group_size))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return (0.0,) * group_size
    return tuple((value - mean) / std for value in values)


def _clone_batches(batches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for batch in batches:
        item: dict[str, Any] = {}
        for key, value in batch.items():
            detach = getattr(value, "detach", None)
            clone = getattr(value, "clone", None)
            if callable(detach) and callable(clone):
                item[key] = value.detach().clone()
            else:
                item[key] = copy.deepcopy(value)
        cloned.append(item)
    return cloned


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _token_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        return [_decode(tokenizer, [token_id]) for token_id in token_ids]
    converted = convert(list(token_ids))
    if isinstance(converted, str):
        return [converted]
    return [str(value) for value in converted]


def _as_rows(value: Any, *, field: str) -> list[list[Any]]:
    if value is None:
        raise ValueError(f"rollout batch is missing {field}")
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        raise ValueError(
            f"rollout field {field} is not a tensor/list "
            f"(received {type(value).__name__})"
        )
    if value and not isinstance(value[0], list):
        return [[item] for item in value]
    return value


def _localize_areal_batches(
    raw_batches: Sequence[Mapping[str, Any]],
    training_batches: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch AReaL single-controller RTensors before local inspection.

    AReaL's controller transports rollout tensors as ``RTensor`` handles. Its
    normal trainer localizes those handles at RPC boundaries, but this audit
    deliberately inspects the controller result itself. Localizing both
    collections together lets AReaL batch the remote shard fetches.
    """

    try:
        from areal.infra.rpc.rtensor import RTensor
    except ImportError:
        return (
            [dict(batch) for batch in raw_batches],
            [dict(batch) for batch in training_batches],
        )

    localized = RTensor.localize([list(raw_batches), list(training_batches)])
    local_raw, local_training = localized
    return (
        [dict(batch) for batch in local_raw],
        [dict(batch) for batch in local_training],
    )


def _roll_left(values: Sequence[float | int]) -> list[float | int]:
    if not values:
        return []
    return [*values[1:], values[0]]


def _close(left: float, right: float, tolerance: float = _FLOAT_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _record_problem(
    collection: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    interaction_index: int | None = None,
) -> None:
    problem: dict[str, Any] = {"code": code, "message": message}
    if interaction_index is not None:
        problem["interaction_index"] = interaction_index
    collection.append(problem)


def audit_rollout_batches(
    raw_batches: Sequence[Mapping[str, Any]],
    training_batches: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    characters: Sequence[str],
    group_size: int,
    max_rounds: int,
    context_length: int,
    max_new_tokens: int,
    include_prompt_text: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and serialize real AReaL rollout and actor tensors."""

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    if not characters:
        raise ValueError("at least one character is required for rollout audit")
    if group_size < 2:
        raise ValueError("rollout audit requires group_size >= 2")
    if len(raw_batches) != len(training_batches):
        raise ValueError("raw and training batch counts differ")

    required = {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "logprobs",
        "versions",
        "rewards",
    }
    training_required = required | {
        "advantages",
        "returns",
        "kl_rewards",
        "tot_rewards",
    }
    expected_rewards = _normalised_slot_rewards(group_size)
    turns_per_trajectory = len(characters) * max_rounds
    expected_interactions = group_size * turns_per_trajectory
    slot_ordinals: defaultdict[int, int] = defaultdict(int)
    character_counts: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    completion_logprobs: list[float] = []
    observed_versions: list[int] = []
    global_index = 0

    for batch_index, (raw_batch, training_batch) in enumerate(
        zip(raw_batches, training_batches, strict=True)
    ):
        missing = required - set(raw_batch)
        training_missing = training_required - set(training_batch)
        if missing:
            raise ValueError(f"raw rollout batch missing fields: {sorted(missing)}")
        if training_missing:
            raise ValueError(f"training batch missing fields: {sorted(training_missing)}")

        raw_rows = {key: _as_rows(raw_batch[key], field=key) for key in required}
        trained_rows = {key: _as_rows(training_batch[key], field=key) for key in training_required}
        batch_size = len(raw_rows["input_ids"])
        for key, rows in (*raw_rows.items(), *trained_rows.items()):
            if len(rows) != batch_size:
                raise ValueError(
                    f"batch {batch_index} field {key} has {len(rows)} rows, expected {batch_size}"
                )

        for row_index in range(batch_size):
            input_ids_full = [int(value) for value in raw_rows["input_ids"][row_index]]
            attention_full = [int(value) for value in raw_rows["attention_mask"][row_index]]
            raw_mask_full = [int(value) for value in raw_rows["loss_mask"][row_index]]
            raw_logprobs_full = [float(value) for value in raw_rows["logprobs"][row_index]]
            versions_full = [int(value) for value in raw_rows["versions"][row_index]]
            reward_values = [float(value) for value in raw_rows["rewards"][row_index]]
            reward = reward_values[0]

            widths = {
                len(input_ids_full),
                len(attention_full),
                len(raw_mask_full),
                len(raw_logprobs_full),
                len(versions_full),
            }
            if len(widths) != 1:
                _record_problem(
                    errors,
                    code="raw_width_mismatch",
                    message=f"raw tensor widths differ: {sorted(widths)}",
                    interaction_index=global_index,
                )
            sequence_length = sum(attention_full)
            if attention_full != [1] * sequence_length + [0] * (
                len(attention_full) - sequence_length
            ):
                _record_problem(
                    errors,
                    code="attention_mask_not_contiguous",
                    message="attention mask must be left-aligned ones followed by padding zeros",
                    interaction_index=global_index,
                )
            if sequence_length <= 0:
                _record_problem(
                    errors,
                    code="empty_sequence",
                    message="interaction has no active tokens",
                    interaction_index=global_index,
                )
                sequence_length = max(0, min(len(input_ids_full), sequence_length))

            input_ids = input_ids_full[:sequence_length]
            raw_mask = raw_mask_full[:sequence_length]
            raw_logprobs = raw_logprobs_full[:sequence_length]
            versions = versions_full[:sequence_length]
            if any(value not in (0, 1) for value in raw_mask_full):
                _record_problem(
                    errors,
                    code="non_binary_raw_loss_mask",
                    message="raw loss mask contains a value other than zero or one",
                    interaction_index=global_index,
                )
            try:
                completion_start = raw_mask.index(1)
            except ValueError:
                completion_start = sequence_length
                _record_problem(
                    errors,
                    code="empty_completion_mask",
                    message="raw interaction has no completion token",
                    interaction_index=global_index,
                )
            if completion_start == 0:
                _record_problem(
                    errors,
                    code="empty_prompt",
                    message=(
                        "completion begins at token zero, so the causal actor has no prompt "
                        "position from which to predict its first token"
                    ),
                    interaction_index=global_index,
                )
            expected_raw_mask = [0] * completion_start + [1] * (sequence_length - completion_start)
            if raw_mask != expected_raw_mask:
                _record_problem(
                    errors,
                    code="raw_loss_mask_not_completion_suffix",
                    message="raw loss mask is not prompt zeros followed by completion ones",
                    interaction_index=global_index,
                )
            if any(raw_mask_full[sequence_length:]):
                _record_problem(
                    errors,
                    code="padding_is_trainable",
                    message="raw loss mask enables padded tokens",
                    interaction_index=global_index,
                )

            prompt_ids = input_ids[:completion_start]
            completion_ids = input_ids[completion_start:]
            prompt_lengths.append(len(prompt_ids))
            completion_lengths.append(len(completion_ids))
            if len(input_ids) > context_length:
                _record_problem(
                    errors,
                    code="context_length_exceeded",
                    message=f"sequence has {len(input_ids)} tokens; limit is {context_length}",
                    interaction_index=global_index,
                )
            if len(completion_ids) > max_new_tokens:
                _record_problem(
                    errors,
                    code="completion_length_exceeded",
                    message=(
                        f"completion has {len(completion_ids)} tokens; limit is {max_new_tokens}"
                    ),
                    interaction_index=global_index,
                )
            if any(not _close(value, 0.0) for value in raw_logprobs[:completion_start]):
                _record_problem(
                    errors,
                    code="prompt_logprob_nonzero",
                    message="AReaL raw prompt logprobs must be zero placeholders",
                    interaction_index=global_index,
                )
            if any(not math.isfinite(value) for value in raw_logprobs[completion_start:]):
                _record_problem(
                    errors,
                    code="completion_logprob_nonfinite",
                    message="completion contains a non-finite rollout logprob",
                    interaction_index=global_index,
                )
            completion_logprobs.extend(raw_logprobs[completion_start:])
            observed_versions.extend(value for value in versions[completion_start:] if value >= 0)
            if any(value != -1 for value in versions[:completion_start]):
                _record_problem(
                    errors,
                    code="prompt_version_not_sentinel",
                    message="prompt token versions must use the -1 sentinel",
                    interaction_index=global_index,
                )
            if any(value < 0 for value in versions[completion_start:]):
                _record_problem(
                    errors,
                    code="completion_version_missing",
                    message="completion token is missing a rollout policy version",
                    interaction_index=global_index,
                )

            nearest_slot = min(
                range(group_size), key=lambda slot: abs(reward - expected_rewards[slot])
            )
            if not _close(reward, expected_rewards[nearest_slot]):
                _record_problem(
                    errors,
                    code="unexpected_audit_reward",
                    message=(
                        f"reward {reward} does not match deterministic normalized slots "
                        f"{expected_rewards}"
                    ),
                    interaction_index=global_index,
                )
            ordinal = slot_ordinals[nearest_slot]
            slot_ordinals[nearest_slot] += 1
            round_index = ordinal // len(characters)
            character_index = ordinal % len(characters)
            character = characters[character_index]
            character_counts[character] += 1
            if round_index >= max_rounds:
                _record_problem(
                    errors,
                    code="too_many_interactions_in_trajectory",
                    message=f"trajectory slot {nearest_slot} exceeds {max_rounds} rounds",
                    interaction_index=global_index,
                )

            trained_mask_full = [float(value) for value in trained_rows["loss_mask"][row_index]]
            trained_logprobs_full = [float(value) for value in trained_rows["logprobs"][row_index]]
            advantages_full = [float(value) for value in trained_rows["advantages"][row_index]]
            returns_full = [float(value) for value in trained_rows["returns"][row_index]]
            kl_full = [float(value) for value in trained_rows["kl_rewards"][row_index]]
            total_rewards_full = [float(value) for value in trained_rows["tot_rewards"][row_index]]
            expected_training_mask = [float(value) for value in _roll_left(raw_mask_full)]
            expected_training_logprobs = [
                float(value) * mask
                for value, mask in zip(
                    _roll_left(raw_logprobs_full), expected_training_mask, strict=True
                )
            ]
            expected_advantages = [reward * mask for mask in expected_training_mask]
            comparisons = (
                ("training_loss_mask_mismatch", trained_mask_full, expected_training_mask),
                (
                    "training_logprobs_mismatch",
                    trained_logprobs_full,
                    expected_training_logprobs,
                ),
                ("advantages_mismatch", advantages_full, expected_advantages),
                ("returns_mismatch", returns_full, expected_advantages),
                ("kl_reward_nonzero", kl_full, [0.0] * len(kl_full)),
                ("total_rewards_mismatch", total_rewards_full, expected_advantages),
            )
            for code, observed, expected in comparisons:
                if len(observed) != len(expected) or any(
                    not _close(left, right) for left, right in zip(observed, expected, strict=True)
                ):
                    _record_problem(
                        errors,
                        code=code,
                        message=f"{code} for actor-aligned tensors",
                        interaction_index=global_index,
                    )

            completion_tokens = _token_strings(tokenizer, completion_ids)
            if len(completion_tokens) != len(completion_ids):
                _record_problem(
                    warnings,
                    code="token_string_count_mismatch",
                    message="tokenizer did not return one token string per completion ID",
                    interaction_index=global_index,
                )
                completion_tokens = [_decode(tokenizer, [token_id]) for token_id in completion_ids]
            token_details: list[dict[str, Any]] = []
            for relative_index, (token_id, token_string, logprob) in enumerate(
                zip(
                    completion_ids,
                    completion_tokens,
                    raw_logprobs[completion_start:],
                    strict=True,
                )
            ):
                absolute_index = completion_start + relative_index
                training_position = absolute_index - 1
                token_details.append(
                    {
                        "relative_index": relative_index,
                        "absolute_index": absolute_index,
                        "training_position": training_position,
                        "token_id": token_id,
                        "token": token_string,
                        "decoded_piece": _decode(tokenizer, [token_id]),
                        "rollout_logprob": logprob,
                        "probability": math.exp(logprob),
                        "raw_loss_mask": raw_mask[absolute_index],
                        "training_loss_mask": trained_mask_full[training_position],
                        "training_logprob": trained_logprobs_full[training_position],
                        "advantage": advantages_full[training_position],
                    }
                )

            record: dict[str, Any] = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "interaction_index": global_index,
                "batch_index": batch_index,
                "batch_row_index": row_index,
                "trajectory_slot": nearest_slot,
                "turn_index": ordinal,
                "round_index": round_index,
                "character": character,
                "character_assignment": "inferred_from_round_robin_and_audit_reward",
                "sequence_length": sequence_length,
                "prompt_token_count": len(prompt_ids),
                "completion_token_count": len(completion_ids),
                "completion_start": completion_start,
                "completion_end": sequence_length,
                "input_ids": input_ids,
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": completion_ids,
                "raw_loss_mask": raw_mask,
                "raw_rollout_logprobs": raw_logprobs,
                "policy_versions": versions,
                "reward": reward,
                "training_loss_mask": trained_mask_full[:sequence_length],
                "training_logprobs": trained_logprobs_full[:sequence_length],
                "advantages": advantages_full[:sequence_length],
                "returns": returns_full[:sequence_length],
                "kl_rewards": kl_full[:sequence_length],
                "total_rewards": total_rewards_full[:sequence_length],
                "completion_text": _decode(tokenizer, completion_ids),
                "completion_tokens": token_details,
            }
            if include_prompt_text:
                record["prompt_text"] = _decode(tokenizer, prompt_ids)
            records.append(record)
            global_index += 1

    if global_index != expected_interactions:
        _record_problem(
            errors,
            code="interaction_count_mismatch",
            message=f"observed {global_index} interactions, expected {expected_interactions}",
        )
    for slot in range(group_size):
        if slot_ordinals[slot] != turns_per_trajectory:
            _record_problem(
                errors,
                code="trajectory_turn_count_mismatch",
                message=(
                    f"trajectory slot {slot} has {slot_ordinals[slot]} interactions, "
                    f"expected {turns_per_trajectory}"
                ),
            )
    expected_per_character = group_size * max_rounds
    for character in characters:
        if character_counts[character] != expected_per_character:
            _record_problem(
                errors,
                code="character_interaction_count_mismatch",
                message=(
                    f"{character!r} has {character_counts[character]} interactions, "
                    f"expected {expected_per_character}"
                ),
            )

    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "interactions": global_index,
        "expected_interactions": expected_interactions,
        "group_size": group_size,
        "max_rounds": max_rounds,
        "characters": list(characters),
        "character_interaction_counts": dict(character_counts),
        "trajectory_interaction_counts": dict(sorted(slot_ordinals.items())),
        "expected_normalized_rewards": list(expected_rewards),
        "prompt_tokens": _length_stats(prompt_lengths),
        "completion_tokens": _length_stats(completion_lengths),
        "completion_logprobs": _float_stats(completion_logprobs),
        "policy_versions": sorted(set(observed_versions)),
        "errors": errors,
        "warnings": warnings,
    }
    return records, summary


def _length_stats(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _float_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": statistics.fmean(values)}


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("outputs") / "rollout-audit" / timestamp


def _write_outputs(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    with (output_dir / "interactions.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RLFF rollout tensor audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Interactions: `{summary['interactions']}/{summary['expected_interactions']}`",
        f"- Group size: `{summary['group_size']}`",
        f"- Max rounds: `{summary['max_rounds']}`",
        f"- Characters: `{', '.join(summary['characters'])}`",
        f"- Policy versions: `{summary['policy_versions']}`",
        f"- Errors: `{len(summary['errors'])}`",
        f"- Warnings: `{len(summary['warnings'])}`",
        "",
        "## Errors",
        "",
    ]
    errors = summary["errors"]
    if errors:
        lines.extend(f"- `{item['code']}`: {item['message']}" for item in errors)
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = summary["warnings"]
    if warnings:
        lines.extend(f"- `{item['code']}`: {item['message']}" for item in warnings)
    else:
        lines.append("- None")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cloud_audit(
    config: RLFFConfig,
    *,
    episode_index: int,
    output_dir: Path,
    include_prompt_text: bool,
) -> dict[str, Any]:
    """Launch one real AReaL rollout group and audit its exported tensors."""

    from . import runtime
    from .proxy import RLFFGroupAwareAgent

    dataset = runtime.build_areal_training_dataset(config)
    if episode_index < 0 or episode_index >= len(dataset):
        raise IndexError(
            f"episode-index {episode_index} is outside dataset range 0..{len(dataset) - 1}"
        )
    selected_dataset = dataset.select([episode_index])
    row = dataset[episode_index]
    episode = decode_proxy_episode_payload(row)
    raw_characters = episode.get("characters")
    if not isinstance(raw_characters, Sequence) or isinstance(
        raw_characters, (str, bytes, bytearray)
    ):
        raise ValueError("selected episode has no ordered characters")
    characters = tuple(str(item["name"]) for item in raw_characters)

    group_size = config.episode_grouping.group_size
    if group_size < 2:
        raise ValueError("cloud rollout audit requires an RLFF group size of at least two")
    native_config = runtime.load_native_areal_config(config)
    _actor_class, trainer_class = runtime._build_areal_runtime_classes()
    workflow_kwargs = runtime.build_agent_workflow_kwargs(config)
    workflow_kwargs.pop("reward_provider", None)
    workflow_kwargs["reward_provider_name"] = DETERMINISTIC_AUDIT_REWARD_PROVIDER
    workflow_kwargs["group_size"] = group_size
    try:
        json.dumps(workflow_kwargs)
    except TypeError as exc:
        raise TypeError("rollout audit workflow arguments must be JSON serializable") from exc

    started = datetime.now(UTC)
    with trainer_class(
        native_config,
        train_dataset=selected_dataset,
        valid_dataset=None,
    ) as trainer:
        trainer._ensure_proxy_started()
        rollout_onloaded = False
        raw_batches: Sequence[Mapping[str, Any]] = []
        training_batches: Sequence[Mapping[str, Any]] = []
        try:
            if trainer._should_offload_rollout:
                trainer._onload_rollout()
                rollout_onloaded = True
            raw_batches = trainer.actor.rollout_batch(
                [row],
                workflow=RLFFGroupAwareAgent,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            if not raw_batches:
                raise RuntimeError(
                    "AReaL rollout_batch returned no interaction batches; inspect the "
                    "rollout/controller logs for the preceding workflow failure"
                )
            training_input = _clone_batches(raw_batches)
            training_batches = trainer.actor.compute_advantages(training_input)
            local_raw_batches, local_training_batches = _localize_areal_batches(
                raw_batches,
                training_batches,
            )
        finally:
            if raw_batches:
                targets = [raw_batches]
                if training_batches:
                    targets.append(training_batches)
                try:
                    trainer.actor.clear_batches(*targets)
                except Exception as exc:  # pragma: no cover - cloud cleanup path
                    warnings.warn(
                        f"failed to clear AReaL rollout audit batches: {exc}",
                        stacklevel=2,
                    )
            if rollout_onloaded:
                trainer._offload_rollout()

        records, summary = audit_rollout_batches(
            local_raw_batches,
            local_training_batches,
            tokenizer=trainer.tokenizer,
            characters=characters,
            group_size=group_size,
            max_rounds=config.rollout.max_rounds,
            context_length=config.sglang.context_length,
            max_new_tokens=config.sglang.max_new_tokens,
            include_prompt_text=include_prompt_text,
        )

    finished = datetime.now(UTC)
    summary.update(
        {
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - started).total_seconds(), 6),
            "episode_index": episode_index,
            "episode_id": str(episode.get("episode_id", "")),
            "episode_title": str(episode.get("title", "")),
            "config_fingerprint": config.config_fingerprint(),
            "output_dir": str(output_dir),
            "reward_provider": "deterministic_local_audit",
            "deepseek_called": False,
            "optimizer_update_performed": False,
            "include_prompt_text": include_prompt_text,
        }
    )
    _write_outputs(output_dir, records, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--include-prompt-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include fully decoded prompts in interactions.jsonl (default: true)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        config = load_config(args.config)
        started = time.perf_counter()
        summary = run_cloud_audit(
            config,
            episode_index=args.episode_index,
            output_dir=output_dir,
            include_prompt_text=args.include_prompt_text,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(
            f"rollout audit finished in {time.perf_counter() - started:.3f}s; "
            f"results: {output_dir}",
            file=sys.stderr,
        )
        return 0 if summary["status"] == "passed" else 1
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"rlff-rollout-audit: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"failure details: {output_dir / 'failure.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by cloud command
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DeterministicAuditRewardProvider",
    "audit_rollout_batches",
    "main",
    "run_cloud_audit",
]
