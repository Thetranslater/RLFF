"""Compare two reward_results.jsonl files and build a trajectory-oriented HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("group_id", "")),
        str(row.get("trajectory_id", "")),
        str(row.get("scope", "")),
        str(row.get("character", "")),
    )


def attempt_for(row: dict[str, Any]) -> dict[str, Any]:
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        return attempts[-1]
    return {}


def usage_for(row: dict[str, Any]) -> dict[str, int]:
    attempt = attempt_for(row)
    usage = attempt.get("usage") if isinstance(attempt.get("usage"), dict) else row.get("usage")
    return normalize_usage(usage)


def normalize_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
        "prompt_cache_hit_tokens": ("prompt_cache_hit_tokens",),
        "prompt_cache_miss_tokens": ("prompt_cache_miss_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = usage.get(candidate)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[target] = int(value)
                break
    return normalized


def all_usages_for(row: dict[str, Any]) -> list[dict[str, int]]:
    attempts = row.get("attempts")
    if isinstance(attempts, list):
        usages = [
            normalize_usage(attempt.get("usage"))
            for attempt in attempts
            if isinstance(attempt, dict) and isinstance(attempt.get("usage"), dict)
        ]
        if usages:
            return usages
    usage = normalize_usage(row.get("usage"))
    return [usage] if usage else []


def score_vector(row: dict[str, Any] | None) -> list[float]:
    if row is None:
        return []
    result = row.get("result")
    if isinstance(result, dict):
        return [
            float(value)
            for value in result.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return [float(result)]
    return []


def score_text(row: dict[str, Any] | None) -> str:
    scores = score_vector(row)
    if not scores:
        return "-"
    if row and row.get("scope") == "completion_local":
        return "[" + ", ".join(fmt(score) for score in scores) + "]"
    return fmt(scores[0])


def mean_score(row: dict[str, Any] | None) -> float | None:
    scores = score_vector(row)
    return statistics.mean(scores) if scores else None


def result_json(row: dict[str, Any] | None) -> str:
    if row is None:
        return "-"
    result = row.get("result")
    if result is None:
        return "-"
    return json.dumps(result, ensure_ascii=False, indent=2)


def content_for(row: dict[str, Any] | None, field: str) -> str:
    if row is None:
        return ""
    attempt = attempt_for(row)
    value = attempt.get(field)
    if value is None:
        value = row.get(field)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value) if value is not None else ""


def dialogue_for(rows: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for row in rows:
        payload = row.get("payload")
        if isinstance(payload, dict):
            value = payload.get("input")
            if isinstance(value, dict) and isinstance(value.get("utterances"), str):
                candidates.append(value["utterances"])
    return max(candidates, key=len, default="")


def dialogue_html(utterances: str) -> str:
    if not utterances.strip():
        return '<div class="muted">No dialogue payload found.</div>'
    result: list[str] = []
    for line in utterances.splitlines():
        if not line.strip():
            continue
        speaker, separator, content = line.partition(":")
        if not separator:
            speaker, content = "content", line
        result.append(
            f'<div class="turn"><span class="speaker">{esc(speaker)}</span>'
            f'<span class="turn-content">{esc(content)}</span></div>'
        )
    return "".join(result)


def provider_label(rows: list[dict[str, Any]], fallback: str) -> str:
    for row in rows:
        attempt = attempt_for(row)
        request = attempt.get("request")
        if isinstance(request, dict) and request.get("model"):
            return str(request["model"])
        if row.get("provider"):
            return str(row["provider"])
    return fallback


def status_html(row: dict[str, Any] | None) -> str:
    if row is None:
        return '<span class="status missing">missing</span>'
    status = str(row.get("status", "unknown"))
    cls = "ok" if status == "ok" else "error"
    elapsed = row.get("elapsed_seconds")
    attempts = row.get("attempts")
    count = len(attempts) if isinstance(attempts, list) else 0
    return (
        f'<span class="status {cls}">{esc(status)}</span>'
        f'<span class="meta"> {fmt(elapsed)}s / {count} attempt(s)</span>'
    )


def tokens_text(row: dict[str, Any] | None) -> str:
    if row is None:
        return "-"
    usages = all_usages_for(row)
    if not usages:
        return "-"
    usage = {key: sum(item.get(key, 0) for item in usages) for key in ("input_tokens", "output_tokens", "total_tokens")}
    total = usage.get("total_tokens")
    if total is None:
        return "-"
    return f"in {usage.get('input_tokens', 0)} / out {usage.get('output_tokens', 0)} / total {total}"


def operation_delta(qwen: dict[str, Any] | None, deepseek: dict[str, Any] | None) -> float | None:
    q = mean_score(qwen)
    d = mean_score(deepseek)
    return q - d if q is not None and d is not None else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def compare_scope(
    qmap: dict[tuple[str, str, str, str], dict[str, Any]],
    dmap: dict[tuple[str, str, str, str], dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    keys = sorted(set(qmap) | set(dmap))
    pairs: list[tuple[float, float]] = []
    value_pairs: list[tuple[float, float]] = []
    for key in keys:
        if key[2] != scope:
            continue
        qrow = qmap.get(key)
        drow = dmap.get(key)
        if not qrow or not drow or qrow.get("status") != "ok" or drow.get("status") != "ok":
            continue
        qscores = score_vector(qrow)
        dscores = score_vector(drow)
        if qscores and dscores:
            pairs.append((statistics.mean(qscores), statistics.mean(dscores)))
            value_pairs.extend(zip(qscores, dscores))
    deltas = [q - d for q, d in pairs]
    abs_deltas = [abs(delta) for delta in deltas]
    return {
        "matched": len(pairs),
        "mean_abs_delta": statistics.mean(abs_deltas) if abs_deltas else None,
        "mean_delta": statistics.mean(deltas) if deltas else None,
        "within_0_5": sum(delta <= 0.5 for delta in abs_deltas) / len(abs_deltas) if abs_deltas else None,
        "within_1": sum(delta <= 1.0 for delta in abs_deltas) / len(abs_deltas) if abs_deltas else None,
        "pearson": pearson([q for q, _ in pairs], [d for _, d in pairs]),
        "value_matched": len(value_pairs),
        "value_mean_abs_delta": statistics.mean([abs(q - d) for q, d in value_pairs]) if value_pairs else None,
    }


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}
    for row in rows:
        for usage in all_usages_for(row):
            for key, value in usage.items():
                totals[key] += value
    return totals


def aggregate_runtime(rows: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> dict[str, float | int | None]:
    values = [float(row["elapsed_seconds"]) for row in rows if isinstance(row.get("elapsed_seconds"), (int, float))]
    successful = [float(row["elapsed_seconds"]) for row in rows if row.get("status") == "ok" and isinstance(row.get("elapsed_seconds"), (int, float))]
    return {
        "rows": len(rows),
        "ok": sum(row.get("status") == "ok" for row in rows),
        "errors": sum(row.get("status") != "ok" for row in rows),
        "http_attempts": sum(len(row.get("attempts", [])) for row in rows if isinstance(row.get("attempts"), list)),
        "elapsed_total": sum(values) if values else None,
        "wall_elapsed": summary.get("elapsed_seconds") if isinstance(summary, dict) else None,
        "elapsed_mean_ok": statistics.mean(successful) if successful else None,
        "elapsed_median_ok": statistics.median(successful) if successful else None,
        "usage": aggregate_usage(rows),
    }


def summary_metric(label: str, value: Any, detail: str = "") -> str:
    return f'<div class="metric"><span>{esc(label)}</span><strong>{esc(fmt(value) if isinstance(value, (int, float)) else value)}</strong><small>{esc(detail)}</small></div>'


def detail_block(label: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return f'<details class="model-detail"><summary>{esc(label)} · missing</summary></details>'
    error = row.get("error")
    error_html = f'<div class="error">{esc(error)}</div>' if error else ""
    thinking = content_for(row, "thinking") or "<none returned>"
    content = content_for(row, "content") or "<none returned>"
    return f"""
    <details class="model-detail">
      <summary>{esc(label)} · {status_html(row)}</summary>
      {error_html}
      <div class="detail-label">thinking / reasoning</div>
      <pre>{esc(thinking)}</pre>
      <div class="detail-label">model content</div>
      <pre>{esc(content)}</pre>
      <div class="detail-label">parsed result</div>
      <pre>{esc(result_json(row))}</pre>
    </details>
    """


def comparison_row(
    key: tuple[str, str, str, str],
    qrow: dict[str, Any] | None,
    drow: dict[str, Any] | None,
    qlabel: str,
    dlabel: str,
) -> str:
    qmean = mean_score(qrow)
    dmean = mean_score(drow)
    delta = operation_delta(qrow, drow)
    details = detail_block(qlabel, qrow) + detail_block(dlabel, drow)
    return f"""
    <tr>
      <td class="character">{esc(key[3])}</td>
      <td>{esc(key[2])}</td>
      <td><div class="score qwen">{esc(score_text(qrow))}</div><small>mean {esc(fmt(qmean))}</small><div>{status_html(qrow)}</div></td>
      <td><div class="score deepseek">{esc(score_text(drow))}</div><small>mean {esc(fmt(dmean))}</small><div>{status_html(drow)}</div></td>
      <td class="delta {('positive' if delta is not None and delta > 0.01 else 'negative' if delta is not None and delta < -0.01 else 'neutral')}">{esc(fmt(delta))}</td>
      <td><div>{esc(tokens_text(qrow))}</div><div class="divider"></div><div>{esc(tokens_text(drow))}</div></td>
      <td>{details}</td>
    </tr>
    """


def build_report(
    qwen_rows: list[dict[str, Any]],
    deepseek_rows: list[dict[str, Any]],
    qwen_source: Path,
    deepseek_source: Path,
    qwen_summary: dict[str, Any] | None = None,
    deepseek_summary: dict[str, Any] | None = None,
) -> str:
    qmap = {row_key(row): row for row in qwen_rows}
    dmap = {row_key(row): row for row in deepseek_rows}
    all_keys = sorted(set(qmap) | set(dmap))
    qlabel = provider_label(qwen_rows, "Qwen")
    dlabel = provider_label(deepseek_rows, "DeepSeek")
    qruntime = aggregate_runtime(qwen_rows, qwen_summary)
    druntime = aggregate_runtime(deepseek_rows, deepseek_summary)
    qsummary = compare_scope(qmap, dmap, "completion_local")
    dsummary = compare_scope(qmap, dmap, "trajectory_role")
    disagreements = sorted(
        [
            (
                abs(operation_delta(qmap[key], dmap[key]) or 0),
                key,
                qmap[key],
                dmap[key],
                operation_delta(qmap[key], dmap[key]),
            )
            for key in set(qmap) & set(dmap)
            if qmap[key].get("status") == "ok"
            and dmap[key].get("status") == "ok"
            and mean_score(qmap[key]) is not None
            and mean_score(dmap[key]) is not None
        ],
        key=lambda item: item[0],
        reverse=True,
    )

    groups: OrderedDict[str, OrderedDict[str, list[tuple[tuple[str, str, str, str], dict[str, Any] | None, dict[str, Any] | None]]]] = OrderedDict()
    for key in all_keys:
        groups.setdefault(key[0], OrderedDict()).setdefault(key[1], []).append((key, qmap.get(key), dmap.get(key)))

    group_html: list[str] = []
    for group_id, trajectories in groups.items():
        first_key = next(iter(trajectories))
        trajectory_rows = [row for key in all_keys if key[0] == group_id and key[1] == first_key for row in (qmap.get(key), dmap.get(key)) if row]
        utterances = dialogue_for(trajectory_rows)
        episode_id = (qmap.get(first_key) or dmap.get(first_key) or {}).get("episode_id", "")
        trajectory_html: list[str] = []
        for trajectory_id, entries in trajectories.items():
            row_for_dialogue = [row for key, qrow, drow in entries for row in (qrow, drow) if row]
            dialogue = dialogue_for(row_for_dialogue)
            rows_html = "".join(comparison_row(key, qrow, drow, qlabel, dlabel) for key, qrow, drow in entries)
            slot = next((row.get("trajectory_slot") for _, qrow, drow in entries for row in (qrow, drow) if row), "-")
            trajectory_html.append(f"""
            <article class="trajectory-card" data-search="{esc(trajectory_id + ' ' + str(episode_id))}">
              <div class="trajectory-heading"><div><h3>Trajectory {esc(slot)}</h3><div class="mono">{esc(trajectory_id)}</div></div></div>
              <details class="dialogue" open><summary>Full dialogue ({len(dialogue.splitlines()) if dialogue else 0} turns)</summary><div class="dialogue-body">{dialogue_html(dialogue)}</div></details>
              <div class="table-wrap"><table><thead><tr><th>Character</th><th>Scope</th><th>{esc(qlabel)}</th><th>{esc(dlabel)}</th><th>Δ ({esc(qlabel)} - {esc(dlabel)})</th><th>Token usage<br><small>Qwen / DeepSeek</small></th><th>Thinking / content</th></tr></thead><tbody>{rows_html}</tbody></table></div>
            </article>
            """)
        group_html.append(f"""
        <section class="group-card" data-search="{esc(group_id + ' ' + str(episode_id))}">
          <div class="group-heading"><div><h2>GRPO Group</h2><div class="mono">{esc(group_id)}</div></div><div class="group-meta">{esc(episode_id)}<br>{len(trajectories)} trajectories</div></div>
          {''.join(trajectory_html)}
        </section>
        """)

    def score_table_row(label: str, result: dict[str, Any]) -> str:
        matched = result["matched"]
        pct05 = fmt(result["within_0_5"] * 100 if result["within_0_5"] is not None else None) + "%" if result["within_0_5"] is not None else "-"
        pct1 = fmt(result["within_1"] * 100 if result["within_1"] is not None else None) + "%" if result["within_1"] is not None else "-"
        return f"<tr><td>{esc(label)}</td><td>{matched}</td><td>{esc(fmt(result['mean_delta']))}</td><td>{esc(fmt(result['mean_abs_delta']))}</td><td>{esc(pct05)}</td><td>{esc(pct1)}</td><td>{esc(fmt(result['pearson']))}</td></tr>"

    disagreement_rows = "".join(
        f"<tr><td>{esc(key[2])}</td><td>{esc(key[3])}</td><td>{esc(key[1].split(':')[-1])}</td><td class=\"qwen\">{esc(score_text(qrow))}<br><small>mean {esc(fmt(mean_score(qrow)))}</small></td><td class=\"deepseek\">{esc(score_text(drow))}<br><small>mean {esc(fmt(mean_score(drow)))}</small></td><td class=\"delta {'positive' if delta and delta > 0.01 else 'negative' if delta and delta < -0.01 else 'neutral'}\">{esc(fmt(delta))}</td></tr>"
        for _, key, qrow, drow, delta in disagreements[:15]
    )

    qusage = qruntime["usage"]
    dusage = druntime["usage"]
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reward model comparison · {esc(qlabel)} vs {esc(dlabel)}</title>
<style>
:root {{ --bg:#f4f6fb; --card:#fff; --ink:#172033; --muted:#68738a; --line:#dce2ee; --blue:#3559d6; --green:#197a4b; --red:#b42318; --purple:#7245b5; --group:#e9efff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
.wrap {{ max-width:1900px; margin:0 auto; padding:28px; }} header {{ margin-bottom:20px; }} h1 {{ margin:0 0 5px; font-size:28px; }} h2 {{ margin:0 0 3px; font-size:19px; }} h3 {{ margin:0 0 2px; font-size:17px; }} h4 {{ margin:0 0 10px; }} .subtitle,.muted,small,.meta {{ color:var(--muted); }} .subtitle {{ word-break:break-all; }}
.summary {{ display:grid; grid-template-columns:repeat(8,minmax(110px,1fr)); gap:10px; margin:18px 0 20px; }} .metric {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }} .metric span,.metric small {{ display:block; }} .metric strong {{ display:block; font-size:21px; margin:2px 0; }}
.compare-panel {{ background:var(--card); border:1px solid var(--line); border-radius:13px; padding:17px; margin:0 0 22px; }} .compare-panel h2 {{ margin-bottom:12px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:10px 11px; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; word-break:break-word; }} th {{ background:#f7f9fd; color:#4d5870; font-size:12px; }} .cost-table th:first-child {{ width:22%; }} .num {{ font-variant-numeric:tabular-nums; }} .qwen {{ color:var(--blue); }} .deepseek {{ color:var(--purple); }} .delta {{ font-weight:700; font-variant-numeric:tabular-nums; }} .positive {{ color:var(--green); }} .negative {{ color:var(--red); }} .neutral {{ color:var(--muted); }}
.toolbar {{ position:sticky; top:0; z-index:3; background:rgba(244,246,251,.94); padding:8px 0 14px; backdrop-filter:blur(8px); }} input {{ width:100%; padding:11px 13px; border:1px solid #b9c4d8; border-radius:9px; font-size:14px; }}
.group-card {{ background:var(--card); border:1px solid #c8d4f2; border-radius:16px; margin:0 0 26px; overflow:hidden; box-shadow:0 3px 12px rgba(27,42,78,.06); }} .group-heading {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; padding:19px 22px; background:linear-gradient(120deg,var(--group),#fff); border-bottom:1px solid #c8d4f2; }} .group-meta {{ color:var(--blue); font-size:12px; text-align:right; word-break:break-all; }}
.trajectory-card {{ margin:18px; border:1px solid var(--line); border-radius:13px; overflow:hidden; }} .trajectory-heading {{ padding:15px 18px 12px; background:#fbfcff; border-bottom:1px solid var(--line); }} .mono {{ font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--muted); word-break:break-all; }} .table-wrap {{ overflow-x:auto; }} .table-wrap table {{ min-width:1250px; table-layout:fixed; }} .table-wrap th:nth-child(1) {{ width:12%; }} .table-wrap th:nth-child(2) {{ width:11%; }} .table-wrap th:nth-child(3),.table-wrap th:nth-child(4) {{ width:18%; }} .table-wrap th:nth-child(5) {{ width:9%; }} .table-wrap th:nth-child(6) {{ width:13%; }} .table-wrap th:nth-child(7) {{ width:19%; }} .character {{ font-weight:650; }} .score {{ font:12px ui-monospace,SFMono-Regular,Consolas,monospace; word-break:break-all; }} .status {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:650; margin-top:5px; }} .status.ok {{ color:var(--green); background:#e6f6ed; }} .status.error {{ color:var(--red); background:#fdecea; }} .status.missing {{ color:var(--muted); background:#eef1f6; }} .meta {{ font-size:11px; }} .divider {{ border-top:1px solid #eef1f7; margin:5px 0; }}
.dialogue {{ border-bottom:1px solid var(--line); }} .dialogue summary {{ cursor:pointer; padding:11px 18px; font-weight:650; background:#f9fbff; }} .dialogue-body {{ padding:10px 18px 14px; max-height:520px; overflow:auto; background:#fff; }} .turn {{ display:flex; gap:10px; padding:5px 0; border-bottom:1px solid #eef1f7; }} .speaker {{ flex:0 0 12em; color:var(--blue); font-weight:650; }} .turn-content {{ white-space:pre-wrap; }} .model-detail {{ border:1px solid var(--line); border-radius:8px; margin:4px 0; background:#fbfcff; }} .model-detail summary {{ cursor:pointer; padding:7px 9px; font-size:12px; }} .detail-label {{ margin:8px 10px 3px; color:var(--muted); font-size:11px; font-weight:650; }} pre {{ margin:4px 10px 10px; padding:9px; white-space:pre-wrap; overflow:auto; max-height:360px; background:#f1f4f9; border-radius:6px; font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }} .error {{ color:var(--red); font-size:12px; margin:8px 10px; }}
@media (max-width:1150px) {{ .summary {{ grid-template-columns:repeat(4,1fr); }} }} @media (max-width:650px) {{ .wrap {{ padding:16px; }} .summary {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:23px; }} .group-heading {{ display:block; }} .group-meta {{ text-align:left; margin-top:8px; }} .speaker {{ flex-basis:8em; }} }}
</style></head><body><main class="wrap">
<header><h1>Reward model comparison</h1><div class="subtitle">{esc(qlabel)} vs {esc(dlabel)} · generated {esc(generated)}</div><div class="subtitle">Qwen source: {esc(qwen_source.resolve())}</div><div class="subtitle">DeepSeek source: {esc(deepseek_source.resolve())}</div></header>
<section class="summary">
{summary_metric("Matched logical rows", len(set(qmap) & set(dmap)), f"Qwen {len(qwen_rows)} / DeepSeek {len(deepseek_rows)}")}
{summary_metric("Matched groups", len(set(key[0] for key in qmap) & set(key[0] for key in dmap)))}
{summary_metric("Matched trajectories", len(set(key[1] for key in qmap) & set(key[1] for key in dmap)))}
{summary_metric("Completion MAE", qsummary["mean_abs_delta"], "mean score difference")}
{summary_metric("Trajectory MAE", dsummary["mean_abs_delta"], "mean score difference")}
{summary_metric("Qwen total tokens", qusage["total_tokens"], f"{qruntime['http_attempts']} HTTP attempts")}
{summary_metric("DeepSeek total tokens", dusage["total_tokens"], f"{druntime['http_attempts']} HTTP attempts")}
    {summary_metric("Wall time seconds", qruntime["wall_elapsed"], f"Qwen / DeepSeek {fmt(druntime['wall_elapsed'])}s")}
</section>
<section class="compare-panel"><h2>Score agreement on shared successful operations</h2><div class="table-wrap"><table class="cost-table"><thead><tr><th>Scope</th><th>Matched operations</th><th>Mean Δ</th><th>MAE</th><th>Within ±0.5</th><th>Within ±1.0</th><th>Pearson r</th></tr></thead><tbody>{score_table_row("completion_local", qsummary)}{score_table_row("trajectory_role", dsummary)}</tbody></table></div><small>Δ = Qwen mean score − DeepSeek mean score. Agreement is computed only where both logical operations returned status=ok.</small></section>
<section class="compare-panel"><h2>Largest score disagreements</h2><div class="table-wrap"><table class="cost-table"><thead><tr><th>Scope</th><th>Character</th><th>Trajectory slot</th><th class="qwen">{esc(qlabel)}</th><th class="deepseek">{esc(dlabel)}</th><th>Δ</th></tr></thead><tbody>{disagreement_rows}</tbody></table></div><small>按每个逻辑操作的平均分差异绝对值排序；详细 thinking、content 和 parsed result 位于下方对应轨迹中。</small></section>
<section class="compare-panel"><h2>Request cost and runtime</h2><div class="table-wrap"><table class="cost-table"><thead><tr><th>Metric</th><th class="qwen">{esc(qlabel)}</th><th class="deepseek">{esc(dlabel)}</th><th>Difference / note</th></tr></thead><tbody>
<tr><td>Logical rows</td><td>{qruntime['rows']}</td><td>{druntime['rows']}</td><td>same expected input rows: {qruntime['rows'] == druntime['rows']}</td></tr>
<tr><td>Successful / error rows</td><td>{qruntime['ok']} / {qruntime['errors']}</td><td>{druntime['ok']} / {druntime['errors']}</td><td>errors include no-task operations</td></tr>
<tr><td>HTTP attempts</td><td>{qruntime['http_attempts']}</td><td>{druntime['http_attempts']}</td><td></td></tr>
<tr><td>Wall-clock runtime</td><td>{esc(fmt(qruntime['wall_elapsed']))} s</td><td>{esc(fmt(druntime['wall_elapsed']))} s</td><td>Qwen − DeepSeek: {esc(fmt((qruntime['wall_elapsed'] or 0) - (druntime['wall_elapsed'] or 0)))} s</td></tr>
<tr><td>Sum of per-operation elapsed</td><td>{esc(fmt(qruntime['elapsed_total']))} s</td><td>{esc(fmt(druntime['elapsed_total']))} s</td><td>logical-call latency sum; not wall-clock time</td></tr>
<tr><td>Mean successful call</td><td>{esc(fmt(qruntime['elapsed_mean_ok']))} s</td><td>{esc(fmt(druntime['elapsed_mean_ok']))} s</td><td></td></tr>
<tr><td>Input tokens</td><td>{qusage['input_tokens']}</td><td>{dusage['input_tokens']}</td><td>DeepSeek cache hit/miss: {dusage.get('prompt_cache_hit_tokens', 0)} / {dusage.get('prompt_cache_miss_tokens', 0)}</td></tr>
<tr><td>Output tokens</td><td>{qusage['output_tokens']}</td><td>{dusage['output_tokens']}</td><td></td></tr>
<tr><td>Total tokens</td><td>{qusage['total_tokens']}</td><td>{dusage['total_tokens']}</td><td>Qwen / DeepSeek ratio: {esc(fmt(qusage['total_tokens'] / dusage['total_tokens'], 3) if dusage['total_tokens'] else '-')}×</td></tr>
</tbody></table></div></section>
<div class="toolbar"><input id="search" aria-label="Search groups or trajectories" placeholder="Search group, trajectory, episode or character..."></div>
<section id="groups">{''.join(group_html)}</section>
</main><script>const input=document.getElementById('search');input.addEventListener('input',()=>{{const q=input.value.trim().toLowerCase();document.querySelectorAll('.group-card').forEach(card=>{{card.hidden=Boolean(q)&&!card.dataset.search.toLowerCase().includes(q);}});}});</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qwen", type=Path)
    parser.add_argument("deepseek", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qwen_summary_path = args.qwen.with_name("summary.json")
    deepseek_summary_path = args.deepseek.with_name("summary.json")
    qwen_summary = json.loads(qwen_summary_path.read_text(encoding="utf-8")) if qwen_summary_path.exists() else None
    deepseek_summary = json.loads(deepseek_summary_path.read_text(encoding="utf-8")) if deepseek_summary_path.exists() else None
    report = build_report(
        load_rows(args.qwen),
        load_rows(args.deepseek),
        args.qwen,
        args.deepseek,
        qwen_summary,
        deepseek_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
