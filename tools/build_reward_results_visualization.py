"""Build a GRPO-group/trajectory-oriented HTML report from reward_results.jsonl."""

from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def attempt_for(row: dict[str, Any]) -> dict[str, Any]:
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        return attempts[-1]
    return {}


def usage_for(row: dict[str, Any]) -> dict[str, int]:
    usage = attempt_for(row).get("usage")
    if not isinstance(usage, dict):
        usage = row.get("usage")
    if not isinstance(usage, dict):
        return {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = usage.get(candidate)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[target] = int(value)
                break
    return normalized


def result_summary(row: dict[str, Any]) -> tuple[str, str]:
    result = row.get("result")
    if row.get("scope") == "completion_local" and isinstance(result, dict):
        scores = [value for value in result.values() if isinstance(value, (int, float))]
        if scores:
            values = ", ".join(number(value) for value in scores)
            return f"[{values}]", f"mean {number(mean(scores))}"
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return number(result), ""
    return "—", ""


def status_badge(row: dict[str, Any]) -> str:
    status = str(row.get("status", "unknown"))
    cls = "ok" if status == "ok" else "error"
    return f'<span class="badge {cls}">{esc(status)}</span>'


def call_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return '<span class="muted">missing</span>'
    _, suffix = result_summary(row)
    elapsed = number(row.get("elapsed_seconds"))
    attempts = row.get("attempts")
    attempt_count = len(attempts) if isinstance(attempts, list) else 0
    detail = f'{status_badge(row)} <span class="elapsed">{elapsed}s · {attempt_count} attempt(s)</span>'
    if suffix:
        detail += f'<div class="subtle">{esc(suffix)}</div>'
    return detail


def tokens_cell(rows: list[dict[str, Any]]) -> str:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False
    for row in rows:
        usage = usage_for(row)
        found = found or bool(usage)
        for key in totals:
            totals[key] += usage.get(key, 0)
    if not found:
        return '<span class="muted">—</span>'
    return (
        f"prompt {totals['prompt_tokens']}<br>"
        f"completion {totals['completion_tokens']}<br>"
        f"total {totals['total_tokens']}"
    )


def utterances_for(rows: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        value = payload.get("input")
        if isinstance(value, dict) and isinstance(value.get("utterances"), str):
            candidates.append(value["utterances"])
    return max(candidates, key=len, default="")


def dialogue_html(rows: list[dict[str, Any]]) -> str:
    utterances = utterances_for(rows)
    if not utterances.strip():
        return '<div class="muted">没有找到 payload.input.utterances</div>'
    turns: list[str] = []
    for line in utterances.splitlines():
        if not line.strip():
            continue
        speaker, separator, content = line.partition(":")
        if not separator:
            speaker, content = "内容", line
        turns.append(
            f'<div class="turn"><span class="speaker">{esc(speaker)}</span>'
            f'<span class="turn-content">{esc(content)}</span></div>'
        )
    return "".join(turns)


def thinking_block(
    row: dict[str, Any] | None,
    label: str,
    provider_label: str = "Reward model",
) -> str:
    if row is None:
        return ""
    attempt = attempt_for(row)
    thinking = attempt.get("thinking") or row.get("thinking") or "<none returned>"
    content = attempt.get("content") or row.get("content") or "<none returned>"
    error = row.get("error")
    error_html = f'<div class="error-text">{esc(error)}</div>' if error else ""
    return f"""
    <details class="thinking-item">
      <summary>{esc(label)} · {status_badge(row)}</summary>
      {error_html}
      <div class="thinking-label">thinking / reasoning</div>
      <pre>{esc(thinking)}</pre>
      <div class="thinking-label">{esc(provider_label)} content</div>
      <pre>{esc(content)}</pre>
    </details>
    """


def trajectory_card(trajectory_id: str, rows: list[dict[str, Any]]) -> str:
    first = rows[0]
    by_character: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
    for row in rows:
        character = str(row.get("character", "<unknown>"))
        by_character.setdefault(character, {})[str(row.get("scope"))] = row

    table_rows: list[str] = []
    thinking: list[str] = []
    for character, scopes in by_character.items():
        completion = scopes.get("completion_local")
        trajectory = scopes.get("trajectory_role")
        completion_score, completion_suffix = result_summary(completion or {})
        trajectory_score, _ = result_summary(trajectory or {})
        errors = [
            str(row.get("error"))
            for row in (completion, trajectory)
            if row is not None and row.get("error")
        ]
        error_cell = '<span class="muted">—</span>' if not errors else esc("; ".join(errors))
        table_rows.append(
            f"""
            <tr>
              <td class="character">{esc(character)}</td>
              <td><div class="score-list">{esc(completion_score)}</div>
                  <div class="subtle">{esc(completion_suffix)}</div></td>
              <td class="score">{esc(trajectory_score)}</td>
              <td>{call_cell(completion)}</td>
              <td>{call_cell(trajectory)}</td>
              <td class="tokens">{tokens_cell([r for r in (completion, trajectory) if r])}</td>
              <td class="error-cell">{error_cell}</td>
            </tr>
            """
        )
        thinking.append(thinking_block(completion, f"{character} · completion reward"))
        thinking.append(thinking_block(trajectory, f"{character} · trajectory reward"))

    episode_id = first.get("episode_id", "")
    slot = first.get("trajectory_slot", "—")
    return f"""
    <article class="trajectory-card" data-search="{esc(trajectory_id + ' ' + str(episode_id))}">
      <div class="card-heading">
        <div><h3>Trajectory {esc(slot)}</h3><div class="mono">{esc(trajectory_id)}</div></div>
      </div>
      <details class="dialogue" open>
        <summary>完整对话（{len(utterances_for(rows).splitlines())} turns）</summary>
        <div class="dialogue-body">{dialogue_html(rows)}</div>
      </details>
      <table>
        <thead><tr>
          <th>角色</th><th>Completion reward<br><span class="th-note">逐轮分数 / 均值</span></th>
          <th>Trajectory reward</th><th>Completion 调用</th><th>Trajectory 调用</th>
          <th>Token usage</th><th>错误</th>
        </tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
      <section class="thinking-section"><h4>Thinking / reasoning 与 Reward model 原始内容</h4>{''.join(thinking)}</section>
    </article>
    """


def build_report(rows: list[dict[str, Any]], source: Path) -> str:
    groups: OrderedDict[str, OrderedDict[str, list[dict[str, Any]]]] = OrderedDict()
    for row in rows:
        group_id = str(row.get("group_id", "unknown"))
        trajectory_id = str(row.get("trajectory_id", "unknown"))
        groups.setdefault(group_id, OrderedDict()).setdefault(trajectory_id, []).append(row)

    ok_count = sum(row.get("status") == "ok" for row in rows)
    error_count = len(rows) - ok_count
    http_attempts = sum(
        len(row.get("attempts", [])) for row in rows if isinstance(row.get("attempts"), list)
    )
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = usage_for(row)
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)

    group_cards: list[str] = []
    for group_id, trajectories in groups.items():
        first = next(iter(next(iter(trajectories.values()))))
        episode_id = first.get("episode_id", "")
        characters = sorted({
            str(row.get("character", ""))
            for trajectory_rows in trajectories.values()
            for row in trajectory_rows
        })
        search_text = " ".join([group_id, str(episode_id), *characters])
        group_cards.append(
            f"""
            <section class="group-card" data-search="{esc(search_text)}">
              <div class="group-heading">
                <div><h2>GRPO Group</h2><div class="mono">{esc(group_id)}</div></div>
                <div class="group-meta"><div>{esc(episode_id)}</div><div>{len(trajectories)} 条轨迹 · {len(characters)} 个角色</div></div>
              </div>
              {''.join(trajectory_card(tid, trajectory_rows) for tid, trajectory_rows in trajectories.items())}
            </section>
            """
        )

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    trajectory_count = sum(len(value) for value in groups.values())
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reward Results · GRPO Groups</title>
<style>
:root {{ --bg:#f4f6fb; --card:#fff; --ink:#172033; --muted:#68738a; --line:#dce2ee; --blue:#3559d6; --green:#197a4b; --red:#b42318; --group:#e9efff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
.wrap {{ max-width:1800px; margin:0 auto; padding:28px; }} header {{ margin-bottom:22px; }} h1 {{ margin:0 0 5px; font-size:28px; }} h2 {{ margin:0 0 3px; font-size:19px; }} h3 {{ margin:0 0 2px; font-size:17px; }} h4 {{ font-size:16px; margin:0 0 12px; }}
.subtitle,.muted,.subtle,.th-note {{ color:var(--muted); }} .subtitle {{ word-break:break-all; }} .summary {{ display:grid; grid-template-columns:repeat(7,minmax(110px,1fr)); gap:10px; margin:18px 0 22px; }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:13px 15px; }} .metric .value {{ display:block; font-size:22px; font-weight:700; margin-top:2px; }}
.toolbar {{ position:sticky; top:0; z-index:3; background:rgba(244,246,251,.94); padding:8px 0 14px; backdrop-filter:blur(8px); }} input {{ width:100%; padding:11px 13px; border:1px solid #b9c4d8; border-radius:9px; font-size:14px; }}
.group-card {{ background:var(--card); border:1px solid #c8d4f2; border-radius:16px; margin:0 0 26px; overflow:hidden; box-shadow:0 3px 12px rgba(27,42,78,.06); }} .group-heading {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; padding:19px 22px; background:linear-gradient(120deg,var(--group),#fff); border-bottom:1px solid #c8d4f2; }} .group-meta {{ color:var(--blue); font-size:12px; text-align:right; word-break:break-all; }}
.trajectory-card {{ margin:18px; border:1px solid var(--line); border-radius:13px; overflow:hidden; box-shadow:0 2px 8px rgba(27,42,78,.04); }} .card-heading {{ padding:15px 18px 12px; background:#fbfcff; border-bottom:1px solid var(--line); }} .mono {{ font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--muted); word-break:break-all; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; }} th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; word-break:break-word; }} th {{ background:#f7f9fd; font-size:12px; color:#4d5870; }} th:nth-child(1) {{ width:12%; }} th:nth-child(2) {{ width:20%; }} th:nth-child(3) {{ width:10%; }} th:nth-child(4),th:nth-child(5) {{ width:13%; }} th:nth-child(6) {{ width:12%; }} th:nth-child(7) {{ width:20%; }}
.character {{ font-weight:650; }} .score {{ font-size:16px; font-weight:650; }} .score-list {{ font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }} .tokens {{ font-size:12px; color:#4d5870; }} .elapsed {{ color:var(--muted); font-size:11px; }} .subtle {{ font-size:11px; margin-top:3px; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:650; }} .badge.ok {{ color:var(--green); background:#e6f6ed; }} .badge.error {{ color:var(--red); background:#fdecea; }} .error-cell,.error-text {{ color:var(--red); font-size:12px; }}
.dialogue {{ border-bottom:1px solid var(--line); }} .dialogue summary {{ cursor:pointer; padding:11px 18px; font-weight:650; background:#f9fbff; }} .dialogue-body {{ padding:10px 18px 14px; max-height:520px; overflow:auto; background:#fff; }} .turn {{ display:flex; gap:10px; padding:5px 0; border-bottom:1px solid #eef1f7; }} .speaker {{ flex:0 0 12em; color:var(--blue); font-weight:650; }} .turn-content {{ white-space:pre-wrap; }}
.thinking-section {{ padding:18px; background:#fbfcff; }} .thinking-item {{ border:1px solid var(--line); border-radius:9px; margin:9px 0; background:#fff; }} .thinking-item summary {{ cursor:pointer; padding:10px 12px; font-weight:600; }} .thinking-item pre {{ margin:7px 12px 14px; padding:11px; white-space:pre-wrap; overflow:auto; max-height:480px; background:#f6f8fc; border-radius:7px; font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }} .thinking-label {{ margin:0 12px; color:var(--muted); font-size:12px; font-weight:650; }}
@media (max-width:1100px) {{ .summary {{ grid-template-columns:repeat(4,1fr); }} table {{ min-width:1100px; }} .trajectory-card {{ overflow-x:auto; }} .dialogue-body,.thinking-section {{ min-width:1100px; }} }} @media (max-width:650px) {{ .wrap {{ padding:16px; }} .summary {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:23px; }} .group-heading {{ display:block; }} .group-meta {{ text-align:left; margin-top:8px; }} }}
</style></head><body><main class="wrap">
<header><h1>Reward Results · GRPO Groups</h1><div class="subtitle">源文件：{esc(source.resolve())}</div><div class="subtitle">生成时间：{esc(generated)} · 外层按 GRPO group，内层按 trajectory</div></header>
<section class="summary"><div class="metric">GRPO 组数<span class="value">{len(groups)}</span></div><div class="metric">轨迹数<span class="value">{trajectory_count}</span></div><div class="metric">逻辑调用<span class="value">{len(rows)}</span></div><div class="metric">成功调用<span class="value">{ok_count}</span></div><div class="metric">错误调用<span class="value">{error_count}</span></div><div class="metric">HTTP attempts<span class="value">{http_attempts}</span></div><div class="metric">总 tokens<span class="value">{total_usage['total_tokens']}</span></div></section>
<div class="toolbar"><input id="search" placeholder="搜索 group、trajectory、episode 或角色名称…"></div><section id="groups">{''.join(group_cards)}</section></main>
<script>const input=document.getElementById('search');input.addEventListener('input',()=>{{const q=input.value.trim().toLowerCase();document.querySelectorAll('.group-card').forEach(card=>{{card.hidden=Boolean(q)&&!card.dataset.search.toLowerCase().includes(q);}});}});</script>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name("reward_results_visualization.html")
    output.write_text(build_report(load_rows(args.input), args.input), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
