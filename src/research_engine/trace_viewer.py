"""Dependency-free JSON/HTML viewer for structured research traces."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _result_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, Mapping) else payload


def _timestamp(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _cost(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_cost(value: Any) -> str:
    resolved = _cost(value)
    return f"{resolved:.6f}" if resolved is not None else "unknown"


def _events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = _result_payload(payload)
    rows = payload.get("trace") or result.get("trace") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _plan(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _result_payload(payload)
    value = payload.get("plan") or result.get("plan") or {}
    return value if isinstance(value, Mapping) else {}


def build_view_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a result or replay bundle into a stable presentation model."""

    result = _result_payload(payload)
    events = _events(payload)
    metadata = result.get("run_metadata") or payload.get("metadata") or {}
    trace_id = metadata.get("trace_id") or next(
        (event.get("trace_id") for event in events if event.get("trace_id")),
        "unavailable",
    )
    starts: dict[str, dict[str, Any]] = {}
    spans: list[dict[str, Any]] = []
    for event in events:
        name = str(event.get("event", ""))
        span_id = event.get("span_id")
        if not span_id:
            continue
        if name in {"agent_started", "component_started"}:
            starts[str(span_id)] = event
        elif name in {
            "agent_completed",
            "agent_failed",
            "component_completed",
            "component_failed",
        }:
            start = starts.get(str(span_id), {})
            spans.append(
                {
                    "span_id": span_id,
                    "agent_id": event.get("agent_id")
                    or event.get("component_id")
                    or start.get("agent_id")
                    or start.get("component_id"),
                    "role": event.get("role") or start.get("role"),
                    "operation": event.get("operation") or start.get("operation"),
                    "status": "failed" if name.endswith("failed") else "completed",
                    "started_at": start.get("at"),
                    "finished_at": event.get("at"),
                    "duration_ms": float(event.get("duration_ms", 0.0) or 0.0),
                    "input_summary": event.get("input_summary") or start.get("input_summary"),
                    "output_summary": event.get("output_summary"),
                    "token_usage": event.get("token_usage")
                    or (event.get("usage") if isinstance(event.get("usage"), Mapping) else {}),
                    "cost_usd": _cost(event.get("cost_usd")),
                    "cost_status": (event.get("usage") or {}).get("cost_status")
                    if isinstance(event.get("usage"), Mapping)
                    else None,
                    "retry_count": int(event.get("retry_count", 0) or 0),
                    "error_type": event.get("error_type"),
                }
            )

    task_events: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") == "task_transition" and event.get("task_id"):
            task_events.setdefault(str(event["task_id"]), []).append(event)
    dag_nodes = []
    task_spans = []
    for task in _plan(payload).get("subtasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("subtask_id", "unknown"))
        transitions = task_events.get(task_id, [])
        running = next((row for row in transitions if row.get("to") == "running"), None)
        terminal = next(
            (
                row
                for row in reversed(transitions)
                if row.get("to")
                in {
                    "succeeded",
                    "failed",
                    "timed_out",
                    "degraded",
                    "skipped",
                    "cancelled",
                }
            ),
            None,
        )
        if running is not None:
            start_time = _timestamp(running.get("at"))
            finish_time = _timestamp((terminal or running).get("at"))
            task_spans.append(
                {
                    "agent_id": task_id,
                    "role": task.get("kind", "task"),
                    "operation": (terminal or running).get("to", "running"),
                    "status": (terminal or running).get("to", "running"),
                    "started_at": running.get("at"),
                    "finished_at": (terminal or running).get("at"),
                    "duration_ms": max(
                        0.0,
                        ((finish_time or start_time or 0) - (start_time or 0)) * 1000,
                    ),
                }
            )
        dag_nodes.append(
            {
                "task_id": task_id,
                "question": task.get("question", ""),
                "kind": task.get("kind", ""),
                "dependencies": list(task.get("dependencies") or []),
                "status": (
                    transitions[-1].get("to") if transitions else task.get("status", "pending")
                ),
            }
        )

    decisions = [
        event
        for event in events
        if event.get("event")
        in {
            "swarm_role_decision",
            "swarm_quality_guardrail",
            "evidence_verifier_decision",
            "fixed_fallback_decision",
            "dynamic_swarm_final_quality_guard",
            "swarm_stop",
            "global_timeout_forced_synthesis",
        }
    ]
    metrics = result.get("metrics") or {}
    total_tokens = sum(int((span.get("token_usage") or {}).get("total", 0) or 0) for span in spans)
    span_costs = [float(span["cost_usd"]) for span in spans if span.get("cost_usd") is not None]
    usage_metrics = metrics.get("llm_usage") or {}
    metric_cost = _cost(usage_metrics.get("cost_usd"))
    total_cost = (
        metric_cost if metric_cost is not None else (sum(span_costs) if span_costs else None)
    )
    started = _timestamp(result.get("started_at"))
    finished = _timestamp(result.get("finished_at"))
    return {
        "trace_id": trace_id,
        "run_id": result.get("run_id") or payload.get("run_id"),
        "status": result.get("status") or payload.get("lifecycle_status"),
        "started_at": result.get("started_at") or payload.get("started_at"),
        "finished_at": result.get("finished_at") or payload.get("updated_at"),
        "duration_ms": max(0.0, ((finished or 0) - (started or finished or 0)) * 1000),
        "dag_nodes": dag_nodes,
        "task_spans": task_spans,
        "spans": sorted(spans, key=lambda row: str(row.get("started_at") or "")),
        "decisions": decisions,
        "events": events,
        "metrics": metrics,
        "summary": {
            "event_count": len(events),
            "span_count": len(spans),
            "token_count": int(
                (metrics.get("llm_usage") or {}).get("total_tokens", total_tokens) or total_tokens
            ),
            "cost_usd": total_cost,
            "cost_status": usage_metrics.get(
                "cost_status", "calculated" if total_cost is not None else "unknown"
            ),
            "retry_count": sum(int(span.get("retry_count", 0)) for span in spans),
        },
    }


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value if value is not None else "—"))


def _waterfall_rows(spans: Sequence[Mapping[str, Any]]) -> str:
    timestamps = [
        point
        for span in spans
        for point in (_timestamp(span.get("started_at")), _timestamp(span.get("finished_at")))
        if point is not None
    ]
    if not timestamps:
        return '<p class="muted">No completed Agent spans were recorded.</p>'
    origin, end = min(timestamps), max(timestamps)
    window = max(end - origin, 0.001)
    rows = []
    for span in spans:
        started = _timestamp(span.get("started_at")) or origin
        finished = _timestamp(span.get("finished_at")) or started
        left = max(0.0, (started - origin) / window * 100)
        width = max(0.7, (finished - started) / window * 100)
        label = f"{span.get('agent_id')} · {span.get('operation') or span.get('role')}"
        rows.append(
            '<div class="wf-row"><div class="wf-label">'
            + _cell(label)
            + '</div><div class="wf-track"><div class="wf-bar '
            + ("failed" if span.get("status") == "failed" else "")
            + f'" style="left:{left:.3f}%;width:{width:.3f}%" title="{_cell(span.get("duration_ms"))} ms"></div></div></div>'
        )
    return "".join(rows)


def render_trace_html(payload: Mapping[str, Any], *, title: str = "Deep Research Trace") -> str:
    """Render a portable HTML report; no server, CDN, or JavaScript is required."""

    view = build_view_model(payload)
    cards = [
        ("Trace ID", view["trace_id"]),
        ("Run status", view["status"]),
        ("DAG nodes", len(view["dag_nodes"])),
        ("Agent spans", view["summary"]["span_count"]),
        ("Tokens", view["summary"]["token_count"]),
        ("Cost (USD)", _format_cost(view["summary"]["cost_usd"])),
    ]
    dag = (
        "".join(
            '<article class="node"><strong>'
            + _cell(node["task_id"])
            + '</strong><span class="pill">'
            + _cell(node["status"])
            + "</span><p>"
            + _cell(node["question"])
            + "</p><small>depends on: "
            + _cell(node["dependencies"] or "none")
            + "</small></article>"
            for node in view["dag_nodes"]
        )
        or '<p class="muted">No DAG plan found in this payload.</p>'
    )
    telemetry = "".join(
        "<tr>"
        + "".join(
            f"<td>{_cell(value)}</td>"
            for value in (
                span.get("agent_id"),
                span.get("role"),
                span.get("operation"),
                span.get("status"),
                f"{span.get('duration_ms', 0):.1f}",
                (span.get("token_usage") or {}).get("total", 0),
                _format_cost(span.get("cost_usd")),
                span.get("retry_count", 0),
                span.get("input_summary"),
                span.get("output_summary"),
            )
        )
        + "</tr>"
        for span in view["spans"]
    )
    decisions = (
        "".join(
            "<details><summary><strong>"
            + _cell(event.get("event"))
            + "</strong> · "
            + _cell(event.get("reason") or event.get("action") or event.get("explanation"))
            + "</summary><pre>"
            + _cell(event)
            + "</pre></details>"
            for event in view["decisions"]
        )
        or '<p class="muted">No Swarm, verifier, fallback, or stop decisions found.</p>'
    )
    event_rows = "".join(
        "<tr>"
        f"<td>{_cell(event.get('sequence'))}</td><td>{_cell(event.get('at'))}</td>"
        f"<td>{_cell(event.get('category'))}</td><td>{_cell(event.get('phase'))}</td>"
        f"<td>{_cell(event.get('event'))}</td><td><code>{_cell(event.get('event_id'))}</code></td>"
        "</tr>"
        for event in view["events"]
    )
    card_html = "".join(
        f'<div class="card"><span>{_cell(label)}</span><strong>{_cell(value)}</strong></div>'
        for label, value in cards
    )
    raw_json = _cell(view)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_cell(title)}</title><style>
:root{{--bg:#08111f;--panel:#101d30;--line:#243752;--text:#e8f1ff;--muted:#90a4c2;--accent:#35d0ba;--blue:#6aa8ff;--bad:#ff6577}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#07101e,#0b1728);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:28px}}h1{{margin:0;font-size:28px}}h2{{font-size:18px;margin:0 0 14px}}.sub,.muted,small{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:22px 0}}.card,.panel,.node{{background:rgba(16,29,48,.94);border:1px solid var(--line);border-radius:12px}}
.card{{padding:14px}}.card span{{display:block;color:var(--muted);font-size:12px}}.card strong{{display:block;margin-top:5px;overflow-wrap:anywhere}}
.panel{{padding:18px;margin:14px 0;overflow:auto}}.dag{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}.node{{padding:12px}}.node p{{min-height:42px}}.pill{{float:right;background:#193c45;color:#76ead8;padding:2px 7px;border-radius:999px;font-size:11px}}
.wf-row{{display:grid;grid-template-columns:240px 1fr;gap:10px;margin:7px 0}}.wf-label{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.wf-track{{height:17px;background:#0a1424;border-radius:4px;position:relative}}.wf-bar{{position:absolute;height:100%;background:linear-gradient(90deg,var(--blue),var(--accent));border-radius:4px;min-width:3px}}.wf-bar.failed{{background:var(--bad)}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px;vertical-align:top;max-width:290px;overflow-wrap:anywhere}}th{{color:#a9bfe0;position:sticky;top:0;background:var(--panel)}}
details{{border-top:1px solid var(--line);padding:10px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#bfd1e9}}.compare td:first-child{{font-weight:700;color:var(--accent)}}
@media(max-width:700px){{main{{padding:14px}}.wf-row{{grid-template-columns:110px 1fr}}}}
</style></head><body><main>
<h1>{_cell(title)}</h1><p class="sub">Structured local observability for planning, parallel execution, verification, adversarial repair and recovery.</p>
<section class="cards">{card_html}</section>
<section class="panel"><h2>Harness 与普通 Agent 框架的差异</h2><table class="compare"><tr><th>能力</th><th>Agent Harness</th><th>普通单 Agent / 链式框架</th></tr>
<tr><td>执行模型</td><td>DAG 拓扑并发、预算与状态机</td><td>顺序工具调用或单循环</td></tr><tr><td>动态角色</td><td>按复杂度增减角色并记录理由</td><td>角色通常静态写死</td></tr>
<tr><td>质量控制</td><td>Claim Verifier、Red/Blue Patch、Fixed fallback</td><td>通常依赖一次生成或整篇重写</td></tr><tr><td>恢复与审计</td><td>run_id 幂等恢复、结构化 Trace、确定性 Replay</td><td>失败后常从头重跑，过程难复现</td></tr></table></section>
	<section class="panel"><h2>DAG</h2><div class="dag">{dag}</div></section>
	<section class="panel"><h2>DAG 时间线</h2>{_waterfall_rows(view["task_spans"])}</section>
	<section class="panel"><h2>并发瀑布图</h2>{_waterfall_rows(view["spans"])}</section>
<section class="panel"><h2>Agent 输入/输出与成本遥测</h2><table><thead><tr><th>Agent</th><th>角色</th><th>操作</th><th>状态</th><th>ms</th><th>Token</th><th>成本</th><th>重试</th><th>输入摘要</th><th>输出摘要</th></tr></thead><tbody>{telemetry}</tbody></table></section>
<section class="panel"><h2>Dynamic Swarm、Verifier、Fallback 与停止原因</h2>{decisions}</section>
<section class="panel"><h2>结构化事件</h2><table><thead><tr><th>#</th><th>时间</th><th>类别</th><th>阶段</th><th>事件</th><th>Event ID</th></tr></thead><tbody>{event_rows}</tbody></table></section>
<section class="panel"><details><summary><strong>Viewer JSON model</strong></summary><pre>{raw_json}</pre></details></section>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="deep-research-trace",
        description="Render a Deep Research result or replay JSON as a local HTML trace viewer",
    )
    parser.add_argument("input", help="Research result or replay JSON")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--title", default="Deep Research Trace")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_trace_html(payload, title=args.title), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
