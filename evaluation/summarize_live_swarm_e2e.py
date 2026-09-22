"""Summarize one end-to-end Live smoke result per Swarm complexity tier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


def _duration_seconds(payload: dict[str, Any]) -> float:
    started = datetime.fromisoformat(str(payload["started_at"]))
    finished = datetime.fromisoformat(str(payload["finished_at"]))
    return (finished - started).total_seconds()


def summarize_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    swarm = payload["run_metadata"]["swarm"]
    configured_roles = [str(agent["role"]) for agent in swarm["agents"]]
    invoked = Counter(
        str(event.get("role", "unknown"))
        for event in payload.get("trace", [])
        if event.get("event") == "agent_started"
    )
    query_examples: dict[str, str] = {}
    for event in payload.get("trace", []):
        if event.get("event") != "role_strategy_applied":
            continue
        role = str(event.get("role", "unknown"))
        query_examples.setdefault(role, str(event.get("query", "")))
    metrics = payload["metrics"]
    return {
        "source_file": str(path.resolve()),
        "question": payload["question"],
        "status": payload["status"],
        "level": swarm["level"],
        "complexity_score": swarm["complexity_score"],
        "policy_version": swarm.get("policy_version"),
        "configured_agent_count": swarm["agent_count"],
        "max_concurrency": swarm["max_concurrency"],
        "configured_roles": configured_roles,
        "invoked_roles": dict(sorted(invoked.items())),
        "configured_role_coverage": (
            len(set(configured_roles) & set(invoked)) / len(set(configured_roles))
            if configured_roles
            else 1.0
        ),
        "query_examples": query_examples,
        "worker_invocations": metrics["budget"]["used"]["worker_invocations"],
        "evidence_count": metrics["evidence_count"],
        "source_count": metrics["source_count"],
        "review_rounds": metrics["review_rounds"],
        "review_score": metrics["review_score"],
        "claim_support_rate": metrics["claim_support_rate"],
        "stop_reasons": metrics["budget"]["stop_reasons"],
        "latency_seconds": _duration_seconds(payload),
    }


def render(rows: Sequence[dict[str, Any]], json_path: Path) -> str:
    lines = [
        "# ResearchBench-Live v1.0 三档端到端 Swarm Smoke",
        "",
        "- 后端：DeepSeek + Tavily Web",
        "- 策略：`heuristic-v4`",
        f"- 结构化结果：`{json_path.resolve()}`",
        "",
        "| 档位 | 分数 | 配置/实际角色 | 并发 | Worker | 证据/来源 | Claim 支持率 | Review | 延迟 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['level']} | {row['complexity_score']:.2f} | "
            f"{row['configured_agent_count']}/{len(row['invoked_roles'])} | "
            f"{row['max_concurrency']} | {row['worker_invocations']} | "
            f"{row['evidence_count']}/{row['source_count']} | "
            f"{100.0 * row['claim_support_rate']:.2f}% | "
            f"{row['review_score']:.3f} | {row['latency_seconds']:.1f}s | "
            f"{row['status']} |"
        )
    lines.extend(["", "## 角色行为", ""])
    for row in rows:
        configured = ", ".join(row["configured_roles"])
        invoked = ", ".join(
            f"{role}×{count}" for role, count in row["invoked_roles"].items()
        )
        lines.extend(
            [
                f"### {row['level']}",
                "",
                f"- 配置：{configured}",
                f"- 实际：{invoked or '无'}",
                f"- 配置角色覆盖率：{100.0 * row['configured_role_coverage']:.2f}%",
                "",
            ]
        )
        for role, query in sorted(row["query_examples"].items()):
            lines.append(f"- `{role}`：{query}")
        lines.append("")
    lines.extend(
        [
            "## 结论",
            "",
            "- 三档配置分别扩展为 1/3/5 Agent，并发预算同步扩展为 1/3/5。",
            "- complex 的五类角色均被实际路由；standard 的 scope 角色未命中，说明配置角色数不等于实际执行角色数。",
            "- Claim 支持率随任务复杂度上升而下降；下一阶段应优先增加搜索结果主题相关性过滤和来源级去重，不能仅继续增加 Agent。",
            "",
            "> 这是每档 1 题的链路 smoke，只证明端到端机制可运行，不代表总体质量结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/live_swarm/e2e-v4-summary"),
    )
    args = parser.parse_args()
    rows = sorted(
        (summarize_result(path) for path in args.results),
        key=lambda row: {"simple": 0, "standard": 1, "complex": 2}.get(
            str(row["level"]), 99
        ),
    )
    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render(rows, json_path), encoding="utf-8")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
