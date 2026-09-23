#!/usr/bin/env python3
"""Summarize why research runs end as completed_with_review_issues.

The analyzer supports both legacy experiment artifacts and new runs that expose
claim-verdict and Red-issue distributions in ``system_metrics``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


def _reason_root(reason: str) -> str:
    return reason.split(":", 1)[0] if reason else "unknown"


def _rows(payload: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for variant, body in payload.get("variants", {}).items():
        for row in body.get("samples", []):
            yield variant, row


def _classify(row: dict[str, Any], citation_threshold: float, claim_threshold: float) -> list[str]:
    metrics = row.get("system_metrics", {})
    explicit = metrics.get("completion_issue_reasons")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit]

    reasons: list[str] = []
    review_passed = metrics.get("review_passed")
    claim_passed = metrics.get("claim_ledger_passed")
    if review_passed is False:
        reasons.append("red_review_failed")
    coverage = metrics.get("citation_coverage")
    if isinstance(coverage, (int, float)) and coverage < citation_threshold:
        reasons.append("citation_coverage_below_threshold")
    support = metrics.get("claim_support_rate")
    if isinstance(support, (int, float)) and support < claim_threshold:
        reasons.append("claim_support_below_threshold")
    if claim_passed is False and "claim_support_below_threshold" not in reasons:
        reasons.append("claim_ledger_failed_other")
    if metrics.get("semantic_claim_verification_available") is False:
        reasons.append("semantic_verifier_unavailable")
    return reasons or ["unclassified_legacy_issue"]


def analyze(
    payload: dict[str, Any],
    *,
    citation_threshold: float = 0.80,
    claim_threshold: float = 0.80,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    overall_status: Counter[str] = Counter()
    overall_reasons: Counter[str] = Counter()
    overall_gate: Counter[str] = Counter()
    overall_stops: Counter[str] = Counter()
    overall_rejections: Counter[str] = Counter()
    overall_dimensions: Counter[str] = Counter()
    overall_verdicts: Counter[str] = Counter()

    grouped: dict[str, list[dict[str, Any]]] = {}
    modern_metrics_seen = False
    for variant, row in _rows(payload):
        grouped.setdefault(variant, []).append(row)
        if "claim_ledger_metrics" in row.get("system_metrics", {}):
            modern_metrics_seen = True

    for variant, rows in grouped.items():
        status = Counter(str(row.get("status", "unknown")) for row in rows)
        issue_rows = [row for row in rows if row.get("status") == "completed_with_review_issues"]
        reasons: Counter[str] = Counter()
        gates: Counter[str] = Counter()
        stops: Counter[str] = Counter()
        rejections: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        verdicts: Counter[str] = Counter()

        for row in issue_rows:
            metrics = row.get("system_metrics", {})
            reasons.update(_classify(row, citation_threshold, claim_threshold))
            red_failed = metrics.get("review_passed") is False
            claim_failed = metrics.get("claim_ledger_passed") is False
            gate = (
                "red_and_claim"
                if red_failed and claim_failed
                else "red_only"
                if red_failed
                else "claim_only"
                if claim_failed
                else "other"
            )
            gates[gate] += 1
            stops.update(metrics.get("budget", {}).get("stop_reasons", []))
            dimensions.update(metrics.get("review_issue_dimension_counts", {}))
            ledger_metrics = metrics.get("claim_ledger_metrics", {})
            for name in (
                "partially_supported_claim_count",
                "contradicted_claim_count",
                "uncited_claim_count",
                "insufficient_claim_count",
                "unknown_claim_count",
                "time_conflict_count",
                "number_conflict_count",
                "entity_conflict_count",
            ):
                verdicts[name] += int(ledger_metrics.get(name, 0))
            for event in row.get("trace", []):
                if event.get("event") != "blue_repair":
                    continue
                rejections.update(
                    _reason_root(str(item.get("reason", "")))
                    for item in event.get("rejected_patches", [])
                )

        variants[variant] = {
            "run_count": len(rows),
            "status_counts": dict(status),
            "issue_run_count": len(issue_rows),
            "issue_run_rate": len(issue_rows) / len(rows) if rows else 0.0,
            "gate_failure_combinations": dict(gates),
            "reason_run_counts": dict(reasons),
            "stop_reason_counts": dict(stops),
            "patch_rejection_counts": dict(rejections),
            "red_issue_dimension_counts": dict(dimensions),
            "claim_verdict_counts": dict(verdicts),
        }
        overall_status.update(status)
        overall_reasons.update(reasons)
        overall_gate.update(gates)
        overall_stops.update(stops)
        overall_rejections.update(rejections)
        overall_dimensions.update(dimensions)
        overall_verdicts.update(verdicts)

    total = sum(overall_status.values())
    issue_total = overall_status["completed_with_review_issues"]
    return {
        "source_run_id": payload.get("run_id"),
        "thresholds": {
            "citation_coverage": citation_threshold,
            "claim_support_rate": claim_threshold,
        },
        "overall": {
            "run_count": total,
            "issue_run_count": issue_total,
            "issue_run_rate": issue_total / total if total else 0.0,
            "status_counts": dict(overall_status),
            "gate_failure_combinations": dict(overall_gate),
            "reason_run_counts": dict(overall_reasons),
            "stop_reason_counts": dict(overall_stops),
            "patch_rejection_counts": dict(overall_rejections),
            "red_issue_dimension_counts": dict(overall_dimensions),
            "claim_verdict_counts": dict(overall_verdicts),
        },
        "variants": variants,
        "limitations": (
            [
                "本产物直接保存 completion_issue_reasons、claim_ledger_metrics 和 Red issue 分布，原因计数无需由代理标签重建。",
            ]
            if modern_metrics_seen
            else [
                "历史实验产物没有保存最终 Claim verdict 和 Red issue 维度，因此对应字段为空。",
                "历史原因计数依据最终 Gate 状态和实验阈值重建。",
                "升级后的运行会直接保存 completion_issue_reasons、claim_ledger_metrics 和 Red issue 分布。",
            ]
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    overall = result["overall"]
    lines = [
        "# completed_with_review_issues 原因分布",
        "",
        f"来源实验：`{result.get('source_run_id')}`",
        "",
        f"共 {overall['run_count']} 次运行，其中 {overall['issue_run_count']} 次为 "
        f"`completed_with_review_issues`（{overall['issue_run_rate']:.2%}）。",
        "",
        "## 总体原因（一次运行可命中多个原因）",
        "",
        "| 原因 | 运行数 | 占问题运行 |",
        "|---|---:|---:|",
    ]
    denominator = max(1, overall["issue_run_count"])
    for reason, count in sorted(
        overall["reason_run_counts"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {reason} | {count} | {count / denominator:.2%} |")

    lines.extend(
        [
            "",
            "## Gate 组合",
            "",
            "| Gate | 运行数 | 占问题运行 |",
            "|---|---:|---:|",
        ]
    )
    for reason, count in sorted(
        overall["gate_failure_combinations"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| {reason} | {count} | {count / denominator:.2%} |")

    lines.extend(["", "## 分组", ""])
    for variant, body in result["variants"].items():
        lines.append(
            f"- `{variant}`：{body['issue_run_count']}/{body['run_count']} "
            f"（{body['issue_run_rate']:.2%}）"
        )
    lines.extend(["", "## 停止原因", ""])
    for reason, count in sorted(
        overall["stop_reason_counts"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- `{reason}`：{count}")
    lines.extend(["", "## Blue 补丁拒绝原因", ""])
    for reason, count in sorted(
        overall["patch_rejection_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"- `{reason}`：{count}")
    lines.extend(["", "## 说明", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--citation-threshold", type=float, default=0.80)
    parser.add_argument("--claim-threshold", type=float, default=0.80)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = analyze(
        payload,
        citation_threshold=args.citation_threshold,
        claim_threshold=args.claim_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"completed-review-issues-{result.get('source_run_id') or args.artifact.stem}"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
