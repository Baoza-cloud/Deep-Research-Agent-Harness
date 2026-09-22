"""Strict completeness and quality gate for a formal Frozen ablation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_VARIANTS = ("fixed_harness", "dynamic_swarm")
ACCEPTED_STATUSES = {"completed", "completed_with_evidence_gaps"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _claim_metrics(row: dict[str, Any]) -> dict[str, Any]:
    system = row.get("system_metrics") or {}
    metrics = system.get("claim_ledger_metrics")
    if isinstance(metrics, dict):
        return metrics
    ledger = row.get("claim_ledger") or {}
    metrics = ledger.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _citation_coverage(row: dict[str, Any]) -> float:
    system = row.get("system_metrics") or {}
    if system.get("citation_coverage") is not None:
        return _number(system.get("citation_coverage"))
    return _number(
        ((row.get("evaluation") or {}).get("rules") or {}).get(
            "citation_coverage"
        )
    )


def _claim_support_rate(row: dict[str, Any]) -> float:
    system = row.get("system_metrics") or {}
    metrics = _claim_metrics(row)
    if metrics.get("claim_support_rate") is not None:
        return _number(metrics.get("claim_support_rate"))
    return _number(system.get("claim_support_rate"))


def _severe_review_issues(row: dict[str, Any], minimum: int) -> list[dict[str, Any]]:
    review = row.get("review") or {}
    issues = review.get("structured_issues") or []
    return [
        issue
        for issue in issues
        if isinstance(issue, dict) and int(issue.get("severity") or 0) >= minimum
    ]


def _trace_cost_overrides(row: dict[str, Any]) -> int:
    return sum(
        1
        for item in row.get("trace") or []
        if isinstance(item, dict)
        and item.get("event") == "dynamic_swarm_final_quality_guard"
        and item.get("cost_used_as_quality_override") is True
    )


def _issue(
    code: str,
    message: str,
    *,
    variant: str | None = None,
    pair_key: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "message": message}
    if variant is not None:
        value["variant"] = variant
    if pair_key is not None:
        value["pair_key"] = pair_key
    return value


def validate(
    payload: dict[str, Any],
    *,
    dataset_path: Path,
    expected_variants: Iterable[str] = DEFAULT_VARIANTS,
    expected_questions: int = 35,
    expected_repeats: int = 3,
    min_citation_coverage: float = 0.80,
    min_claim_support_rate: float = 0.80,
    severe_issue_threshold: int = 2,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    variants = tuple(expected_variants)
    expected_total = expected_questions * expected_repeats

    actual_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if payload.get("dataset_sha256") != actual_sha:
        errors.append(_issue("dataset_sha_mismatch", "结果与冻结数据集 SHA-256 不一致"))
    if int(payload.get("repeats") or 0) != expected_repeats:
        errors.append(_issue("repeat_count_mismatch", "顶层 repeats 与正式口径不一致"))

    selected_ids = [str(item) for item in payload.get("selected_sample_ids") or []]
    if len(selected_ids) != expected_questions or len(set(selected_ids)) != expected_questions:
        errors.append(
            _issue(
                "selected_question_count_mismatch",
                f"需要 {expected_questions} 个唯一问题，实际 {len(set(selected_ids))}",
            )
        )

    status_counts: Counter[str] = Counter()
    variant_summaries: dict[str, Any] = {}
    for variant in variants:
        variant_payload = (payload.get("variants") or {}).get(variant)
        if not isinstance(variant_payload, dict):
            errors.append(_issue("missing_variant", "缺少正式对照组", variant=variant))
            continue
        rows = variant_payload.get("samples") or []
        keys = [str(row.get("pair_key")) for row in rows]
        expected_keys = {
            f"{sample_id}#r{repeat}"
            for repeat in range(1, expected_repeats + 1)
            for sample_id in selected_ids
        }
        actual_keys = set(keys)
        if len(rows) != expected_total:
            errors.append(
                _issue(
                    "row_count_mismatch",
                    f"需要 {expected_total} 行，实际 {len(rows)}",
                    variant=variant,
                )
            )
        duplicate_count = len(keys) - len(actual_keys)
        if duplicate_count:
            errors.append(
                _issue(
                    "duplicate_pair_key",
                    f"存在 {duplicate_count} 个重复 pair_key",
                    variant=variant,
                )
            )
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            errors.append(
                _issue(
                    "pair_key_set_mismatch",
                    f"缺少 {len(missing)} 项，额外 {len(unexpected)} 项",
                    variant=variant,
                )
            )

        minimum_coverage = 1.0
        minimum_support = 1.0
        failed_rows = 0
        severe_count = 0
        conflict_count = 0
        blocking_count = 0
        cost_override_count = 0
        for row in rows:
            pair_key = str(row.get("pair_key"))
            if row.get("error"):
                failed_rows += 1
                errors.append(
                    _issue(
                        "execution_error",
                        str(row.get("error")),
                        variant=variant,
                        pair_key=pair_key,
                    )
                )
                continue

            status = str(row.get("status"))
            status_counts[status] += 1
            if status not in ACCEPTED_STATUSES:
                errors.append(
                    _issue(
                        "unaccepted_completion_status",
                        f"正式结果不接受状态 {status}",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )

            coverage = _citation_coverage(row)
            support = _claim_support_rate(row)
            minimum_coverage = min(minimum_coverage, coverage)
            minimum_support = min(minimum_support, support)
            if coverage < min_citation_coverage:
                errors.append(
                    _issue(
                        "citation_coverage_below_gate",
                        f"引用覆盖率 {coverage:.4f} < {min_citation_coverage:.4f}",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )
            if support < min_claim_support_rate:
                errors.append(
                    _issue(
                        "claim_support_below_gate",
                        f"Claim 支持率 {support:.4f} < {min_claim_support_rate:.4f}",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )

            system = row.get("system_metrics") or {}
            metrics = _claim_metrics(row)
            blocking = int(system.get("blocking_review_issue_count") or 0)
            blocking_count += blocking
            if blocking:
                errors.append(
                    _issue(
                        "blocking_review_issue",
                        f"仍有 {blocking} 个阻塞性 Red 问题",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )

            conflicts = sum(
                int(_number(metrics.get(name)))
                for name in (
                    "contradicted_claim_count",
                    "time_conflict_count",
                    "number_conflict_count",
                    "entity_conflict_count",
                )
            )
            conflict_count += conflicts
            if conflicts:
                errors.append(
                    _issue(
                        "claim_conflict",
                        f"Claim Ledger 仍有 {conflicts} 个冲突",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )

            severe = _severe_review_issues(row, severe_issue_threshold)
            severe_count += len(severe)
            if severe:
                errors.append(
                    _issue(
                        "severe_red_issue",
                        f"最终 Red 结果仍有 {len(severe)} 个 severity >= {severe_issue_threshold} 问题",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )

            overrides = _trace_cost_overrides(row)
            cost_override_count += overrides
            if overrides:
                errors.append(
                    _issue(
                        "cost_overrode_quality",
                        "Dynamic 最终护栏允许成本覆盖质量",
                        variant=variant,
                        pair_key=pair_key,
                    )
                )

            if variant == "dynamic_swarm" and system.get(
                "dynamic_quality_fallback_triggered"
            ):
                candidates = system.get("dynamic_quality_candidates") or {}
                chosen = (
                    "fixed_harness_fallback"
                    if system.get("dynamic_quality_fallback_selected")
                    else "dynamic"
                )
                chosen_metrics = candidates.get(chosen) or {}
                if candidates and (
                    _number(chosen_metrics.get("claim_support_rate"))
                    < min_claim_support_rate
                    or _number(chosen_metrics.get("citation_coverage"))
                    < min_citation_coverage
                    or int(chosen_metrics.get("blocking_issue_count") or 0) > 0
                ):
                    errors.append(
                        _issue(
                            "fallback_selected_below_gate",
                            f"质量回退后选中的 {chosen} 仍未通过质量门禁",
                            variant=variant,
                            pair_key=pair_key,
                        )
                    )

        aggregate = variant_payload.get("aggregate") or {}
        if int(aggregate.get("failed") or 0) != failed_rows:
            warnings.append(
                _issue(
                    "aggregate_failed_mismatch",
                    "aggregate.failed 与逐行统计不一致",
                    variant=variant,
                )
            )
        variant_summaries[variant] = {
            "rows": len(rows),
            "failed_rows": failed_rows,
            "minimum_citation_coverage": minimum_coverage,
            "minimum_claim_support_rate": minimum_support,
            "blocking_review_issues": blocking_count,
            "claim_conflicts": conflict_count,
            "severe_red_issues": severe_count,
            "cost_quality_overrides": cost_override_count,
        }

    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": payload.get("run_id"),
        "passed": not errors,
        "gate": {
            "expected_questions": expected_questions,
            "expected_repeats": expected_repeats,
            "expected_rows_per_variant": expected_total,
            "min_citation_coverage": min_citation_coverage,
            "min_claim_support_rate": min_claim_support_rate,
            "accepted_statuses": sorted(ACCEPTED_STATUSES),
            "severe_issue_threshold": severe_issue_threshold,
        },
        "variants": variant_summaries,
        "status_counts": dict(status_counts),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def _markdown(report: dict[str, Any], result_path: Path) -> str:
    lines = [
        "# Formal Frozen 35×3 完整性与质量门禁",
        "",
        f"- 结果：{'PASS' if report['passed'] else 'FAIL'}",
        f"- 实验产物：`{result_path}`",
        f"- Run ID：`{report.get('run_id')}`",
        f"- 错误：{report['error_count']}",
        f"- 警告：{report['warning_count']}",
        "",
        "| 变体 | 行数 | 执行失败 | 最低引用覆盖率 | 最低 Claim 支持率 | 阻塞 Red | Claim 冲突 | 严重 Red | 成本覆盖质量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["variants"].items():
        lines.append(
            f"| {name} | {item['rows']} | {item['failed_rows']} | "
            f"{item['minimum_citation_coverage']:.2%} | "
            f"{item['minimum_claim_support_rate']:.2%} | "
            f"{item['blocking_review_issues']} | {item['claim_conflicts']} | "
            f"{item['severe_red_issues']} | {item['cost_quality_overrides']} |"
        )
    lines.extend(["", "## 状态分布", ""])
    for status, count in sorted(report["status_counts"].items()):
        lines.append(f"- `{status}`：{count}")
    lines.extend(["", "## 门禁问题", ""])
    if not report["errors"]:
        lines.append("- 无")
    else:
        for item in report["errors"]:
            location = "/".join(
                value
                for value in (item.get("variant"), item.get("pair_key"))
                if value
            )
            prefix = f" ({location})" if location else ""
            lines.append(f"- `{item['code']}`{prefix}：{item['message']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/researchbench_frozen_v1.0.json"),
    )
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--questions", type=int, default=35)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-citation-coverage", type=float, default=0.80)
    parser.add_argument("--min-claim-support-rate", type=float, default=0.80)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    report = validate(
        payload,
        dataset_path=args.dataset,
        expected_questions=args.questions,
        expected_repeats=args.repeats,
        min_citation_coverage=args.min_citation_coverage,
        min_claim_support_rate=args.min_claim_support_rate,
    )
    prefix = args.output_prefix or args.result.with_name(
        f"{args.result.stem}-quality-gate"
    )
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(_markdown(report, args.result), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": report["error_count"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
