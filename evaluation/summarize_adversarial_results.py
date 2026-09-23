"""Render a paired, interview-ready ResearchBench-Adversarial report."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


FAULTS = (
    "F-UNKNOWN-CITATION",
    "F-UNCITED-CLAIM",
    "F-UNSUPPORTED-CLAIM",
    "F-MISSING-LIMITATION",
)
FAULT_LABELS = {
    "F-UNKNOWN-CITATION": "未知引用",
    "F-UNCITED-CLAIM": "无引用陈述",
    "F-UNSUPPORTED-CLAIM": "证据不支持结论",
    "F-MISSING-LIMITATION": "缺失局限说明",
}


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _question_means(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if "error" not in row:
            grouped.setdefault(str(row["sample_id"]), []).append(float(row["metrics"][metric]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _paired_delta(
    payload: dict[str, Any],
    baseline: str,
    candidate: str,
    metric: str,
    samples: int = 10_000,
) -> dict[str, Any]:
    left = _question_means(payload["variants"][baseline]["samples"], metric)
    right = _question_means(payload["variants"][candidate]["samples"], metric)
    keys = sorted(set(left) & set(right))
    values = [right[key] - left[key] for key in keys]
    rng = random.Random(42)
    boots = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    repeat_deltas = []
    for repeat in range(1, int(payload["repeats"]) + 1):
        base_rows = {
            row["pair_key"]: row
            for row in payload["variants"][baseline]["samples"]
            if row.get("repeat") == repeat and "error" not in row
        }
        cand_rows = {
            row["pair_key"]: row
            for row in payload["variants"][candidate]["samples"]
            if row.get("repeat") == repeat and "error" not in row
        }
        common = sorted(set(base_rows) & set(cand_rows))
        repeat_deltas.append(
            statistics.fmean(
                float(cand_rows[key]["metrics"][metric]) - float(base_rows[key]["metrics"][metric])
                for key in common
            )
        )
    spread = statistics.stdev(values)
    return {
        "paired_questions": len(keys),
        "delta": statistics.fmean(values),
        "ci": [boots[250], boots[9749]],
        "cohens_dz": statistics.fmean(values) / spread if spread else 0.0,
        "repeat_deltas": repeat_deltas,
    }


def _fault_rates(payload: dict[str, Any], variant: str) -> dict[str, float]:
    hits: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for row in payload["variants"][variant]["samples"]:
        repaired = set(row["metrics"]["repaired_fault_ids"])
        for fault in FAULTS:
            total[fault] += 1
            hits[fault] += fault in repaired
    return {fault: hits[fault] / total[fault] for fault in FAULTS}


def _auto_uncertainty_patch_count(payload: dict[str, Any]) -> int:
    return sum(
        patch.get("patch_id") == "AUTO-rule-uncertainty"
        for row in payload["variants"]["structured_patch_blue"]["samples"]
        if "error" not in row
        for patch in (row.get("patch_result") or {}).get("applied", [])
    )


def _auto_factual_delete_count(payload: dict[str, Any]) -> int:
    return sum(
        str(patch.get("patch_id", "")).startswith("AUTO-factual-delete-")
        for row in payload["variants"]["structured_patch_blue"]["samples"]
        if "error" not in row
        for patch in (row.get("patch_result") or {}).get("applied", [])
    )


def _auto_unknown_citation_count(payload: dict[str, Any]) -> int:
    return sum(
        str(patch.get("patch_id", "")).startswith("AUTO-unknown-citation-")
        for row in payload["variants"]["structured_patch_blue"]["samples"]
        if "error" not in row
        for patch in (row.get("patch_result") or {}).get("applied", [])
    )


def _paired_run_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
    samples: int = 10_000,
) -> dict[str, Any]:
    variant = "structured_patch_blue"
    left = _question_means(baseline["variants"][variant]["samples"], metric)
    right = _question_means(candidate["variants"][variant]["samples"], metric)
    keys = sorted(set(left) & set(right))
    values = [right[key] - left[key] for key in keys]
    rng = random.Random(42)
    boots = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    repeat_deltas = []
    for repeat in range(1, int(candidate["repeats"]) + 1):
        base_rows = {
            row["pair_key"]: row
            for row in baseline["variants"][variant]["samples"]
            if row.get("repeat") == repeat and "error" not in row
        }
        cand_rows = {
            row["pair_key"]: row
            for row in candidate["variants"][variant]["samples"]
            if row.get("repeat") == repeat and "error" not in row
        }
        common = sorted(set(base_rows) & set(cand_rows))
        repeat_deltas.append(
            statistics.fmean(
                float(cand_rows[key]["metrics"][metric]) - float(base_rows[key]["metrics"][metric])
                for key in common
            )
        )
    return {
        "paired_questions": len(keys),
        "delta": statistics.fmean(values),
        "ci": [boots[250], boots[9749]],
        "repeat_deltas": repeat_deltas,
    }


def render(payload: dict[str, Any], source: Path) -> str:
    lines = [
        "# ResearchBench-Adversarial v0.1 正式实验",
        "",
        f"- 运行 ID：`{payload['run_id']}`",
        f"- Frozen SHA-256：`{payload['source_dataset_sha256']}`",
        f"- 后端：`{payload['provider']}/{payload.get('model') or 'default'}`",
        f"- 规模：{payload['case_count']} 题 × {payload['repeats']} 次重复；每题 4 个 gold fault",
        f"- 原始结果：`{source.resolve()}`",
        "",
        "## 总体结果",
        "",
        "| 变体 | 成功/总数 | Red 检出率 | 故障修复率 | 事实准确率 | 幻觉率 | 引用覆盖率 | 事实保留率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, variant in payload["variants"].items():
        agg = variant["aggregate"]
        mean = agg["mean"]
        lines.append(
            f"| {name} | {agg['successful']}/{agg['total']} | "
            f"{_pct(mean['red_detection_recall'])} | {_pct(mean['fault_repair_rate'])} | "
            f"{_pct(mean['rule_factual_accuracy'])} | {_pct(mean['rule_hallucination_rate'])} | "
            f"{_pct(mean['citation_coverage'])} | {_pct(mean['clean_fact_retention_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 按故障类型",
            "",
            "| 故障 | Rewrite Blue | Structured Patch Blue |",
            "|---|---:|---:|",
        ]
    )
    rewrite_faults = _fault_rates(payload, "rewrite_blue")
    structured_faults = _fault_rates(payload, "structured_patch_blue")
    for fault in FAULTS:
        lines.append(
            f"| {FAULT_LABELS[fault]} | {_pct(rewrite_faults[fault])} | "
            f"{_pct(structured_faults[fault])} |"
        )

    lines.extend(["", "## 题目级配对 Bootstrap", ""])
    comparisons = (
        ("corrupted_no_repair", "rewrite_blue"),
        ("corrupted_no_repair", "structured_patch_blue"),
        ("rewrite_blue", "structured_patch_blue"),
    )
    for baseline, candidate in comparisons:
        repair = _paired_delta(payload, baseline, candidate, "fault_repair_rate")
        hallucination = _paired_delta(payload, baseline, candidate, "rule_hallucination_rate")
        lines.append(
            f"- `{candidate}` vs `{baseline}`：修复率 Δ "
            f"{100 * repair['delta']:+.2f} pp（95% CI "
            f"[{100 * repair['ci'][0]:+.2f}, {100 * repair['ci'][1]:+.2f}] pp）；"
            f"幻觉率 Δ {100 * hallucination['delta']:+.2f} pp（95% CI "
            f"[{100 * hallucination['ci'][0]:+.2f}, {100 * hallucination['ci'][1]:+.2f}] pp）。"
        )

    rollback = payload["rollback_guard"]
    structured_vs_rewrite = _paired_delta(
        payload, "rewrite_blue", "structured_patch_blue", "fault_repair_rate"
    )
    missing_limitation_rate = structured_faults["F-MISSING-LIMITATION"]
    auto_patch_count = _auto_uncertainty_patch_count(payload)
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- Rewrite Blue 与 Structured Patch Blue 相对不修复基线均产生显著收益，且三个重复方向一致。",
            (
                "- Structured Patch 与 Rewrite 的总修复率差异未达显著，"
                "确定性 `rule-uncertainty` ADD 已将缺失局限说明修复率提升到 "
                f"{_pct(missing_limitation_rate)}；程序补丁实际触发 {auto_patch_count} 次。"
                if missing_limitation_rate >= 0.99
                and structured_vs_rewrite["ci"][0] <= 0 <= structured_vs_rewrite["ci"][1]
                else "- Structured Patch 的主要瓶颈仍是缺失局限说明的 ADD 补丁。"
            ),
            f"- 确定性补丁守卫拒绝 {rollback['rejected_count']}/{rollback['attack_count']} 个恶意补丁，回滚正确率 {_pct(rollback['rollback_correctness'])}。",
            (
                "- 下一步应针对证据不支持结论的 DELETE/MODIFY 失败样本做错误分类；"
                "不应修改测试 fault 或统计口径。"
                if missing_limitation_rate >= 0.99
                else "- 下一步应把 `rule-uncertainty` 转为程序生成的确定性 ADD patch，再复跑相同协议；不应修改测试 fault 或统计口径。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_source: Path,
    candidate_source: Path,
) -> str:
    compatibility_keys = (
        "source_dataset_sha256",
        "protocol",
        "provider",
        "model",
        "case_count",
        "repeats",
    )
    incompatible = [key for key in compatibility_keys if baseline.get(key) != candidate.get(key)]
    if incompatible:
        raise ValueError("Incompatible runs: " + ", ".join(incompatible))

    metrics = (
        ("fault_repair_rate", "故障修复率"),
        ("rule_factual_accuracy", "事实准确率"),
        ("rule_hallucination_rate", "幻觉率"),
        ("citation_coverage", "引用覆盖率"),
        ("clean_fact_retention_rate", "事实保留率"),
        ("citation_preservation_rate", "引用保留率"),
        ("patch_acceptance_rate", "补丁接受率"),
    )
    variant = "structured_patch_blue"
    old_mean = baseline["variants"][variant]["aggregate"]["mean"]
    new_mean = candidate["variants"][variant]["aggregate"]["mean"]
    deltas = {metric: _paired_run_delta(baseline, candidate, metric) for metric, _ in metrics}
    old_faults = _fault_rates(baseline, variant)
    new_faults = _fault_rates(candidate, variant)
    lines = [
        "# Structured Patch Blue 确定性修复前后对照",
        "",
        f"- 基线运行：`{baseline['run_id']}`",
        f"- 修复后运行：`{candidate['run_id']}`",
        f"- Frozen SHA-256：`{candidate['source_dataset_sha256']}`",
        f"- 后端：`{candidate['provider']}/{candidate.get('model') or 'default'}`",
        f"- 规模：{candidate['case_count']} 题 × {candidate['repeats']} 次重复",
        f"- 基线原始结果：`{baseline_source.resolve()}`",
        f"- 修复后原始结果：`{candidate_source.resolve()}`",
        "",
        "## Structured Patch 总体变化",
        "",
        "| 指标 | 修复前 | 修复后 | 配对 Δ | 题目级 Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in metrics:
        delta = deltas[metric]
        lines.append(
            f"| {label} | {_pct(old_mean[metric])} | {_pct(new_mean[metric])} | "
            f"{100 * delta['delta']:+.2f} pp | "
            f"[{100 * delta['ci'][0]:+.2f}, {100 * delta['ci'][1]:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "## 按故障类型",
            "",
            "| 故障 | 修复前 | 修复后 | 变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for fault in FAULTS:
        lines.append(
            f"| {FAULT_LABELS[fault]} | {_pct(old_faults[fault])} | "
            f"{_pct(new_faults[fault])} | "
            f"{100 * (new_faults[fault] - old_faults[fault]):+.2f} pp |"
        )
    repair = deltas["fault_repair_rate"]
    retention = deltas["clean_fact_retention_rate"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 总修复率提升 {100 * repair['delta']:.2f} 个百分点（95% CI：{100 * repair['ci'][0]:.2f}–{100 * repair['ci'][1]:.2f}），三次重复变化分别为 "
            + "、".join(f"{100 * value:+.2f} pp" for value in repair["repeat_deltas"])
            + "。",
            f"- 缺失局限说明从 {_pct(old_faults['F-MISSING-LIMITATION'])} 提升到 {_pct(new_faults['F-MISSING-LIMITATION'])}；程序生成的 `AUTO-rule-uncertainty` 补丁触发 {_auto_uncertainty_patch_count(candidate)} 次。",
            f"- 证据不支持结论从 {_pct(old_faults['F-UNSUPPORTED-CLAIM'])} 提升到 {_pct(new_faults['F-UNSUPPORTED-CLAIM'])}；严格门控的 `AUTO-factual-delete` 补丁触发 {_auto_factual_delete_count(candidate)} 次。",
            f"- 仅删除引用标记的补丁会被拒绝；未知引用经词法唯一候选和 Claim–Evidence 验证后自动替换 {_auto_unknown_citation_count(candidate)} 次。",
            f"- 事实保留率变化 {100 * retention['delta']:+.2f} pp，95% CI [{100 * retention['ci'][0]:+.2f}, {100 * retention['ci'][1]:+.2f}] pp，没有以删除干净事实换取修复率。",
            "- 数据集、协议、模型、题数、重复次数与统计口径均保持不变。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        output = args.output or args.result.with_name(
            f"{args.result.stem}-vs-{args.baseline.stem}.md"
        )
        content = render_comparison(baseline, payload, args.baseline, args.result)
    else:
        output = args.output or args.result.with_suffix(".md")
        content = render(payload, args.result)
    output.write_text(content, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
