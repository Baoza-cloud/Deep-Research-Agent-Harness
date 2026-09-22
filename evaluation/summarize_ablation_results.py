"""Create an interview-ready Markdown summary from a frozen ablation run."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Sequence


METRICS = {
    "overall_quality": "综合质量",
    "factual_accuracy": "事实准确率",
    "hallucination_rate": "幻觉率",
    "citation_coverage": "引用覆盖率",
    "key_fact_recall": "关键事实召回率",
    "worker_invocations": "Worker 调用数",
    "estimated_total_tokens": "估算 Token",
    "estimated_cost_usd": "估算成本",
}

METRIC_PATHS = {
    "overall_quality": ("evaluation", "overall_quality"),
    "factual_accuracy": ("evaluation", "rules", "factual_accuracy"),
    "hallucination_rate": ("evaluation", "rules", "hallucination_rate"),
    "citation_coverage": ("evaluation", "rules", "citation_coverage"),
    "key_fact_recall": ("evaluation", "benchmark", "key_fact_recall"),
    "latency_seconds": ("latency_seconds",),
    "llm_calls": ("llm_usage", "llm_calls"),
    "prompt_chars": ("llm_usage", "prompt_chars"),
    "completion_chars": ("llm_usage", "completion_chars"),
    "estimated_total_tokens": ("llm_usage", "estimated_total_tokens"),
    "estimated_cost_usd": ("llm_usage", "estimated_cost_usd"),
    "worker_invocations": (
        "system_metrics",
        "budget",
        "used",
        "worker_invocations",
    ),
    "review_rounds": ("system_metrics", "budget", "used", "review_rounds"),
}


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _signed_points(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _relative_change_with_ci(
    payload: dict[str, Any],
    baseline: str,
    candidate: str,
    metric: str,
    result: dict[str, Any],
) -> str:
    """Format a paired relative change and its question-level bootstrap CI."""
    lower, upper = _relative_bootstrap_ci(
        _paired_question_metric(payload, baseline, candidate, metric)
    )
    change = float(result["relative_change_percent"])
    if change <= 0:
        return f"降低 {-change:.2f}%（95% CI {-upper:.2f}%–{-lower:.2f}%）"
    return f"增加 {change:.2f}%（95% CI {lower:.2f}%–{upper:.2f}%）"


def _repeat_deltas(
    payload: dict[str, Any], baseline: str, candidate: str, metric: str
) -> list[float]:
    path = METRIC_PATHS[metric]

    def rows_for(variant: str) -> dict[tuple[int, str], float]:
        values: dict[tuple[int, str], float] = {}
        for row in payload["variants"][variant]["samples"]:
            if "error" in row:
                continue
            value: Any = row
            for key in path:
                value = value[key]
            values[(int(row["repeat"]), str(row["sample_id"]))] = float(value)
        return values

    base = rows_for(baseline)
    cand = rows_for(candidate)
    repeats = sorted({key[0] for key in base} & {key[0] for key in cand})
    deltas: list[float] = []
    for repeat in repeats:
        keys = sorted(
            key for key in set(base) & set(cand) if key[0] == repeat
        )
        if keys:
            deltas.append(statistics.fmean(cand[key] - base[key] for key in keys))
    return deltas


def _paired_question_metric(
    payload: dict[str, Any], baseline: str, candidate: str, metric: str
) -> list[tuple[float, float]]:
    path = METRIC_PATHS[metric]

    def grouped(variant: str) -> dict[str, list[float]]:
        values: dict[str, list[float]] = {}
        for row in payload["variants"][variant]["samples"]:
            if "error" not in row:
                value: Any = row
                for key in path:
                    value = value[key]
                values.setdefault(str(row["sample_id"]), []).append(
                    float(value)
                )
        return values

    base = grouped(baseline)
    cand = grouped(candidate)
    return [
        (statistics.fmean(base[key]), statistics.fmean(cand[key]))
        for key in sorted(set(base) & set(cand))
    ]


def _relative_bootstrap_ci(
    pairs: Sequence[tuple[float, float]], samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [rng.choice(pairs) for _ in pairs]
        baseline = statistics.fmean(item[0] for item in draw)
        candidate = statistics.fmean(item[1] for item in draw)
        estimates.append(100.0 * (candidate - baseline) / baseline)
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]


def _comparison_lines(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    table = [
        "| 对比 | 综合质量相对变化 | 绝对变化（95% CI） | Cohen's dz | 重复方向 |",
        "|---|---:|---:|---:|---:|",
    ]
    claims: list[str] = []
    for key, comparison in payload.get("comparisons", {}).items():
        if comparison.get("insufficient"):
            continue
        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        result = comparison["metrics"]["overall_quality"]
        delta = float(result["paired_delta_mean"])
        lower, upper = map(float, result["paired_delta_bootstrap_95_ci"])
        repeat_deltas = _repeat_deltas(
            payload, baseline, candidate, "overall_quality"
        )
        direction = sum(item > 0 for item in repeat_deltas)
        table.append(
            "| "
            f"{candidate} vs {baseline} | "
            f"{float(result['relative_change_percent']):+.2f}% | "
            f"{_signed_points(delta)} [{_signed_points(lower)}, {_signed_points(upper)}] | "
            f"{float(result['paired_cohens_dz']):+.2f} | {direction}/{len(repeat_deltas)} 正向 |"
        )
        complete = all(
            payload["variants"][name]["aggregate"]["failed"] == 0
            and payload["variants"][name]["aggregate"]["independent_questions"] == 35
            for name in (baseline, candidate)
        )
        if complete and lower > 0 and repeat_deltas and direction == len(repeat_deltas):
            relative_lower, relative_upper = _relative_bootstrap_ci(
                _paired_question_metric(
                    payload, baseline, candidate, "overall_quality"
                )
            )
            claims.append(
                f"在 ResearchBench-Frozen v1.0（11 领域/35 题，3 次重复）上，"
                f"{candidate} 相较 {baseline} 的综合质量稳定提升 "
                f"{float(result['relative_change_percent']):.2f}%"
                f"（95% CI {relative_lower:.2f}%–{relative_upper:.2f}%，"
                f"paired Cohen's dz={float(result['paired_cohens_dz']):.2f}）。"
            )

    efficiency = payload.get("comparisons", {}).get(
        "structured_patch_blue_vs_rewrite_blue", {}
    )
    if efficiency.get("metrics"):
        latency = efficiency["metrics"]["latency_seconds"]
        latency_lower, latency_upper = latency["paired_delta_bootstrap_95_ci"]
        latency_repeats = _repeat_deltas(
            payload, "rewrite_blue", "structured_patch_blue", "latency_seconds"
        )
        complete = all(
            payload["variants"][name]["aggregate"]["failed"] == 0
            for name in ("rewrite_blue", "structured_patch_blue")
        )
        if complete and latency_upper < 0 and all(item < 0 for item in latency_repeats):
            rel_low, rel_high = _relative_bootstrap_ci(
                _paired_question_metric(
                    payload,
                    "rewrite_blue",
                    "structured_patch_blue",
                    "latency_seconds",
                )
            )
            completion_pairs = _paired_question_metric(
                payload,
                "rewrite_blue",
                "structured_patch_blue",
                "completion_chars",
            )
            completion_mean_base = statistics.fmean(item[0] for item in completion_pairs)
            completion_mean_candidate = statistics.fmean(item[1] for item in completion_pairs)
            completion_change = 100.0 * (
                completion_mean_candidate - completion_mean_base
            ) / completion_mean_base
            completion_low, completion_high = _relative_bootstrap_ci(completion_pairs)
            quality = efficiency["metrics"]["overall_quality"]
            calls = efficiency["metrics"]["llm_calls"]
            claims.append(
                "相较整篇重写 Blue，Structured Patch Blue 的综合质量差异未达显著"
                f"（Δ={_signed_points(float(quality['paired_delta_mean']))}，"
                f"95% CI [{_signed_points(float(quality['paired_delta_bootstrap_95_ci'][0]))}, "
                f"{_signed_points(float(quality['paired_delta_bootstrap_95_ci'][1]))}]）；"
                f"端到端延迟降低 {-float(latency['relative_change_percent']):.2f}%"
                f"（相对变化 95% CI {-rel_high:.2f}%–{-rel_low:.2f}%，3/3 重复同向），"
                f"输出字符量降低 {-completion_change:.2f}%"
                f"（95% CI {-completion_high:.2f}%–{-completion_low:.2f}%），"
                f"代价是模型调用数增加 {float(calls['relative_change_percent']):.2f}%。"
            )
    swarm = payload.get("comparisons", {}).get(
        "dynamic_swarm_vs_fixed_harness", {}
    )
    if swarm.get("metrics") and payload.get("provider") != "offline":
        baseline = str(swarm["baseline"])
        candidate = str(swarm["candidate"])
        metrics = swarm["metrics"]
        quality = metrics["overall_quality"]
        workers = metrics["worker_invocations"]
        calls = metrics["llm_calls"]
        tokens = metrics["estimated_total_tokens"]
        latency = metrics["latency_seconds"]
        cost = metrics["estimated_cost_usd"]
        quality_low, quality_high = quality["paired_delta_bootstrap_95_ci"]
        if quality_low <= 0 <= quality_high:
            quality_claim = (
                "综合质量差异未达显著"
                f"（Δ={_signed_points(float(quality['paired_delta_mean']))}，"
                f"95% CI [{_signed_points(float(quality_low))}, "
                f"{_signed_points(float(quality_high))}]，"
                f"paired Cohen's dz={float(quality['paired_cohens_dz']):.2f}）"
            )
        else:
            quality_claim = (
                f"综合质量变化 {_signed_points(float(quality['paired_delta_mean']))}"
                f"（95% CI [{_signed_points(float(quality_low))}, "
                f"{_signed_points(float(quality_high))}]）"
            )
        claims.append(
            "Dynamic Swarm 相较 Fixed Harness："
            f"{quality_claim}；"
            f"Worker 调用{_relative_change_with_ci(payload, baseline, candidate, 'worker_invocations', workers)}"
            f"（{float(workers['baseline_mean']):.2f}→{float(workers['candidate_mean']):.2f}），"
            f"LLM 调用{_relative_change_with_ci(payload, baseline, candidate, 'llm_calls', calls)}，"
            f"估算 Token {_relative_change_with_ci(payload, baseline, candidate, 'estimated_total_tokens', tokens)}，"
            f"平均延迟{_relative_change_with_ci(payload, baseline, candidate, 'latency_seconds', latency)}，"
            f"估算成本{_relative_change_with_ci(payload, baseline, candidate, 'estimated_cost_usd', cost)}。"
        )
    return table, claims


def _swarm_behavior_lines(payload: dict[str, Any]) -> list[str]:
    variant = payload.get("variants", {}).get("dynamic_swarm")
    if not variant:
        return []
    aggregate = variant["aggregate"]
    roles = aggregate.get("role_invocations", {})
    stops = aggregate.get("stop_reasons", {})
    successful = int(aggregate.get("successful", 0))
    total_roles = sum(int(value) for value in roles.values())
    complexity: dict[str, int] = {}
    agent_counts: dict[int, int] = {}
    for row in variant.get("samples", []):
        if "error" in row:
            continue
        metrics = row.get("system_metrics", {})
        level = str(metrics.get("swarm_complexity", "unknown"))
        complexity[level] = complexity.get(level, 0) + 1
        count = int(metrics.get("swarm_agent_count", 0))
        agent_counts[count] = agent_counts.get(count, 0) + 1

    lines = [
        "## Dynamic Swarm 行为证据",
        "",
        "| 角色 | 调用数 | 占比 |",
        "|---|---:|---:|",
    ]
    for role, count in sorted(roles.items(), key=lambda item: (-int(item[1]), item[0])):
        share = 100.0 * int(count) / total_roles if total_roles else 0.0
        lines.append(f"| {role} | {int(count)} | {share:.2f}% |")
    lines.extend(
        [
            "",
            "- 复杂度分布："
            + "，".join(f"{key}={value}" for key, value in sorted(complexity.items()))
            + f"（按 {sum(complexity.values())} 次成功运行计）。",
            "- Agent 数量分布："
            + "，".join(f"{key} agents={value}" for key, value in sorted(agent_counts.items()))
            + "。",
        ]
    )
    if successful:
        lines.append(
            "- 停止信号："
            + "，".join(
                f"{key}={int(value)}（{100.0 * int(value) / successful:.2f}%）"
                for key, value in sorted(stops.items())
            )
            + "；同一次运行可能记录多个停止信号。"
        )
    return lines


def render(payload: dict[str, Any], source: Path) -> str:
    swarm_policy = payload.get("swarm_policy") or {}
    lines = [
        "# ResearchBench-Frozen v1.0 多组消融实验",
        "",
        f"- 运行 ID：`{payload['run_id']}`",
        f"- 数据集 SHA-256：`{payload['dataset_sha256']}`",
        f"- 后端：`{payload['provider']}/{payload.get('model') or 'default'}`",
        (
            f"- SwarmPolicy：`{swarm_policy.get('version', 'unversioned')}`；"
            f"阈值：simple < {swarm_policy.get('simple_threshold', 'N/A')}，"
            f"complex >= {swarm_policy.get('complex_threshold', 'N/A')}"
        ),
        f"- 重复次数：{payload['repeats']}；Bootstrap：按题目聚类配对",
        f"- 原始结果：`{source.as_posix()}`",
    ]
    resumed_from = payload.get("resumed_from") or []
    resume_stats = payload.get("resume_stats") or {}
    if resumed_from:
        details = "；".join(
            f"{name}: reused={stats.get('reused_successful_rows', 0)}, "
            f"new={stats.get('new_rows_requested', 0)}"
            for name, stats in resume_stats.items()
        )
        lines.extend(
            [
                "",
                "> 运行溯源：本报告通过结果恢复机制补跑先前失败或缺失的单元；"
                f"{details}。未复用其他策略版本或兼容回放结果。",
            ]
        )
    lines.extend(
        [
            "",
            "## 各组结果",
            "",
            "| 变体 | 成功/总数 | 综合质量（95% CI） | 事实准确率 | 引用覆盖率 | Worker 调用 | LLM 调用 | 估算 Token | 估算成本 | 平均延迟 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, variant in payload["variants"].items():
        agg = variant["aggregate"]
        metrics = agg["metrics"]
        if "overall_quality" not in metrics:
            lines.append(
                f"| {name} | {agg['successful']}/{agg['total']} | N/A | N/A | "
                "N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        quality = metrics["overall_quality"]
        q_low, q_high = quality["bootstrap_95_ci"]
        lines.append(
            f"| {name} | {agg['successful']}/{agg['total']} | "
            f"{_percent(quality['mean'])} [{_percent(q_low)}, {_percent(q_high)}] | "
            f"{_percent(metrics['factual_accuracy']['mean'])} | "
            f"{_percent(metrics['citation_coverage']['mean'])} | "
            f"{metrics['worker_invocations']['mean']:.2f} | "
            f"{metrics['llm_calls']['mean']:.2f} | "
            f"{metrics['estimated_total_tokens']['mean']:.0f} | "
            f"${metrics['estimated_cost_usd']['mean']:.6f} | "
            f"{metrics['latency_seconds']['mean']:.2f}s |"
        )

    lines.extend(
        [
            "",
            "## 运行状态",
            "",
            "| 变体 | completed | evidence_gaps | review_issues | partial_timeout | 运行异常 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, variant in payload["variants"].items():
        status_counts: dict[str, int] = {}
        runtime_errors = 0
        for row in variant.get("samples", []):
            if "error" in row:
                runtime_errors += 1
                continue
            status = str(row.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        lines.append(
            f"| {name} | {status_counts.get('completed', 0)} | "
            f"{status_counts.get('completed_with_evidence_gaps', 0)} | "
            f"{status_counts.get('completed_with_review_issues', 0)} | "
            f"{status_counts.get('partial_timeout', 0)} | "
            f"{runtime_errors} |"
        )
    lines.extend(
        [
            "",
            "> `successful` 表示流水线没有抛出运行异常，不代表报告通过 Claim 支持率与"
            "审查门禁；质量状态按上表单独报告。",
        ]
    )

    dynamic = payload.get("variants", {}).get("dynamic_swarm", {})
    observed_versions: dict[str, int] = {}
    for row in dynamic.get("samples", []):
        version = str(row.get("swarm_plan", {}).get("policy_version") or "unversioned")
        observed_versions[version] = observed_versions.get(version, 0) + 1
    if len(observed_versions) > 1 or (
        observed_versions
        and next(iter(observed_versions)) != swarm_policy.get("version")
    ):
        composition = "，".join(
            f"{version}={count}" for version, count in sorted(observed_versions.items())
        )
        lines.extend(
            [
                "",
                f"> 兼容回放提示：Dynamic 样本来源为 {composition}。只有档位分类与"
                "当前策略一致且具体预算兼容的历史行才会被复用；本报告用于策略筛选，"
                "不等同于一次全新同批运行。",
            ]
        )

    table, claims = _comparison_lines(payload)
    lines.extend(["", "## 配对消融", "", *table])
    behavior = _swarm_behavior_lines(payload)
    if behavior:
        lines.extend(["", *behavior])
    independent_questions = max(
        (
            int(item.get("aggregate", {}).get("independent_questions", 0))
            for item in payload.get("variants", {}).values()
        ),
        default=0,
    )
    repeats = int(payload.get("repeats", 1))
    formal = independent_questions == 35 and repeats >= 3 and all(
        int(item.get("aggregate", {}).get("failed", 0)) == 0
        for item in payload.get("variants", {}).values()
    )
    lines.extend(
        [
            "",
            "## 可写入简历的结论" if formal else "## 策略筛选结论（不可直接写入简历）",
            "",
        ]
    )
    if not formal:
        lines.append(
            f"> 当前仅覆盖 {independent_questions} 道独立题、每题 {repeats} 次运行；"
            "用于回归门禁和策略筛选，不代表正式 35 题 × 3 次结果。"
        )
        lines.append("")
    if claims:
        lines.extend(f"- {claim}" for claim in claims)
    else:
        lines.append(
            "- 本次结果没有同时满足完整配对、3/3 方向一致且 95% CI 不跨 0；"
            "不建议声称存在稳定提升。"
        )
    lines.extend(
        [
            "",
            f"> 统计口径：同一道题的 {repeats} 次运行先取均值，再以 "
            f"{independent_questions} 道题为独立单位做配对 Bootstrap；",
            "> 多次重复仅用于检查随机稳定性，不扩充有效样本量。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    output = args.output or args.result.with_suffix(".md")
    output.write_text(render(payload, args.result), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
