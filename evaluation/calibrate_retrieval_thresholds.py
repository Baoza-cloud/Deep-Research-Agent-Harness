"""Calibrate role-specific retrieval thresholds on ResearchBench-Live.

The expensive Tavily calls are made once. Every threshold candidate is then
evaluated against the same cached rows, so comparisons do not mix Web drift
with threshold effects. Benchmark required domains/terms provide reproducible
weak relevance labels, while ResearchBench-Retrieval-Gold provides adjudicated
candidate-level retain/reject decisions. Gold metrics are reported separately
because the current Gold slice samples filtered proxy-positive candidates and
is not an unbiased estimate of population-level retrieval performance.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from research_engine.agents import ResearchWorker  # noqa: E402
from research_engine.env import load_env_files  # noqa: E402
from research_engine.web_backends import TavilySearchBackend  # noqa: E402


ROLE_ORDER = (
    "researcher",
    "scope_researcher",
    "evidence_researcher",
    "counter_researcher",
    "impact_analyst",
    "evidence_verifier",
)
PRIMARY_SOURCE_CLASSES = {
    "government",
    "regulation",
    "standard",
    "paper",
    "official_documentation",
}
MAX_PROXY_FALSE_REJECTION = {
    "researcher": 0.10,
    "scope_researcher": 0.15,
    "evidence_researcher": 0.10,
    "counter_researcher": 0.20,
    "impact_analyst": 0.15,
    "evidence_verifier": 0.08,
}


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").casefold().strip(".")


def _domain_matches(url: str, domain: str) -> bool:
    host = _host(url)
    expected = domain.casefold().strip(".")
    return host == expected or host.endswith(f".{expected}")


def _proxy_relevance(
    row: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Return a benchmark-derived weak relevance label and its reasons."""

    url = str(row.get("url") or row.get("source") or "")
    text = " ".join(str(row.get(key) or "") for key in ("title", "content", "text")).casefold()
    domain_hits = [
        str(domain)
        for domain in sample.get("required_domains", [])
        if _domain_matches(url, str(domain))
    ]
    term_hits = [
        str(term) for term in sample.get("required_terms", []) if str(term).casefold() in text
    ]
    reasons = [*(f"domain:{item}" for item in domain_hits)]
    reasons.extend(f"term:{item}" for item in term_hits)
    return bool(reasons), reasons


def _candidate_key(unit: Mapping[str, Any], rank: int) -> str:
    return f"{unit['sample']['id']}::{unit['role']}::{rank}"


def _normalized_content(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get("content") or row.get("text") or "").split())


def load_gold_records(
    path: Path,
    units: Sequence[Mapping[str, Any]],
    *,
    compatible_run_ids: Sequence[str] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and strictly bind adjudicated Gold labels to cached candidates."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("track") != "retrieval_gold":
        raise ValueError("Gold dataset must use track='retrieval_gold'")
    if payload.get("status") != "adjudicated_gold":
        raise ValueError("Gold dataset must have status='adjudicated_gold'")
    protocol = payload.get("annotation_protocol", {})
    if not protocol.get("human_verified"):
        raise ValueError("Gold dataset must be human verified")
    source_run_id = str(payload.get("source_run_id") or "")
    allowed_runs = {str(item) for item in compatible_run_ids if item}
    if allowed_runs and source_run_id not in allowed_runs:
        raise ValueError(
            f"Gold source run {source_run_id!r} does not match cache runs {sorted(allowed_runs)!r}"
        )

    unit_index = {str(unit["unit_id"]): unit for unit in units}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in payload.get("records", []):
        record = dict(original)
        role = str(record.get("role") or "")
        if role not in ROLE_ORDER:
            raise ValueError(f"Unknown Gold role: {role!r}")
        unit_id = f"{record.get('sample_id')}::{role}"
        unit = unit_index.get(unit_id)
        if unit is None:
            raise ValueError(f"Gold candidate unit is missing from cache: {unit_id}")
        rank = int(record.get("input_rank", 0))
        expected_key = _candidate_key(unit, rank)
        if record.get("candidate_key") != expected_key or expected_key in seen:
            raise ValueError(f"Invalid or duplicate Gold candidate key: {expected_key}")
        rows = unit.get("rows", [])
        if rank < 1 or rank > len(rows):
            raise ValueError(f"Gold candidate rank is outside cached rows: {expected_key}")
        row = rows[rank - 1]
        cached_url = str(row.get("url") or row.get("source") or "")
        if cached_url != str(record.get("url") or ""):
            raise ValueError(f"Gold candidate URL mismatch: {expected_key}")
        content_sha256 = hashlib.sha256(_normalized_content(row).encode("utf-8")).hexdigest()
        if content_sha256 != record.get("content_sha256"):
            raise ValueError(f"Gold candidate content mismatch: {expected_key}")
        annotation = record.get("annotation", {})
        if not isinstance(annotation.get("should_retain"), bool):
            raise ValueError(f"Gold candidate lacks a retain decision: {expected_key}")
        seen.add(expected_key)
        records.append(record)
    if not records:
        raise ValueError("Gold dataset contains no records")
    return payload, records


def _threshold_grid() -> list[float]:
    candidates = {round(value / 100, 2) for value in range(6, 81, 2)}
    candidates.update(ResearchWorker.ROLE_RELEVANCE_THRESHOLDS.values())
    return sorted(candidates)


def _evaluate_unit(
    unit: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    role = str(unit["role"])
    sample = unit["sample"]
    rows = unit.get("rows", [])
    previous = ResearchWorker.ROLE_RELEVANCE_THRESHOLDS.get(role)
    ResearchWorker.ROLE_RELEVANCE_THRESHOLDS[role] = threshold
    try:
        _, audit = ResearchWorker._filter_and_rank(
            rows,
            str(sample["question"]),
            role,
            len(rows),
        )
    finally:
        if previous is None:
            ResearchWorker.ROLE_RELEVANCE_THRESHOLDS.pop(role, None)
        else:
            ResearchWorker.ROLE_RELEVANCE_THRESHOLDS[role] = previous

    labels: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        relevant, reasons = _proxy_relevance(row, sample)
        labels[index] = {"relevant": relevant, "reasons": reasons}
    selected_ranks = {int(item["input_rank"]) for item in audit["selected"]}
    below_threshold_ranks = {
        int(item["input_rank"])
        for item in audit["removals"]
        if item["reason"] == "below_role_threshold"
    }
    nonempty_ranks = {
        index
        for index, row in enumerate(rows, start=1)
        if str(row.get("content") or row.get("text") or "").strip()
    }
    positive_ranks = {index for index in nonempty_ranks if labels[index]["relevant"]}
    false_rejected_ranks = positive_ranks & below_threshold_ranks
    selected_positive_ranks = selected_ranks & positive_ranks
    selected_authorities = [float(item["source_authority_score"]) for item in audit["selected"]]
    primary_count = sum(
        str(item["source_class"]) in PRIMARY_SOURCE_CLASSES for item in audit["selected"]
    )
    return {
        "unit_id": unit["unit_id"],
        "sample_id": sample["id"],
        "role": role,
        "threshold": threshold,
        "raw_count": len(rows),
        "nonempty_count": len(nonempty_ranks),
        "retained_count": len(selected_ranks),
        "proxy_relevant_count": len(positive_ranks),
        "proxy_false_rejected_count": len(false_rejected_ranks),
        "selected_proxy_relevant_count": len(selected_positive_ranks),
        "primary_source_count": primary_count,
        "selected_authority_sum": sum(selected_authorities),
        "selected_input_ranks": sorted(selected_ranks),
        "below_threshold_input_ranks": sorted(below_threshold_ranks),
        "removal_counts": audit["removal_counts"],
        "false_rejected": [
            {
                "input_rank": rank,
                "url": str(rows[rank - 1].get("url") or ""),
                "title": rows[rank - 1].get("title"),
                "label_reasons": labels[rank]["reasons"],
            }
            for rank in sorted(false_rejected_ranks)
        ],
    }


def _aggregate_gold(
    unit_rows: Sequence[Mapping[str, Any]],
    gold_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute candidate-level confusion metrics on the adjudicated Gold slice."""

    selected_by_unit = {
        str(row["unit_id"]): {int(rank) for rank in row["selected_input_ranks"]}
        for row in unit_rows
    }
    true_positive = false_positive = true_negative = false_negative = 0
    for record in gold_records:
        unit_id = f"{record['sample_id']}::{record['role']}"
        predicted_retain = int(record["input_rank"]) in selected_by_unit.get(unit_id, set())
        should_retain = bool(record["annotation"]["should_retain"])
        if predicted_retain and should_retain:
            true_positive += 1
        elif predicted_retain:
            false_positive += 1
        elif should_retain:
            false_negative += 1
        else:
            true_negative += 1
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / positive if positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_labeled_count": positive + negative,
        "gold_positive_count": positive,
        "gold_negative_count": negative,
        "gold_true_positive": true_positive,
        "gold_false_positive": false_positive,
        "gold_true_negative": true_negative,
        "gold_false_negative": false_negative,
        "gold_false_rejection_rate": false_negative / positive if positive else 0.0,
        "gold_precision": precision,
        "gold_recall": recall,
        "gold_f1": f1,
        "gold_precision_defined": predicted_positive > 0,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = sum(int(row["nonempty_count"]) for row in rows)
    retained = sum(int(row["retained_count"]) for row in rows)
    relevant = sum(int(row["proxy_relevant_count"]) for row in rows)
    false_rejected = sum(int(row["proxy_false_rejected_count"]) for row in rows)
    selected_relevant = sum(int(row["selected_proxy_relevant_count"]) for row in rows)
    primary = sum(int(row["primary_source_count"]) for row in rows)
    authority_sum = sum(float(row["selected_authority_sum"]) for row in rows)
    removals: Counter[str] = Counter()
    for row in rows:
        removals.update({str(key): int(value) for key, value in row["removal_counts"].items()})
    return {
        "unit_count": len(rows),
        "raw_nonempty_count": raw,
        "retained_count": retained,
        "retention_rate": retained / raw if raw else 0.0,
        "proxy_relevant_count": relevant,
        "proxy_false_rejected_count": false_rejected,
        "proxy_false_rejection_rate": false_rejected / relevant if relevant else 0.0,
        "selected_proxy_relevant_count": selected_relevant,
        "proxy_precision": selected_relevant / retained if retained else 0.0,
        "end_to_end_proxy_recall": selected_relevant / relevant if relevant else 1.0,
        "mean_selected_authority": authority_sum / retained if retained else 0.0,
        "primary_source_rate": primary / retained if retained else 0.0,
        "removal_counts": dict(sorted(removals.items())),
    }


def evaluate_thresholds(
    units: Sequence[Mapping[str, Any]],
    extra_thresholds: Sequence[float] = (),
    gold_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    grid = sorted({*_threshold_grid(), *(round(value, 4) for value in extra_thresholds)})
    for role in ROLE_ORDER:
        role_units = [unit for unit in units if unit["role"] == role and unit.get("rows")]
        role_gold = [record for record in gold_records if record["role"] == role]
        curves[role] = []
        for threshold in grid:
            unit_rows = [_evaluate_unit(unit, threshold) for unit in role_units]
            point = {"threshold": threshold, **_aggregate(unit_rows)}
            point.update(_aggregate_gold(unit_rows, role_gold))
            curves[role].append(point)
    return curves


def recommend_thresholds(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    minimum_thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    recommendations: dict[str, dict[str, Any]] = {}
    floors = minimum_thresholds or ResearchWorker.ROLE_RELEVANCE_THRESHOLDS
    for role in ROLE_ORDER:
        current_floor = floors.get(role, 0.0)
        rows = [row for row in curves.get(role, []) if float(row["threshold"]) >= current_floor]
        ceiling = MAX_PROXY_FALSE_REJECTION[role]
        feasible = [
            row
            for row in rows
            if float(row["proxy_false_rejection_rate"]) <= ceiling
            and int(row["retained_count"]) > 0
        ]
        pool = feasible or rows
        if not pool:
            continue
        chosen = max(
            pool,
            key=lambda row: (
                float(row["proxy_precision"]),
                float(row["mean_selected_authority"]),
                1.0 - float(row["retention_rate"]),
                -float(row["threshold"]),
            ),
        )
        recommendations[role] = {
            **chosen,
            "max_proxy_false_rejection_rate": ceiling,
            "constraint_satisfied": bool(feasible),
        }
    return recommendations


def recommend_gold_thresholds(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Choose the maximum-F1 threshold for every role covered by Gold."""

    recommendations: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        rows = [
            row
            for row in curves.get(role, [])
            if int(row.get("gold_labeled_count", 0)) > 0
            and int(row.get("gold_positive_count", 0)) > 0
        ]
        if not rows:
            continue
        chosen = max(
            rows,
            key=lambda row: (
                float(row["gold_f1"]),
                float(row["gold_recall"]),
                float(row["gold_precision"]),
                float(row["threshold"]),
            ),
        )
        recommendations[role] = {
            **chosen,
            "selection_objective": "max_gold_f1",
        }
    return recommendations


def _combine_recommendations(
    proxy: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        if role in gold:
            combined[role] = {**gold[role], "recommendation_source": "gold_f1"}
        elif role in proxy:
            combined[role] = {
                **proxy[role],
                "recommendation_source": "proxy_fallback_no_gold",
            }
    return combined


def _aggregate_gold_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in summaries.values() if int(row.get("gold_labeled_count", 0))]
    values = {
        name: sum(int(row.get(name, 0)) for row in rows)
        for name in (
            "gold_labeled_count",
            "gold_positive_count",
            "gold_negative_count",
            "gold_true_positive",
            "gold_false_positive",
            "gold_true_negative",
            "gold_false_negative",
        )
    }
    tp = values["gold_true_positive"]
    fp = values["gold_false_positive"]
    fn = values["gold_false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    values.update(
        {
            "gold_false_rejection_rate": fn / (tp + fn) if tp + fn else 0.0,
            "gold_precision": precision,
            "gold_recall": recall,
            "gold_f1": (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            ),
            "covered_roles": [
                role
                for role in ROLE_ORDER
                if int(summaries.get(role, {}).get("gold_labeled_count", 0))
            ],
        }
    )
    return values


def _aggregate_role_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(summaries.values())
    raw = sum(int(row["raw_nonempty_count"]) for row in rows)
    retained = sum(int(row["retained_count"]) for row in rows)
    relevant = sum(int(row["proxy_relevant_count"]) for row in rows)
    false_rejected = sum(int(row["proxy_false_rejected_count"]) for row in rows)
    selected_relevant = sum(int(row["selected_proxy_relevant_count"]) for row in rows)
    authority_sum = sum(
        float(row["mean_selected_authority"]) * int(row["retained_count"]) for row in rows
    )
    primary_sum = sum(
        float(row["primary_source_rate"]) * int(row["retained_count"]) for row in rows
    )
    return {
        "raw_nonempty_count": raw,
        "retained_count": retained,
        "retention_rate": retained / raw if raw else 0.0,
        "proxy_relevant_count": relevant,
        "proxy_false_rejected_count": false_rejected,
        "proxy_false_rejection_rate": false_rejected / relevant if relevant else 0.0,
        "selected_proxy_relevant_count": selected_relevant,
        "proxy_precision": selected_relevant / retained if retained else 0.0,
        "mean_selected_authority": authority_sum / retained if retained else 0.0,
        "primary_source_rate": primary_sum / retained if retained else 0.0,
    }


def _baseline(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for role, threshold in thresholds.items():
        match = min(
            curves.get(role, []),
            key=lambda row: abs(float(row["threshold"]) - threshold),
            default=None,
        )
        if match is not None:
            output[role] = dict(match)
    return output


async def _fetch_units(
    samples: Sequence[Mapping[str, Any]],
    *,
    search_depth: str,
    candidate_limit: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    backend = TavilySearchBackend(search_depth=search_depth)
    semaphore = asyncio.Semaphore(concurrency)
    specs = [
        (sample, str(role)) for sample in samples for role in sample["expected_swarm"]["roles"]
    ]
    completed = 0
    progress_lock = asyncio.Lock()

    async def fetch(sample: Mapping[str, Any], role: str) -> dict[str, Any]:
        nonlocal completed
        hint = ResearchWorker.ROLE_QUERY_HINTS.get(role, "")
        query = f"{sample['question']} {hint}".strip()
        started = time.perf_counter()
        try:
            async with semaphore:
                rows = list(await backend.search(query, limit=candidate_limit))
            error = None
        except Exception as exc:
            rows = []
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        async with progress_lock:
            completed += 1
            print(
                f"[{completed}/{len(specs)}] {sample['id']}::{role} "
                f"rows={len(rows)} latency={latency:.2f}s" + (f" error={error}" if error else ""),
                flush=True,
            )
        return {
            "unit_id": f"{sample['id']}::{role}",
            "sample": dict(sample),
            "role": role,
            "query": query,
            "rows": rows,
            "latency_seconds": latency,
            "error": error,
        }

    return list(await asyncio.gather(*(fetch(sample, role) for sample, role in specs)))


def render_markdown(payload: Mapping[str, Any], source: Path) -> str:
    lines = [
        "# ResearchBench-Live 检索阈值校准",
        "",
        f"- 运行 ID：`{payload['run_id']}`",
        f"- 数据集：`{payload['dataset_name']} {payload['dataset_version']}`",
        f"- Tavily：`{payload['search_depth']}`，候选上限 {payload['candidate_limit']}",
        f"- 在线检索单元：{payload['successful_units']}/{payload['total_units']}",
        f"- 原始结果：`{source.resolve()}`",
        "",
        "> Proxy 指标使用 required_domains/required_terms 弱标签；Gold 指标使用正式人工标注。",
        "> 当前 Gold 只抽样了被过滤的 Proxy-positive 候选，因此 Gold Precision/Recall/F1",
        "> 仅描述该审核切片，不能作为全体检索结果的无偏总体估计。",
        "",
        "## Proxy 总体对照",
        "",
        "| 配置 | 保留数 | 保留率 | Proxy 误杀率 | Proxy 精确率 | 平均权威度 | 一手来源率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("当前阈值", "overall_baseline"),
        ("Proxy 推荐", "overall_proxy_recommendation"),
        ("最终 Gold/Proxy 阈值", "overall_recommendation"),
    ):
        aggregate = payload[key]
        lines.append(
            f"| {label} | {aggregate['retained_count']} | "
            f"{aggregate['retention_rate']:.1%} | "
            f"{aggregate['proxy_false_rejection_rate']:.1%} | "
            f"{aggregate['proxy_precision']:.1%} | "
            f"{aggregate['mean_selected_authority']:.3f} | "
            f"{aggregate['primary_source_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Gold 审核切片总体对照",
            "",
            "| 配置 | Gold 数 | TP | FP | TN | FN | 误杀率 | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("当前阈值", "overall_gold_baseline"),
        ("最终 Gold/Proxy 阈值", "overall_gold_recommendation"),
    ):
        aggregate = payload[key]
        lines.append(
            f"| {label} | {aggregate['gold_labeled_count']} | "
            f"{aggregate['gold_true_positive']} | {aggregate['gold_false_positive']} | "
            f"{aggregate['gold_true_negative']} | {aggregate['gold_false_negative']} | "
            f"{aggregate['gold_false_rejection_rate']:.1%} | "
            f"{aggregate['gold_precision']:.1%} | {aggregate['gold_recall']:.1%} | "
            f"{aggregate['gold_f1']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Proxy 与 Gold 阈值差异",
            "",
            "| 角色 | 当前 | Proxy 推荐 | Gold 推荐 | 最终 | Gold-Proxy | Gold n | Gold 误杀率 | Gold P/R/F1 | 最终依据 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    baseline = payload["baseline"]
    proxy_recommendations = payload["proxy_recommendations"]
    gold_recommendations = payload["gold_recommendations"]
    recommendations = payload["recommendations"]
    for role in ROLE_ORDER:
        before = baseline.get(role)
        proxy = proxy_recommendations.get(role)
        gold = gold_recommendations.get(role)
        final = recommendations.get(role)
        current_text = f"{before['threshold']:.2f}" if before else "N/A"
        proxy_text = f"{proxy['threshold']:.2f}" if proxy else "N/A"
        gold_text = f"{gold['threshold']:.2f}" if gold else "N/A"
        final_text = f"{final['threshold']:.2f}" if final else "N/A"
        delta_text = (
            f"{float(gold['threshold']) - float(proxy['threshold']):+.2f}"
            if gold and proxy
            else "N/A"
        )
        metrics_text = (
            f"{gold['gold_precision']:.1%}/{gold['gold_recall']:.1%}/{gold['gold_f1']:.1%}"
            if gold
            else "N/A"
        )
        gold_count = int(gold["gold_labeled_count"]) if gold else 0
        gold_fnr_text = f"{gold['gold_false_rejection_rate']:.1%}" if gold else "N/A"
        source_text = final.get("recommendation_source", "N/A") if final else "N/A"
        lines.append(
            f"| {role} | {current_text} | {proxy_text} | {gold_text} | "
            f"{final_text} | {delta_text} | {gold_count} | {gold_fnr_text} | "
            f"{metrics_text} | {source_text} |"
        )
    lines.extend(
        [
            "",
            "## 建议阈值配置",
            "",
            "```python",
            "ROLE_RELEVANCE_THRESHOLDS = {",
        ]
    )
    for role in ROLE_ORDER:
        recommendation = recommendations.get(role)
        if recommendation:
            lines.append(f'    "{role}": {recommendation["threshold"]:.2f},')
    lines.extend(["}", "```", ""])
    failures = [unit for unit in payload["units"] if unit.get("error")]
    if failures:
        lines.extend(["## 检索失败", ""])
        for unit in failures:
            lines.append(f"- `{unit['unit_id']}`：{unit['error']}")
        lines.append("")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> Path:
    load_env_files((BASE_DIR / ".env", EVALUATION_DIR / ".env"))
    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("track") != "live":
        raise ValueError("Threshold calibration requires track='live'")
    samples = dataset.get("samples", [])[: args.sample_limit or None]
    current_thresholds = dict(ResearchWorker.ROLE_RELEVANCE_THRESHOLDS)
    cache_thresholds: dict[str, float] = {}
    source_run_id = None
    compatible_run_ids: list[str] = []
    if args.cache:
        cached = json.loads(Path(args.cache).read_text(encoding="utf-8"))
        units = list(cached.get("units", []))
        if not units:
            raise ValueError("Calibration cache does not contain units")
        source_run_id = cached.get("run_id")
        compatible_run_ids = [
            str(item) for item in (cached.get("run_id"), cached.get("source_run_id")) if item
        ]
        cached_thresholds = cached.get("current_thresholds")
        if isinstance(cached_thresholds, Mapping):
            cache_thresholds = {
                str(role): float(value) for role, value in cached_thresholds.items()
            }
    else:
        if not os.getenv("TAVILY_API_KEY"):
            raise RuntimeError("TAVILY_API_KEY is not configured")
        units = await _fetch_units(
            samples,
            search_depth=args.search_depth,
            candidate_limit=args.candidate_limit,
            concurrency=args.concurrency,
        )
    gold_payload: dict[str, Any] | None = None
    gold_records: list[dict[str, Any]] = []
    gold_path: Path | None = None
    if not args.no_gold:
        gold_path = Path(args.gold)
        gold_payload, gold_records = load_gold_records(
            gold_path,
            units,
            compatible_run_ids=compatible_run_ids,
        )
    curves = evaluate_thresholds(
        units,
        tuple(current_thresholds.values()),
        gold_records,
    )
    proxy_recommendations = recommend_thresholds(curves, current_thresholds)
    gold_recommendations = recommend_gold_thresholds(curves)
    recommendations = _combine_recommendations(
        proxy_recommendations,
        gold_recommendations,
    )
    baseline = _baseline(curves, current_thresholds)
    overall_baseline = _aggregate_role_summaries(baseline)
    overall_proxy_recommendation = _aggregate_role_summaries(proxy_recommendations)
    overall_recommendation = _aggregate_role_summaries(recommendations)
    overall_gold_baseline = _aggregate_gold_summaries(baseline)
    overall_gold_recommendation = _aggregate_gold_summaries(recommendations)
    threshold_differences = {
        role: {
            "current_threshold": current_thresholds.get(role),
            "proxy_threshold": (
                proxy_recommendations[role]["threshold"] if role in proxy_recommendations else None
            ),
            "gold_threshold": (
                gold_recommendations[role]["threshold"] if role in gold_recommendations else None
            ),
            "final_threshold": (
                recommendations[role]["threshold"] if role in recommendations else None
            ),
            "gold_minus_proxy": (
                float(gold_recommendations[role]["threshold"])
                - float(proxy_recommendations[role]["threshold"])
                if role in gold_recommendations and role in proxy_recommendations
                else None
            ),
            "recommendation_source": (
                recommendations[role]["recommendation_source"] if role in recommendations else None
            ),
        }
        for role in ROLE_ORDER
    }
    latencies = [float(unit["latency_seconds"]) for unit in units]
    run_id = datetime.now(timezone.utc).strftime("retrieval-calibration-%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "live",
        "dataset_name": dataset.get("name", "ResearchBench-Live"),
        "dataset_version": dataset.get("version", "unknown"),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "provider": "tavily",
        "source_run_id": source_run_id,
        "search_depth": args.search_depth,
        "candidate_limit": args.candidate_limit,
        "proxy_label": "required-domain OR required-term",
        "gold": (
            {
                "name": gold_payload.get("name"),
                "version": gold_payload.get("version"),
                "status": gold_payload.get("status"),
                "dataset": str(gold_path.resolve()),
                "dataset_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
                "source_run_id": gold_payload.get("source_run_id"),
                "selection_rule": gold_payload.get("selection_rule"),
                "record_count": len(gold_records),
                "covered_roles": sorted({str(item["role"]) for item in gold_records}),
                "missing_roles": [
                    role
                    for role in ROLE_ORDER
                    if role not in {str(item["role"]) for item in gold_records}
                ],
                "population_metrics_valid": False,
                "scope_note": (
                    "Metrics describe the filtered proxy-positive review slice; "
                    "they are not unbiased population retrieval estimates."
                ),
            }
            if gold_payload is not None and gold_path is not None
            else None
        ),
        "threshold_grid": _threshold_grid(),
        "cache_thresholds": cache_thresholds,
        "current_thresholds": current_thresholds,
        "total_units": len(units),
        "successful_units": sum(not unit.get("error") for unit in units),
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "baseline": baseline,
        "proxy_recommendations": proxy_recommendations,
        "gold_recommendations": gold_recommendations,
        "recommendations": recommendations,
        "threshold_differences": threshold_differences,
        "overall_baseline": overall_baseline,
        "overall_proxy_recommendation": overall_proxy_recommendation,
        "overall_recommendation": overall_recommendation,
        "overall_gold_baseline": overall_gold_baseline,
        "overall_gold_recommendation": overall_gold_recommendation,
        "curves": curves,
        "units": units,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        render_markdown(payload, output_path), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "total_units": payload["total_units"],
                "successful_units": payload["successful_units"],
                "recommendations": {
                    role: {
                        "threshold": item["threshold"],
                        "source": item["recommendation_source"],
                    }
                    for role, item in recommendations.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"result_file={output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate role retrieval thresholds with live Tavily results"
    )
    parser.add_argument(
        "--dataset",
        default=str(EVALUATION_DIR / "datasets" / "researchbench_live_v1.0.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_DIR / "results" / "retrieval_calibration"),
    )
    parser.add_argument("--search-depth", choices=("basic", "advanced"), default="advanced")
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--cache",
        help="Reuse the units from an earlier calibration JSON without new API calls",
    )
    parser.add_argument(
        "--gold",
        default=str(EVALUATION_DIR / "datasets" / "researchbench_retrieval_gold_v1.0.json"),
        help="Adjudicated Retrieval-Gold dataset bound to the cached candidates",
    )
    parser.add_argument(
        "--no-gold",
        action="store_true",
        help="Run the legacy proxy-only calibration without Retrieval-Gold",
    )
    args = parser.parse_args()
    if args.candidate_limit < 1 or args.concurrency < 1:
        parser.error("candidate-limit and concurrency must be positive")
    try:
        asyncio.run(run(args))
    except (RuntimeError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
