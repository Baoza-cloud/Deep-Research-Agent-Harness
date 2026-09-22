"""Build the manually adjudicated Retrieval-Gold slice from Live candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from calibrate_retrieval_thresholds import _evaluate_unit  # noqa: E402
from research_engine.agents import ResearchWorker  # noqa: E402


SOURCE_RUN = (
    EVALUATION_DIR
    / "results"
    / "retrieval_calibration"
    / "retrieval-calibration-20260916T145324Z.json"
)
OUTPUT = EVALUATION_DIR / "datasets" / "researchbench_retrieval_gold_v1.0.json"
REPORT = (
    EVALUATION_DIR
    / "results"
    / "retrieval_calibration"
    / "retrieval-gold-v1.0-report.md"
)

# Frozen source thresholds used to select the v1.0 annotation candidates.
# Do not replace these with the current runtime thresholds: changing production
# calibration must not silently change the versioned Gold membership.
SOURCE_THRESHOLDS = {
    "researcher": 0.14,
    "scope_researcher": 0.58,
    "evidence_researcher": 0.54,
    "counter_researcher": 0.56,
    "impact_analyst": 0.56,
    "evidence_verifier": 0.42,
}


# key -> (relevance, source_quality, should_retain, confidence, rationale)
ANNOTATIONS: dict[str, tuple[str, str, bool, float, str]] = {
    "live_standard_python_concurrency_001::scope_researcher::8": (
        "partially_relevant", "community", True, 0.90,
        "直接概括 gather 与 TaskGroup 的任务聚合和失败取消差异，可支持范围梳理，但缺少完整异常传播细节。",
    ),
    "live_standard_python_concurrency_001::scope_researcher::9": (
        "relevant", "authoritative_community", True, 0.96,
        "Python 官方社区讨论直接比较 TaskGroup 与 gather(return_exceptions=True) 的取消及异常语义。",
    ),
    "live_standard_python_concurrency_001::evidence_researcher::10": (
        "partially_relevant", "community", False, 0.94,
        "内容涉及两者差异，但只有截断的 Reddit 讨论，不能作为 Evidence 角色所需的稳定证据。",
    ),
    "live_standard_python_concurrency_001::counter_researcher::9": (
        "relevant", "community_video", True, 0.91,
        "给出了 Python 版本兼容限制和异常组行为，直接补充 TaskGroup 的适用边界与反例。",
    ),
    "live_standard_python_concurrency_001::counter_researcher::10": (
        "partially_relevant", "community", False, 0.93,
        "只陈述 TaskGroup 失败时取消其他任务，没有形成 gather 对照或新的限制证据。",
    ),
    "live_standard_pgvector_index_001::counter_researcher::8": (
        "relevant", "technical_secondary", True, 0.92,
        "系统比较 HNSW 与 IVFFlat 的召回、查询速度、构建成本和局限；虽非 pgvector 专文，仍直接服务反例与权衡分析。",
    ),
    "live_standard_redis_eviction_001::scope_researcher::10": (
        "partially_relevant", "community", False, 0.96,
        "只解释 allkeys-lru，未覆盖 volatile-lru、allkeys-lfu 或三者适用边界。",
    ),
    "live_standard_redis_eviction_001::evidence_researcher::7": (
        "relevant", "official_vendor", True, 0.99,
        "Redis 官方页面直接解释 LFU 与 LRU 的频率/时近语义和选择条件，属于高价值一手证据。",
    ),
    "live_standard_redis_eviction_001::counter_researcher::5": (
        "partially_relevant", "community", False, 0.94,
        "仅列出 volatile-lfu 与 allkeys-lfu 的集合差异，缺少题目要求的三策略限制和负载对照。",
    ),
    "live_standard_redis_eviction_001::counter_researcher::7": (
        "partially_relevant", "community", False, 0.97,
        "仅重复 allkeys-lru 的基本语义，没有提供反例、失败模式或比较证据。",
    ),
    "live_standard_oauth_pkce_001::scope_researcher::10": (
        "relevant", "authoritative_secondary", True, 0.98,
        "直接澄清 PKCE 不替代客户端密钥，并说明公共与机密客户端的适用范围。",
    ),
    "live_standard_oauth_pkce_001::evidence_researcher::6": (
        "relevant", "authoritative_secondary", True, 0.98,
        "OAuth.net 页面提供 PKCE、安全目标、code verifier 与 client secret 关系的直接证据。",
    ),
    "live_standard_oauth_pkce_001::counter_researcher::4": (
        "relevant", "authoritative_secondary", True, 0.99,
        "明确指出 PKCE 不是客户端认证且不能替代 client secret，正是所需限制证据。",
    ),
    "live_standard_oauth_pkce_001::counter_researcher::7": (
        "relevant", "security_research", True, 0.98,
        "详细讨论 plain challenge 导致的 PKCE downgrade attack，属于直接安全反例。",
    ),
    "live_standard_oauth_pkce_001::counter_researcher::8": (
        "relevant", "official_vendor", True, 0.97,
        "Auth0 文档说明无法安全保存 client secret 的客户端类型及授权码流程限制。",
    ),
    "live_complex_medical_ai_001::scope_researcher::5": (
        "relevant", "authoritative_secondary", True, 0.97,
        "直接梳理 FDA 对健康 AI 的监管范围、风险分级和受监管/不受监管功能边界。",
    ),
    "live_complex_medical_ai_001::scope_researcher::6": (
        "relevant", "professional_organization", True, 0.95,
        "国家医学院材料直接讨论 FDA 监管边界、临床采用障碍、透明度和患者安全。",
    ),
    "live_complex_medical_ai_001::scope_researcher::7": (
        "partially_relevant", "commercial_secondary", False, 0.88,
        "主题相关且覆盖临床证据要求，但关键监管表述来自商业二手文章，不满足本题一手来源约束。",
    ),
    "live_complex_medical_ai_001::scope_researcher::9": (
        "partially_relevant", "peer_reviewed", False, 0.91,
        "系统综述与医疗 AI 风险相关，但缓存片段不足以支持 WHO/FDA/NIST 治理范围比较。",
    ),
    "live_complex_medical_ai_001::evidence_researcher::8": (
        "partially_relevant", "commercial_secondary", False, 0.92,
        "涉及临床证据、持续学习和变更控制，但为二手汇总，不能替代题目要求的一手证据。",
    ),
    "live_complex_medical_ai_001::evidence_researcher::9": (
        "irrelevant", "tertiary", False, 0.98,
        "返回内容主要是 Wikipedia 参考文献列表，没有形成可直接引用的治理事实证据。",
    ),
    "live_complex_medical_ai_001::counter_researcher::3": (
        "relevant", "peer_reviewed", True, 0.97,
        "同行评审综述直接覆盖生成式 AI 医疗应用的准确性、偏见、隐私和临床安全挑战。",
    ),
    "live_complex_medical_ai_001::counter_researcher::7": (
        "relevant", "peer_reviewed", True, 0.97,
        "同行评审调查直接提供透明度、责任、隐私和监管缺口等反面证据。",
    ),
    "live_complex_medical_ai_001::counter_researcher::8": (
        "relevant", "official", True, 0.99,
        "WHO 官方页面直接覆盖健康 AI 监管、证据生成和风险治理，应保留。",
    ),
    "live_complex_medical_ai_001::counter_researcher::9": (
        "partially_relevant", "scholarly_secondary", False, 0.91,
        "讨论一般医疗 AI 风险，但没有聚焦生成式 AI、具体治理要求或失败案例。",
    ),
    "live_complex_medical_ai_001::impact_analyst::4": (
        "relevant", "authoritative_secondary", True, 0.97,
        "覆盖 FDA 监管路径、2025 指南时间线和生成式 AI 医疗部署影响，可直接支持实施分析。",
    ),
    "live_complex_medical_ai_001::impact_analyst::5": (
        "partially_relevant", "community_video", False, 0.90,
        "片段涉及监管沟通和变更控制，但来源与上下文不足以支撑医院级实施建议。",
    ),
    "live_complex_medical_ai_001::impact_analyst::9": (
        "relevant", "official", True, 0.99,
        "WHO 官方伦理与治理指南直接支持患者安全、治理风险和落地影响分析。",
    ),
    "live_complex_financial_rag_001::evidence_researcher::5": (
        "partially_relevant", "low_quality_aggregator", False, 0.98,
        "只讨论通用 DLP，与银行云端/本地 RAG、模型风险及监管边界缺乏直接关系。",
    ),
    "live_complex_financial_rag_001::evidence_researcher::6": (
        "partially_relevant", "low_quality_aggregator", False, 0.99,
        "与上一条为跨子域重复聚合内容，且只因泛词 data 命中，正确淘汰。",
    ),
    "live_complex_research_agent_001::scope_researcher::4": (
        "partially_relevant", "community", True, 0.90,
        "直接介绍 Tavily 作为 RAG/Agent 搜索后端的能力，可作为三后端比较中的单侧范围材料。",
    ),
    "live_complex_research_agent_001::scope_researcher::10": (
        "irrelevant", "community_extraction_failed", False, 0.98,
        "缓存正文只有工具清单和页面噪声，没有可核验的能力、限制或比较内容。",
    ),
    "live_complex_research_agent_001::counter_researcher::9": (
        "irrelevant", "community_extraction_failed", False, 0.98,
        "同一知乎页只返回工具清单，没有反例、限制、提示注入或引用污染证据。",
    ),
}


def _candidate_key(unit: dict[str, Any], rank: int) -> str:
    return f"{unit['sample']['id']}::{unit['role']}::{rank}"


def build() -> dict[str, Any]:
    source = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
    thresholds = dict(SOURCE_THRESHOLDS)
    candidates: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for unit in source["units"]:
        evaluation = _evaluate_unit(unit, thresholds[unit["role"]])
        for rejected in evaluation["false_rejected"]:
            rank = int(rejected["input_rank"])
            candidates.append((unit, rank, rejected))

    actual_keys = {_candidate_key(unit, rank) for unit, rank, _ in candidates}
    expected_keys = set(ANNOTATIONS)
    if actual_keys != expected_keys:
        missing = sorted(actual_keys - expected_keys)
        stale = sorted(expected_keys - actual_keys)
        raise ValueError(f"Annotation mismatch; missing={missing}, stale={stale}")

    records: list[dict[str, Any]] = []
    for index, (unit, rank, proxy) in enumerate(candidates, start=1):
        key = _candidate_key(unit, rank)
        relevance, quality, should_retain, confidence, rationale = ANNOTATIONS[key]
        row = unit["rows"][rank - 1]
        content = " ".join(str(row.get("content") or "").split())
        score = ResearchWorker._score_relevance(
            row,
            unit["sample"]["question"],
            unit["role"],
        )
        records.append(
            {
                "annotation_id": f"retrieval-gold-{index:03d}",
                "candidate_key": key,
                "sample_id": unit["sample"]["id"],
                "domain": unit["sample"].get("domain"),
                "question": unit["sample"]["question"],
                "role": unit["role"],
                "retrieval_query": unit["query"],
                "input_rank": rank,
                "title": row.get("title"),
                "url": row.get("url"),
                "source_type": row.get("source_type"),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "cached_excerpt": content[:1200],
                "retrieval_relevance_score": score["relevance_score"],
                "role_threshold": thresholds[unit["role"]],
                "source_class": score["source_class"],
                "source_authority_score": score["source_authority_score"],
                "proxy_label_reasons": proxy["label_reasons"],
                "annotation": {
                    "relevance": relevance,
                    "source_quality": quality,
                    "should_retain": should_retain,
                    "decision": "false_rejection" if should_retain else "correct_rejection",
                    "confidence": confidence,
                    "rationale": rationale,
                    "review_basis": "cached_title_and_content",
                },
            }
        )

    decisions = Counter(record["annotation"]["decision"] for record in records)
    relevance = Counter(record["annotation"]["relevance"] for record in records)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_role[record["role"]][record["annotation"]["decision"]] += 1
        by_role[record["role"]]["total"] += 1
    return {
        "name": "ResearchBench-Retrieval-Gold",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "retrieval_gold",
        "status": "adjudicated_gold",
        "description": "ResearchBench-Live 校准中 33 条 Proxy 误杀候选的正式人工审核集。",
        "source_run_id": source["run_id"],
        "source_file": str(SOURCE_RUN.resolve()),
        "selection_rule": "proxy-relevant candidate removed by its calibrated role threshold",
        "annotation_protocol": {
            "relevance_labels": [
                "relevant",
                "partially_relevant",
                "irrelevant",
                "cannot_determine",
            ],
            "decision_rule": "should_retain=true only when the page materially contributes to the assigned role; source quality is judged separately",
            "reviewer_type": "manual_semantic_review",
            "human_verified": True,
            "independent_second_pass_required": False,
            "signoff": {
                "reviewer_role": "project_owner",
                "status": "approved",
                "scope": "all_records",
                "approved_at": "2026-09-17",
            },
        },
        "thresholds": thresholds,
        "statistics": {
            "record_count": len(records),
            "unique_url_count": len({record["url"] for record in records}),
            "decision_distribution": dict(sorted(decisions.items())),
            "relevance_distribution": dict(sorted(relevance.items())),
            "false_rejection_share_of_reviewed": (
                decisions["false_rejection"] / len(records) if records else 0.0
            ),
            "by_role": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(by_role.items())
            },
        },
        "records": records,
    }


def render_report(dataset: dict[str, Any]) -> str:
    stats = dataset["statistics"]
    lines = [
        "# ResearchBench Retrieval-Gold v1.0",
        "",
        f"- 标注记录：{stats['record_count']}",
        f"- 唯一 URL：{stats['unique_url_count']}",
        f"- 真误杀：{stats['decision_distribution'].get('false_rejection', 0)}",
        f"- 正确淘汰：{stats['decision_distribution'].get('correct_rejection', 0)}",
        f"- 真误杀占复核候选：{stats['false_rejection_share_of_reviewed']:.1%}",
        "- 状态：adjudicated_gold，项目负责人已确认",
        "",
        "## 按角色统计",
        "",
        "| 角色 | 复核数 | 真误杀 | 正确淘汰 |",
        "|---|---:|---:|---:|",
    ]
    for role, counts in stats["by_role"].items():
        lines.append(
            f"| {role} | {counts.get('total', 0)} | "
            f"{counts.get('false_rejection', 0)} | "
            f"{counts.get('correct_rejection', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 逐条决定",
            "",
            "| ID | 样本 | 角色 | 页面 | 相关性 | 来源质量 | 决定 | 置信度 |",
            "|---|---|---|---|---|---|---|---:|",
        ]
    )
    for record in dataset["records"]:
        annotation = record["annotation"]
        title = str(record.get("title") or "无标题")[:60].replace("|", "\\|")
        lines.append(
            f"| {record['annotation_id']} | {record['sample_id']} | {record['role']} | "
            f"[{title}]({record['url']}) | {annotation['relevance']} | "
            f"{annotation['source_quality']} | {annotation['decision']} | "
            f"{annotation['confidence']:.2f} |"
        )
    lines.extend(
        [
            "",
            "> 本数据集的相关性、来源质量和保留决定已经完成正式人工审核。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    dataset = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(render_report(dataset), encoding="utf-8")
    print(json.dumps(dataset["statistics"], ensure_ascii=False, indent=2))
    print(f"dataset={OUTPUT}")
    print(f"report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
