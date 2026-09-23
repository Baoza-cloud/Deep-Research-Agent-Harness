"""Three-layer evaluation for deep-research reports.

Layer 1: deterministic factual/citation metrics.
Layer 2: optional five-dimension LLM-as-Judge.
Layer 3: bootstrap 95% confidence intervals and Cohen's d comparisons.
"""

from __future__ import annotations

import inspect
import math
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from research_engine.claims import (
    extract_claims,
    is_reviewable_claim,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
SEMANTIC_TERM_ALIASES: tuple[tuple[str, ...], ...] = (
    ("consumer group", "消费者组", "消费组"),
    ("不同组", "不同 consumer group", "不同消费者组", "各消费组"),
    ("独立消费", "分别消费", "各自消费", "独立读取"),
    ("一个消费者", "单个消费者", "一名消费者"),
    ("异步", "async", "asynchronous"),
    ("证据不足", "无法确认", "缺少直接证据", "待验证"),
)


def _tokens(text: str) -> set[str]:
    return {item.lower() for item in TOKEN_PATTERN.findall(text)}


def _claims(report: str) -> list[Any]:
    return [item for item in extract_claims(report, min_chars=6) if is_reviewable_claim(item)]


@dataclass
class RuleMetrics:
    factual_accuracy: float
    hallucination_rate: float
    citation_coverage: float
    citation_validity: float
    source_diversity: int
    supported_claims: int
    total_claims: int


@dataclass
class EvaluationResult:
    rules: RuleMetrics
    judge_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float | None = None


@dataclass
class BenchmarkMetrics:
    key_fact_recall: float
    forbidden_claim_rate: float
    key_facts_hit: int
    key_facts_total: int
    forbidden_terms_hit: list[str] = field(default_factory=list)
    rule_key_fact_recall: float = 0.0
    semantic_key_fact_recall: float = 0.0
    rule_key_facts_hit: int = 0
    semantic_key_facts_hit: int = 0


def _semantic_normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.casefold())


def _term_variants(term: str, fact: dict[str, Any]) -> set[str]:
    variants = {term}
    lowered = term.casefold()
    for group in SEMANTIC_TERM_ALIASES:
        if lowered in {item.casefold() for item in group}:
            variants.update(group)
    custom = fact.get("term_aliases", {})
    if isinstance(custom, dict):
        for key, values in custom.items():
            group = [str(key)]
            if isinstance(values, list):
                group.extend(str(item) for item in values)
            if lowered in {item.casefold() for item in group}:
                variants.update(group)
    return variants


def _semantic_term_present(report: str, term: str, fact: dict[str, Any]) -> bool:
    normalized_report = _semantic_normalize(report)
    return any(
        _semantic_normalize(variant) in normalized_report
        for variant in _term_variants(term, fact)
        if _semantic_normalize(variant)
    )


def evaluate_rules(report: str, evidences: Sequence[dict[str, Any]]) -> RuleMetrics:
    evidence_map = {
        str(item.get("evidence_id")): str(item.get("content", ""))
        for item in evidences
        if item.get("evidence_id")
    }
    claims = _claims(report)
    cited_claims = 0
    supported_claims = 0
    all_citations: list[str] = []
    for claim in claims:
        citations = claim.citations
        all_citations.extend(citations)
        if citations:
            cited_claims += 1
        claim_tokens = _tokens(claim.text)
        for citation in citations:
            evidence_tokens = _tokens(evidence_map.get(citation, ""))
            union = claim_tokens | evidence_tokens
            similarity = len(claim_tokens & evidence_tokens) / len(union) if union else 0.0
            if citation in evidence_map and similarity >= 0.08:
                supported_claims += 1
                break

    total = len(claims)
    accuracy = supported_claims / total if total else 0.0
    coverage = cited_claims / total if total else 0.0
    validity = (
        sum(citation in evidence_map for citation in all_citations) / len(all_citations)
        if all_citations
        else 0.0
    )
    sources = {str(item.get("url") or item.get("source")) for item in evidences}
    sources.discard("None")
    sources.discard("")
    return RuleMetrics(
        factual_accuracy=accuracy,
        hallucination_rate=1.0 - accuracy,
        citation_coverage=coverage,
        citation_validity=validity,
        source_diversity=len(sources),
        supported_claims=supported_claims,
        total_claims=total,
    )


def evaluate_expectations(report: str, sample: dict[str, Any]) -> BenchmarkMetrics:
    """Score exact rules and alias-aware semantic facts as separate audit layers."""

    lowered = report.casefold()
    key_facts = sample.get("key_facts", [])
    rule_hits = 0
    semantic_hits = 0
    combined_hits = 0
    for fact in key_facts:
        terms = [str(item).casefold() for item in fact.get("required_terms", [])]
        minimum = int(fact.get("minimum_hits", len(terms)))
        rule_matched = sum(term in lowered for term in terms) >= minimum
        semantic_matched = (
            sum(_semantic_term_present(report, term, fact) for term in terms) >= minimum
        )
        rule_hits += int(rule_matched)
        semantic_hits += int(semantic_matched)
        combined_hits += int(rule_matched or semantic_matched)
    forbidden = [str(item) for item in sample.get("forbidden_terms", [])]
    forbidden_hits = [item for item in forbidden if item.casefold() in lowered]
    return BenchmarkMetrics(
        key_fact_recall=combined_hits / len(key_facts) if key_facts else 1.0,
        forbidden_claim_rate=(len(forbidden_hits) / len(forbidden) if forbidden else 0.0),
        key_facts_hit=combined_hits,
        key_facts_total=len(key_facts),
        forbidden_terms_hit=forbidden_hits,
        rule_key_fact_recall=(rule_hits / len(key_facts) if key_facts else 1.0),
        semantic_key_fact_recall=(semantic_hits / len(key_facts) if key_facts else 1.0),
        rule_key_facts_hit=rule_hits,
        semantic_key_facts_hit=semantic_hits,
    )


def overall_quality(rules: RuleMetrics, benchmark: BenchmarkMetrics) -> float:
    return statistics.fmean(
        (
            rules.factual_accuracy,
            rules.citation_coverage,
            rules.citation_validity,
            benchmark.key_fact_recall,
            1.0 - benchmark.forbidden_claim_rate,
        )
    )


async def evaluate_with_judge(
    question: str,
    report: str,
    evidences: Sequence[dict[str, Any]],
    judge: Callable[[str], str | Awaitable[str]],
) -> dict[str, float]:
    from research_engine.parsing import parse_json_payload

    prompt = f"""
请作为严格评审，按 0-1 评价研究报告的五个维度，只输出 JSON：
{{"factuality":0-1,"logic":0-1,"citation_quality":0-1,
"completeness":0-1,"actionability":0-1}}
问题：{question}
证据：{evidences}
报告：{report}
""".strip()
    response = judge(prompt)
    if inspect.isawaitable(response):
        response = await response
    payload = parse_json_payload(str(response))
    required = ("factuality", "logic", "citation_quality", "completeness", "actionability")
    return {name: max(0.0, min(1.0, float(payload[name]))) for name in required}


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    lower = means[max(0, int(alpha * samples))]
    upper = means[min(samples - 1, int((1.0 - alpha) * samples) - 1)]
    return lower, upper


def cohens_d(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    if len(baseline) < 2 or len(candidate) < 2:
        raise ValueError("Cohen's d requires at least two observations per group")
    baseline_variance = statistics.variance(baseline)
    candidate_variance = statistics.variance(candidate)
    pooled = math.sqrt(
        ((len(baseline) - 1) * baseline_variance + (len(candidate) - 1) * candidate_variance)
        / (len(baseline) + len(candidate) - 2)
    )
    if pooled == 0:
        return 0.0
    return (statistics.fmean(candidate) - statistics.fmean(baseline)) / pooled


def paired_cohens_d(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Cohen's dz for paired ablations evaluated on the same questions."""

    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("paired Cohen's d requires equal groups with at least two values")
    differences = [right - left for left, right in zip(baseline, candidate)]
    spread = statistics.stdev(differences)
    return statistics.fmean(differences) / spread if spread else 0.0
