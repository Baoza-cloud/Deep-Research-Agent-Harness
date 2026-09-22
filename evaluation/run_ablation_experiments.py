"""Run reproducible Deep Research ablations on ResearchBench-Frozen."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from research_engine import (  # noqa: E402
    AgentSpec,
    BlueTeamRepairer,
    ClaimEvidenceVerifier,
    DeepResearchAgent,
    Evidence,
    HeuristicPlanner,
    HeuristicSwarmPolicy,
    LLMPlanner,
    Planner,
    ResearchConfig,
    ResearchPlan,
    ResearchSubtask,
    ReviewResult,
    RunContext,
    SharedMemory,
    Synthesizer,
    auto_provider,
    build_llm,
    load_env_files,
)
from research_evaluation import (  # noqa: E402
    bootstrap_ci,
    evaluate_expectations,
    evaluate_rules,
    overall_quality,
    paired_cohens_d,
)


VARIANT_DESCRIPTIONS = {
    "single_agent": "单检索任务 + 单次合成；保留统一的最终 Claim Ledger",
    "no_red_blue": "多任务 DAG 研究与合成，但关闭 Red Review 和 Blue Repair",
    "rewrite_blue": "启用 Red Review；Blue 只使用整篇重写修复",
    "structured_patch_blue": "启用 Red Review、结构化 Patch、语义验证和失败回滚",
    "fixed_harness": "固定单一 Research Worker 角色与配置并发，启用完整 Structured Patch Blue",
    "dynamic_swarm": "按复杂度动态选择研究角色、并发预算与停止条件，启用完整 Structured Patch Blue",
}

RUN_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "researchbench_run_context", default="unscoped"
)


def _estimate_tokens(text: str) -> int:
    """Provider-neutral approximation used when API usage metadata is unavailable."""

    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = len(re.sub(r"[\u3400-\u9fff\s]", "", text))
    return cjk + math.ceil(non_cjk / 4)


class InstrumentedLLM:
    """Count calls and character volume without recording prompts or secrets."""

    def __init__(
        self,
        delegate: Any,
        input_cost_per_million_usd: float = 0.0,
        output_cost_per_million_usd: float = 0.0,
    ):
        self.delegate = delegate
        self.config = getattr(delegate, "config", None)
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd
        self._usage: dict[str, dict[str, float]] = {}

    def _bucket(self) -> dict[str, float]:
        return self._usage.setdefault(
            RUN_CONTEXT.get(),
            {
                "llm_calls": 0.0,
                "prompt_chars": 0.0,
                "completion_chars": 0.0,
                "estimated_prompt_tokens": 0.0,
                "estimated_completion_tokens": 0.0,
                "estimated_total_tokens": 0.0,
                "estimated_cost_usd": 0.0,
                "llm_latency_seconds": 0.0,
            },
        )

    async def __call__(self, prompt: str) -> str:
        started = time.perf_counter()
        bucket = self._bucket()
        bucket["llm_calls"] += 1
        bucket["prompt_chars"] += len(prompt)
        prompt_tokens = _estimate_tokens(prompt)
        bucket["estimated_prompt_tokens"] += prompt_tokens
        response = self.delegate(prompt)
        if inspect.isawaitable(response):
            response = await response
        text = str(response)
        bucket["completion_chars"] += len(text)
        completion_tokens = _estimate_tokens(text)
        bucket["estimated_completion_tokens"] += completion_tokens
        bucket["estimated_total_tokens"] += prompt_tokens + completion_tokens
        bucket["estimated_cost_usd"] += (
            prompt_tokens * self.input_cost_per_million_usd
            + completion_tokens * self.output_cost_per_million_usd
        ) / 1_000_000
        bucket["llm_latency_seconds"] += time.perf_counter() - started
        return text

    def usage(self, context_key: str) -> dict[str, float]:
        return dict(self._usage.get(context_key, _zero_usage()))


def _zero_usage() -> dict[str, float]:
    return {
        "llm_calls": 0.0,
        "prompt_chars": 0.0,
        "completion_chars": 0.0,
        "estimated_prompt_tokens": 0.0,
        "estimated_completion_tokens": 0.0,
        "estimated_total_tokens": 0.0,
        "estimated_cost_usd": 0.0,
        "llm_latency_seconds": 0.0,
    }


def _reprice_row(
    row: dict[str, Any],
    input_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
) -> dict[str, Any]:
    """Recompute estimated cost so resumed rows share one pricing scenario."""

    usage = row.get("llm_usage")
    if not isinstance(usage, dict):
        return row
    prompt_tokens = float(usage.get("estimated_prompt_tokens", 0.0))
    completion_tokens = float(usage.get("estimated_completion_tokens", 0.0))
    usage["estimated_cost_usd"] = (
        prompt_tokens * input_cost_per_million_usd
        + completion_tokens * output_cost_per_million_usd
    ) / 1_000_000
    return row


def _resume_row_is_compatible(variant: str, row: dict[str, Any]) -> bool:
    """Reuse a Dynamic row only when its selected tier is behavior-compatible."""

    if variant != "dynamic_swarm":
        return True
    swarm_plan = row.get("swarm_plan")
    if not isinstance(swarm_plan, dict):
        return False
    score = swarm_plan.get("complexity_score")
    if not isinstance(score, (int, float)):
        return False
    current_level = (
        "simple"
        if score < HeuristicSwarmPolicy.SIMPLE_THRESHOLD
        else (
            "standard"
            if score < HeuristicSwarmPolicy.COMPLEX_THRESHOLD
            else "complex"
        )
    )
    cached_level = swarm_plan.get("level")
    if cached_level != current_level:
        return False
    cached_version = swarm_plan.get("policy_version")
    if cached_version == HeuristicSwarmPolicy.VERSION:
        return True
    if HeuristicSwarmPolicy.VERSION != "heuristic-v4":
        return False
    # v4 changes only the simple boundary.  v3 has the same per-tier budgets;
    # v2 has the same standard/complex budgets.  Pre-versioned v1 standard
    # rows are accepted only after checking their concrete tier parameters.
    if cached_version == "heuristic-v3":
        return True
    if cached_version == "heuristic-v2" and cached_level in {"standard", "complex"}:
        return True
    return (
        cached_version is None
        and cached_level == "standard"
        and len(swarm_plan.get("agents", [])) == 3
        and swarm_plan.get("max_review_rounds") == 3
        and swarm_plan.get("max_verification_queries") == 6
        and swarm_plan.get("evidence_stagnation_patience") == 2
    )


class FrozenEvidenceWorker:
    """Return immutable benchmark evidence regardless of generated query wording."""

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.evidences = [
            Evidence(
                evidence_id=str(row["evidence_id"]),
                subtask_id=str(row.get("subtask_id", "facts")),
                content=str(row["content"]),
                source=str(row.get("source") or row.get("url") or "frozen"),
                title=row.get("title"),
                url=row.get("url"),
                source_type=str(row.get("source_type", "frozen_reference")),
                published_at=row.get("published_at"),
                metadata={
                    "benchmark_snapshot": True,
                    "snapshot_content_sha256": hashlib.sha256(
                        str(row["content"]).encode("utf-8")
                    ).hexdigest(),
                },
            )
            for row in rows
        ]

    async def run(self, task: ResearchSubtask) -> list[Evidence]:
        return [replace(item, metadata=dict(item.metadata)) for item in self.evidences]

    async def run_for_agent(
        self,
        task: ResearchSubtask,
        spec: AgentSpec,
        context: RunContext,
    ) -> list[Evidence]:
        hints = {
            "scope_researcher": "官方定义 术语 边界 适用范围",
            "evidence_researcher": "一手来源 官方文档 原始数据 直接证据",
            "counter_researcher": "反例 局限 失败案例 争议 相反证据",
            "impact_analyst": "风险 权衡 成本 影响 实施建议",
            "evidence_verifier": "事实核验 原始出处 直接支持 交叉验证",
        }
        query = f"{task.question} {hints.get(spec.role, '')}".strip()
        context.emit(
            "role_strategy_applied",
            task_id=task.subtask_id,
            agent_id=spec.agent_id,
            role=spec.role,
            query=query,
            strategy="frozen_projection",
        )
        rows = await self.run(task)
        for evidence in rows:
            evidence.metadata.update(
                {"retrieval_query": query, "retrieval_role": spec.role}
            )
        return rows


class SingleTaskPlanner(HeuristicPlanner):
    async def plan(self, question: str) -> ResearchPlan:
        return ResearchPlan(
            main_question=question,
            objective="使用冻结证据直接回答问题",
            subtasks=[
                ResearchSubtask(
                    "single",
                    question,
                    "Single-Agent baseline uses one retrieval task",
                    max_results=20,
                )
            ],
            success_criteria=["回答核心问题", "引用冻结证据"],
        )

    async def replan(
        self,
        plan: ResearchPlan,
        failed_tasks: Sequence[ResearchSubtask],
        reason: str,
    ) -> list[ResearchSubtask]:
        return []


class NoOpReviewer:
    async def review(
        self,
        question: str,
        report: str,
        evidences: Sequence[Evidence],
    ) -> ReviewResult:
        return ReviewResult(
            passed=True,
            scores={"ablation_no_review": 1.0},
            metrics={"citation_coverage": 0.0},
            total_score=1.0,
        )


def load_frozen_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("track") != "frozen":
        raise ValueError("Ablation runner requires a dataset with track='frozen'")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Frozen dataset must contain a non-empty samples list")
    ids = [str(item.get("id", "")) for item in samples]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Frozen sample IDs must be non-empty and unique")
    for sample in samples:
        if not sample.get("question") or not sample.get("evidence"):
            raise ValueError(f"Sample {sample.get('id')} needs question and evidence")
        if not sample.get("domain") or not sample.get("key_facts"):
            raise ValueError(f"Sample {sample.get('id')} needs domain and key_facts")
    domains = {str(item["domain"]) for item in samples}
    expected_samples = payload.get("expected_sample_count")
    expected_domains = payload.get("expected_domain_count")
    if expected_samples is not None and len(samples) != int(expected_samples):
        raise ValueError(
            f"Frozen dataset declares {expected_samples} samples but contains {len(samples)}"
        )
    if expected_domains is not None and len(domains) != int(expected_domains):
        raise ValueError(
            f"Frozen dataset declares {expected_domains} domains but contains {len(domains)}"
        )
    return payload


def _new_llm(
    provider: str,
    model: str | None,
    input_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
) -> InstrumentedLLM | None:
    if provider == "offline":
        return None
    resolved_provider: Any = auto_provider() if provider == "auto" else provider
    if resolved_provider is None:
        raise RuntimeError("No configured LLM provider; use --provider offline or configure a key")
    return InstrumentedLLM(
        build_llm(resolved_provider, model),
        input_cost_per_million_usd,
        output_cost_per_million_usd,
    )


def _build_agent(
    variant: str,
    sample: dict[str, Any],
    llm: InstrumentedLLM | None,
    args: argparse.Namespace,
) -> tuple[DeepResearchAgent, SharedMemory]:
    memory = SharedMemory(":memory:")
    planner: Planner = (
        SingleTaskPlanner()
        if variant == "single_agent"
        else (LLMPlanner(llm) if llm is not None else HeuristicPlanner())
    )
    reviewer = NoOpReviewer() if variant in {"single_agent", "no_red_blue"} else None
    verifier = ClaimEvidenceVerifier(llm)
    repairer = None
    if variant == "rewrite_blue":
        # Returning no structured patch deterministically exercises the existing
        # whole-report rewrite fallback while retaining the common final ledger.
        repairer = BlueTeamRepairer(
            llm=None,
            max_generation_attempts=1,
            verifier=verifier,
        )
    elif variant in {"structured_patch_blue", "fixed_harness", "dynamic_swarm"}:
        repairer = BlueTeamRepairer(
            llm=llm,
            max_patches=args.max_blue_patches,
            max_generation_attempts=args.max_patch_attempts,
            verifier=verifier,
        )

    config = ResearchConfig(
        max_concurrency=args.concurrency,
        task_timeout_seconds=args.task_timeout,
        global_timeout_seconds=args.global_timeout,
        max_replans=0,
        max_review_rounds=args.max_review_rounds,
        min_citation_coverage=args.min_citation_coverage,
        min_claim_support_rate=args.min_claim_support_rate,
        max_blue_patches_per_round=args.max_blue_patches,
        max_blue_patch_generation_attempts=args.max_patch_attempts,
        enable_semantic_claim_verification=llm is not None,
        enable_dynamic_swarm=variant == "dynamic_swarm",
        max_swarm_agents=args.max_swarm_agents,
        max_worker_invocations=args.max_worker_invocations,
        max_stagnant_review_rounds=args.max_stagnant_review_rounds,
    )
    agent = DeepResearchAgent(
        planner=planner,
        worker=FrozenEvidenceWorker(sample["evidence"]),
        synthesizer=Synthesizer(llm),
        memory=memory,
        reviewer=reviewer,
        repairer=repairer,
        config=config,
    )
    return agent, memory


async def _run_sample(
    variant: str,
    sample: dict[str, Any],
    llm: InstrumentedLLM | None,
    args: argparse.Namespace,
    context_key: str,
) -> dict[str, Any]:
    agent, memory = _build_agent(variant, sample, llm, args)
    started = time.perf_counter()
    token = RUN_CONTEXT.set(context_key)
    try:
        result = await agent.run(str(sample["question"]))
    finally:
        RUN_CONTEXT.reset(token)
        memory.close()
    elapsed = time.perf_counter() - started
    usage = llm.usage(context_key) if llm is not None else _zero_usage()

    unique_evidence: dict[str, dict[str, Any]] = {}
    for evidence in result.evidences:
        unique_evidence[evidence.evidence_id] = asdict(evidence)
    rules = evaluate_rules(result.answer, list(unique_evidence.values()))
    benchmark = evaluate_expectations(result.answer, sample)
    quality = overall_quality(rules, benchmark)
    role_invocations = Counter(
        str(item.get("role", "unknown"))
        for item in result.trace
        if item.get("event") == "agent_started"
    )
    serialized_result = result.to_dict()
    return {
        "sample_id": sample["id"],
        "domain": sample.get("domain"),
        "question": sample["question"],
        "status": result.status,
        "latency_seconds": elapsed,
        "llm_usage": usage,
        "evaluation": {
            "rules": asdict(rules),
            "benchmark": asdict(benchmark),
            "overall_quality": quality,
        },
        "system_metrics": result.metrics,
        "swarm_plan": result.run_metadata.get("swarm", {}),
        "role_invocations": dict(sorted(role_invocations.items())),
        "run_id": result.run_id,
        "report": result.answer,
        "review": serialized_result.get("review"),
        "claim_ledger": serialized_result.get("claim_ledger"),
        "evidences": list(unique_evidence.values()),
        "trace": result.trace,
    }


def _aggregate(rows: Sequence[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    successful = [row for row in rows if "error" not in row]
    metric_extractors = {
        "overall_quality": lambda row: row["evaluation"]["overall_quality"],
        "factual_accuracy": lambda row: row["evaluation"]["rules"]["factual_accuracy"],
        "hallucination_rate": lambda row: row["evaluation"]["rules"]["hallucination_rate"],
        "citation_coverage": lambda row: row["evaluation"]["rules"]["citation_coverage"],
        "citation_validity": lambda row: row["evaluation"]["rules"]["citation_validity"],
        "key_fact_recall": lambda row: row["evaluation"]["benchmark"]["key_fact_recall"],
        "rule_key_fact_recall": lambda row: row["evaluation"]["benchmark"].get(
            "rule_key_fact_recall",
            row["evaluation"]["benchmark"]["key_fact_recall"],
        ),
        "semantic_key_fact_recall": lambda row: row["evaluation"]["benchmark"].get(
            "semantic_key_fact_recall",
            row["evaluation"]["benchmark"]["key_fact_recall"],
        ),
        "forbidden_claim_rate": lambda row: row["evaluation"]["benchmark"]["forbidden_claim_rate"],
        "latency_seconds": lambda row: row["latency_seconds"],
        "llm_calls": lambda row: row["llm_usage"]["llm_calls"],
        "prompt_chars": lambda row: row["llm_usage"]["prompt_chars"],
        "completion_chars": lambda row: row["llm_usage"]["completion_chars"],
        "estimated_total_tokens": lambda row: row["llm_usage"]["estimated_total_tokens"],
        "estimated_cost_usd": lambda row: row["llm_usage"]["estimated_cost_usd"],
        "worker_invocations": lambda row: row["system_metrics"]["budget"]["used"]["worker_invocations"],
        "review_rounds": lambda row: row["system_metrics"]["budget"]["used"]["review_rounds"],
    }
    metrics: dict[str, Any] = {}
    for name, extractor in metric_extractors.items():
        values = [float(extractor(row)) for row in successful]
        if not values:
            continue
        by_question: dict[str, list[float]] = {}
        for row in successful:
            by_question.setdefault(str(row["sample_id"]), []).append(float(extractor(row)))
        question_means = [statistics.fmean(group) for group in by_question.values()]
        lower, upper = bootstrap_ci(question_means, samples=bootstrap_samples)
        metrics[name] = {
            "mean": statistics.fmean(values),
            "bootstrap_95_ci": [lower, upper],
            "ci_unit": "question_cluster_mean",
        }
    role_totals: Counter[str] = Counter()
    stop_reason_totals: Counter[str] = Counter()
    for row in successful:
        role_totals.update(row.get("role_invocations", {}))
        stop_reason_totals.update(
            row.get("system_metrics", {}).get("budget", {}).get("stop_reasons", [])
        )
    return {
        "total": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "independent_questions": len({row["sample_id"] for row in successful}),
        "metrics": metrics,
        "role_invocations": dict(sorted(role_totals.items())),
        "stop_reasons": dict(sorted(stop_reason_totals.items())),
    }


def _comparisons(
    variant_rows: dict[str, list[dict[str, Any]]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    def question_metric(rows: Sequence[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            if "error" in row:
                continue
            value: Any = row
            for key in path:
                value = value[key]
            grouped.setdefault(str(row["sample_id"]), []).append(float(value))
        return {key: statistics.fmean(values) for key, values in grouped.items()}

    pairs = [
        ("single_agent", "no_red_blue"),
        ("single_agent", "rewrite_blue"),
        ("single_agent", "structured_patch_blue"),
        ("no_red_blue", "structured_patch_blue"),
        ("rewrite_blue", "structured_patch_blue"),
        ("fixed_harness", "dynamic_swarm"),
    ]
    metric_paths = {
        "overall_quality": ("evaluation", "overall_quality"),
        "factual_accuracy": ("evaluation", "rules", "factual_accuracy"),
        "hallucination_rate": ("evaluation", "rules", "hallucination_rate"),
        "citation_coverage": ("evaluation", "rules", "citation_coverage"),
        "key_fact_recall": ("evaluation", "benchmark", "key_fact_recall"),
        "rule_key_fact_recall": (
            "evaluation", "benchmark", "rule_key_fact_recall"
        ),
        "semantic_key_fact_recall": (
            "evaluation", "benchmark", "semantic_key_fact_recall"
        ),
        "latency_seconds": ("latency_seconds",),
        "llm_calls": ("llm_usage", "llm_calls"),
        "estimated_total_tokens": ("llm_usage", "estimated_total_tokens"),
        "estimated_cost_usd": ("llm_usage", "estimated_cost_usd"),
        "worker_invocations": (
            "system_metrics",
            "budget",
            "used",
            "worker_invocations",
        ),
        "review_rounds": (
            "system_metrics",
            "budget",
            "used",
            "review_rounds",
        ),
    }
    comparisons: dict[str, Any] = {}
    for baseline_name, candidate_name in pairs:
        if baseline_name not in variant_rows or candidate_name not in variant_rows:
            continue
        key = f"{candidate_name}_vs_{baseline_name}"
        base_quality = question_metric(
            variant_rows[baseline_name], metric_paths["overall_quality"]
        )
        candidate_quality = question_metric(
            variant_rows[candidate_name], metric_paths["overall_quality"]
        )
        common = sorted(set(base_quality) & set(candidate_quality))
        if len(common) < 2:
            comparisons[key] = {"paired_questions": len(common), "insufficient": True}
            continue
        metric_results: dict[str, Any] = {}
        for metric_name, path in metric_paths.items():
            base_map = question_metric(variant_rows[baseline_name], path)
            candidate_map = question_metric(variant_rows[candidate_name], path)
            metric_common = sorted(set(base_map) & set(candidate_map))
            base_values = [base_map[item] for item in metric_common]
            candidate_values = [candidate_map[item] for item in metric_common]
            deltas = [right - left for left, right in zip(base_values, candidate_values)]
            lower, upper = bootstrap_ci(deltas, samples=bootstrap_samples)
            base_mean = statistics.fmean(base_values)
            delta_mean = statistics.fmean(deltas)
            metric_results[metric_name] = {
                "baseline_mean": base_mean,
                "candidate_mean": statistics.fmean(candidate_values),
                "paired_delta_mean": delta_mean,
                "paired_delta_bootstrap_95_ci": [lower, upper],
                "relative_change_percent": (
                    100.0 * delta_mean / base_mean if base_mean else None
                ),
                "paired_cohens_dz": paired_cohens_d(base_values, candidate_values),
            }
        comparisons[key] = {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "paired_questions": len(common),
            "metrics": metric_results,
        }
    return comparisons


async def run(args: argparse.Namespace) -> Path:
    load_env_files((BASE_DIR / ".env", EVALUATION_DIR / ".env"))
    dataset_path = Path(args.dataset)
    dataset = load_frozen_dataset(dataset_path)
    samples = list(dataset["samples"])
    if args.sample_id:
        requested = set(args.sample_id)
        known = {str(sample["id"]) for sample in samples}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"Unknown --sample-id values: {missing}")
        samples = [sample for sample in samples if str(sample["id"]) in requested]
    samples = samples[: args.limit or None]
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    resume_payloads: list[dict[str, Any]] = []
    resume_paths = [Path(item) for item in args.resume_from]
    for resume_path in resume_paths:
        loaded_resume: Any = json.loads(resume_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_resume, dict):
            raise ValueError("Resume result must be a JSON object")
        if loaded_resume.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Resume result uses a different Frozen dataset SHA-256")
        if loaded_resume.get("provider") != args.provider:
            raise ValueError("Resume result uses a different provider")
        if loaded_resume.get("model") != args.model:
            raise ValueError("Resume result uses a different model")
        if int(loaded_resume.get("repeats", 0)) != args.repeats:
            raise ValueError("Resume result uses a different repeat count")
        resume_payloads.append(loaded_resume)
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    resume_stats: dict[str, dict[str, int]] = {}

    for variant in args.variants:
        prior_success: dict[str, dict[str, Any]] = {}
        rejected_cached_rows = 0
        for resume_payload in resume_payloads:
            for row in (
                resume_payload.get("variants", {})
                .get(variant, {})
                .get("samples", [])
            ):
                if "error" in row:
                    continue
                if not _resume_row_is_compatible(variant, row):
                    rejected_cached_rows += 1
                    continue
                repriced = _reprice_row(
                    row,
                    args.input_cost_per_million_usd,
                    args.output_cost_per_million_usd,
                )
                prior_success[str(repriced["pair_key"])] = repriced
        expected = [
            (repeat, sample, f"{sample.get('id')}#r{repeat}")
            for repeat in range(1, args.repeats + 1)
            for sample in samples
        ]
        pending = [item for item in expected if item[2] not in prior_success]
        resume_stats[variant] = {
            "reused_successful_rows": len(prior_success),
            "rejected_incompatible_rows": rejected_cached_rows,
            "new_rows_requested": len(pending),
        }
        llm = (
            _new_llm(
                args.provider,
                args.model,
                args.input_cost_per_million_usd,
                args.output_cost_per_million_usd,
            )
            if pending
            else None
        )
        semaphore = asyncio.Semaphore(args.experiment_concurrency)

        async def execute(repeat: int, sample: dict[str, Any]) -> dict[str, Any]:
            pair_key = f"{sample.get('id')}#r{repeat}"
            context_key = f"{variant}:{pair_key}"
            async with semaphore:
                try:
                    row = await _run_sample(variant, sample, llm, args, context_key)
                    row["repeat"] = repeat
                    row["pair_key"] = pair_key
                    return row
                except Exception as exc:
                    return {
                        "sample_id": sample.get("id"),
                        "domain": sample.get("domain"),
                        "pair_key": pair_key,
                        "repeat": repeat,
                        "question": sample.get("question"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        jobs = [execute(repeat, sample) for repeat, sample, _ in pending]
        new_rows = list(await asyncio.gather(*jobs))
        merged = {str(row["pair_key"]): row for row in new_rows}
        merged.update(prior_success)
        rows = [merged[pair_key] for _, _, pair_key in expected]
        variant_rows[variant] = rows

    created_at = datetime.now(timezone.utc)
    payload = {
        "run_id": created_at.strftime("ablation-%Y%m%dT%H%M%S%fZ"),
        "created_at": created_at.isoformat(),
        "track": "frozen",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "selected_sample_ids": [str(sample["id"]) for sample in samples],
        "provider": args.provider,
        "model": args.model,
        "repeats": args.repeats,
        "swarm_policy": {
            "name": "HeuristicSwarmPolicy",
            "version": HeuristicSwarmPolicy.VERSION,
            "simple_threshold": HeuristicSwarmPolicy.SIMPLE_THRESHOLD,
            "complex_threshold": HeuristicSwarmPolicy.COMPLEX_THRESHOLD,
        },
        "cost_assumptions": {
            "token_estimation": "CJK characters + ceil(non-CJK non-space characters / 4)",
            "input_cost_per_million_usd": args.input_cost_per_million_usd,
            "output_cost_per_million_usd": args.output_cost_per_million_usd,
        },
        "resumed_from": [str(path.resolve()) for path in resume_paths],
        "resume_stats": resume_stats,
        "interpretation": (
            "offline 模式只验证流水线和冻结数据契约；各变体不会调用模型，"
            "因此不得用其质量差异作为消融结论。"
            if args.provider == "offline"
            else "各变体在相同冻结证据和问题上成对比较。"
        ),
        "variants": {
            variant: {
                "description": VARIANT_DESCRIPTIONS[variant],
                "aggregate": _aggregate(rows, args.bootstrap_samples),
                "samples": rows,
            }
            for variant, rows in variant_rows.items()
        },
        "comparisons": _comparisons(variant_rows, args.bootstrap_samples),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['run_id']}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        variant: data["aggregate"]
        for variant, data in payload["variants"].items()
    }
    print(json.dumps({"run_id": payload["run_id"], "summary": summary}, ensure_ascii=False, indent=2))
    print(f"result_file={output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Deep Research ablations on ResearchBench-Frozen"
    )
    parser.add_argument(
        "--dataset",
        default=str(EVALUATION_DIR / "datasets" / "researchbench_frozen_v1.0.json"),
    )
    parser.add_argument("--output-dir", default=str(EVALUATION_DIR / "results" / "ablations"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_DESCRIPTIONS),
        default=list(VARIANT_DESCRIPTIONS),
    )
    parser.add_argument(
        "--provider",
        choices=("offline", "auto", "deepseek", "mimo", "vllm", "openai"),
        default="offline",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--resume-from",
        action="append",
        default=[],
        help="Reuse successful pair_key rows from a compatible earlier result and rerun failures",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Run only the selected benchmark sample ID; repeatable",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--experiment-concurrency", type=int, default=4)
    parser.add_argument("--task-timeout", type=float, default=45.0)
    parser.add_argument("--global-timeout", type=float, default=240.0)
    parser.add_argument("--max-review-rounds", type=int, default=4)
    parser.add_argument("--min-citation-coverage", type=float, default=0.80)
    parser.add_argument("--min-claim-support-rate", type=float, default=0.80)
    parser.add_argument("--max-blue-patches", type=int, default=20)
    parser.add_argument("--max-patch-attempts", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-swarm-agents", type=int, default=5)
    parser.add_argument("--max-worker-invocations", type=int, default=16)
    parser.add_argument("--max-stagnant-review-rounds", type=int, default=2)
    parser.add_argument(
        "--input-cost-per-million-usd",
        type=float,
        default=float(os.getenv("RESEARCH_INPUT_COST_PER_MILLION_USD", "0")),
    )
    parser.add_argument(
        "--output-cost-per-million-usd",
        type=float,
        default=float(os.getenv("RESEARCH_OUTPUT_COST_PER_MILLION_USD", "0")),
    )
    try:
        args = parser.parse_args()
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        if args.limit < 0:
            raise ValueError("--limit must be >= 0")
        if args.bootstrap_samples < 1:
            raise ValueError("--bootstrap-samples must be >= 1")
        if args.experiment_concurrency < 1:
            raise ValueError("--experiment-concurrency must be >= 1")
        if args.input_cost_per_million_usd < 0 or args.output_cost_per_million_usd < 0:
            raise ValueError("cost rates must be >= 0")
        asyncio.run(run(args))
    except (RuntimeError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
