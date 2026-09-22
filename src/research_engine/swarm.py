"""Adaptive swarm composition and per-run resource budgets."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

from .config import ResearchConfig
from .harness import AgentSpec
from .schemas import Evidence, ResearchPlan, ResearchSubtask, ReviewResult


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class BudgetResource(str, Enum):
    WORKER_INVOCATION = "worker_invocations"
    REPLAN = "replans"
    REVIEW_ROUND = "review_rounds"
    VERIFICATION_QUERY = "verification_queries"


@dataclass(frozen=True)
class SwarmPlan:
    """Run-scoped execution shape selected after the research DAG is known."""

    level: ComplexityLevel
    complexity_score: float
    signals: tuple[str, ...]
    agents: tuple[AgentSpec, ...]
    max_concurrency: int
    max_worker_invocations: int
    max_replans: int
    max_review_rounds: int
    max_verification_queries: int
    max_elapsed_seconds: float
    diminishing_returns_patience: int
    evidence_stagnation_patience: int
    min_score_improvement: float

    def __post_init__(self) -> None:
        if not 0 <= self.complexity_score <= 1:
            raise ValueError("complexity_score must be between 0 and 1")
        if not self.agents:
            raise ValueError("SwarmPlan requires at least one agent")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_worker_invocations < 1 or self.max_review_rounds < 1:
            raise ValueError("worker and review budgets must be >= 1")
        if self.max_replans < 0 or self.max_verification_queries < 0:
            raise ValueError("replan and verification budgets must be >= 0")
        if self.max_elapsed_seconds <= 0:
            raise ValueError("max_elapsed_seconds must be positive")
        if self.diminishing_returns_patience < 1:
            raise ValueError("diminishing_returns_patience must be >= 1")
        if self.evidence_stagnation_patience < 0:
            raise ValueError("evidence_stagnation_patience must be >= 0")
        if self.min_score_improvement < 0:
            raise ValueError("min_score_improvement must be >= 0")

    def select_agent(
        self,
        task: ResearchSubtask,
        fallback: AgentSpec,
    ) -> AgentSpec:
        """Route a task to the most specialized active role."""

        by_role = {agent.role: agent for agent in self.agents}
        task_text = f"{task.subtask_id} {task.kind} {task.question}".lower()
        if task.subtask_id.startswith("verify_"):
            preferred = "evidence_verifier"
        elif any(token in task_text for token in ("alternative", "counter", "反方", "反例", "限制")):
            preferred = "counter_researcher"
        elif any(token in task_text for token in ("implication", "risk", "影响", "风险", "建议")):
            preferred = "impact_analyst"
        elif any(token in task_text for token in ("scope", "边界", "概念", "定义")):
            preferred = "scope_researcher"
        else:
            preferred = "evidence_researcher"
        return (
            by_role.get(preferred)
            or by_role.get("evidence_researcher")
            or (self.agents[0] if self.agents else fallback)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "complexity_score": self.complexity_score,
            "signals": list(self.signals),
            "agent_count": len(self.agents),
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "tool_names": list(agent.tool_names),
                }
                for agent in self.agents
            ],
            "max_concurrency": self.max_concurrency,
            "max_worker_invocations": self.max_worker_invocations,
            "max_replans": self.max_replans,
            "max_review_rounds": self.max_review_rounds,
            "max_verification_queries": self.max_verification_queries,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "diminishing_returns_patience": self.diminishing_returns_patience,
            "evidence_stagnation_patience": self.evidence_stagnation_patience,
            "min_score_improvement": self.min_score_improvement,
        }


class SwarmPolicy(Protocol):
    def decide(
        self,
        question: str,
        plan: ResearchPlan,
        config: ResearchConfig,
        base_worker: AgentSpec,
    ) -> SwarmPlan: ...


class HeuristicSwarmPolicy:
    """Deterministic policy that is cheap, inspectable and evaluation-friendly."""

    VERSION = "heuristic-v4"
    SIMPLE_THRESHOLD = 0.23
    COMPLEX_THRESHOLD = 0.72

    _comparison_terms = ("比较", "对比", "区别", "versus", " vs ", "权衡")
    _temporal_terms = ("最新", "当前", "近期", "实时", "today", "latest", "current")
    _high_stakes_terms = (
        "法律",
        "医疗",
        "临床",
        "金融",
        "合规",
        "安全",
        "legal",
        "medical",
        "financial",
    )
    _breadth_terms = ("风险", "反例", "局限", "影响", "建议", "多来源", "争议", "证据")

    def decide(
        self,
        question: str,
        plan: ResearchPlan,
        config: ResearchConfig,
        base_worker: AgentSpec,
    ) -> SwarmPlan:
        if not config.enable_dynamic_swarm:
            return SwarmPlan(
                level=ComplexityLevel.STANDARD,
                complexity_score=0.5,
                signals=("dynamic_swarm_disabled",),
                agents=(base_worker,),
                max_concurrency=config.max_concurrency,
                max_worker_invocations=config.max_worker_invocations,
                max_replans=config.max_replans,
                max_review_rounds=config.max_review_rounds,
                max_verification_queries=(
                    config.max_verification_queries_per_round
                    * config.max_review_rounds
                ),
                max_elapsed_seconds=config.global_timeout_seconds,
                diminishing_returns_patience=config.max_stagnant_review_rounds,
                evidence_stagnation_patience=0,
                min_score_improvement=config.min_score_improvement,
            )

        score, signals = self._score(question, plan)
        # LLM planners commonly emit six or more subtasks even for a focused
        # lookup, which contributes 0.22 before question-level signals.  A
        # 0.23 boundary includes that focused 0.22 case but leaves questions
        # with any additional complexity signal in the three-agent tier.
        if score < self.SIMPLE_THRESHOLD:
            level = ComplexityLevel.SIMPLE
            desired_concurrency = 1
            # Agent composition and review depth are separate controls.  A
            # focused question only needs one retrieval role, but still gets a
            # third Red/Blue round and enough verification headroom to avoid
            # the quality regression observed when both were reduced at once.
            review_rounds = 3
            verification_queries = 4
            invocation_headroom = 4
            elapsed_fraction = 0.75
            evidence_stagnation_patience = 1
        # A medium comparison planned as a six-to-eight task DAG typically
        # lands around 0.52-0.60.  Reserve the five-agent tier for questions
        # with several strong signals (for example current + high-stakes +
        # multi-dimensional), rather than promoting on DAG breadth alone.
        elif score < self.COMPLEX_THRESHOLD:
            level = ComplexityLevel.STANDARD
            desired_concurrency = 3
            review_rounds = 3
            verification_queries = 6
            invocation_headroom = 6
            elapsed_fraction = 0.80
            evidence_stagnation_patience = 2
        else:
            level = ComplexityLevel.COMPLEX
            desired_concurrency = 5
            review_rounds = config.max_review_rounds
            verification_queries = (
                config.max_verification_queries_per_round * review_rounds
            )
            invocation_headroom = 10
            elapsed_fraction = 1.0
            evidence_stagnation_patience = 3

        agents = self._agents(level, base_worker)[: config.max_swarm_agents]
        max_concurrency = max(
            1,
            min(config.max_concurrency, desired_concurrency, len(agents)),
        )
        max_worker_invocations = min(
            config.max_worker_invocations,
            len(plan.subtasks) + invocation_headroom,
        )
        max_review_rounds = min(config.max_review_rounds, review_rounds)
        max_verification_queries = min(
            verification_queries,
            config.max_verification_queries_per_round
            * max(0, max_review_rounds - 1),
        )
        return SwarmPlan(
            level=level,
            complexity_score=score,
            signals=tuple(signals),
            agents=tuple(agents),
            max_concurrency=max_concurrency,
            max_worker_invocations=max_worker_invocations,
            max_replans=config.max_replans,
            max_review_rounds=max_review_rounds,
            max_verification_queries=max_verification_queries,
            max_elapsed_seconds=min(
                config.global_timeout_seconds,
                max(
                    config.task_timeout_seconds * 2,
                    config.global_timeout_seconds * elapsed_fraction,
                ),
            ),
            diminishing_returns_patience=config.max_stagnant_review_rounds,
            evidence_stagnation_patience=evidence_stagnation_patience,
            min_score_improvement=config.min_score_improvement,
        )

    def _score(self, question: str, plan: ResearchPlan) -> tuple[float, list[str]]:
        normalized = question.lower()
        score = 0.0
        signals: list[str] = []

        if len(plan.subtasks) >= 6:
            score += 0.22
            signals.append("large_dag")
        elif len(plan.subtasks) >= 4:
            score += 0.08
            signals.append("multi_stage_dag")
        if len(question) >= 120:
            score += 0.18
            signals.append("long_question")
        elif len(question) >= 60:
            score += 0.10
            signals.append("medium_question")
        if any(term in normalized for term in self._comparison_terms):
            score += 0.14
            signals.append("comparison")
        if any(term in normalized for term in self._temporal_terms):
            score += 0.12
            signals.append("time_sensitive")
        if any(term in normalized for term in self._high_stakes_terms):
            score += 0.18
            signals.append("high_stakes")
        breadth_hits = sum(term in normalized for term in self._breadth_terms)
        if breadth_hits >= 3:
            score += 0.20
            signals.append("multi_dimensional")
        elif breadth_hits >= 1:
            score += 0.08
            signals.append("additional_dimensions")
        clause_count = len(re.findall(r"[，,；;、]|以及|并且|同时", question))
        if clause_count >= 3:
            score += 0.12
            signals.append("many_constraints")
        elif clause_count >= 1:
            score += 0.06
            signals.append("compound_question")
        if any(term in normalized for term in ("分别", "哪些", "为何", "为什么", "又", "以及")):
            score += 0.08
            signals.append("multi_part_question")
        return round(min(score, 1.0), 3), signals or ["focused_question"]

    @staticmethod
    def _agents(level: ComplexityLevel, base: AgentSpec) -> list[AgentSpec]:
        def specialized(suffix: str, role: str, description: str) -> AgentSpec:
            return AgentSpec(
                agent_id=f"{base.agent_id}-{suffix}",
                role=role,
                description=description,
                tool_names=base.tool_names,
                model=base.model,
                metadata={**base.metadata, "swarm_managed": True},
            )

        if level is ComplexityLevel.SIMPLE:
            return [base]
        agents = [
            specialized("scope", "scope_researcher", "Clarify scope and definitions"),
            specialized("evidence", "evidence_researcher", "Gather primary evidence"),
            specialized("counter", "counter_researcher", "Find counter-evidence and limits"),
        ]
        if level is ComplexityLevel.COMPLEX:
            agents.extend(
                [
                    specialized("impact", "impact_analyst", "Analyze risks and trade-offs"),
                    specialized("verify", "evidence_verifier", "Verify disputed claims"),
                ]
            )
        return agents


@dataclass
class BudgetController:
    """Deterministic, run-local accounting and stopping controller."""

    swarm: SwarmPlan
    started_at: float = field(default_factory=time.monotonic)
    used: dict[str, int] = field(init=False)
    exhausted: set[str] = field(default_factory=set, init=False)
    stop_reasons: list[str] = field(default_factory=list)
    review_scores: list[float] = field(default_factory=list)
    claim_support_rates: list[float] = field(default_factory=list)
    quality_utility_scores: list[float] = field(default_factory=list)
    quality_guardrail_actions: list[str] = field(default_factory=list)
    _stagnant_reviews: int = 0
    _stagnant_evidence_batches: int = 0
    _evidence_fingerprints: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.used = {resource.value: 0 for resource in BudgetResource}

    @property
    def limits(self) -> dict[str, int]:
        return {
            BudgetResource.WORKER_INVOCATION.value: self.swarm.max_worker_invocations,
            BudgetResource.REPLAN.value: self.swarm.max_replans,
            BudgetResource.REVIEW_ROUND.value: self.swarm.max_review_rounds,
            BudgetResource.VERIFICATION_QUERY.value: self.swarm.max_verification_queries,
        }

    def remaining(self, resource: BudgetResource | str) -> int:
        name = BudgetResource(resource).value
        return max(0, self.limits[name] - self.used[name])

    def try_consume(self, resource: BudgetResource | str, amount: int = 1) -> bool:
        if amount < 0:
            raise ValueError("Budget amount must be non-negative")
        name = BudgetResource(resource).value
        if self.deadline_exceeded():
            return False
        if amount > self.remaining(name):
            self.exhausted.add(name)
            return False
        self.used[name] += amount
        return True

    def deadline_exceeded(self) -> bool:
        if self.elapsed_seconds >= self.swarm.max_elapsed_seconds:
            self.mark_stop("deadline_exceeded")
            return True
        return False

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def observe_review(
        self,
        review: ReviewResult,
        *,
        quality_gate_passed: bool = True,
        claim_support_rate: float | None = None,
    ) -> str | None:
        self.review_scores.append(review.total_score)
        support_rate = (
            max(0.0, min(1.0, claim_support_rate))
            if claim_support_rate is not None
            else (1.0 if quality_gate_passed else 0.0)
        )
        cost_ratio = max(
            (
                self.used[name] / limit
                for name, limit in self.limits.items()
                if limit > 0
            ),
            default=0.0,
        )
        time_ratio = min(1.0, self.elapsed_seconds / self.swarm.max_elapsed_seconds)
        quality = 0.55 * review.total_score + 0.45 * support_rate
        self.quality_utility_scores.append(
            round(quality - 0.08 * cost_ratio - 0.07 * time_ratio, 6)
        )
        if review.passed and quality_gate_passed:
            return self.mark_stop("review_passed")
        if review.converged and quality_gate_passed:
            return self.mark_stop("review_converged")
        if review.oscillating:
            return self.mark_stop("review_oscillating")
        if len(self.quality_utility_scores) >= 2:
            # Utility rewards Red/Claim quality while charging for consumed
            # calls (proxy cost) and wall-clock budget. Diminishing-return
            # stopping therefore cannot optimize call count in isolation.
            improvement = (
                self.quality_utility_scores[-1]
                - self.quality_utility_scores[-2]
            )
            if improvement < self.swarm.min_score_improvement:
                self._stagnant_reviews += 1
            else:
                self._stagnant_reviews = 0
            if (
                quality_gate_passed
                and not any(
                    issue.category
                    in {"factual_error", "inference_overreach", "citation_error"}
                    and issue.severity > 1
                    for issue in review.structured_issues
                )
                and self._stagnant_reviews
                >= self.swarm.diminishing_returns_patience
            ):
                return self.mark_stop("diminishing_returns")
        return None

    def observe_claim_support(
        self,
        claim_support_rate: float,
        *,
        minimum: float,
        drop_tolerance: float = 0.05,
    ) -> str | None:
        """Escalate verification before a dynamic swarm can trade away quality."""

        rate = max(0.0, min(1.0, claim_support_rate))
        previous = self.claim_support_rates[-1] if self.claim_support_rates else None
        self.claim_support_rates.append(rate)
        action = None
        if previous is not None and previous - rate > drop_tolerance:
            action = "fallback_fixed_harness"
        elif rate < minimum:
            action = (
                "fallback_fixed_harness"
                if "add_evidence_verifier" in self.quality_guardrail_actions
                else "add_evidence_verifier"
            )
        if action and action not in self.quality_guardrail_actions:
            self.quality_guardrail_actions.append(action)
        return action

    def observe_evidence_batch(
        self,
        evidences: Sequence[Evidence],
    ) -> str | None:
        if not evidences or self.swarm.evidence_stagnation_patience == 0:
            return None
        fingerprints = {
            re.sub(r"\s+", " ", evidence.content).strip().casefold()
            for evidence in evidences
            if evidence.content.strip()
        }
        novel = fingerprints - self._evidence_fingerprints
        self._evidence_fingerprints.update(fingerprints)
        if novel:
            self._stagnant_evidence_batches = 0
            return None
        self._stagnant_evidence_batches += 1
        if self._stagnant_evidence_batches >= self.swarm.evidence_stagnation_patience:
            return self.mark_stop("evidence_saturated")
        return None

    def mark_stop(self, reason: str) -> str:
        if reason not in self.stop_reasons:
            self.stop_reasons.append(reason)
        return reason

    def mark_exhausted(
        self,
        resource: BudgetResource | str,
        reason: str,
    ) -> str:
        self.exhausted.add(BudgetResource(resource).value)
        return self.mark_stop(reason)

    def snapshot(self) -> dict[str, Any]:
        limits = self.limits
        return {
            "limits": limits,
            "used": dict(self.used),
            "remaining": {
                name: max(0, limit - self.used[name])
                for name, limit in limits.items()
            },
            "exhausted": sorted(self.exhausted),
            "stop_reasons": list(self.stop_reasons),
            "review_scores": list(self.review_scores),
            "claim_support_rates": list(self.claim_support_rates),
            "quality_utility_scores": list(self.quality_utility_scores),
            "quality_guardrail_actions": list(self.quality_guardrail_actions),
            "proxy_cost_units": round(
                self.used[BudgetResource.WORKER_INVOCATION.value]
                + self.used[BudgetResource.VERIFICATION_QUERY.value]
                + 2 * self.used[BudgetResource.REVIEW_ROUND.value],
                3,
            ),
            "unique_evidence_count": len(self._evidence_fingerprints),
            "stagnant_evidence_batches": self._stagnant_evidence_batches,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }
