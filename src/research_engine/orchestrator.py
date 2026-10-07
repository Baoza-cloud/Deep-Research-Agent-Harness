"""Async DAG orchestration for the full deep-research lifecycle."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

from .adversarial import (
    BlueTeamRepairer,
    RedTeamReviewer,
    ReviewConvergence,
    is_blocking_review_issue,
)
from .agents import Synthesizer
from .claims import ClaimEvidenceVerifier, extract_claims, is_reviewable_claim
from .comparison import build_comparison_coverage, enforce_comparison_coverage
from .config import ResearchConfig
from .harness import (
    AgentRuntime,
    AgentSpec,
    LocalAgentRuntime,
    RoleAwareAgentAdapter,
    RunContext,
    ToolRegistry,
)
from .memory import SharedMemory
from .planner import Planner
from .persistence import (
    PersistedTrace,
    RunAlreadyExistsError,
    RunCheckpoint,
    RunStore,
)
from .schemas import (
    ClaimLedger,
    CompletionStatus,
    Evidence,
    ResearchPlan,
    ResearchResult,
    ResearchSubtask,
    ReviewResult,
    TaskStatus,
    utc_now,
)
from .state import ResearchRunState, TaskRuntime
from .swarm import (
    BudgetController,
    BudgetResource,
    HeuristicSwarmPolicy,
    SwarmPlan,
    SwarmPolicy,
)
from .text_quality import remove_omission_markers
from .tracing import (
    TRACE_SCHEMA_VERSION,
    duration_ms,
    new_trace_id,
    stable_span_id,
    summarize_value,
    usage_delta,
    usage_snapshot,
)
from .versioning import engine_version, source_commit


@dataclass
class TaskExecution:
    runtime: TaskRuntime
    evidences: list[Evidence]
    succeeded: bool
    failure_kind: str | None = None


@dataclass
class ReportQualityCandidate:
    report: str
    ledger: ClaimLedger
    review: ReviewResult | None
    origin: str

    @property
    def claim_support_rate(self) -> float:
        return float(self.ledger.metrics.get("claim_support_rate", 0.0))

    @property
    def citation_coverage(self) -> float:
        if self.review is None:
            return 0.0
        return float(self.review.metrics.get("citation_coverage", 0.0))

    def blocking_issue_count(self, max_pass_issue_severity: int) -> int:
        return sum(
            is_blocking_review_issue(issue, max_pass_issue_severity)
            for issue in (self.review.structured_issues if self.review else [])
        )


def select_quality_candidate(
    primary: ReportQualityCandidate,
    fallback: ReportQualityCandidate,
    *,
    min_claim_support_rate: float,
    min_citation_coverage: float,
    max_pass_issue_severity: int,
    support_drop_tolerance: float,
) -> tuple[ReportQualityCandidate, str]:
    """Choose only by report quality; efficiency savings never override defects."""

    def hard_pass(candidate: ReportQualityCandidate) -> bool:
        return (
            metric_gate(candidate) and candidate.blocking_issue_count(max_pass_issue_severity) == 0
        )

    def metric_gate(candidate: ReportQualityCandidate) -> bool:
        return (
            candidate.claim_support_rate >= min_claim_support_rate
            and candidate.citation_coverage >= min_citation_coverage
        )

    primary_passes = hard_pass(primary)
    fallback_passes = hard_pass(fallback)
    primary_metrics_pass = metric_gate(primary)
    fallback_metrics_pass = metric_gate(fallback)
    if primary_metrics_pass and not fallback_metrics_pass:
        return primary, "fallback_rejected_by_hard_metrics"
    if fallback_metrics_pass and not primary_metrics_pass:
        return fallback, "fallback_repairs_hard_metrics"
    if fallback_passes and not primary_passes:
        return fallback, "fallback_clears_hard_quality_gate"
    if primary_passes and not fallback_passes:
        return primary, "dynamic_clears_hard_quality_gate"

    support_gain = fallback.claim_support_rate - primary.claim_support_rate
    if (
        support_gain > support_drop_tolerance
        and fallback.citation_coverage >= min_citation_coverage
        and fallback.blocking_issue_count(max_pass_issue_severity)
        <= primary.blocking_issue_count(max_pass_issue_severity)
    ):
        return fallback, "fallback_repairs_claim_support_drop"

    def quality_rank(candidate: ReportQualityCandidate) -> tuple[float, ...]:
        review_score = candidate.review.total_score if candidate.review else 0.0
        return (
            float(hard_pass(candidate)),
            -float(candidate.blocking_issue_count(max_pass_issue_severity)),
            min(candidate.claim_support_rate, candidate.citation_coverage),
            candidate.claim_support_rate,
            candidate.citation_coverage,
            review_score,
        )

    if quality_rank(fallback) > quality_rank(primary):
        return fallback, "fallback_wins_quality_comparison"
    return primary, "dynamic_wins_quality_comparison"


def adjudicate_red_with_claim_ledger(
    review: ReviewResult | None,
    ledger: ClaimLedger,
) -> list[dict[str, Any]]:
    """Downgrade severity-2 Red disputes contradicted by a perfect ledger.

    The semantic Claim Verifier is authoritative only when every reviewable
    claim is supported and no typed conflict is present. Severity-3 Red issues
    are never downgraded. Original issue details are returned for tracing.
    """

    if review is None or ledger.verification_mode != "semantic":
        return []
    metrics = ledger.metrics
    reviewable = float(metrics.get("reviewable_claim_count", 0.0))
    supported = float(metrics.get("supported_claim_count", 0.0))
    conflict_count = sum(
        float(metrics.get(name, 0.0))
        for name in (
            "contradicted_claim_count",
            "time_conflict_count",
            "number_conflict_count",
            "entity_conflict_count",
        )
    )
    if reviewable <= 0 or supported != reviewable or conflict_count > 0:
        return []

    adjudicated: list[dict[str, Any]] = []
    for issue in review.structured_issues:
        if issue.severity != 2 or issue.category not in {
            "factual_error",
            "inference_overreach",
            "citation_error",
        }:
            continue
        adjudicated.append(
            {
                "issue_id": issue.issue_id,
                "original_category": issue.category,
                "original_severity": issue.severity,
                "description": issue.description,
                "target": issue.target,
                "reason": "semantic_claim_ledger_full_support_without_conflicts",
            }
        )
        issue.category = "writing_advice"
        issue.severity = 1
    return adjudicated


def refresh_review_after_deterministic_claim_patch(
    review: ReviewResult | None,
    report: str,
    *,
    pass_score: float,
    min_citation_coverage: float,
    max_pass_issue_severity: int,
) -> ReviewResult | None:
    """Refresh deterministic gates without another whole-report LLM review."""

    if review is None:
        return None
    claims = [claim for claim in extract_claims(report, min_chars=6) if is_reviewable_claim(claim)]
    cited = sum(bool(claim.citations) for claim in claims)
    coverage = cited / len(claims) if claims else 1.0
    retained_issues = []
    for issue in review.structured_issues:
        if issue.issue_id == "rule-citation-coverage" and coverage >= min_citation_coverage:
            continue
        target = (issue.target or "").strip()
        if (
            target
            and issue.category in {"factual_error", "inference_overreach", "citation_error"}
            and target not in report
            and target not in {"未引用的事实陈述", "局限与不确定性"}
        ):
            continue
        retained_issues.append(issue)
    review.structured_issues = retained_issues
    review.issues = [issue.description for issue in retained_issues]
    review.missing_information = [
        issue.description for issue in retained_issues if issue.action.value == "ADD"
    ]
    review.metrics["citation_coverage"] = coverage
    review.metrics["claim_count"] = float(len(claims))
    review.metrics["uncited_claim_count"] = float(len(claims) - cited)
    review.passed = (
        review.total_score >= pass_score
        and coverage >= min_citation_coverage
        and not any(
            is_blocking_review_issue(issue, max_pass_issue_severity) for issue in retained_issues
        )
    )
    return review


class ResearchWorkerProtocol(Protocol):
    async def run(self, task: ResearchSubtask) -> list[Evidence]: ...


class ReviewerProtocol(Protocol):
    async def review(
        self,
        question: str,
        report: str,
        evidences: Sequence[Evidence],
    ) -> Any: ...


class DeepResearchAgent:
    """Coordinates planning, concurrent research, memory, review and repair."""

    def __init__(
        self,
        planner: Planner,
        worker: ResearchWorkerProtocol,
        synthesizer: Synthesizer,
        memory: SharedMemory | None = None,
        reviewer: ReviewerProtocol | None = None,
        repairer: BlueTeamRepairer | None = None,
        config: ResearchConfig | None = None,
        runtime: AgentRuntime | None = None,
        tool_registry: ToolRegistry | None = None,
        worker_spec: AgentSpec | None = None,
        swarm_policy: SwarmPolicy | None = None,
        run_store: RunStore | None = None,
    ):
        self.planner = planner
        self.worker = worker
        self.synthesizer = synthesizer
        self.memory = memory or SharedMemory(":memory:")
        self.config = config or ResearchConfig()
        self.reviewer = reviewer or RedTeamReviewer(
            llm=getattr(synthesizer, "llm", None),
            pass_score=self.config.review_pass_score,
            min_citation_coverage=self.config.min_citation_coverage,
            max_pass_issue_severity=self.config.max_pass_issue_severity,
        )
        llm = getattr(synthesizer, "llm", None)
        self.repairer = repairer or BlueTeamRepairer(
            llm=llm,
            max_patches=self.config.max_blue_patches_per_round,
            max_growth_chars=self.config.max_patch_growth_chars,
            max_generation_attempts=self.config.max_blue_patch_generation_attempts,
            verifier=ClaimEvidenceVerifier(
                llm if self.config.enable_semantic_claim_verification else None
            ),
        )
        if isinstance(runtime, LocalAgentRuntime):
            if tool_registry is not None and tool_registry is not runtime.tools:
                raise ValueError(
                    "tool_registry must be the same registry used by LocalAgentRuntime"
                )
            self.tool_registry = runtime.tools
        else:
            self.tool_registry = tool_registry or ToolRegistry()
        worker_backend = getattr(worker, "backend", None)
        if worker_backend is not None and not self.tool_registry.contains("retrieval"):
            self.tool_registry.register("retrieval", worker_backend)
        self.runtime = runtime or LocalAgentRuntime(self.tool_registry)
        self.worker_spec = worker_spec or AgentSpec(
            agent_id="research-worker",
            role="researcher",
            description="Retrieve and normalize evidence for one research subtask",
            tool_names=("retrieval",) if worker_backend is not None else (),
        )
        self.swarm_policy = swarm_policy or HeuristicSwarmPolicy()
        self.run_store = run_store

    async def run(
        self,
        question: str,
        *,
        run_id: str | None = None,
        resume: bool = False,
    ) -> ResearchResult:
        """Execute, resume, or idempotently return one durable research run."""

        question = question.strip()
        if not question:
            raise ValueError("Research question cannot be empty")
        if resume and run_id is None:
            raise ValueError("resume=True requires run_id")

        run_id = run_id or f"research-{uuid.uuid4().hex}"
        initial_trace_id = new_trace_id()
        checkpoint: RunCheckpoint | None = None
        owner_id = uuid.uuid4().hex
        if self.run_store is not None:
            created = self.run_store.create_run(
                run_id,
                question,
                metadata={
                    "engine_version": engine_version(),
                    "source_commit": source_commit(),
                    "trace_id": initial_trace_id,
                    "trace_schema_version": TRACE_SCHEMA_VERSION,
                },
            )
            if not created:
                checkpoint = self.run_store.load(run_id)
                if checkpoint.result is not None and checkpoint.lifecycle_status == "completed":
                    return checkpoint.result
                if checkpoint.result is not None and not resume:
                    return checkpoint.result
                if not resume:
                    raise RunAlreadyExistsError(
                        f"run_id {run_id!r} already exists; pass resume=True to continue"
                    )
            self.run_store.acquire(run_id, owner_id, resume=resume)
        elif resume:
            raise RuntimeError("A RunStore is required to resume a run")

        try:
            result = await self._execute_run(
                question,
                run_id=run_id,
                resume_checkpoint=checkpoint if resume else None,
                initial_trace_id=(
                    str(checkpoint.metadata.get("trace_id") or initial_trace_id)
                    if checkpoint is not None
                    else initial_trace_id
                ),
            )
        except BaseException as exc:
            if self.run_store is not None:
                self.run_store.mark_interrupted(run_id, exc)
            raise
        if self.run_store is not None:
            self.run_store.finalize(run_id, result)
        return result

    async def resume(self, run_id: str) -> ResearchResult:
        if self.run_store is None:
            raise RuntimeError("A RunStore is required to resume a run")
        checkpoint = self.run_store.load(run_id)
        return await self.run(checkpoint.objective, run_id=run_id, resume=True)

    def replay(self, run_id: str) -> dict[str, Any]:
        if self.run_store is None:
            raise RuntimeError("A RunStore is required to replay a run")
        return self.run_store.replay(run_id)

    async def _execute_run(
        self,
        question: str,
        *,
        run_id: str,
        resume_checkpoint: RunCheckpoint | None,
        initial_trace_id: str,
    ) -> ResearchResult:

        reset_review_history = getattr(self.reviewer, "reset_history", None)
        if callable(reset_review_history):
            reset_review_history()

        started_at = resume_checkpoint.started_at if resume_checkpoint else utc_now()
        llm_config = getattr(getattr(self.synthesizer, "llm", None), "config", None)
        provider = getattr(llm_config, "provider", None)
        current_run_metadata: dict[str, Any] = {
            "engine_version": engine_version(),
            "source_commit": source_commit(),
            "research_config": asdict(self.config),
            "prompt_schema_versions": {
                "planner": "v2",
                "synthesizer": "v3",
                "red_review": "v3",
                "blue_patch": "v5",
                "claim_verifier": "v4",
            },
            "llm": {
                "provider": getattr(provider, "value", provider),
                "model": getattr(llm_config, "model", None),
                "base_url": getattr(llm_config, "base_url", None),
                "temperature": getattr(llm_config, "temperature", None),
                "request_timeout_seconds": getattr(llm_config, "request_timeout_seconds", None),
                "max_retries": getattr(llm_config, "max_retries", None),
            },
            "harness": {
                "runtime": type(self.runtime).__name__,
                "worker_agent_id": self.worker_spec.agent_id,
                "worker_role": self.worker_spec.role,
                "registered_tools": list(self.tool_registry.names),
            },
        }
        run_metadata = (
            dict(resume_checkpoint.metadata)
            if resume_checkpoint is not None and resume_checkpoint.metadata
            else current_run_metadata
        )
        trace_id = str(run_metadata.get("trace_id") or initial_trace_id)
        run_metadata["trace_id"] = trace_id
        run_metadata["trace_schema_version"] = TRACE_SCHEMA_VERSION
        previous_events = (
            self.run_store.events(run_id)
            if self.run_store is not None and resume_checkpoint is not None
            else []
        )
        trace = PersistedTrace(
            previous_events,
            store=self.run_store,
            run_id=run_id,
            trace_id=trace_id,
        )
        llm = getattr(self.synthesizer, "llm", None)
        context = RunContext(
            run_id=run_id,
            objective=question,
            tools=self.tool_registry,
            metadata=run_metadata,
            event_sink=trace.persist_only,
            usage_provider=lambda: usage_snapshot(llm),
        )
        if resume_checkpoint is not None and resume_checkpoint.plan is not None:
            plan = resume_checkpoint.plan
            swarm = resume_checkpoint.swarm or self.swarm_policy.decide(
                question,
                plan,
                self.config,
                self.worker_spec,
            )
            # A resumed attempt gets a fresh wall-clock window from the
            # current configuration while retaining its persisted role and
            # budget shape. This makes timeout recovery deterministic even
            # when tracing adds small local overhead.
            swarm = replace(
                swarm,
                max_elapsed_seconds=self.config.global_timeout_seconds,
            )
            budget = BudgetController.restore(swarm, resume_checkpoint.budget)
            completed_review_rounds = sum(
                1 for event in previous_events if event.get("event") == "red_review"
            )
            if budget.used[BudgetResource.REVIEW_ROUND.value] > completed_review_rounds:
                budget.used[BudgetResource.REVIEW_ROUND.value] = completed_review_rounds
                budget.exhausted.discard(BudgetResource.REVIEW_ROUND.value)
        else:
            if self.run_store is not None:
                self.run_store.set_phase(run_id, "planning")
            plan = await self._trace_component(
                trace,
                context,
                component_id="lead-planner",
                role="planner",
                operation="plan",
                input_value=question,
                awaitable=self.planner.plan(question),
            )
            swarm = self.swarm_policy.decide(
                question,
                plan,
                self.config,
                self.worker_spec,
            )
            budget = BudgetController(swarm)
        semaphore = asyncio.Semaphore(swarm.max_concurrency)
        swarm_metadata = {
            **swarm.to_dict(),
            "policy": type(self.swarm_policy).__name__,
            "policy_version": getattr(self.swarm_policy, "VERSION", None),
        }
        role_decision = self._swarm_role_decision(swarm)
        swarm_metadata["role_decision"] = role_decision
        run_metadata["swarm"] = swarm_metadata
        context.put_artifact("research_plan", plan)
        context.put_artifact("swarm_plan", swarm)
        context.put_artifact("budget_controller", budget)
        evidences: list[Evidence] = []
        forced_synthesis = False
        if resume_checkpoint is None:
            trace.extend(
                [
                    {
                        "at": utc_now(),
                        "event": "plan_created",
                        "run_id": run_id,
                        "version": plan.version,
                        "tasks": [task.subtask_id for task in plan.subtasks],
                    },
                    {
                        "at": utc_now(),
                        "event": "swarm_configured",
                        **swarm_metadata,
                    },
                    {
                        "at": utc_now(),
                        "event": "swarm_role_decision",
                        **role_decision,
                    },
                ]
            )

        def on_task_transition(runtime: TaskRuntime, history: dict[str, Any]) -> None:
            event = {
                "at": history["at"],
                "event": "task_transition",
                "task_id": runtime.spec.subtask_id,
                **{key: value for key, value in history.items() if key != "at"},
            }
            if self.run_store is not None:
                prepared = trace.prepare(event)
                self.run_store.checkpoint_node(run_id, runtime, event=prepared)
                trace.append_already_persisted(prepared)
            else:
                trace.append(event)

        state = ResearchRunState(plan, on_transition=on_task_transition)
        resume_execution_needed = False
        if resume_checkpoint is not None:
            for task_id, runtime in state.tasks.items():
                saved = resume_checkpoint.nodes.get(task_id)
                if saved is None:
                    resume_execution_needed = True
                    continue
                runtime.status = saved.status
                runtime.spec.status = saved.status
                runtime.attempts = saved.attempts
                runtime.error = saved.error
                runtime.started_at = saved.started_at
                runtime.finished_at = saved.finished_at
                runtime.history = list(saved.history)
                runtime.evidences = list(saved.evidences)
                if saved.status is TaskStatus.SUCCEEDED:
                    evidences.extend(saved.evidences)
                else:
                    resume_execution_needed = True
                    runtime.recover("run_resumed")
            trace.append(
                {
                    "at": utc_now(),
                    "event": "run_resumed",
                    "run_id": run_id,
                    "from_phase": resume_checkpoint.phase,
                    "reused_nodes": sorted(
                        task_id
                        for task_id, runtime in state.tasks.items()
                        if runtime.status is TaskStatus.SUCCEEDED
                    ),
                    "retry_nodes": sorted(
                        task_id
                        for task_id, runtime in state.tasks.items()
                        if runtime.status is not TaskStatus.SUCCEEDED
                    ),
                }
            )

        if self.run_store is not None:
            self.run_store.save_plan(run_id, plan, swarm, run_metadata)

            def checkpoint_budget(snapshot: dict[str, Any]) -> None:
                self.run_store.save_budget(run_id, snapshot)
                trace.append(
                    {
                        "at": utc_now(),
                        "event": "budget_checkpointed",
                        "budget": {
                            key: value for key, value in snapshot.items() if key != "internal"
                        },
                    }
                )

            budget.bind_checkpoint(checkpoint_budget)
            self.run_store.set_phase(run_id, "executing_dag")

        try:
            await asyncio.wait_for(
                self._execute_plan(
                    plan,
                    state,
                    evidences,
                    trace,
                    context,
                    swarm,
                    budget,
                    semaphore,
                ),
                timeout=swarm.max_elapsed_seconds,
            )
        except asyncio.TimeoutError:
            forced_synthesis = True
            budget.mark_stop("deadline_exceeded")
            trace.append(
                {
                    "at": utc_now(),
                    "event": "global_timeout_forced_synthesis",
                    "timeout_seconds": swarm.max_elapsed_seconds,
                }
            )
            self._cancel_unfinished(state, "deadline_exceeded")
        if "deadline_exceeded" in budget.stop_reasons:
            forced_synthesis = True

        can_reuse_report = bool(
            resume_checkpoint is not None
            and resume_checkpoint.current_report
            and not resume_execution_needed
        )
        if can_reuse_report:
            draft = str(resume_checkpoint.current_report)
            removed_markers = 0
            trace.append(
                {
                    "at": utc_now(),
                    "event": "intermediate_report_reused",
                    "source_phase": resume_checkpoint.phase,
                }
            )
        else:
            if self.run_store is not None:
                self.run_store.set_phase(run_id, "synthesizing")
            draft = await self._trace_component(
                trace,
                context,
                component_id="report-synthesizer",
                role="synthesizer",
                operation="initial_synthesis",
                input_value={"question": question, "evidence_count": len(evidences)},
                awaitable=self.synthesizer.synthesize(
                    question,
                    plan,
                    evidences,
                    self.memory,
                    forced=forced_synthesis,
                ),
            )
            draft = self.repairer.normalize_citations(draft, evidences)
            draft, removed_markers = remove_omission_markers(draft)
            self._checkpoint_report(run_id, "initial_synthesis", draft)
        if removed_markers:
            trace.append(
                {
                    "at": utc_now(),
                    "event": "report_omission_markers_removed",
                    "stage": "initial_synthesis",
                    "count": removed_markers,
                }
            )

        convergence = ReviewConvergence(
            self.config.min_score_improvement,
            self.config.oscillation_window,
        )
        final_review = (
            resume_checkpoint.result.review
            if can_reuse_report and resume_checkpoint and resume_checkpoint.result
            else None
        )
        verification_agent_override: AgentSpec | None = None
        previous_ledger: ClaimLedger | None = (
            resume_checkpoint.result.claim_ledger
            if can_reuse_report and resume_checkpoint and resume_checkpoint.result
            else None
        )
        fixed_fallback_requested = False
        if self.run_store is not None:
            self.run_store.set_phase(run_id, "reviewing")
        starting_review_round = (
            sum(1 for event in previous_events if event.get("event") == "red_review") + 1
            if resume_checkpoint is not None
            else 1
        )
        final_comparison_coverage = build_comparison_coverage(question, draft)
        for review_round in range(starting_review_round, swarm.max_review_rounds + 1):
            if not budget.try_consume(BudgetResource.REVIEW_ROUND):
                reason = (
                    "deadline_exceeded" if budget.deadline_exceeded() else "review_budget_exhausted"
                )
                budget.mark_stop(reason)
                self._record_stop(trace, budget, reason, round=review_round)
                break
            final_review = await self._trace_component(
                trace,
                context,
                component_id="red-reviewer",
                role="red_agent",
                operation=f"review_round_{review_round}",
                input_value={"question": question, "report": draft, "evidences": evidences},
                awaitable=self.reviewer.review(question, draft, evidences),
            )
            final_comparison_coverage = build_comparison_coverage(question, draft)
            final_review = enforce_comparison_coverage(
                final_review,
                final_comparison_coverage,
            )
            final_review = convergence.update(draft, final_review)
            trace.append(
                {
                    "at": utc_now(),
                    "event": "comparison_evidence_matrix",
                    "round": review_round,
                    **final_comparison_coverage.to_dict(),
                    "repair_queries": final_comparison_coverage.repair_queries(
                        limit=self.config.max_verification_queries_per_round
                    ),
                }
            )
            trace.append(
                {
                    "at": utc_now(),
                    "event": "red_review",
                    "round": review_round,
                    "score": final_review.total_score,
                    "citation_coverage": final_review.metrics.get("citation_coverage"),
                    "issue_count": len(final_review.structured_issues),
                    "max_issue_severity": max(
                        (item.severity for item in final_review.structured_issues),
                        default=0,
                    ),
                    "passed": final_review.passed,
                    "converged": final_review.converged,
                    "oscillating": final_review.oscillating,
                }
            )
            pre_repair_ledger = await self._trace_component(
                trace,
                context,
                component_id="evidence-verifier",
                role="evidence_verifier",
                operation=f"pre_repair_round_{review_round}",
                input_value={"report": draft, "evidences": evidences},
                awaitable=self.repairer.verifier.build_ledger(
                    draft,
                    evidences,
                    previous_ledger=previous_ledger,
                ),
            )
            previous_ledger = pre_repair_ledger
            pre_support_rate = pre_repair_ledger.metrics.get("claim_support_rate", 0.0)
            pre_semantic_available = (
                self.repairer.verifier.llm is None
                or pre_repair_ledger.verification_mode == "semantic"
            )
            pre_claim_gate_passed = (
                pre_support_rate >= self.config.min_claim_support_rate
                and pre_semantic_available
                and final_comparison_coverage.passed
            )
            quality_guardrail_action = None
            if self.config.enable_dynamic_swarm:
                quality_guardrail_action = budget.observe_claim_support(
                    pre_support_rate,
                    minimum=self.config.min_claim_support_rate,
                )
                if quality_guardrail_action == "add_evidence_verifier":
                    verification_agent_override = AgentSpec(
                        agent_id=f"{self.worker_spec.agent_id}-quality-verifier",
                        role="evidence_verifier",
                        description="Quality guardrail: verify unsupported claims",
                        tool_names=self.worker_spec.tool_names,
                        model=self.worker_spec.model,
                        metadata={
                            **self.worker_spec.metadata,
                            "swarm_managed": True,
                            "quality_guardrail": True,
                        },
                    )
                elif quality_guardrail_action == "fallback_fixed_harness":
                    verification_agent_override = self.worker_spec
                    fixed_fallback_requested = True
                if not final_comparison_coverage.passed and quality_guardrail_action is None:
                    quality_guardrail_action = "add_evidence_verifier"
                    verification_agent_override = AgentSpec(
                        agent_id=f"{self.worker_spec.agent_id}-comparison-verifier",
                        role="evidence_verifier",
                        description="Quality guardrail: fill A/B comparison evidence cells",
                        tool_names=self.worker_spec.tool_names,
                        model=self.worker_spec.model,
                        metadata={
                            **self.worker_spec.metadata,
                            "swarm_managed": True,
                            "comparison_quality_guardrail": True,
                        },
                    )
                if quality_guardrail_action:
                    trace.append(
                        {
                            "at": utc_now(),
                            "event": "swarm_quality_guardrail",
                            "round": review_round,
                            "action": quality_guardrail_action,
                            "claim_support_rate": pre_support_rate,
                            "minimum_claim_support_rate": (self.config.min_claim_support_rate),
                        }
                    )
            trace.append(
                {
                    "at": utc_now(),
                    "event": "evidence_verifier_decision",
                    "round": review_round,
                    "triggered": quality_guardrail_action == "add_evidence_verifier",
                    "action": quality_guardrail_action or "keep_current_roles",
                    "reason": (
                        "comparison_evidence_matrix_incomplete"
                        if not final_comparison_coverage.passed
                        else (
                            "claim_support_below_minimum"
                            if pre_support_rate < self.config.min_claim_support_rate
                            else "claim_support_meets_minimum"
                        )
                    ),
                    "claim_support_rate": pre_support_rate,
                    "minimum_claim_support_rate": self.config.min_claim_support_rate,
                    "selected_role": (
                        verification_agent_override.role
                        if verification_agent_override is not None
                        else None
                    ),
                }
            )
            trace.append(
                {
                    "at": utc_now(),
                    "event": "claim_evidence_alignment",
                    "round": review_round,
                    "stage": "before_targeted_retrieval",
                    "verification_mode": pre_repair_ledger.verification_mode,
                    "passed": pre_claim_gate_passed,
                    **pre_repair_ledger.metrics,
                }
            )
            stop_reason = budget.observe_review(
                final_review,
                quality_gate_passed=pre_claim_gate_passed,
                claim_support_rate=pre_support_rate,
            )
            if stop_reason:
                self._record_stop(trace, budget, stop_reason, round=review_round)
                break

            # Do not return a newly repaired draft without a final Red review.
            if review_round == swarm.max_review_rounds:
                budget.mark_exhausted(
                    BudgetResource.REVIEW_ROUND,
                    "review_budget_exhausted",
                )
                self._record_stop(
                    trace,
                    budget,
                    "review_budget_exhausted",
                    round=review_round,
                )
                break

            claim_query_limit = max(
                1,
                self.config.max_verification_queries_per_round
                if verification_agent_override is not None
                else self.config.max_verification_queries_per_round // 2,
            )
            claim_queries = self.repairer.verifier.repair_queries(
                pre_repair_ledger,
                limit=claim_query_limit,
            )
            comparison_queries = final_comparison_coverage.repair_queries(
                limit=self.config.max_verification_queries_per_round,
            )
            verification_queries = list(
                dict.fromkeys(
                    [
                        *comparison_queries,
                        *final_review.repair_queries,
                        *claim_queries,
                    ]
                )
            )[: self.config.max_verification_queries_per_round]
            if verification_queries:
                new_evidence = await self._verify_issues(
                    verification_queries,
                    review_round,
                    trace,
                    context,
                    swarm,
                    budget,
                    semaphore,
                    agent_override=verification_agent_override,
                )
                evidences.extend(new_evidence)

            if verification_queries and new_evidence:
                post_retrieval_ledger = await self._trace_component(
                    trace,
                    context,
                    component_id="evidence-verifier",
                    role="evidence_verifier",
                    operation=f"post_retrieval_round_{review_round}",
                    input_value={"report": draft, "evidences": evidences},
                    awaitable=self.repairer.verifier.build_ledger(
                        draft,
                        evidences,
                        previous_ledger=pre_repair_ledger,
                    ),
                )
                post_ledger_reused = False
            else:
                post_retrieval_ledger = pre_repair_ledger
                post_ledger_reused = True
            previous_ledger = post_retrieval_ledger
            claim_patch_result = self.repairer.repair_claim_gaps(
                draft,
                evidences,
                post_retrieval_ledger,
            )
            deterministic_patch_changed = claim_patch_result.changed
            trace.append(
                {
                    "at": utc_now(),
                    "event": "claim_evidence_alignment",
                    "round": review_round,
                    "stage": "after_targeted_retrieval",
                    "verification_mode": post_retrieval_ledger.verification_mode,
                    "ledger_reused": post_ledger_reused,
                    "targeted_queries": verification_queries,
                    "comparison_queries": comparison_queries,
                    "targeted_evidence_count": (len(new_evidence) if verification_queries else 0),
                    **post_retrieval_ledger.metrics,
                }
            )
            if claim_patch_result.changed:
                draft = self.repairer.normalize_citations(claim_patch_result.report, evidences)
                self._checkpoint_report(
                    run_id,
                    "deterministic_claim_patch",
                    draft,
                    round=review_round,
                )
                trace.append(
                    {
                        "at": utc_now(),
                        "event": "blue_repair",
                        "round": review_round,
                        "mode": "deterministic_claim_patch",
                        "requested_patch_count": claim_patch_result.requested_count,
                        "applied_patches": [
                            {
                                "patch_id": patch.patch_id,
                                "action": patch.action.value,
                                "evidence_ids": patch.evidence_ids,
                                "reason": patch.reason,
                            }
                            for patch in claim_patch_result.applied
                        ],
                        "rejected_patches": [
                            {"patch_id": item.patch_id, "reason": item.reason}
                            for item in claim_patch_result.rejected
                        ],
                    }
                )
            if claim_patch_result.rejected:
                trace.append(
                    {
                        "at": utc_now(),
                        "event": "blue_repair",
                        "round": review_round,
                        "mode": "deterministic_claim_patch",
                        "requested_patch_count": claim_patch_result.requested_count,
                        "applied_patches": [],
                        "rejected_patches": [
                            {"patch_id": item.patch_id, "reason": item.reason}
                            for item in claim_patch_result.rejected
                        ],
                    }
                )

            conservative = review_round >= 2
            patch_result = await self._trace_component(
                trace,
                context,
                component_id="blue-repairer",
                role="blue_agent",
                operation=f"structured_patch_round_{review_round}",
                input_value={
                    "question": question,
                    "report": draft,
                    "evidences": evidences,
                    "review": final_review,
                },
                awaitable=self.repairer.repair(
                    question,
                    draft,
                    evidences,
                    final_review,
                    conservative=conservative,
                ),
            )
            if patch_result.changed:
                draft = self.repairer.normalize_citations(patch_result.report, evidences)
                draft, removed_markers = remove_omission_markers(draft)
                self._checkpoint_report(
                    run_id,
                    "structured_blue_patch",
                    draft,
                    round=review_round,
                )
                if removed_markers:
                    trace.append(
                        {
                            "at": utc_now(),
                            "event": "report_omission_markers_removed",
                            "stage": "structured_patch",
                            "round": review_round,
                            "count": removed_markers,
                        }
                    )
                trace.append(
                    {
                        "at": utc_now(),
                        "event": "blue_repair",
                        "round": review_round,
                        "mode": "structured_patch",
                        "requested_patch_count": patch_result.requested_count,
                        "applied_patches": [
                            {
                                "patch_id": patch.patch_id,
                                "action": patch.action.value,
                                "issue_ids": patch.issue_ids,
                                "evidence_ids": patch.evidence_ids,
                                "reason": patch.reason,
                            }
                            for patch in patch_result.applied
                        ],
                        "rejected_patches": [
                            {"patch_id": item.patch_id, "reason": item.reason}
                            for item in patch_result.rejected
                        ],
                        "claim_validation": {
                            patch_id: ledger.metrics
                            for patch_id, ledger in patch_result.validation_ledgers.items()
                        },
                    }
                )
                continue

            if deterministic_patch_changed:
                trace.append(
                    {
                        "at": utc_now(),
                        "event": "blue_repair",
                        "round": review_round,
                        "mode": "structured_patch",
                        "requested_patch_count": patch_result.requested_count,
                        "applied_patches": [],
                        "rejected_patches": [
                            {"patch_id": item.patch_id, "reason": item.reason}
                            for item in patch_result.rejected
                        ],
                        "reason": "deterministic_patch_retained_pending_next_red_review",
                    }
                )
                continue

            instructions = self.repairer.instructions(final_review)
            if conservative:
                instructions.append(
                    "CONSERVATIVE_REWRITE: 连续审查未通过。将报告压缩到 2500 个中文字符"
                    "以内，不输出证据原文长引语、字段表格或完整来源表；只保留证据片段"
                    "能够直接支持的 5-10 个核心结论，每个事实句必须紧邻一个直接支持它"
                    "的证据 ID。删除所有仍受争议、重复或仅为完整性而添加的事实。宁可"
                    "明确证据不足，也不要维持未经证实的覆盖面。"
                )
            draft = await self._trace_component(
                trace,
                context,
                component_id="report-synthesizer",
                role="synthesizer",
                operation=f"rewrite_fallback_round_{review_round}",
                input_value={"report": draft, "instructions": instructions},
                awaitable=self.synthesizer.synthesize(
                    question,
                    plan,
                    evidences,
                    self.memory,
                    draft=draft,
                    repair_instructions=instructions,
                    forced=forced_synthesis,
                ),
            )
            draft = self.repairer.normalize_citations(draft, evidences)
            draft, removed_markers = remove_omission_markers(draft)
            self._checkpoint_report(
                run_id,
                "rewrite_fallback",
                draft,
                round=review_round,
            )
            if removed_markers:
                trace.append(
                    {
                        "at": utc_now(),
                        "event": "report_omission_markers_removed",
                        "stage": "rewrite_fallback",
                        "round": review_round,
                        "count": removed_markers,
                    }
                )
            trace.append(
                {
                    "at": utc_now(),
                    "event": "blue_repair",
                    "round": review_round,
                    "mode": "rewrite_fallback",
                    "actions": instructions,
                    "requested_patch_count": patch_result.requested_count,
                    "rejected_patches": [
                        {"patch_id": item.patch_id, "reason": item.reason}
                        for item in patch_result.rejected
                    ],
                }
            )

        claim_ledger = await self._trace_component(
            trace,
            context,
            component_id="evidence-verifier",
            role="evidence_verifier",
            operation="final_ledger",
            input_value={"report": draft, "evidences": evidences},
            awaitable=self.repairer.verifier.build_ledger(
                draft,
                evidences,
                previous_ledger=previous_ledger,
            ),
        )
        final_claim_patch = self.repairer.repair_claim_gaps(
            draft,
            evidences,
            claim_ledger,
        )
        if final_claim_patch.changed:
            draft = self.repairer.normalize_citations(
                final_claim_patch.report,
                evidences,
            )
            claim_ledger = await self._trace_component(
                trace,
                context,
                component_id="evidence-verifier",
                role="evidence_verifier",
                operation="verify_final_claim_patch",
                input_value={"report": draft, "evidences": evidences},
                awaitable=self.repairer.verifier.build_ledger(
                    draft,
                    evidences,
                    previous_ledger=claim_ledger,
                ),
            )
            final_review = refresh_review_after_deterministic_claim_patch(
                final_review,
                draft,
                pass_score=self.config.review_pass_score,
                min_citation_coverage=self.config.min_citation_coverage,
                max_pass_issue_severity=self.config.max_pass_issue_severity,
            )
            self._checkpoint_report(run_id, "final_claim_patch", draft)
            trace.append(
                {
                    "at": utc_now(),
                    "event": "blue_repair",
                    "round": "final",
                    "mode": "deterministic_claim_patch",
                    "incremental_verification": True,
                    "requested_patch_count": final_claim_patch.requested_count,
                    "applied_patches": [
                        {
                            "patch_id": patch.patch_id,
                            "action": patch.action.value,
                            "evidence_ids": patch.evidence_ids,
                            "reason": patch.reason,
                        }
                        for patch in final_claim_patch.applied
                    ],
                    "rejected_patches": [
                        {"patch_id": item.patch_id, "reason": item.reason}
                        for item in final_claim_patch.rejected
                    ],
                    "reused_unchanged_claim_count": claim_ledger.metrics.get(
                        "reused_unchanged_claim_count", 0.0
                    ),
                }
            )
        claim_support_rate = claim_ledger.metrics.get("claim_support_rate", 0.0)
        final_citation_coverage = float(
            final_review.metrics.get("citation_coverage", 0.0) if final_review else 0.0
        )
        fallback_blocking_issue_count = sum(
            is_blocking_review_issue(issue, self.config.max_pass_issue_severity)
            for issue in (final_review.structured_issues if final_review else [])
        )
        fallback_red_review_failed = bool(final_review and not final_review.passed)
        fallback_triggered = bool(
            self.config.enable_dynamic_swarm
            and not forced_synthesis
            and (
                fixed_fallback_requested
                or final_citation_coverage < self.config.min_citation_coverage
                or claim_support_rate < self.config.min_claim_support_rate
                or fallback_red_review_failed
            )
        )
        fallback_trigger_reasons: list[str] = []
        if fixed_fallback_requested:
            fallback_trigger_reasons.append("guardrail_requested_fixed_harness")
        if final_citation_coverage < self.config.min_citation_coverage:
            fallback_trigger_reasons.append("citation_coverage_below_minimum")
        if claim_support_rate < self.config.min_claim_support_rate:
            fallback_trigger_reasons.append("claim_support_below_minimum")
        if fallback_blocking_issue_count:
            fallback_trigger_reasons.append("red_review_blocking_issues")
        elif fallback_red_review_failed:
            fallback_trigger_reasons.append("red_review_failed_without_blocking_details")
        if not self.config.enable_dynamic_swarm:
            fallback_trigger_reasons.append("dynamic_swarm_disabled")
        if forced_synthesis:
            fallback_trigger_reasons.append("forced_synthesis_skips_fallback")
        trace.append(
            {
                "at": utc_now(),
                "event": "fixed_fallback_decision",
                "triggered": fallback_triggered,
                "reasons": fallback_trigger_reasons or ["quality_gates_passed"],
                "observed": {
                    "citation_coverage": final_citation_coverage,
                    "claim_support_rate": claim_support_rate,
                    "red_review_passed": not fallback_red_review_failed,
                    "blocking_review_issue_count": fallback_blocking_issue_count,
                },
                "thresholds": {
                    "min_citation_coverage": self.config.min_citation_coverage,
                    "min_claim_support_rate": self.config.min_claim_support_rate,
                },
            }
        )
        fallback_selected = False
        fallback_selection_reason = "not_triggered"
        fallback_candidate_metrics: dict[str, Any] = {}
        if fallback_triggered:
            if "fallback_fixed_harness" not in budget.quality_guardrail_actions:
                budget.quality_guardrail_actions.append("fallback_fixed_harness")
            fallback_report = await self._trace_component(
                trace,
                context,
                component_id="report-synthesizer",
                role="synthesizer",
                operation="fixed_harness_fallback",
                input_value={"report": draft, "evidence_count": len(evidences)},
                awaitable=self.synthesizer.synthesize(
                    question,
                    plan,
                    evidences,
                    self.memory,
                    draft=draft,
                    repair_instructions=[
                        "FIXED_HARNESS_QUALITY_FALLBACK：忽略动态角色的扩展性写作，只使用"
                        "证据目录中能够逐句直接支持的事实生成紧凑报告；每个事实句紧邻有效"
                        "证据 ID；删除推断越界和部分支持的复合断言；证据缺口只合并声明一次。"
                    ],
                    forced=False,
                ),
            )
            fallback_report = self.repairer.normalize_citations(fallback_report, evidences)
            fallback_report, removed_markers = remove_omission_markers(fallback_report)
            fallback_ledger = await self._trace_component(
                trace,
                context,
                component_id="evidence-verifier",
                role="evidence_verifier",
                operation="fixed_fallback_ledger",
                input_value={"report": fallback_report, "evidences": evidences},
                awaitable=self.repairer.verifier.build_ledger(
                    fallback_report,
                    evidences,
                    previous_ledger=claim_ledger,
                ),
            )
            fallback_claim_patch = self.repairer.repair_claim_gaps(
                fallback_report,
                evidences,
                fallback_ledger,
            )
            if fallback_claim_patch.changed:
                fallback_report = self.repairer.normalize_citations(
                    fallback_claim_patch.report,
                    evidences,
                )
                fallback_ledger = await self._trace_component(
                    trace,
                    context,
                    component_id="evidence-verifier",
                    role="evidence_verifier",
                    operation="verify_fixed_fallback_patch",
                    input_value={"report": fallback_report, "evidences": evidences},
                    awaitable=self.repairer.verifier.build_ledger(
                        fallback_report,
                        evidences,
                        previous_ledger=fallback_ledger,
                    ),
                )
            fallback_review = await self._trace_component(
                trace,
                context,
                component_id="red-reviewer",
                role="red_agent",
                operation="fixed_fallback_review",
                input_value={"report": fallback_report, "evidences": evidences},
                awaitable=self.reviewer.review(
                    question,
                    fallback_report,
                    evidences,
                ),
            )
            primary_candidate = ReportQualityCandidate(
                draft,
                claim_ledger,
                final_review,
                "dynamic_swarm",
            )
            fixed_candidate = ReportQualityCandidate(
                fallback_report,
                fallback_ledger,
                fallback_review,
                "fixed_harness_fallback",
            )
            selected_candidate, fallback_selection_reason = select_quality_candidate(
                primary_candidate,
                fixed_candidate,
                min_claim_support_rate=self.config.min_claim_support_rate,
                min_citation_coverage=self.config.min_citation_coverage,
                max_pass_issue_severity=self.config.max_pass_issue_severity,
                support_drop_tolerance=(self.config.dynamic_claim_support_drop_tolerance),
            )
            fallback_selected = selected_candidate is fixed_candidate
            draft = selected_candidate.report
            self._checkpoint_report(
                run_id,
                "final_quality_guard",
                draft,
                selected=selected_candidate.origin,
                reason=fallback_selection_reason,
            )
            claim_ledger = selected_candidate.ledger
            final_review = selected_candidate.review
            claim_support_rate = selected_candidate.claim_support_rate
            final_citation_coverage = selected_candidate.citation_coverage
            fallback_candidate_metrics = {
                "dynamic": {
                    "claim_support_rate": primary_candidate.claim_support_rate,
                    "citation_coverage": primary_candidate.citation_coverage,
                    "blocking_issue_count": primary_candidate.blocking_issue_count(
                        self.config.max_pass_issue_severity
                    ),
                },
                "fixed_harness_fallback": {
                    "claim_support_rate": fixed_candidate.claim_support_rate,
                    "citation_coverage": fixed_candidate.citation_coverage,
                    "blocking_issue_count": fixed_candidate.blocking_issue_count(
                        self.config.max_pass_issue_severity
                    ),
                },
            }
            trace.append(
                {
                    "at": utc_now(),
                    "event": "dynamic_swarm_final_quality_guard",
                    "triggered": True,
                    "selected": selected_candidate.origin,
                    "reason": fallback_selection_reason,
                    "cost_used_as_quality_override": False,
                    "omission_markers_removed": removed_markers,
                    "fallback_claim_patch_count": len(fallback_claim_patch.applied),
                    "candidates": fallback_candidate_metrics,
                }
            )
        semantic_ledger_available = (
            self.repairer.verifier.llm is None or claim_ledger.verification_mode == "semantic"
        )
        claim_ledger_passed = (
            claim_support_rate >= self.config.min_claim_support_rate and semantic_ledger_available
        )
        adjudicated_red_disagreements = adjudicate_red_with_claim_ledger(
            final_review,
            claim_ledger,
        )
        if adjudicated_red_disagreements:
            trace.append(
                {
                    "at": utc_now(),
                    "event": "red_claim_verifier_adjudication",
                    "count": len(adjudicated_red_disagreements),
                    "issues": adjudicated_red_disagreements,
                }
            )
        trace.append(
            {
                "at": utc_now(),
                "event": "claim_ledger_built",
                "verification_mode": claim_ledger.verification_mode,
                "semantic_verification_available": semantic_ledger_available,
                "passed": claim_ledger_passed,
                **claim_ledger.metrics,
            }
        )
        sources = list(dict.fromkeys(evidence.url or evidence.source for evidence in evidences))
        review_issue_dimension_counts: dict[str, int] = {}
        review_issue_action_counts: dict[str, int] = {}
        review_issue_category_counts: dict[str, int] = {}
        if final_review:
            for issue in final_review.structured_issues:
                review_issue_dimension_counts[issue.dimension] = (
                    review_issue_dimension_counts.get(issue.dimension, 0) + 1
                )
                action = issue.action.value
                review_issue_action_counts[action] = review_issue_action_counts.get(action, 0) + 1
                review_issue_category_counts[issue.category] = (
                    review_issue_category_counts.get(issue.category, 0) + 1
                )
        completion_issue_reasons: list[str] = []
        if claim_support_rate < self.config.min_claim_support_rate:
            completion_issue_reasons.append("claim_support_below_threshold")
        if not semantic_ledger_available:
            completion_issue_reasons.append("semantic_verifier_unavailable")
        for metric_name, reason in (
            ("partially_supported_claim_count", "partially_supported_claims"),
            ("contradicted_claim_count", "contradicted_claims"),
            ("uncited_claim_count", "uncited_claims"),
            ("insufficient_claim_count", "insufficient_claims"),
            ("unknown_claim_count", "unknown_claims"),
            ("time_conflict_count", "time_conflicts"),
            ("number_conflict_count", "number_conflicts"),
            ("entity_conflict_count", "entity_conflicts"),
        ):
            if claim_ledger.metrics.get(metric_name, 0.0) > 0:
                completion_issue_reasons.append(reason)
        evidence_gap_count = int(claim_ledger.metrics.get("evidence_gap_statement_count", 0.0))
        final_comparison_coverage = build_comparison_coverage(question, draft)
        trace.append(
            {
                "at": utc_now(),
                "event": "comparison_evidence_matrix",
                "round": "final",
                **final_comparison_coverage.to_dict(),
                "repair_queries": final_comparison_coverage.repair_queries(
                    limit=self.config.max_verification_queries_per_round
                ),
            }
        )
        evidence_gap_reasons: list[str] = []
        if evidence_gap_count:
            evidence_gap_reasons.append("explicit_evidence_gap_statements")
        if any(
            claim_ledger.metrics.get(metric_name, 0.0) > 0
            for metric_name in (
                "partially_supported_claim_count",
                "uncited_claim_count",
                "insufficient_claim_count",
                "unknown_claim_count",
            )
        ):
            evidence_gap_reasons.append("claim_evidence_gaps")
        if final_review and any(
            issue.category == "evidence_gap" for issue in final_review.structured_issues
        ):
            evidence_gap_reasons.append("red_detected_evidence_gaps")
        if adjudicated_red_disagreements:
            evidence_gap_reasons.append("red_claim_verifier_disagreements")
        if final_comparison_coverage.applicable and not final_comparison_coverage.passed:
            evidence_gap_reasons.append("comparison_evidence_matrix_incomplete")

        blocking_review_issues = [
            issue
            for issue in (final_review.structured_issues if final_review else [])
            if is_blocking_review_issue(issue, self.config.max_pass_issue_severity)
        ]
        unresolved_claim_defects = (
            claim_support_rate < self.config.min_claim_support_rate
            or any(
                claim_ledger.metrics.get(metric_name, 0.0) > 0
                for metric_name in (
                    "contradicted_claim_count",
                    "time_conflict_count",
                    "number_conflict_count",
                    "entity_conflict_count",
                )
            )
            or not semantic_ledger_available
        )
        opaque_review_failure = bool(
            final_review
            and not final_review.passed
            and (
                not final_review.structured_issues
                or any(issue.category == "unknown" for issue in final_review.structured_issues)
            )
        )
        has_review_defect = bool(
            blocking_review_issues or unresolved_claim_defects or opaque_review_failure
        )
        if blocking_review_issues or opaque_review_failure:
            completion_issue_reasons.insert(0, "red_review_failed")
        if final_review and any(
            issue.category == "citation_error" for issue in blocking_review_issues
        ):
            completion_issue_reasons.append("citation_quality_issues")
        if forced_synthesis:
            status = CompletionStatus.PARTIAL_TIMEOUT.value
        elif has_review_defect:
            status = CompletionStatus.COMPLETED_WITH_REVIEW_ISSUES.value
        elif evidence_gap_reasons:
            status = CompletionStatus.COMPLETED_WITH_EVIDENCE_GAPS.value
        else:
            status = CompletionStatus.COMPLETED.value
        counts: dict[str, int] = {}
        for runtime in state.tasks.values():
            counts[runtime.status.value] = counts.get(runtime.status.value, 0) + 1
        finished_at = utc_now()
        run_usage = context.usage_snapshot()
        metrics = {
            "trace_id": trace_id,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "llm_usage": run_usage,
            "task_status_counts": counts,
            "evidence_count": len(evidences),
            "source_count": len(sources),
            "prompt_injection_evidence_count": sum(
                bool(item.metadata.get("prompt_injection_detected")) for item in evidences
            ),
            "evidence_omission_markers_removed": sum(
                int(item.metadata.get("extraction_omission_markers_removed", 0))
                for item in evidences
            ),
            "report_omission_markers_removed": sum(
                int(item.get("count", 0))
                for item in trace
                if item.get("event") == "report_omission_markers_removed"
            ),
            "plan_versions": plan.version,
            "review_score": final_review.total_score if final_review else None,
            "review_passed": final_review.passed if final_review else None,
            "citation_coverage": (
                final_review.metrics.get("citation_coverage") if final_review else None
            ),
            "claim_support_rate": claim_support_rate,
            "dynamic_quality_fallback_triggered": fallback_triggered,
            "dynamic_quality_fallback_selected": fallback_selected,
            "dynamic_quality_fallback_reason": fallback_selection_reason,
            "dynamic_quality_candidates": fallback_candidate_metrics,
            "claim_ledger_passed": claim_ledger_passed,
            "claim_verification_mode": claim_ledger.verification_mode,
            "semantic_claim_verification_available": semantic_ledger_available,
            "claim_ledger_metrics": claim_ledger.metrics,
            "completion_issue_reasons": completion_issue_reasons,
            "evidence_gap_reasons": evidence_gap_reasons,
            "evidence_gap_count": evidence_gap_count,
            "comparison_evidence_coverage": final_comparison_coverage.coverage_rate,
            "comparison_missing_cell_count": len(final_comparison_coverage.missing),
            "comparison_evidence_matrix": final_comparison_coverage.to_dict(),
            "blocking_review_issue_count": len(blocking_review_issues),
            "red_claim_verifier_adjudication_count": len(adjudicated_red_disagreements),
            "review_issue_dimension_counts": review_issue_dimension_counts,
            "review_issue_action_counts": review_issue_action_counts,
            "review_issue_category_counts": review_issue_category_counts,
            "review_rounds": sum(1 for item in trace if item.get("event") == "red_review"),
            "blue_patch_applied_count": sum(
                len(item.get("applied_patches", []))
                for item in trace
                if item.get("event") == "blue_repair"
            ),
            "blue_patch_rejected_count": sum(
                len(item.get("rejected_patches", []))
                for item in trace
                if item.get("event") == "blue_repair"
            ),
            "blue_rewrite_fallbacks": sum(
                1
                for item in trace
                if item.get("event") == "blue_repair" and item.get("mode") == "rewrite_fallback"
            ),
            "forced_synthesis": forced_synthesis,
            "swarm_complexity": swarm.level.value,
            "swarm_agent_count": len(swarm.agents),
            "swarm_max_concurrency": swarm.max_concurrency,
            "budget": budget.snapshot(),
        }
        trace.append(
            {
                "at": finished_at,
                "event": "run_finalized",
                "run_id": run_id,
                "status": status,
                "duration_ms": duration_ms(started_at, finished_at),
                "token_usage": {
                    "prompt": run_usage.get("prompt_tokens", 0),
                    "completion": run_usage.get("completion_tokens", 0),
                    "total": run_usage.get("total_tokens", 0),
                    "measurement": run_usage.get("measurement", "unavailable"),
                },
                "cost_usd": run_usage.get("cost_usd", 0.0),
                "retry_count": run_usage.get("retry_count", 0),
                "stop_reasons": list(budget.stop_reasons),
            }
        )
        trace.extend(context.events)
        trace.sort(key=lambda item: item["at"])
        if self.run_store is not None:
            self.run_store.set_phase(run_id, "finalizing")
        return ResearchResult(
            question=question,
            answer=draft,
            run_id=run_id,
            run_metadata=run_metadata,
            evidences=evidences,
            sources=sources,
            plan=plan,
            review=final_review,
            claim_ledger=claim_ledger,
            status=status,
            metrics=metrics,
            trace=list(trace),
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _execute_plan(
        self,
        plan: ResearchPlan,
        state: ResearchRunState,
        evidences: list[Evidence],
        trace: list[dict],
        context: RunContext,
        swarm: SwarmPlan,
        budget: BudgetController,
        semaphore: asyncio.Semaphore,
    ) -> None:
        replans = int(budget.used.get(BudgetResource.REPLAN.value, 0))
        while not state.complete:
            if budget.deadline_exceeded():
                self._cancel_unfinished(state, "deadline_exceeded")
                self._record_stop(trace, budget, "deadline_exceeded")
                break
            ready = state.refresh_ready()
            if not ready:
                state.degrade_blocked()
                if state.complete:
                    break
                # Defensive close for malformed runtime states that validation missed.
                for runtime in state.by_status(TaskStatus.PENDING, TaskStatus.READY):
                    runtime.transition(TaskStatus.SKIPPED, "scheduler_deadlock")
                break

            batch = sorted(ready, key=lambda runtime: runtime.spec.priority, reverse=True)
            executions = await asyncio.gather(
                *(
                    self._execute_task(
                        runtime,
                        context,
                        swarm,
                        budget,
                        semaphore,
                    )
                    for runtime in batch
                )
            )
            batch_failures = [
                execution
                for execution in executions
                if not execution.succeeded and execution.failure_kind != "worker_budget_exhausted"
            ]
            for execution in executions:
                unique_execution_evidence: list[Evidence] = []
                for evidence in execution.evidences:
                    memory_id, outcome = self.memory.add_evidence(evidence)
                    evidence.metadata["memory_outcome"] = outcome
                    evidence.metadata["memory_id"] = memory_id
                    if outcome not in {"duplicate", "conflict_kept_existing"}:
                        unique_execution_evidence.append(evidence)
                evidences.extend(unique_execution_evidence)

            batch_evidences = [
                evidence for execution in executions for evidence in execution.evidences
            ]
            evidence_stop = (
                budget.observe_evidence_batch(batch_evidences) if not state.complete else None
            )
            if evidence_stop:
                self._cancel_unfinished(state, evidence_stop)
                self._record_stop(
                    trace,
                    budget,
                    evidence_stop,
                    evidence_count=len(evidences),
                )
                break

            ratio = len(batch_failures) / len(executions) if executions else 0.0
            if (
                batch_failures
                and ratio >= self.config.batch_failure_threshold
                and replans < swarm.max_replans
                and budget.try_consume(BudgetResource.REPLAN)
            ):
                reason = ", ".join(
                    sorted({item.failure_kind or "unknown" for item in batch_failures})
                )
                replacements = await self._trace_component(
                    trace,
                    context,
                    component_id="lead-planner",
                    role="planner",
                    operation=f"replan_{replans + 1}",
                    input_value={
                        "failed_tasks": [item.runtime.spec.subtask_id for item in batch_failures],
                        "reason": reason,
                    },
                    awaitable=self.planner.replan(
                        plan,
                        [item.runtime.spec for item in batch_failures],
                        reason,
                    ),
                )
                if replacements:
                    replans += 1
                    plan.version += 1
                    plan.subtasks.extend(replacements)
                    state.add_subtasks(replacements)
                    if self.run_store is not None:
                        self.run_store.save_plan(
                            context.run_id,
                            plan,
                            swarm,
                            context.metadata,
                        )
                    trace.append(
                        {
                            "at": utc_now(),
                            "event": "dynamic_replan",
                            "version": plan.version,
                            "replacements": [item.subtask_id for item in replacements],
                            "reason": reason,
                        }
                    )

            # Level-one degradation makes downstream work possible with partial data.
            for execution in batch_failures:
                runtime = execution.runtime
                if runtime.status in {TaskStatus.FAILED, TaskStatus.TIMED_OUT}:
                    runtime.transition(TaskStatus.DEGRADED, execution.failure_kind)

    async def _execute_task(
        self,
        runtime: TaskRuntime,
        context: RunContext,
        swarm: SwarmPlan,
        budget: BudgetController,
        semaphore: asyncio.Semaphore,
    ) -> TaskExecution:
        if not budget.try_consume(BudgetResource.WORKER_INVOCATION):
            runtime.transition(TaskStatus.SKIPPED, "worker_budget_exhausted")
            context.emit(
                "budget_denied",
                resource=BudgetResource.WORKER_INVOCATION.value,
                task_id=runtime.spec.subtask_id,
            )
            return TaskExecution(runtime, [], False, "worker_budget_exhausted")
        async with semaphore:
            runtime.transition(TaskStatus.RUNNING)
            try:
                evidences = await asyncio.wait_for(
                    self._run_worker(runtime.spec, context, swarm),
                    timeout=self.config.task_timeout_seconds,
                )
                if not evidences:
                    runtime.transition(TaskStatus.FAILED, "no_evidence")
                    return TaskExecution(runtime, [], False, "no_evidence")
                runtime.evidences = list(evidences)
                runtime.transition(TaskStatus.SUCCEEDED)
                return TaskExecution(runtime, evidences, True)
            except asyncio.TimeoutError:
                runtime.transition(TaskStatus.TIMED_OUT, "task_timeout")
                return TaskExecution(runtime, [], False, "task_timeout")
            except asyncio.CancelledError:
                if runtime.status is TaskStatus.RUNNING:
                    runtime.transition(TaskStatus.CANCELLED, "deadline_exceeded")
                raise
            except Exception as exc:  # Worker isolation is an orchestration boundary.
                runtime.transition(TaskStatus.FAILED, f"{type(exc).__name__}: {exc}")
                return TaskExecution(runtime, [], False, type(exc).__name__)

    async def _verify_issues(
        self,
        queries: Sequence[str],
        review_round: int,
        trace: list[dict],
        context: RunContext,
        swarm: SwarmPlan,
        budget: BudgetController,
        semaphore: asyncio.Semaphore,
        agent_override: AgentSpec | None = None,
    ) -> list[Evidence]:
        tasks: list[ResearchSubtask] = []
        for index, query in enumerate(queries, start=1):
            if not budget.try_consume(BudgetResource.VERIFICATION_QUERY):
                break
            tasks.append(
                ResearchSubtask(
                    subtask_id=f"verify_r{review_round}_{index}",
                    question=query,
                    reason="Red Agent requested VERIFY/ADD evidence",
                    max_results=3,
                )
            )

        async def execute(task: ResearchSubtask) -> list[Evidence]:
            if not budget.try_consume(BudgetResource.WORKER_INVOCATION):
                context.emit(
                    "budget_denied",
                    resource=BudgetResource.WORKER_INVOCATION.value,
                    task_id=task.subtask_id,
                )
                return []
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._run_worker(
                            task,
                            context,
                            swarm,
                            agent_override=agent_override,
                        ),
                        timeout=self.config.task_timeout_seconds,
                    )
                except (asyncio.TimeoutError, Exception):
                    return []

        batches = await asyncio.gather(*(execute(task) for task in tasks))
        retrieved = [evidence for batch in batches for evidence in batch]
        evidences: list[Evidence] = []
        for evidence in retrieved:
            memory_id, outcome = self.memory.add_evidence(evidence)
            evidence.metadata["memory_outcome"] = outcome
            evidence.metadata["memory_id"] = memory_id
            if outcome not in {"duplicate", "conflict_kept_existing"}:
                evidences.append(evidence)
        trace.append(
            {
                "at": utc_now(),
                "event": "review_verification",
                "round": review_round,
                "queries": list(queries),
                "evidence_count": len(evidences),
                "agent_role": (agent_override.role if agent_override is not None else None),
            }
        )
        return evidences

    async def _run_worker(
        self,
        task: ResearchSubtask,
        context: RunContext,
        swarm: SwarmPlan,
        *,
        agent_override: AgentSpec | None = None,
    ) -> list[Evidence]:
        agent_spec = agent_override or swarm.select_agent(task, self.worker_spec)
        execution = await self.runtime.invoke(
            agent_spec,
            RoleAwareAgentAdapter(self.worker, agent_spec),
            task,
            context,
        )
        output = execution.output
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise TypeError("Research worker must return a sequence of Evidence")
        evidences = list(output)
        if any(not isinstance(item, Evidence) for item in evidences):
            raise TypeError("Research worker returned a non-Evidence item")
        for evidence in evidences:
            evidence.metadata.setdefault("agent_id", agent_spec.agent_id)
            evidence.metadata.setdefault("agent_role", agent_spec.role)
        return evidences

    async def _trace_component(
        self,
        trace: list[dict[str, Any]],
        context: RunContext,
        *,
        component_id: str,
        role: str,
        operation: str,
        input_value: Any,
        awaitable: Any,
    ) -> Any:
        """Trace non-worker agents with the same telemetry contract as workers."""

        started_at = utc_now()
        trace_id = str(context.metadata.get("trace_id") or context.run_id)
        span_id = stable_span_id(
            trace_id,
            "component",
            f"{component_id}:{operation}:{uuid.uuid4().hex}",
        )
        usage_before = context.usage_snapshot()
        input_summary = summarize_value(input_value)
        trace.append(
            {
                "at": started_at,
                "event": "component_started",
                "span_id": span_id,
                "component_id": component_id,
                "agent_id": component_id,
                "role": role,
                "operation": operation,
                "input_summary": input_summary,
            }
        )
        try:
            output = await awaitable
        except BaseException as exc:
            finished_at = utc_now()
            telemetry = usage_delta(usage_before, context.usage_snapshot())
            trace.append(
                {
                    "at": finished_at,
                    "event": "component_failed",
                    "span_id": span_id,
                    "component_id": component_id,
                    "agent_id": component_id,
                    "role": role,
                    "operation": operation,
                    "duration_ms": duration_ms(started_at, finished_at),
                    "retry_count": telemetry["retry_count"],
                    "token_usage": {
                        "prompt": telemetry["prompt_tokens"],
                        "completion": telemetry["completion_tokens"],
                        "total": telemetry["total_tokens"],
                        "measurement": telemetry["measurement"],
                    },
                    "cost_usd": telemetry["cost_usd"],
                    "error_type": type(exc).__name__,
                }
            )
            raise
        finished_at = utc_now()
        telemetry = usage_delta(usage_before, context.usage_snapshot())
        trace.append(
            {
                "at": finished_at,
                "event": "component_completed",
                "span_id": span_id,
                "component_id": component_id,
                "agent_id": component_id,
                "role": role,
                "operation": operation,
                "duration_ms": duration_ms(started_at, finished_at),
                "input_summary": input_summary,
                "output_summary": summarize_value(output),
                "retry_count": telemetry["retry_count"],
                "token_usage": {
                    "prompt": telemetry["prompt_tokens"],
                    "completion": telemetry["completion_tokens"],
                    "total": telemetry["total_tokens"],
                    "measurement": telemetry["measurement"],
                },
                "cost_usd": telemetry["cost_usd"],
            }
        )
        return output

    @staticmethod
    def _swarm_role_decision(swarm: SwarmPlan) -> dict[str, Any]:
        selected = [agent.role for agent in swarm.agents]
        canonical = [
            "researcher",
            "scope_researcher",
            "evidence_researcher",
            "counter_researcher",
            "impact_analyst",
            "evidence_verifier",
        ]
        reasons = {
            "researcher": "Focused or fixed orchestration uses the base retrieval worker.",
            "scope_researcher": "Clarifies scope and definitions for a multi-stage task.",
            "evidence_researcher": "Collects primary evidence for standard and complex tasks.",
            "counter_researcher": "Actively searches for counter-evidence and limitations.",
            "impact_analyst": "Complex tasks require risk and trade-off analysis.",
            "evidence_verifier": "Complex tasks reserve a role for disputed claim verification.",
        }
        if "dynamic_swarm_disabled" in swarm.signals:
            action = "fixed"
            explanation = (
                "Dynamic Swarm is disabled; the base worker and configured budgets are kept."
            )
        elif len(selected) == 1:
            action = "scale_down"
            explanation = "Complexity stayed below the role-expansion threshold."
        elif len(selected) >= 5:
            action = "scale_up"
            explanation = "Complexity signals crossed the complex threshold, enabling all roles."
        else:
            action = "balanced"
            explanation = "The standard tier adds scope, evidence and counter-evidence roles."
        return {
            "action": action,
            "complexity_level": swarm.level.value,
            "complexity_score": swarm.complexity_score,
            "signals": list(swarm.signals),
            "selected_roles": selected,
            "omitted_roles": [role for role in canonical if role not in selected],
            "role_reasons": {
                role: reasons.get(role, "Selected by the Swarm policy.") for role in selected
            },
            "explanation": explanation,
            "max_concurrency": swarm.max_concurrency,
            "max_worker_invocations": swarm.max_worker_invocations,
        }

    @staticmethod
    def _record_stop(
        trace: list[dict[str, Any]],
        budget: BudgetController,
        reason: str,
        **details: Any,
    ) -> None:
        limits = budget.limits
        observed: dict[str, Any] = {}
        threshold: dict[str, Any] = {}
        if reason == "deadline_exceeded":
            observed["elapsed_seconds"] = round(budget.elapsed_seconds, 3)
            threshold["max_elapsed_seconds"] = budget.swarm.max_elapsed_seconds
        elif reason == "review_budget_exhausted":
            observed["review_rounds"] = budget.used[BudgetResource.REVIEW_ROUND.value]
            threshold["max_review_rounds"] = limits[BudgetResource.REVIEW_ROUND.value]
        elif reason == "evidence_saturated":
            observed["stagnant_evidence_batches"] = budget.snapshot().get(
                "stagnant_evidence_batches"
            )
            threshold["patience"] = budget.swarm.evidence_stagnation_patience
        trace.append(
            {
                "at": utc_now(),
                "event": "swarm_stop",
                "reason": reason,
                "trigger": {
                    "observed": observed,
                    "threshold": threshold,
                    "budget_used": dict(budget.used),
                    "budget_remaining": {
                        name: max(0, limit - budget.used[name]) for name, limit in limits.items()
                    },
                },
                **details,
            }
        )

    @staticmethod
    def _cancel_unfinished(state: ResearchRunState, reason: str) -> None:
        for runtime in state.tasks.values():
            if runtime.status in {TaskStatus.PENDING, TaskStatus.READY}:
                runtime.transition(TaskStatus.CANCELLED, reason)

    def _checkpoint_report(
        self,
        run_id: str,
        stage: str,
        report: str,
        **metadata: Any,
    ) -> None:
        if self.run_store is not None:
            self.run_store.save_report(run_id, stage, report, metadata=metadata)


def default_memory_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "research_memory.sqlite3"


def default_run_store_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "research_runs.sqlite3"
