"""Data contracts shared by the deep-research agents.

The module intentionally uses only the standard library.  It can therefore be
imported by evaluation scripts without loading an embedding model or requiring
an API key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    """The nine states in a research task lifecycle."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class CompletionStatus(str, Enum):
    """Mutually exclusive outcomes for a complete research run."""

    COMPLETED = "completed"
    COMPLETED_WITH_EVIDENCE_GAPS = "completed_with_evidence_gaps"
    COMPLETED_WITH_REVIEW_ISSUES = "completed_with_review_issues"
    PARTIAL_TIMEOUT = "partial_timeout"


TERMINAL_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.TIMED_OUT,
    TaskStatus.DEGRADED,
    TaskStatus.SKIPPED,
    TaskStatus.CANCELLED,
}


@dataclass
class ResearchSubtask:
    """One node in the planner-produced research DAG."""

    subtask_id: str
    question: str
    reason: str
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    kind: str = "search"
    max_results: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """A normalized piece of evidence returned by a research worker."""

    evidence_id: str
    subtask_id: str
    content: str
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    source_type: str = "unknown"
    score: Optional[float] = None
    published_at: Optional[str] = None
    retrieved_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchPlan:
    """Lead Agent output, represented as a validated DAG."""

    main_question: str
    objective: str
    subtasks: List[ResearchSubtask] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    version: int = 1


class RepairAction(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    VERIFY = "VERIFY"


class PatchPosition(str, Enum):
    BEFORE = "before"
    AFTER = "after"


class SupportVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    UNCITED = "uncited"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ReviewIssue:
    issue_id: str
    dimension: str
    description: str
    severity: int = 2
    action: RepairAction = RepairAction.VERIFY
    target: Optional[str] = None
    repair_query: Optional[str] = None
    category: str = "unknown"
    fingerprint: str = ""
    repeat_count: int = 1


@dataclass
class ReviewResult:
    """Red Agent critique and convergence signals."""

    passed: bool
    issues: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    repair_queries: List[str] = field(default_factory=list)
    structured_issues: List[ReviewIssue] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    converged: bool = False
    oscillating: bool = False


@dataclass
class ClaimEvidenceLink:
    evidence_id: str
    verdict: SupportVerdict
    confidence: float = 0.0
    rationale: str = ""
    cited: bool = True
    conflict_types: List[str] = field(default_factory=list)
    supported_aspects: List[str] = field(default_factory=list)
    unsupported_aspects: List[str] = field(default_factory=list)


@dataclass
class ClaimRecord:
    claim_id: str
    text: str
    section: str = ""
    source_text: str = ""
    line_number: int = 0
    sentence_index: int = 0
    citations: List[str] = field(default_factory=list)
    verdict: SupportVerdict = SupportVerdict.UNKNOWN
    confidence: float = 0.0
    links: List[ClaimEvidenceLink] = field(default_factory=list)


@dataclass
class ClaimLedger:
    claims: List[ClaimRecord] = field(default_factory=list)
    verification_mode: str = "rules"
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReportPatch:
    """One deterministic edit proposed by the Blue Agent."""

    patch_id: str
    action: RepairAction
    target: str
    replacement: str = ""
    position: PatchPosition = PatchPosition.AFTER
    evidence_ids: List[str] = field(default_factory=list)
    issue_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class PatchRejection:
    patch_id: str
    reason: str


@dataclass
class PatchApplicationResult:
    report: str
    requested_count: int = 0
    applied: List[ReportPatch] = field(default_factory=list)
    rejected: List[PatchRejection] = field(default_factory=list)
    validation_ledgers: Dict[str, ClaimLedger] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.applied)


@dataclass
class ResearchResult:
    """Serializable final output of a complete research run."""

    question: str
    answer: str
    run_id: Optional[str] = None
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    evidences: List[Evidence] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    plan: Optional[ResearchPlan] = None
    review: Optional[ReviewResult] = None
    claim_ledger: Optional[ClaimLedger] = None
    status: str = CompletionStatus.COMPLETED.value
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly primitives, including enum values."""

        def normalize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        return normalize(asdict(self))
