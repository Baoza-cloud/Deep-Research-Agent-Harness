"""Durable run checkpoints and deterministic trace replay.

The run store is deliberately separate from cross-agent semantic memory.  It
stores orchestration state, not reusable knowledge, and can therefore enforce
``run_id`` idempotency without coupling recovery to a retrieval backend.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .harness import AgentSpec
from .schemas import (
    ClaimEvidenceLink,
    ClaimLedger,
    ClaimRecord,
    Evidence,
    RepairAction,
    ResearchPlan,
    ResearchResult,
    ResearchSubtask,
    ReviewIssue,
    ReviewResult,
    SupportVerdict,
    TaskStatus,
    utc_now,
)
from .swarm import ComplexityLevel, SwarmPlan


class RunStoreError(RuntimeError):
    """Base class for durable-run failures."""


class RunAlreadyExistsError(RunStoreError):
    """Raised when a non-resume submission reuses an unfinished ``run_id``."""


class RunNotFoundError(RunStoreError):
    """Raised when a requested checkpoint does not exist."""


class RunInProgressError(RunStoreError):
    """Raised when another owner holds the active run lease."""


class RunObjectiveMismatchError(RunStoreError):
    """Raised when the same ``run_id`` is submitted with another objective."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _dump(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence_from_dict(data: dict[str, Any]) -> Evidence:
    return Evidence(**data)


def subtask_from_dict(data: dict[str, Any]) -> ResearchSubtask:
    payload = dict(data)
    payload["status"] = TaskStatus(payload.get("status", TaskStatus.PENDING.value))
    return ResearchSubtask(**payload)


def plan_from_dict(data: dict[str, Any]) -> ResearchPlan:
    payload = dict(data)
    payload["subtasks"] = [subtask_from_dict(item) for item in payload.get("subtasks", [])]
    return ResearchPlan(**payload)


def swarm_from_dict(data: dict[str, Any]) -> SwarmPlan:
    agents = tuple(
        AgentSpec(
            agent_id=item["agent_id"],
            role=item["role"],
            description=item.get("description", ""),
            tool_names=tuple(item.get("tool_names", [])),
            model=item.get("model"),
            metadata=item.get("metadata", {}),
        )
        for item in data.get("agents", [])
    )
    return SwarmPlan(
        level=ComplexityLevel(data["level"]),
        complexity_score=float(data["complexity_score"]),
        signals=tuple(data.get("signals", [])),
        agents=agents,
        max_concurrency=int(data["max_concurrency"]),
        max_worker_invocations=int(data["max_worker_invocations"]),
        max_replans=int(data["max_replans"]),
        max_review_rounds=int(data["max_review_rounds"]),
        max_verification_queries=int(data["max_verification_queries"]),
        max_elapsed_seconds=float(data["max_elapsed_seconds"]),
        diminishing_returns_patience=int(data["diminishing_returns_patience"]),
        evidence_stagnation_patience=int(data["evidence_stagnation_patience"]),
        min_score_improvement=float(data["min_score_improvement"]),
    )


def _review_from_dict(data: dict[str, Any] | None) -> ReviewResult | None:
    if data is None:
        return None
    payload = dict(data)
    payload["structured_issues"] = [
        ReviewIssue(
            **{
                **item,
                "action": RepairAction(item.get("action", RepairAction.VERIFY.value)),
            }
        )
        for item in payload.get("structured_issues", [])
    ]
    return ReviewResult(**payload)


def _ledger_from_dict(data: dict[str, Any] | None) -> ClaimLedger | None:
    if data is None:
        return None
    claims: list[ClaimRecord] = []
    for item in data.get("claims", []):
        payload = dict(item)
        payload["verdict"] = SupportVerdict(payload.get("verdict", SupportVerdict.UNKNOWN.value))
        payload["links"] = [
            ClaimEvidenceLink(
                **{
                    **link,
                    "verdict": SupportVerdict(link.get("verdict", SupportVerdict.UNKNOWN.value)),
                }
            )
            for link in payload.get("links", [])
        ]
        claims.append(ClaimRecord(**payload))
    return ClaimLedger(
        claims=claims,
        verification_mode=data.get("verification_mode", "rules"),
        metrics=data.get("metrics", {}),
    )


def result_from_dict(data: dict[str, Any]) -> ResearchResult:
    """Rehydrate a persisted final result without executing external calls."""

    return ResearchResult(
        question=data["question"],
        answer=data["answer"],
        run_id=data.get("run_id"),
        run_metadata=data.get("run_metadata", {}),
        evidences=[evidence_from_dict(item) for item in data.get("evidences", [])],
        sources=data.get("sources", []),
        plan=plan_from_dict(data["plan"]) if data.get("plan") else None,
        review=_review_from_dict(data.get("review")),
        claim_ledger=_ledger_from_dict(data.get("claim_ledger")),
        status=data.get("status", "completed"),
        metrics=data.get("metrics", {}),
        trace=data.get("trace", []),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
    )


@dataclass(frozen=True)
class NodeCheckpoint:
    subtask_id: str
    spec: ResearchSubtask
    status: TaskStatus
    attempts: int
    error: str | None
    started_at: str | None
    finished_at: str | None
    history: list[dict[str, Any]]
    evidences: list[Evidence]


@dataclass(frozen=True)
class RunCheckpoint:
    run_id: str
    objective: str
    lifecycle_status: str
    phase: str
    started_at: str
    updated_at: str
    finished_at: str | None
    metadata: dict[str, Any]
    plan: ResearchPlan | None
    swarm: SwarmPlan | None
    budget: dict[str, Any]
    current_report: str | None
    last_error: str | None
    result: ResearchResult | None
    revision: int
    nodes: dict[str, NodeCheckpoint]


class RunStore:
    """SQLite-backed orchestration store with transactional node checkpoints."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._lock = threading.RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    plan_json TEXT,
                    swarm_json TEXT,
                    budget_json TEXT NOT NULL DEFAULT '{}',
                    current_report TEXT,
                    last_error TEXT,
                    result_json TEXT,
                    owner_id TEXT,
                    lease_expires_at REAL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    schema_version INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS research_dag_nodes (
                    run_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    history_json TEXT NOT NULL DEFAULT '[]',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    output_sha256 TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, subtask_id),
                    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    event_sha256 TEXT NOT NULL,
                    UNIQUE (run_id, event_key),
                    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    report TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, stage, report_sha256),
                    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_research_events_run_sequence
                ON research_events(run_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_research_reports_run_id
                ON research_reports(run_id, report_id);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_run(
        self,
        run_id: str,
        objective: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        now = utc_now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO research_runs (
                    run_id, objective, lifecycle_status, phase, started_at, updated_at,
                    metadata_json, schema_version
                ) VALUES (?, ?, 'created', 'created', ?, ?, ?, ?)
                """,
                (run_id, objective, now, now, _dump(metadata or {}), self.SCHEMA_VERSION),
            )
            created = cursor.rowcount == 1
            row = self._connection.execute(
                "SELECT objective FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunStoreError(f"Unable to create run: {run_id}")
            if row["objective"] != objective:
                raise RunObjectiveMismatchError(
                    f"run_id {run_id!r} belongs to another research objective"
                )
            return created

    def acquire(
        self,
        run_id: str,
        owner_id: str,
        *,
        resume: bool,
        lease_seconds: float = 30.0,
    ) -> None:
        now = time.time()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT owner_id, lease_expires_at FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            active = (
                row["owner_id"]
                and row["owner_id"] != owner_id
                and float(row["lease_expires_at"] or 0) > now
            )
            if active:
                raise RunInProgressError(f"run_id {run_id!r} is already executing")
            self._connection.execute(
                """
                UPDATE research_runs
                SET owner_id = ?, lease_expires_at = ?, lifecycle_status = 'running',
                    updated_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (owner_id, now + lease_seconds, utc_now(), run_id),
            )

    def _heartbeat(self, run_id: str, lease_seconds: float = 300.0) -> None:
        self._connection.execute(
            "UPDATE research_runs SET lease_expires_at = ?, updated_at = ? WHERE run_id = ?",
            (time.time() + lease_seconds, utc_now(), run_id),
        )

    def set_phase(self, run_id: str, phase: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE research_runs
                SET phase = ?, updated_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (phase, utc_now(), run_id),
            )
            self._heartbeat(run_id)

    def save_plan(
        self,
        run_id: str,
        plan: ResearchPlan,
        swarm: SwarmPlan,
        metadata: dict[str, Any],
    ) -> None:
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE research_runs
                SET plan_json = ?, swarm_json = ?, metadata_json = ?, updated_at = ?,
                    revision = revision + 1
                WHERE run_id = ?
                """,
                (_dump(plan), _dump(swarm), _dump(metadata), now, run_id),
            )
            for task in plan.subtasks:
                self._connection.execute(
                    """
                    INSERT INTO research_dag_nodes (
                        run_id, subtask_id, spec_json, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, subtask_id) DO UPDATE SET
                        spec_json = excluded.spec_json,
                        updated_at = excluded.updated_at
                    """,
                    (run_id, task.subtask_id, _dump(task), task.status.value, now),
                )
            self._heartbeat(run_id)

    @staticmethod
    def _event_identity(event: dict[str, Any]) -> tuple[str, str, str]:
        payload = _dump(event)
        digest = _sha256(payload)
        event_key = str(event.get("event_key") or digest)
        occurred_at = str(event.get("at") or utc_now())
        return event_key, occurred_at, payload

    def _insert_event(self, run_id: str, event: dict[str, Any]) -> None:
        event_key, occurred_at, payload = self._event_identity(event)
        self._connection.execute(
            """
            INSERT OR IGNORE INTO research_events (
                run_id, event_key, occurred_at, event_json, event_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, event_key, occurred_at, payload, _sha256(payload)),
        )

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._insert_event(run_id, event)
            self._heartbeat(run_id)

    def append_events(self, run_id: str, events: Iterable[dict[str, Any]]) -> None:
        with self._lock, self._connection:
            for event in events:
                self._insert_event(run_id, event)
            self._heartbeat(run_id)

    def checkpoint_node(
        self,
        run_id: str,
        runtime: Any,
        *,
        event: dict[str, Any] | None = None,
    ) -> None:
        evidence_payload = _dump(getattr(runtime, "evidences", []))
        output_hash = _sha256(evidence_payload) if evidence_payload != "[]" else None
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT status, output_sha256 FROM research_dag_nodes
                WHERE run_id = ? AND subtask_id = ?
                """,
                (run_id, runtime.spec.subtask_id),
            ).fetchone()
            if (
                existing
                and existing["status"] == TaskStatus.SUCCEEDED.value
                and runtime.status is TaskStatus.SUCCEEDED
                and existing["output_sha256"]
                and existing["output_sha256"] != output_hash
            ):
                raise RunStoreError(
                    f"Completed node {runtime.spec.subtask_id!r} produced a different output"
                )
            self._connection.execute(
                """
                INSERT INTO research_dag_nodes (
                    run_id, subtask_id, spec_json, status, attempts, error,
                    started_at, finished_at, history_json, evidence_json,
                    output_sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, subtask_id) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    status = excluded.status,
                    attempts = excluded.attempts,
                    error = excluded.error,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    history_json = excluded.history_json,
                    evidence_json = excluded.evidence_json,
                    output_sha256 = excluded.output_sha256,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    runtime.spec.subtask_id,
                    _dump(runtime.spec),
                    runtime.status.value,
                    runtime.attempts,
                    runtime.error,
                    runtime.started_at,
                    runtime.finished_at,
                    _dump(runtime.history),
                    evidence_payload,
                    output_hash,
                    utc_now(),
                ),
            )
            if event is not None:
                self._insert_event(run_id, event)
            self._heartbeat(run_id)

    def save_budget(self, run_id: str, budget: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE research_runs
                SET budget_json = ?, updated_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (_dump(budget), utc_now(), run_id),
            )
            self._heartbeat(run_id)

    def save_report(
        self,
        run_id: str,
        stage: str,
        report: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        digest = _sha256(report)
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO research_reports (
                    run_id, stage, report, report_sha256, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, stage, report, digest, _dump(metadata or {}), now),
            )
            self._connection.execute(
                """
                UPDATE research_runs
                SET current_report = ?, updated_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (report, now, run_id),
            )
            self._insert_event(
                run_id,
                {
                    "at": now,
                    "event": "intermediate_report_checkpointed",
                    "stage": stage,
                    "report_sha256": digest,
                    "metadata": metadata or {},
                },
            )
            self._heartbeat(run_id)
        return digest

    def mark_interrupted(self, run_id: str, exc: BaseException) -> None:
        now = utc_now()
        error = f"{type(exc).__name__}: {exc}"
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE research_runs
                SET lifecycle_status = 'failed', phase = 'interrupted', last_error = ?,
                    owner_id = NULL, lease_expires_at = NULL, updated_at = ?,
                    revision = revision + 1
                WHERE run_id = ?
                """,
                (error, now, run_id),
            )
            self._insert_event(
                run_id,
                {
                    "at": now,
                    "event": "run_interrupted",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def finalize(self, run_id: str, result: ResearchResult) -> None:
        now = result.finished_at or utc_now()
        lifecycle = "timed_out" if result.status == "partial_timeout" else "completed"
        payload = result.to_dict()
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE research_runs
                SET lifecycle_status = ?, phase = 'finalized', finished_at = ?,
                    result_json = ?, current_report = ?, owner_id = NULL,
                    lease_expires_at = NULL, updated_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (lifecycle, now, _dump(payload), result.answer, now, run_id),
            )
            self._insert_event(
                run_id,
                {
                    "at": now,
                    "event": "run_finalized",
                    "status": result.status,
                    "result_sha256": _sha256(_dump(payload)),
                },
            )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_json, event_sha256 FROM research_events
                WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            if _sha256(row["event_json"]) != row["event_sha256"]:
                raise RunStoreError(f"Trace integrity check failed for run {run_id!r}")
            events.append(json.loads(row["event_json"]))
        return events

    def reports(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT stage, report, report_sha256, metadata_json, created_at
                FROM research_reports WHERE run_id = ? ORDER BY report_id
                """,
                (run_id,),
            ).fetchall()
        return [self._verified_report(row, run_id) for row in rows]

    @staticmethod
    def _verified_report(row: sqlite3.Row, run_id: str) -> dict[str, Any]:
        if _sha256(row["report"]) != row["report_sha256"]:
            raise RunStoreError(f"Report integrity check failed for run {run_id!r}")
        return {
            "stage": row["stage"],
            "report": row["report"],
            "report_sha256": row["report_sha256"],
            "metadata": _load(row["metadata_json"], {}),
            "created_at": row["created_at"],
        }

    def load(self, run_id: str) -> RunCheckpoint:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            node_rows = self._connection.execute(
                """
                SELECT * FROM research_dag_nodes
                WHERE run_id = ? ORDER BY rowid
                """,
                (run_id,),
            ).fetchall()
        nodes: dict[str, NodeCheckpoint] = {}
        for item in node_rows:
            evidence_payload = item["evidence_json"] or "[]"
            if item["output_sha256"] and _sha256(evidence_payload) != item["output_sha256"]:
                raise RunStoreError(f"Node output integrity check failed: {item['subtask_id']}")
            nodes[item["subtask_id"]] = NodeCheckpoint(
                subtask_id=item["subtask_id"],
                spec=subtask_from_dict(_load(item["spec_json"], {})),
                status=TaskStatus(item["status"]),
                attempts=int(item["attempts"]),
                error=item["error"],
                started_at=item["started_at"],
                finished_at=item["finished_at"],
                history=_load(item["history_json"], []),
                evidences=[evidence_from_dict(value) for value in _load(evidence_payload, [])],
            )
        result_payload = _load(row["result_json"], None)
        return RunCheckpoint(
            run_id=row["run_id"],
            objective=row["objective"],
            lifecycle_status=row["lifecycle_status"],
            phase=row["phase"],
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
            metadata=_load(row["metadata_json"], {}),
            plan=plan_from_dict(_load(row["plan_json"], {})) if row["plan_json"] else None,
            swarm=swarm_from_dict(_load(row["swarm_json"], {})) if row["swarm_json"] else None,
            budget=_load(row["budget_json"], {}),
            current_report=row["current_report"],
            last_error=row["last_error"],
            result=result_from_dict(result_payload) if result_payload else None,
            revision=int(row["revision"]),
            nodes=nodes,
        )

    def replay(self, run_id: str) -> dict[str, Any]:
        """Rebuild a stable view from persisted events and immutable outputs only."""

        checkpoint = self.load(run_id)
        events = self.events(run_id)
        reports = self.reports(run_id)
        replayed_states: dict[str, str] = {}
        replayed_budget: dict[str, Any] = {}
        for event in events:
            if event.get("event") == "task_transition":
                replayed_states[str(event["task_id"])] = str(event["to"])
            elif event.get("event") == "budget_checkpointed":
                replayed_budget = dict(event.get("budget") or {})
        for task_id, node in checkpoint.nodes.items():
            replayed = replayed_states.get(task_id)
            if replayed is not None and replayed != node.status.value:
                raise RunStoreError(
                    f"Trace/node state mismatch for {task_id!r}: "
                    f"{replayed!r} != {node.status.value!r}"
                )
            replayed_states.setdefault(task_id, node.status.value)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": run_id,
            "objective": checkpoint.objective,
            "lifecycle_status": checkpoint.lifecycle_status,
            "phase": checkpoint.phase,
            "revision": checkpoint.revision,
            "trace_integrity_verified": True,
            "plan": _jsonable(checkpoint.plan) if checkpoint.plan else None,
            "node_states": replayed_states,
            "nodes": {
                task_id: {
                    "status": node.status.value,
                    "attempts": node.attempts,
                    "error": node.error,
                    "evidences": _jsonable(node.evidences),
                }
                for task_id, node in checkpoint.nodes.items()
            },
            "budget": replayed_budget or checkpoint.budget,
            "reports": reports,
            "current_report": checkpoint.current_report,
            "trace": events,
            "result": checkpoint.result.to_dict() if checkpoint.result else None,
        }

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class PersistedTrace(list[dict[str, Any]]):
    """A list-compatible Trace that commits every new event before returning."""

    def __init__(
        self,
        initial: Iterable[dict[str, Any]],
        *,
        store: RunStore | None,
        run_id: str,
    ):
        super().__init__(initial)
        self.store = store
        self.run_id = run_id

    def append(self, event: dict[str, Any]) -> None:
        if self.store is not None:
            self.store.append_event(self.run_id, event)
        super().append(event)

    def extend(self, events: Iterable[dict[str, Any]]) -> None:
        rows = list(events)
        if self.store is not None:
            self.store.append_events(self.run_id, rows)
        super().extend(rows)

    def append_already_persisted(self, event: dict[str, Any]) -> None:
        super().append(event)
