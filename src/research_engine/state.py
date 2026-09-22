"""Validated nine-state lifecycle for every DAG node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .schemas import ResearchPlan, ResearchSubtask, TERMINAL_STATUSES, TaskStatus, utc_now


class StateTransitionError(RuntimeError):
    pass


VALID_TRANSITIONS = {
    TaskStatus.PENDING: {TaskStatus.READY, TaskStatus.SKIPPED, TaskStatus.CANCELLED},
    TaskStatus.READY: {TaskStatus.RUNNING, TaskStatus.SKIPPED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.TIMED_OUT,
        TaskStatus.CANCELLED,
    },
    TaskStatus.FAILED: {TaskStatus.READY, TaskStatus.DEGRADED},
    TaskStatus.TIMED_OUT: {TaskStatus.READY, TaskStatus.DEGRADED},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.DEGRADED: set(),
    TaskStatus.SKIPPED: set(),
    TaskStatus.CANCELLED: set(),
}


@dataclass
class TaskRuntime:
    spec: ResearchSubtask
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    history: List[dict] = field(default_factory=list)

    def transition(self, target: TaskStatus, reason: Optional[str] = None) -> None:
        target = TaskStatus(target)
        allowed = VALID_TRANSITIONS[self.status]
        if target not in allowed:
            raise StateTransitionError(
                f"Invalid transition for {self.spec.subtask_id}: "
                f"{self.status.value} -> {target.value}"
            )

        previous = self.status
        self.status = target
        self.spec.status = target
        now = utc_now()
        if target is TaskStatus.RUNNING:
            self.attempts += 1
            self.started_at = now
        if target in TERMINAL_STATUSES:
            self.finished_at = now
        if reason:
            self.error = reason
        self.history.append(
            {"at": now, "from": previous.value, "to": target.value, "reason": reason}
        )


class ResearchRunState:
    """Holds task runtimes and derives which DAG nodes are runnable."""

    def __init__(self, plan: ResearchPlan):
        self.tasks: Dict[str, TaskRuntime] = {}
        self.trace: List[dict] = []
        self.add_subtasks(plan.subtasks)

    def add_subtasks(self, subtasks: Iterable[ResearchSubtask]) -> None:
        new_tasks = list(subtasks)
        known = set(self.tasks)
        incoming = {task.subtask_id for task in new_tasks}
        if len(incoming) != len(new_tasks):
            raise ValueError("Duplicate subtask_id in research plan")

        all_ids = known | incoming
        for task in new_tasks:
            if task.subtask_id in known:
                raise ValueError(f"Duplicate subtask_id: {task.subtask_id}")
            missing = set(task.dependencies) - all_ids
            if missing:
                raise ValueError(
                    f"Subtask {task.subtask_id} has unknown dependencies: {sorted(missing)}"
                )
            self.tasks[task.subtask_id] = TaskRuntime(task, TaskStatus(task.status))

        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"Research plan contains a cycle at {task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self.tasks[task_id].spec.dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)

    def refresh_ready(self) -> List[TaskRuntime]:
        ready: List[TaskRuntime] = []
        for runtime in self.tasks.values():
            if runtime.status is not TaskStatus.PENDING:
                continue
            dependencies = [self.tasks[item].status for item in runtime.spec.dependencies]
            if all(
                state in {TaskStatus.SUCCEEDED, TaskStatus.DEGRADED, TaskStatus.SKIPPED}
                for state in dependencies
            ):
                runtime.transition(TaskStatus.READY, "dependencies_satisfied")
                ready.append(runtime)
        return ready

    def degrade_blocked(self) -> None:
        """Close nodes whose upstream dependencies failed permanently."""

        for runtime in self.tasks.values():
            if runtime.status is not TaskStatus.PENDING:
                continue
            dependencies = [self.tasks[item].status for item in runtime.spec.dependencies]
            if any(
                state in {TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED}
                for state in dependencies
            ):
                runtime.transition(TaskStatus.SKIPPED, "upstream_dependency_failed")

    @property
    def complete(self) -> bool:
        return all(runtime.status in TERMINAL_STATUSES for runtime in self.tasks.values())

    def by_status(self, *statuses: TaskStatus) -> List[TaskRuntime]:
        wanted = {TaskStatus(status) for status in statuses}
        return [runtime for runtime in self.tasks.values() if runtime.status in wanted]

    def snapshot(self) -> Dict[str, dict]:
        return {
            task_id: {
                "status": runtime.status.value,
                "attempts": runtime.attempts,
                "error": runtime.error,
                "dependencies": list(runtime.spec.dependencies),
            }
            for task_id, runtime in self.tasks.items()
        }
