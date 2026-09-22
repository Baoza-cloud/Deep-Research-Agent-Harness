"""Planning agents for building and repairing research DAGs."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Awaitable, Callable, Protocol, Sequence

from .parsing import StructuredOutputError, parse_json_payload
from .schemas import ResearchPlan, ResearchSubtask, TaskStatus


LLMCallable = Callable[[str], str | Awaitable[str]]


async def call_llm(llm: LLMCallable, prompt: str) -> str:
    response = llm(prompt)
    if inspect.isawaitable(response):
        response = await response
    return str(response)


class Planner(Protocol):
    async def plan(self, question: str) -> ResearchPlan: ...

    async def replan(
        self,
        plan: ResearchPlan,
        failed_tasks: Sequence[ResearchSubtask],
        reason: str,
    ) -> list[ResearchSubtask]: ...


def validate_plan(plan: ResearchPlan) -> ResearchPlan:
    ids = [task.subtask_id for task in plan.subtasks]
    if not ids:
        raise ValueError("Planner returned no subtasks")
    if len(ids) != len(set(ids)):
        raise ValueError("Planner returned duplicate subtask IDs")
    known = set(ids)
    for task in plan.subtasks:
        missing = set(task.dependencies) - known
        if missing:
            raise ValueError(f"{task.subtask_id} depends on unknown tasks: {sorted(missing)}")

    # Kahn validation catches cycles before execution.
    indegree = {task.subtask_id: len(task.dependencies) for task in plan.subtasks}
    children: dict[str, list[str]] = {task_id: [] for task_id in ids}
    for task in plan.subtasks:
        for dependency in task.dependencies:
            children[dependency].append(task.subtask_id)
    queue = [task_id for task_id, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(ids):
        raise ValueError("Planner returned a cyclic DAG")
    return plan


class HeuristicPlanner:
    """Deterministic fallback that remains useful when an LLM is unavailable."""

    async def plan(self, question: str) -> ResearchPlan:
        subtasks = [
            ResearchSubtask(
                "scope",
                f"定义研究问题的边界、关键概念和判定标准：{question}",
                "先消除问题歧义，防止后续检索偏离目标。",
                priority=5,
            ),
            ResearchSubtask(
                "facts",
                f"搜集直接回答该问题的核心事实与一手证据：{question}",
                "建立事实主干。",
                priority=5,
                dependencies=["scope"],
            ),
            ResearchSubtask(
                "alternatives",
                f"搜集相反观点、替代解释、限制条件和失败案例：{question}",
                "减少确认偏误并为 Red Agent 提供反证。",
                priority=4,
                dependencies=["scope"],
            ),
            ResearchSubtask(
                "implications",
                f"分析事实的影响、风险、权衡与可执行建议：{question}",
                "把证据转化为决策信息。",
                priority=3,
                dependencies=["facts", "alternatives"],
            ),
        ]
        return validate_plan(
            ResearchPlan(
                main_question=question,
                objective="生成可追溯、能处理反证并明确不确定性的深度研究报告",
                subtasks=subtasks,
                assumptions=["优先使用当前企业知识库；证据不足时明确降级"],
                success_criteria=["关键结论有引用", "包含反方证据", "给出局限与不确定性"],
            )
        )

    async def replan(
        self,
        plan: ResearchPlan,
        failed_tasks: Sequence[ResearchSubtask],
        reason: str,
    ) -> list[ResearchSubtask]:
        replacements: list[ResearchSubtask] = []
        existing = {task.subtask_id for task in plan.subtasks}
        for task in failed_tasks:
            base = f"{task.subtask_id}_replan_{plan.version + 1}"
            candidate = base
            counter = 1
            while candidate in existing:
                counter += 1
                candidate = f"{base}_{counter}"
            existing.add(candidate)
            replacements.append(
                replace(
                    task,
                    subtask_id=candidate,
                    question=f"换用更窄的检索表达重新调查：{task.question}",
                    reason=f"原任务因 {reason} 失败后的降级重规划",
                    status=TaskStatus.PENDING,
                    dependencies=[],
                    metadata={**task.metadata, "replaces": task.subtask_id},
                )
            )
        return replacements


class LLMPlanner(HeuristicPlanner):
    """LLM planner with deterministic planning/replanning fallbacks."""

    def __init__(self, llm: LLMCallable):
        self.llm = llm

    async def plan(self, question: str) -> ResearchPlan:
        prompt = f"""
你是深度研究系统的 Lead Planner。把问题拆成 3-8 个可并发执行的 DAG 子任务。
只输出 JSON：
{{
  "objective": "...",
  "assumptions": ["..."],
  "success_criteria": ["..."],
  "subtasks": [
    {{"id":"...","question":"...","reason":"...","priority":1,
      "dependencies":[],"kind":"search","max_results":5}}
  ]
}}
约束：依赖必须引用更早出现的 id；问题之间要覆盖事实、反证、影响与建议。
研究问题：{question}
""".strip()
        try:
            payload = parse_json_payload(await call_llm(self.llm, prompt))
            tasks = [
                ResearchSubtask(
                    subtask_id=str(item["id"]),
                    question=str(item["question"]),
                    reason=str(item.get("reason", "")),
                    priority=int(item.get("priority", 1)),
                    dependencies=[str(value) for value in item.get("dependencies", [])],
                    kind=str(item.get("kind", "search")),
                    max_results=max(1, int(item.get("max_results", 5))),
                )
                for item in payload["subtasks"]
            ]
            if not 3 <= len(tasks) <= 8:
                raise ValueError("LLM plan must contain between 3 and 8 subtasks")
            return validate_plan(
                ResearchPlan(
                    main_question=question,
                    objective=str(payload.get("objective", question)),
                    subtasks=tasks,
                    assumptions=[str(item) for item in payload.get("assumptions", [])],
                    success_criteria=[str(item) for item in payload.get("success_criteria", [])],
                )
            )
        except (KeyError, TypeError, ValueError, StructuredOutputError):
            return await super().plan(question)

    async def replan(
        self,
        plan: ResearchPlan,
        failed_tasks: Sequence[ResearchSubtask],
        reason: str,
    ) -> list[ResearchSubtask]:
        if not failed_tasks:
            return []
        failed = "\n".join(f"- {task.subtask_id}: {task.question}" for task in failed_tasks)
        prompt = f"""
以下研究任务批量失败。请给出更窄、互不依赖的替代检索任务。
只输出 JSON 数组，每项字段为 id/question/reason/priority/max_results。
失败原因：{reason}
原问题：{plan.main_question}
失败任务：
{failed}
""".strip()
        try:
            payload = parse_json_payload(await call_llm(self.llm, prompt))
            if not isinstance(payload, list):
                raise TypeError("replan output is not a list")
            existing = {task.subtask_id for task in plan.subtasks}
            replacements = []
            for index, item in enumerate(payload, start=1):
                task_id = str(item.get("id") or f"replan_{plan.version + 1}_{index}")
                if task_id in existing:
                    task_id = f"{task_id}_v{plan.version + 1}"
                existing.add(task_id)
                replacements.append(
                    ResearchSubtask(
                        task_id,
                        str(item["question"]),
                        str(item.get("reason", "批量失败后的动态重规划")),
                        priority=int(item.get("priority", 1)),
                        dependencies=[],
                        max_results=max(1, int(item.get("max_results", 5))),
                    )
                )
            return replacements
        except (KeyError, TypeError, ValueError, StructuredOutputError):
            return await super().replan(plan, failed_tasks, reason)
