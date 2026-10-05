from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from research_engine import (
    DeepResearchAgent,
    Evidence,
    ResearchConfig,
    ResearchPlan,
    ResearchSubtask,
    ReviewResult,
    RunAlreadyExistsError,
    RunStore,
    Synthesizer,
)


class FatalWorkerCrash(BaseException):
    pass


class SequentialPlanner:
    async def plan(self, question: str) -> ResearchPlan:
        return ResearchPlan(
            question,
            "durable execution test",
            [
                ResearchSubtask("first", "first fact", "seed"),
                ResearchSubtask(
                    "second",
                    "second fact",
                    "dependent",
                    dependencies=["first"],
                ),
            ],
        )

    async def replan(self, plan, failed_tasks, reason):
        return []


class CountingWorker:
    def __init__(
        self,
        calls: Counter[str],
        *,
        crash_once: str | None = None,
        delay: float = 0.0,
    ):
        self.calls = calls
        self.crash_once = crash_once
        self.delay = delay
        self.crashed = False

    async def run(self, task: ResearchSubtask) -> list[Evidence]:
        self.calls[task.subtask_id] += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if task.subtask_id == self.crash_once and not self.crashed:
            self.crashed = True
            raise FatalWorkerCrash("simulated process death")
        return [
            Evidence(
                evidence_id=f"{task.subtask_id}-E1",
                subtask_id=task.subtask_id,
                content=f"{task.question} has durable evidence",
                source=f"fixture://{task.subtask_id}",
            )
        ]


class PassingReviewer:
    async def review(self, question, report, evidences):
        return ReviewResult(
            True,
            total_score=0.95,
            metrics={"citation_coverage": 1.0},
        )


def build_agent(store: RunStore, worker: CountingWorker, *, timeout: float = 2.0):
    return DeepResearchAgent(
        SequentialPlanner(),
        worker,
        Synthesizer(),
        reviewer=PassingReviewer(),
        run_store=store,
        config=ResearchConfig(
            max_concurrency=1,
            task_timeout_seconds=1,
            global_timeout_seconds=timeout,
            max_review_rounds=2,
            max_worker_invocations=8,
            enable_dynamic_swarm=False,
        ),
    )


class DurableRunRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="run-recovery-")
        self.store = RunStore(Path(self.temp_dir.name) / "runs.sqlite3")

    async def asyncTearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    async def test_crash_resume_reuses_completed_node_without_external_call(self):
        calls: Counter[str] = Counter()
        crashing = build_agent(
            self.store,
            CountingWorker(calls, crash_once="second"),
        )

        with self.assertRaises(FatalWorkerCrash):
            await crashing.run("crash recovery", run_id="run-crash")

        failed = self.store.load("run-crash")
        self.assertEqual(failed.lifecycle_status, "failed")
        self.assertEqual(failed.nodes["first"].status.value, "succeeded")
        self.assertEqual(failed.nodes["second"].status.value, "running")
        self.assertEqual(len(failed.nodes["first"].evidences), 1)

        resumed = build_agent(self.store, CountingWorker(calls))
        result = await resumed.resume("run-crash")

        self.assertNotEqual(result.status, "partial_timeout")
        self.assertEqual(calls["first"], 1)
        self.assertEqual(calls["second"], 2)
        self.assertEqual(
            self.store.load("run-crash").nodes["first"].attempts,
            1,
        )
        resume_event = next(item for item in result.trace if item["event"] == "run_resumed")
        self.assertEqual(resume_event["reused_nodes"], ["first"])
        self.assertEqual(resume_event["retry_nodes"], ["second"])

    async def test_global_timeout_result_can_resume_with_new_deadline_window(self):
        calls: Counter[str] = Counter()
        timed_out = build_agent(
            self.store,
            CountingWorker(calls, delay=0.1),
            timeout=0.02,
        )

        partial = await timed_out.run("timeout recovery", run_id="run-timeout")
        self.assertEqual(partial.status, "partial_timeout")
        self.assertEqual(self.store.load("run-timeout").lifecycle_status, "timed_out")

        recovered = build_agent(self.store, CountingWorker(calls), timeout=2.0)
        complete = await recovered.resume("run-timeout")

        self.assertNotEqual(complete.status, "partial_timeout")
        self.assertEqual(complete.metrics["task_status_counts"]["succeeded"], 2)
        self.assertGreater(
            complete.metrics["budget"]["used"]["worker_invocations"],
            partial.metrics["budget"]["used"]["worker_invocations"],
        )

    async def test_completed_run_id_is_an_idempotency_key(self):
        calls: Counter[str] = Counter()
        agent = build_agent(self.store, CountingWorker(calls))

        first = await agent.run("duplicate submission", run_id="run-duplicate")
        second = await agent.run("duplicate submission", run_id="run-duplicate")
        explicit_resume = await agent.resume("run-duplicate")

        self.assertEqual(calls, Counter({"first": 1, "second": 1}))
        self.assertEqual(second.to_dict(), first.to_dict())
        self.assertEqual(explicit_resume.to_dict(), first.to_dict())

    async def test_concurrent_duplicate_submission_is_rejected(self):
        calls: Counter[str] = Counter()
        first_agent = build_agent(
            self.store,
            CountingWorker(calls, delay=0.05),
        )
        duplicate_agent = build_agent(self.store, CountingWorker(calls))
        first_submission = asyncio.create_task(
            first_agent.run("concurrent duplicate", run_id="run-concurrent")
        )
        await asyncio.sleep(0.01)

        with self.assertRaises(RunAlreadyExistsError):
            await duplicate_agent.run("concurrent duplicate", run_id="run-concurrent")

        result = await first_submission
        self.assertNotEqual(result.status, "partial_timeout")
        self.assertEqual(calls, Counter({"first": 1, "second": 1}))

    async def test_trace_replay_is_deterministic_and_side_effect_free(self):
        calls: Counter[str] = Counter()
        agent = build_agent(self.store, CountingWorker(calls))
        result = await agent.run("trace replay", run_id="run-replay")
        calls_before = calls.copy()

        replay_one = agent.replay("run-replay")
        replay_two = agent.replay("run-replay")

        self.assertEqual(replay_one, replay_two)
        self.assertEqual(calls, calls_before)
        self.assertTrue(replay_one["trace_integrity_verified"])
        self.assertEqual(replay_one["result"]["answer"], result.answer)
        self.assertEqual(
            replay_one["node_states"],
            {"first": "succeeded", "second": "succeeded"},
        )
        self.assertGreaterEqual(len(replay_one["reports"]), 1)
        self.assertTrue(any(item["event"] == "budget_checkpointed" for item in replay_one["trace"]))


if __name__ == "__main__":
    unittest.main()
