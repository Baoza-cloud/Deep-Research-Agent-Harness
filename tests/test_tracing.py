from __future__ import annotations

# ruff: noqa: F403, F405

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests._support import *
from research_engine.trace_viewer import build_view_model, render_trace_html
from research_engine.tracing import TRACE_SCHEMA_VERSION, TraceEventFactory
from research_engine.persistence import RunStore
from research_engine.llm_backends import LLMBackendConfig, LLMProvider, OpenAICompatibleLLM


class StructuredTracingTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_usage_tracks_provider_tokens_cost_and_retries(self):
        class RetryableError(RuntimeError):
            status_code = 429

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RetryableError("retry once")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    usage=SimpleNamespace(
                        prompt_tokens=1000,
                        completion_tokens=500,
                        total_tokens=1500,
                    ),
                )

        llm = OpenAICompatibleLLM(
            LLMBackendConfig(
                provider=LLMProvider.DEEPSEEK,
                model="test-model",
                api_key="test-key",
                max_retries=1,
                input_cost_per_million_usd=1.0,
                output_cost_per_million_usd=2.0,
            )
        )
        completions = FakeCompletions()
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        output = await llm("hello")
        usage = llm.usage_snapshot()

        self.assertEqual(output, "ok")
        self.assertEqual(usage["prompt_tokens"], 1000)
        self.assertEqual(usage["completion_tokens"], 500)
        self.assertEqual(usage["total_tokens"], 1500)
        self.assertEqual(usage["retry_count"], 1)
        self.assertEqual(usage["call_count"], 1)
        self.assertAlmostEqual(usage["cost_usd"], 0.002)

    def test_event_factory_adds_stable_envelope(self):
        factory = TraceEventFactory("tr-test")
        first = factory.normalize({"at": "2026-01-01T00:00:00+00:00", "event": "plan_created"})
        second = factory.normalize({"at": "2026-01-01T00:00:01+00:00", "event": "agent_started"})

        self.assertEqual(first["trace_schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(first["trace_id"], "tr-test")
        self.assertEqual((first["sequence"], second["sequence"]), (1, 2))
        self.assertNotEqual(first["event_id"], second["event_id"])
        self.assertEqual(first["category"], "dag")
        self.assertEqual(second["phase"], "execution")

    async def test_run_records_agent_telemetry_and_control_decisions(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            Synthesizer(),
            reviewer=AlwaysPassReviewer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=1,
            ),
        )

        result = await agent.run("对比当前方案的证据、限制、风险与影响")

        self.assertTrue(result.run_metadata["trace_id"].startswith("tr-"))
        self.assertTrue(result.trace)
        self.assertTrue(
            all(event.get("trace_id") == result.run_metadata["trace_id"] for event in result.trace)
        )
        self.assertEqual(
            len({event["event_id"] for event in result.trace}),
            len(result.trace),
        )
        completed = next(event for event in result.trace if event["event"] == "agent_completed")
        self.assertIn("input_summary", completed)
        self.assertIn("output_summary", completed)
        self.assertIn("duration_ms", completed)
        self.assertIn("token_usage", completed)
        self.assertIn("cost_usd", completed)
        self.assertIn("retry_count", completed)
        self.assertTrue(any(event["event"] == "swarm_role_decision" for event in result.trace))
        self.assertTrue(
            any(event["event"] == "evidence_verifier_decision" for event in result.trace)
        )
        self.assertTrue(any(event["event"] == "fixed_fallback_decision" for event in result.trace))

    async def test_persisted_replay_uses_one_trace_envelope(self):
        with tempfile.TemporaryDirectory(prefix="trace-store-") as directory:
            store = RunStore(Path(directory) / "runs.sqlite3")
            agent = DeepResearchAgent(
                HeuristicPlanner(),
                ResearchWorker(FakeBackend()),
                Synthesizer(),
                reviewer=AlwaysPassReviewer(),
                run_store=store,
                config=ResearchConfig(global_timeout_seconds=2, max_review_rounds=1),
            )
            result = await agent.run("trace persistence", run_id="trace-persistence")
            replay = store.replay("trace-persistence")
            store.close()

        self.assertTrue(replay["trace_integrity_verified"])
        self.assertTrue(replay["trace"])
        self.assertTrue(
            all(
                event.get("trace_id") == result.run_metadata["trace_id"]
                for event in replay["trace"]
            )
        )
        self.assertTrue(
            all(
                event.get("trace_schema_version") == TRACE_SCHEMA_VERSION
                for event in replay["trace"]
            )
        )
        sequences = [event["sequence"] for event in replay["trace"]]
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_trace_viewer_builds_waterfall_and_harness_comparison(self):
        payload = {
            "run_id": "run-viewer",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "run_metadata": {"trace_id": "tr-viewer"},
            "plan": {
                "subtasks": [
                    {
                        "subtask_id": "scope",
                        "question": "Define scope",
                        "kind": "scope",
                        "dependencies": [],
                        "status": "succeeded",
                    }
                ]
            },
            "trace": [
                {
                    "at": "2026-01-01T00:00:00+00:00",
                    "event": "agent_started",
                    "trace_id": "tr-viewer",
                    "span_id": "span-1",
                    "agent_id": "worker",
                    "role": "researcher",
                },
                {
                    "at": "2026-01-01T00:00:00.500000+00:00",
                    "event": "agent_completed",
                    "trace_id": "tr-viewer",
                    "span_id": "span-1",
                    "agent_id": "worker",
                    "role": "researcher",
                    "duration_ms": 500,
                    "token_usage": {"total": 42},
                    "cost_usd": 0.001,
                    "retry_count": 1,
                },
                {
                    "at": "2026-01-01T00:00:00.600000+00:00",
                    "event": "fixed_fallback_decision",
                    "trace_id": "tr-viewer",
                    "triggered": False,
                    "reasons": ["quality_gates_passed"],
                },
            ],
        }

        model = build_view_model(payload)
        rendered = render_trace_html(payload)

        self.assertEqual(model["trace_id"], "tr-viewer")
        self.assertEqual(model["summary"]["token_count"], 42)
        self.assertEqual(model["duration_ms"], 1_000.0)
        self.assertEqual(len(model["spans"]), 1)
        self.assertIn("DAG 时间线", rendered)
        self.assertIn("并发瀑布图", rendered)
        self.assertIn("Harness 与普通 Agent 框架的差异", rendered)
        self.assertIn("fixed_fallback_decision", rendered)
        self.assertNotIn("https://", rendered)


if __name__ == "__main__":
    unittest.main()
