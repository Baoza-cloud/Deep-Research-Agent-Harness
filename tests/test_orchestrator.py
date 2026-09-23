# ruff: noqa: F403, F405

from tests._support import *


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def test_perfect_semantic_ledger_adjudicates_only_severity_two_red_disputes(self):
        review = ReviewResult(
            False,
            structured_issues=[
                ReviewIssue(
                    "R1",
                    "logic",
                    "Red 与逐句支持判断存在分歧",
                    severity=2,
                    category="inference_overreach",
                ),
                ReviewIssue(
                    "R2",
                    "factuality",
                    "高风险事实错误",
                    severity=3,
                    category="factual_error",
                ),
            ],
        )
        ledger = ClaimLedger(
            verification_mode="semantic",
            metrics={
                "reviewable_claim_count": 3.0,
                "supported_claim_count": 3.0,
                "contradicted_claim_count": 0.0,
                "time_conflict_count": 0.0,
                "number_conflict_count": 0.0,
                "entity_conflict_count": 0.0,
            },
        )

        adjudicated = adjudicate_red_with_claim_ledger(review, ledger)

        self.assertEqual(len(adjudicated), 1)
        self.assertEqual(review.structured_issues[0].category, "writing_advice")
        self.assertEqual(review.structured_issues[0].severity, 1)
        self.assertEqual(review.structured_issues[1].category, "factual_error")
        self.assertEqual(review.structured_issues[1].severity, 3)

    async def test_dynamic_final_guard_generates_and_selects_fixed_candidate(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(
                FixedBackend(
                    [
                        {
                            "content": "系统支持异步搜索。",
                            "source": "fixture",
                        }
                    ]
                )
            ),
            QualityFallbackSynthesizer(),
            reviewer=QualityFallbackReviewer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=1,
                min_citation_coverage=0.8,
                enable_dynamic_swarm=True,
            ),
        )

        result = await agent.run("异步搜索能力")

        self.assertTrue(result.answer.startswith("# fixed fallback"))
        self.assertTrue(result.metrics["dynamic_quality_fallback_triggered"])
        self.assertTrue(result.metrics["dynamic_quality_fallback_selected"])
        guard = next(
            item
            for item in result.trace
            if item.get("event") == "dynamic_swarm_final_quality_guard"
        )
        self.assertEqual(guard["selected"], "fixed_harness_fallback")
        self.assertFalse(guard["cost_used_as_quality_override"])

    async def test_explicit_evidence_gap_has_distinct_completion_status(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            EvidenceGapSynthesizer(),
            reviewer=AlwaysPassReviewer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=1,
            ),
        )
        result = await agent.run("evidence gap status")

        self.assertEqual(result.status, "completed_with_evidence_gaps")
        self.assertEqual(result.metrics["evidence_gap_count"], 1)
        self.assertFalse(result.metrics["completion_issue_reasons"])

    async def test_final_report_removes_omission_markers_and_traces_cleanup(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            OmissionSynthesizer(),
            reviewer=AlwaysPassReviewer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=1,
            ),
        )
        result = await agent.run("omission marker cleanup")

        self.assertNotIn("[...]", result.answer)
        self.assertNotIn("[……]", result.answer)
        self.assertEqual(result.metrics["report_omission_markers_removed"], 2)
        cleanup = [
            item for item in result.trace if item["event"] == "report_omission_markers_removed"
        ]
        self.assertEqual(cleanup[0]["stage"], "initial_synthesis")

    async def test_final_claim_gate_repairs_unsupported_claim_incrementally(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            UnsupportedClaimSynthesizer(),
            reviewer=AlwaysPassReviewer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=1,
                min_claim_support_rate=0.8,
            ),
        )
        result = await agent.run("Claim ledger gate test")

        self.assertTrue(result.review.passed)
        self.assertTrue(result.metrics["claim_ledger_passed"])
        self.assertEqual(result.status, "completed")
        self.assertIn("[facts-E1]", result.answer)
        final_repairs = [
            item
            for item in result.trace
            if item.get("event") == "blue_repair" and item.get("round") == "final"
        ]
        self.assertEqual(len(final_repairs), 1)
        self.assertTrue(final_repairs[0]["incremental_verification"])

    async def test_failed_review_is_visible_in_run_status(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            Synthesizer(),
            reviewer=AlwaysFailReviewer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=1,
            ),
        )
        result = await agent.run("failed review status")
        self.assertEqual(result.status, "completed_with_review_issues")
        self.assertFalse(result.metrics["review_passed"])

    async def test_online_synthesizer_llm_is_injected_into_default_red_reviewer(self):
        async def fake_llm(prompt):
            return "{}"

        synthesizer = Synthesizer(fake_llm)
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            synthesizer,
        )
        self.assertIs(agent.reviewer.llm, fake_llm)

    async def test_blue_repair_is_red_reviewed_before_return(self):
        synthesizer = RepairingSynthesizer()
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            synthesizer,
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=3,
                min_citation_coverage=0.8,
            ),
        )
        result = await agent.run("Blue repair test")
        self.assertTrue(result.review.passed)
        self.assertEqual(result.metrics["review_rounds"], 2)
        self.assertEqual(result.metrics["citation_coverage"], 1)
        self.assertEqual([item["event"] for item in result.trace].count("blue_repair"), 1)

    async def test_orchestrator_prefers_structured_patch_over_whole_rewrite(self):
        synthesizer = RepairingSynthesizer()
        long_claim = "这是需要证据支持的长事实陈述" + "内容" * 450

        async def blue_llm(prompt):
            return json.dumps(
                {
                    "patches": [
                        {
                            "patch_id": "P1",
                            "issue_ids": ["rule-citation-coverage"],
                            "action": "MODIFY",
                            "target": long_claim + "。",
                            "replacement": long_claim + "。 [scope-E1]",
                            "evidence_ids": ["scope-E1"],
                            "reason": "补充相邻有效引用",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            synthesizer,
            repairer=BlueTeamRepairer(
                blue_llm,
                verifier=ClaimEvidenceVerifier(),
            ),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_review_rounds=3,
                min_citation_coverage=0.8,
            ),
        )
        result = await agent.run("Structured Blue patch test")
        blue_trace = [item for item in result.trace if item.get("event") == "blue_repair"]

        self.assertTrue(result.review.passed)
        self.assertEqual(synthesizer.calls, 1)
        self.assertEqual(blue_trace[0]["mode"], "deterministic_claim_patch")
        self.assertTrue(blue_trace[0]["applied_patches"][0]["patch_id"].startswith("AUTO-claim-"))

    async def test_end_to_end_offline(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            Synthesizer(),
            config=ResearchConfig(task_timeout_seconds=1, global_timeout_seconds=2),
        )
        result = await agent.run("公司的远程办公政策是什么？")
        self.assertEqual(result.status, "completed")
        self.assertGreaterEqual(len(result.evidences), 4)
        self.assertIn("[scope-E1]", result.answer)
        self.assertEqual(result.metrics["task_status_counts"]["succeeded"], 4)
        self.assertIsNotNone(result.claim_ledger)
        self.assertIn("claim_support_rate", result.metrics)
        self.assertTrue(result.run_id.startswith("research-"))
        self.assertIn("prompt_schema_versions", result.run_metadata)
        self.assertEqual(
            result.run_metadata["harness"],
            {
                "runtime": "LocalAgentRuntime",
                "worker_agent_id": "research-worker",
                "worker_role": "researcher",
                "registered_tools": ["retrieval"],
            },
        )
        self.assertEqual(
            [item["event"] for item in result.trace].count("agent_started"),
            4,
        )
        self.assertEqual(
            [item["event"] for item in result.trace].count("agent_completed"),
            4,
        )
        self.assertEqual(result.run_metadata["swarm"]["level"], "simple")
        self.assertEqual(result.metrics["swarm_agent_count"], 1)
        self.assertEqual(result.metrics["budget"]["used"]["worker_invocations"], 4)
        self.assertTrue(any(item["event"] == "swarm_configured" for item in result.trace))
        self.assertNotIn("api_key", json.dumps(result.run_metadata))

    async def test_worker_budget_skips_remaining_dag_tasks_deterministically(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(FakeBackend()),
            Synthesizer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                max_worker_invocations=2,
            ),
        )

        result = await agent.run("什么是 RAG？")

        self.assertEqual(result.metrics["budget"]["used"]["worker_invocations"], 2)
        self.assertIn(
            "worker_invocations",
            result.metrics["budget"]["exhausted"],
        )
        self.assertEqual(result.metrics["task_status_counts"]["skipped"], 2)
        self.assertEqual(
            [item["event"] for item in result.trace].count("agent_started"),
            2,
        )

    async def test_global_timeout_forces_partial_synthesis(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(SlowBackend()),
            Synthesizer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=0.02,
                max_review_rounds=1,
            ),
        )
        result = await agent.run("timeout test")
        self.assertEqual(result.status, "partial_timeout")
        self.assertTrue(result.metrics["forced_synthesis"])

    async def test_batch_failure_triggers_dynamic_replan(self):
        agent = DeepResearchAgent(
            HeuristicPlanner(),
            ResearchWorker(ReplanBackend()),
            Synthesizer(),
            config=ResearchConfig(
                task_timeout_seconds=1,
                global_timeout_seconds=2,
                batch_failure_threshold=0.5,
                max_replans=1,
                max_review_rounds=1,
            ),
        )
        result = await agent.run("replan test")
        self.assertEqual(result.metrics["plan_versions"], 2)
        self.assertTrue(any(item["event"] == "dynamic_replan" for item in result.trace))
        self.assertGreaterEqual(len(result.evidences), 1)
