# ruff: noqa: F403, F405

from tests._support import *


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_removes_extraction_omission_markers(self):
        worker = ResearchWorker(
            FixedBackend(
                [
                    {
                        "content": "事实一。 [...] 事实二。 […] 事实三。 [……] 事实四。",
                        "source": "fixture",
                    }
                ]
            )
        )
        evidence = (await worker.run(ResearchSubtask("clean", "q", "test")))[0]

        self.assertNotIn("[...]", evidence.content)
        self.assertNotIn("[…]", evidence.content)
        self.assertNotIn("[……]", evidence.content)
        self.assertEqual(evidence.metadata["extraction_omission_markers_removed"], 3)
        cleaned, count = remove_omission_markers("A【...】B [……] C")
        self.assertEqual(count, 2)
        self.assertEqual(cleaned, "A\n\nB\n\nC")

    async def test_retrieval_adapter_supports_sync_and_async_backends(self):
        def sync_search(query, limit=5):
            return [{"content": query, "limit": limit}]

        class AsyncBackend:
            async def search(self, query, limit=5):
                return [{"content": query.upper(), "limit": limit}]

        sync_rows = await RetrievalToolAdapter(sync_search).search("sync", limit=2)
        async_rows = await RetrievalToolAdapter(AsyncBackend()).search("async", limit=3)

        self.assertEqual(sync_rows[0], {"content": "sync", "limit": 2})
        self.assertEqual(async_rows[0], {"content": "ASYNC", "limit": 3})

    async def test_registry_and_runtime_enforce_declared_capabilities(self):
        registry = ToolRegistry()
        registry.register("retrieval", FakeBackend())
        with self.assertRaises(ValueError):
            registry.register("retrieval", FakeBackend())

        runtime = LocalAgentRuntime(registry)
        context = RunContext("run-1", "test objective")
        task = ResearchSubtask("facts", "question", "reason")
        execution = await runtime.invoke(
            AgentSpec("worker", "researcher", tool_names=("retrieval",)),
            LegacyAgentAdapter(ResearchWorker(registry.get("retrieval"))),
            task,
            context,
        )

        self.assertEqual(execution.agent_id, "worker")
        self.assertEqual(execution.output[0].evidence_id, "facts-E1")
        self.assertIs(context.tool("retrieval"), registry.get("retrieval"))
        self.assertEqual(
            [item["event"] for item in context.events],
            ["agent_started", "agent_completed"],
        )

        with self.assertRaises(RuntimeError):
            await runtime.invoke(
                AgentSpec("paper-worker", "researcher", tool_names=("papers",)),
                LegacyAgentAdapter(ResearchWorker(FakeBackend())),
                task,
                context,
            )

    async def test_research_worker_applies_role_specific_query_and_ranking(self):
        class CapturingBackend:
            def __init__(self):
                self.query = ""
                self.limit = 0

            async def search(self, query, limit=5):
                self.query = query
                self.limit = limit
                return [
                    {
                        "content": "社区文章给出一种观点。",
                        "source": "blog",
                        "source_type": "web",
                        "score": 0.30,
                    },
                    {
                        "content": "官方标准提供原始验证证据。",
                        "source": "standard",
                        "source_type": "standard",
                        "score": 0.10,
                    },
                ]

        backend = CapturingBackend()
        worker = ResearchWorker(backend)
        context = RunContext("role-run", "verify")
        evidences = await worker.run_for_agent(
            ResearchSubtask("verify_r1_1", "核验该结论", "verify", max_results=1),
            AgentSpec("verifier", "evidence_verifier", tool_names=("retrieval",)),
            context,
        )

        self.assertIn("交叉验证", backend.query)
        self.assertEqual(backend.limit, 2)
        self.assertEqual(evidences[0].source, "standard")
        self.assertEqual(evidences[0].metadata["retrieval_role"], "evidence_verifier")
        self.assertEqual(context.events[0]["event"], "role_strategy_applied")

    async def test_worker_scores_filters_deduplicates_and_traces_retrieval(self):
        canonical_content = (
            "Python asyncio TaskGroup cancellation semantics preserve cancellation "
            "counts and safely coordinate nested asynchronous tasks."
        )

        class QualityBackend:
            async def search(self, query, limit=5):
                return [
                    {
                        "title": "Task groups",
                        "content": canonical_content,
                        "url": "https://docs.python.org/3/library/asyncio-task.html?utm_source=test",
                        "source_type": "web",
                        "score": 0.80,
                    },
                    {
                        "title": "Task groups duplicate",
                        "content": canonical_content,
                        "url": "https://docs.python.org/3/library/asyncio-task.html#task-groups",
                        "source_type": "web",
                        "score": 0.70,
                    },
                    {
                        "title": "Task groups mirror section",
                        "content": canonical_content + " Documented.",
                        "url": "https://docs.python.org/3/library/asyncio-taskgroup.html",
                        "source_type": "web",
                        "score": 0.65,
                    },
                    {
                        "title": "SQLite reference",
                        "content": "SQLite transactions use commit and rollback for database writes.",
                        "url": "https://docs.python.org/3/library/sqlite3.html",
                        "source_type": "web",
                        "score": 0.05,
                    },
                    {
                        "title": "Community TaskGroup guide",
                        "content": "Python asyncio TaskGroup cancellation semantics and nested tasks.",
                        "url": "https://example.com/taskgroup-guide",
                        "source_type": "web",
                        "score": 0.95,
                    },
                ]

        context = RunContext("quality-run", "retrieval quality")
        evidences = await ResearchWorker(QualityBackend()).run_for_agent(
            ResearchSubtask(
                "facts",
                "Python asyncio TaskGroup cancellation semantics",
                "collect evidence",
                max_results=4,
            ),
            AgentSpec("evidence", "evidence_researcher", tool_names=("retrieval",)),
            context,
        )

        self.assertEqual(len(evidences), 2)
        self.assertEqual(evidences[0].metadata["source_class"], "official_documentation")
        self.assertGreater(evidences[0].metadata["retrieval_relevance_score"], 0.5)
        self.assertEqual(
            evidences[0].metadata["canonical_url"],
            "https://docs.python.org/3/library/asyncio-task.html",
        )
        audit = next(
            event for event in context.events if event["event"] == "retrieval_filter_applied"
        )
        self.assertEqual(audit["candidate_count"], 5)
        self.assertEqual(audit["selected_count"], 2)
        self.assertEqual(audit["filtered_count"], 3)
        self.assertEqual(audit["removal_counts"]["duplicate_url"], 1)
        self.assertEqual(audit["removal_counts"]["same_domain_near_duplicate"], 1)
        self.assertEqual(audit["removal_counts"]["below_role_threshold"], 1)

    async def test_each_role_uses_an_independent_relevance_threshold(self):
        class BorderlineBackend:
            async def search(self, query, limit=5):
                return [
                    {
                        "content": "AlphaFramework community note.",
                        "url": "https://example.com/alpha",
                        "source_type": "web",
                        "score": 0.10,
                    }
                ]

        task = ResearchSubtask(
            "borderline",
            "Compare AlphaFramework BetaEngine GammaProtocol DeltaService",
            "threshold test",
            max_results=1,
        )
        worker = ResearchWorker(BorderlineBackend())
        researcher_context = RunContext("researcher-run", "threshold")
        scope_context = RunContext("scope-run", "threshold")
        researcher = await worker.run_for_agent(
            task,
            AgentSpec("general", "researcher", tool_names=("retrieval",)),
            researcher_context,
        )
        scope = await worker.run_for_agent(
            task,
            AgentSpec("scope", "scope_researcher", tool_names=("retrieval",)),
            scope_context,
        )

        self.assertEqual(len(researcher), 1)
        self.assertEqual(scope, [])
        researcher_audit = researcher_context.events[-1]
        scope_audit = scope_context.events[-1]
        self.assertEqual(researcher_audit["relevance_threshold"], 0.14)
        self.assertEqual(scope_audit["relevance_threshold"], 0.44)
        self.assertEqual(scope_audit["removal_counts"], {"below_role_threshold": 1})


class SwarmPolicyTests(unittest.TestCase):
    def setUp(self):
        self.plan = ResearchPlan(
            "q",
            "objective",
            [
                ResearchSubtask("scope", "定义边界", "scope"),
                ResearchSubtask("facts", "搜集事实", "facts", dependencies=["scope"]),
                ResearchSubtask(
                    "alternatives",
                    "搜集反例和限制",
                    "counter",
                    dependencies=["scope"],
                ),
                ResearchSubtask(
                    "implications",
                    "分析风险和建议",
                    "impact",
                    dependencies=["facts", "alternatives"],
                ),
            ],
        )
        self.worker = AgentSpec(
            "research-worker",
            "researcher",
            tool_names=("retrieval",),
        )

    def test_policy_scales_roles_concurrency_and_budget_with_complexity(self):
        policy = HeuristicSwarmPolicy()
        config = ResearchConfig(max_concurrency=5, max_review_rounds=4)
        simple = policy.decide("什么是 RAG？", self.plan, config, self.worker)
        complex_plan = policy.decide(
            "请比较当前医疗和法律场景中的方案，同时分析多来源证据、风险、反例、"
            "局限、影响以及实施建议，并说明最新变化与关键权衡。",
            self.plan,
            config,
            self.worker,
        )

        self.assertEqual(simple.level, ComplexityLevel.SIMPLE)
        self.assertEqual(len(simple.agents), 1)
        self.assertEqual(simple.max_concurrency, 1)
        self.assertEqual(complex_plan.level, ComplexityLevel.COMPLEX)
        self.assertEqual(len(complex_plan.agents), 5)
        self.assertGreater(complex_plan.max_concurrency, simple.max_concurrency)
        self.assertGreater(
            complex_plan.max_worker_invocations,
            simple.max_worker_invocations,
        )
        self.assertEqual(
            complex_plan.select_agent(
                ResearchSubtask("verify_r1_1", "核验", "verify"), self.worker
            ).role,
            "evidence_verifier",
        )

    def test_six_task_focused_lookup_remains_simple(self):
        plan = ResearchPlan(
            "q",
            "objective",
            [ResearchSubtask(f"t{index}", "搜集事实", "facts") for index in range(6)],
        )
        policy = HeuristicSwarmPolicy()
        config = ResearchConfig(max_concurrency=5)

        simple = policy.decide("Python Semaphore 的作用是什么？", plan, config, self.worker)
        standard = policy.decide("比较 Python TaskGroup 与 gather。", plan, config, self.worker)

        self.assertEqual(simple.complexity_score, 0.22)
        self.assertEqual(simple.level, ComplexityLevel.SIMPLE)
        self.assertEqual(len(simple.agents), 1)
        self.assertEqual(standard.complexity_score, 0.36)
        self.assertEqual(standard.level, ComplexityLevel.STANDARD)
        self.assertEqual(len(standard.agents), 3)

    def test_budget_accounts_resources_and_stops_on_diminishing_returns(self):
        swarm = HeuristicSwarmPolicy().decide(
            "什么是 RAG？",
            self.plan,
            ResearchConfig(
                max_worker_invocations=2,
                max_stagnant_review_rounds=2,
            ),
            self.worker,
        )
        budget = BudgetController(swarm)

        self.assertTrue(budget.try_consume(BudgetResource.WORKER_INVOCATION, 2))
        self.assertFalse(budget.try_consume(BudgetResource.WORKER_INVOCATION))
        self.assertIn("worker_invocations", budget.snapshot()["exhausted"])
        self.assertIsNone(budget.observe_review(ReviewResult(False, total_score=0.50)))
        self.assertIsNone(budget.observe_review(ReviewResult(False, total_score=0.505)))
        self.assertEqual(
            budget.observe_review(ReviewResult(False, total_score=0.506)),
            "diminishing_returns",
        )

    def test_claim_gate_failure_defers_review_convergence_stop(self):
        swarm = HeuristicSwarmPolicy().decide(
            "什么是 RAG？",
            self.plan,
            ResearchConfig(max_stagnant_review_rounds=1),
            self.worker,
        )
        budget = BudgetController(swarm)

        self.assertIsNone(
            budget.observe_review(
                ReviewResult(True, total_score=0.9),
                quality_gate_passed=False,
            )
        )

    def test_blocking_red_issue_defers_diminishing_returns_stop(self):
        swarm = HeuristicSwarmPolicy().decide(
            "什么是 RAG？",
            self.plan,
            ResearchConfig(max_stagnant_review_rounds=1),
            self.worker,
        )
        budget = BudgetController(swarm)
        issue = ReviewIssue(
            "R1",
            "logic",
            "存在推断越界",
            severity=2,
            category="inference_overreach",
        )

        self.assertIsNone(
            budget.observe_review(
                ReviewResult(False, structured_issues=[issue], total_score=0.80),
                quality_gate_passed=True,
                claim_support_rate=1.0,
            )
        )
        self.assertIsNone(
            budget.observe_review(
                ReviewResult(False, structured_issues=[issue], total_score=0.80),
                quality_gate_passed=True,
                claim_support_rate=1.0,
            )
        )

    def test_dynamic_quality_guard_adds_verifier_then_falls_back(self):
        swarm = HeuristicSwarmPolicy().decide(
            "比较当前法律风险、证据和影响，并给出建议",
            self.plan,
            ResearchConfig(),
            self.worker,
        )
        budget = BudgetController(swarm)

        self.assertEqual(
            budget.observe_claim_support(0.70, minimum=0.80),
            "add_evidence_verifier",
        )
        self.assertEqual(
            budget.observe_claim_support(0.68, minimum=0.80),
            "fallback_fixed_harness",
        )
        snapshot = budget.snapshot()
        self.assertEqual(
            snapshot["quality_guardrail_actions"],
            ["add_evidence_verifier", "fallback_fixed_harness"],
        )
        self.assertIn("proxy_cost_units", snapshot)
        self.assertIsNone(
            budget.observe_review(
                ReviewResult(False, total_score=0.9, converged=True),
                quality_gate_passed=False,
            )
        )

    def test_simple_swarm_stops_after_duplicate_only_evidence_batch(self):
        swarm = HeuristicSwarmPolicy().decide(
            "什么是 RAG？",
            self.plan,
            ResearchConfig(),
            self.worker,
        )
        budget = BudgetController(swarm)
        evidence = Evidence("E1", "scope", "相同的冻结证据", "fixture")

        self.assertIsNone(budget.observe_evidence_batch([evidence]))
        self.assertEqual(
            budget.observe_evidence_batch([evidence]),
            "evidence_saturated",
        )
