# ruff: noqa: F403, F405

from tests._support import *
from tests._support import (
    _aggregate_gold,
    _proxy_relevance,
    _reprice_row,
    _resume_row_is_compatible,
)


class EvaluationEngineeringTests(unittest.TestCase):
    def test_review_issue_analyzer_reconstructs_legacy_gate_reasons(self):
        payload = {
            "run_id": "fixture-run",
            "variants": {
                "fixed": {
                    "samples": [
                        {
                            "status": "completed_with_review_issues",
                            "system_metrics": {
                                "review_passed": False,
                                "citation_coverage": 0.5,
                                "claim_ledger_passed": False,
                                "claim_support_rate": 0.4,
                                "semantic_claim_verification_available": True,
                                "budget": {"stop_reasons": ["diminishing_returns"]},
                            },
                            "trace": [],
                        }
                    ]
                }
            },
        }

        result = analyze_review_issues(payload)

        self.assertEqual(result["overall"]["issue_run_count"], 1)
        self.assertEqual(
            result["overall"]["gate_failure_combinations"],
            {"red_and_claim": 1},
        )
        self.assertEqual(
            result["overall"]["reason_run_counts"]["claim_support_below_threshold"],
            1,
        )

    def test_retrieval_gold_dataset_contract(self):
        path = EVALUATION_DIR / "datasets" / "researchbench_retrieval_gold_v1.0.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["records"]

        self.assertEqual(payload["track"], "retrieval_gold")
        self.assertEqual(payload["version"], "1.0.0")
        self.assertEqual(payload["status"], "adjudicated_gold")
        self.assertEqual(payload["statistics"]["record_count"], 33)
        self.assertTrue(payload["annotation_protocol"]["human_verified"])
        self.assertEqual(payload["annotation_protocol"]["signoff"]["status"], "approved")
        self.assertEqual(len(records), 33)
        self.assertEqual(len({item["candidate_key"] for item in records}), 33)
        self.assertEqual(len({item["annotation_id"] for item in records}), 33)
        self.assertEqual(
            sum(item["annotation"]["decision"] == "false_rejection" for item in records),
            18,
        )
        for item in records:
            self.assertIn(
                item["annotation"]["relevance"],
                {"relevant", "partially_relevant", "irrelevant", "cannot_determine"},
            )
            self.assertEqual(
                item["annotation"]["should_retain"],
                item["annotation"]["decision"] == "false_rejection",
            )
            self.assertLess(item["retrieval_relevance_score"], item["role_threshold"])

    def test_live_threshold_calibration_proxy_labels_and_aggregation(self):
        sample = {
            "required_domains": ["docs.python.org"],
            "required_terms": ["TaskGroup", "exception"],
        }
        relevant, reasons = _proxy_relevance(
            {
                "url": "https://docs.python.org/3/library/asyncio-task.html",
                "title": "TaskGroup exceptions",
                "content": "TaskGroup propagates an exception.",
            },
            sample,
        )
        irrelevant, _ = _proxy_relevance(
            {
                "url": "https://example.com/sqlite",
                "content": "Database transaction notes.",
            },
            sample,
        )
        aggregate = aggregate_threshold_rows(
            [
                {
                    "nonempty_count": 4,
                    "retained_count": 2,
                    "proxy_relevant_count": 2,
                    "proxy_false_rejected_count": 1,
                    "selected_proxy_relevant_count": 1,
                    "primary_source_count": 1,
                    "selected_authority_sum": 1.2,
                    "removal_counts": {"below_role_threshold": 2},
                }
            ]
        )

        self.assertTrue(relevant)
        self.assertFalse(irrelevant)
        self.assertIn("domain:docs.python.org", reasons)
        self.assertEqual(aggregate["retention_rate"], 0.5)
        self.assertEqual(aggregate["proxy_false_rejection_rate"], 0.5)
        self.assertEqual(aggregate["mean_selected_authority"], 0.6)

    def test_gold_threshold_metrics_use_human_retain_decisions(self):
        unit_rows = [
            {
                "unit_id": "sample::scope_researcher",
                "selected_input_ranks": [1, 3],
            }
        ]
        records = [
            {
                "sample_id": "sample",
                "role": "scope_researcher",
                "input_rank": rank,
                "annotation": {"should_retain": should_retain},
            }
            for rank, should_retain in ((1, True), (2, True), (3, False), (4, False))
        ]

        metrics = _aggregate_gold(unit_rows, records)

        self.assertEqual(metrics["gold_true_positive"], 1)
        self.assertEqual(metrics["gold_false_positive"], 1)
        self.assertEqual(metrics["gold_true_negative"], 1)
        self.assertEqual(metrics["gold_false_negative"], 1)
        self.assertEqual(metrics["gold_false_rejection_rate"], 0.5)
        self.assertEqual(metrics["gold_precision"], 0.5)
        self.assertEqual(metrics["gold_recall"], 0.5)
        self.assertEqual(metrics["gold_f1"], 0.5)

    def test_gold_threshold_recommendation_maximizes_f1(self):
        curves = {
            role: []
            for role in (
                "researcher",
                "scope_researcher",
                "evidence_researcher",
                "counter_researcher",
                "impact_analyst",
                "evidence_verifier",
            )
        }
        curves["scope_researcher"] = [
            {
                "threshold": 0.30,
                "gold_labeled_count": 10,
                "gold_positive_count": 6,
                "gold_f1": 0.80,
                "gold_recall": 1.0,
                "gold_precision": 2 / 3,
            },
            {
                "threshold": 0.42,
                "gold_labeled_count": 10,
                "gold_positive_count": 6,
                "gold_f1": 0.90,
                "gold_recall": 0.90,
                "gold_precision": 0.90,
            },
        ]

        recommendations = recommend_gold_thresholds(curves)

        self.assertEqual(recommendations["scope_researcher"]["threshold"], 0.42)
        self.assertNotIn("researcher", recommendations)

    def test_dynamic_resume_requires_matching_swarm_policy_version(self):
        current = {
            "swarm_plan": {
                "policy_version": HeuristicSwarmPolicy.VERSION,
                "level": "simple",
                "complexity_score": 0.22,
            }
        }
        stale = {
            "swarm_plan": {
                "policy_version": "heuristic-v1",
                "level": "simple",
                "complexity_score": 0.08,
            }
        }
        compatible_v2_standard = {
            "swarm_plan": {
                "policy_version": "heuristic-v2",
                "level": "standard",
                "complexity_score": 0.52,
            }
        }
        incompatible_v2_simple = {
            "swarm_plan": {
                "policy_version": "heuristic-v2",
                "level": "simple",
                "complexity_score": 0.22,
            }
        }
        compatible_unversioned_standard = {
            "swarm_plan": {
                "level": "standard",
                "complexity_score": 0.52,
                "agents": [{}, {}, {}],
                "max_review_rounds": 3,
                "max_verification_queries": 6,
                "evidence_stagnation_patience": 2,
            }
        }
        unversioned = {"swarm_plan": {}}

        self.assertTrue(_resume_row_is_compatible("dynamic_swarm", current))
        self.assertFalse(_resume_row_is_compatible("dynamic_swarm", stale))
        self.assertTrue(_resume_row_is_compatible("dynamic_swarm", compatible_v2_standard))
        self.assertFalse(_resume_row_is_compatible("dynamic_swarm", incompatible_v2_simple))
        self.assertTrue(_resume_row_is_compatible("dynamic_swarm", compatible_unversioned_standard))
        self.assertFalse(_resume_row_is_compatible("dynamic_swarm", unversioned))
        self.assertTrue(_resume_row_is_compatible("fixed_harness", unversioned))

    def test_resume_rows_can_be_repriced_without_rerunning_model(self):
        row = {
            "llm_usage": {
                "estimated_prompt_tokens": 1_000_000,
                "estimated_completion_tokens": 500_000,
                "estimated_cost_usd": 0.0,
            }
        }

        _reprice_row(row, 0.30, 1.20)

        self.assertAlmostEqual(row["llm_usage"]["estimated_cost_usd"], 0.90)

    def test_instrumented_llm_estimates_tokens_and_configurable_cost(self):
        async def delegate(_prompt):
            return "中文回答 with text"

        llm = InstrumentedLLM(
            delegate,
            input_cost_per_million_usd=1.0,
            output_cost_per_million_usd=2.0,
        )
        token = RUN_CONTEXT.set("cost-test")
        try:
            asyncio.run(llm("中文问题 with prompt"))
        finally:
            RUN_CONTEXT.reset(token)
        usage = llm.usage("cost-test")

        self.assertEqual(usage["llm_calls"], 1)
        self.assertGreater(usage["estimated_total_tokens"], 0)
        self.assertGreater(usage["estimated_cost_usd"], 0)

    def test_frozen_and_live_datasets_are_explicitly_separated(self):
        frozen_path = EVALUATION_DIR / "datasets" / "researchbench_frozen_v1.0.json"
        live_path = EVALUATION_DIR / "datasets" / "researchbench_live_v1.0.json"
        frozen = load_frozen_dataset(frozen_path)
        live = load_live_swarm_dataset(live_path)

        self.assertEqual(frozen["track"], "frozen")
        self.assertEqual(live["track"], "live")
        self.assertTrue(all(item.get("evidence") for item in frozen["samples"]))
        self.assertTrue(all("evidence" not in item for item in live["samples"]))
        self.assertEqual(len(live["samples"]), 15)
        self.assertEqual(
            {item["expected_swarm"]["level"] for item in live["samples"]},
            {"simple", "standard", "complex"},
        )
        self.assertEqual(
            {
                level: sum(item["expected_swarm"]["level"] == level for item in live["samples"])
                for level in ("simple", "standard", "complex")
            },
            {"simple": 5, "standard": 5, "complex": 5},
        )
        self.assertEqual(len(frozen["samples"]), 35)
        self.assertEqual(
            len({item["domain"] for item in frozen["samples"]}),
            11,
        )
        self.assertTrue(all(len(item.get("key_facts", [])) >= 3 for item in frozen["samples"]))

    def test_ablation_variants_include_fixed_and_dynamic_swarm(self):
        self.assertEqual(
            set(VARIANT_DESCRIPTIONS),
            {
                "single_agent",
                "no_red_blue",
                "rewrite_blue",
                "structured_patch_blue",
                "fixed_harness",
                "dynamic_swarm",
            },
        )

    def test_rule_metrics_share_claim_extraction_with_engine(self):
        report = (
            "# 报告\n\n这是一个长度足够并由证据直接支持的事实陈述。 [facts-E1]\n\n"
            "## 局限\n本报告证据范围有限。\n\n"
            "## 来源\n[facts-E1] fixture"
        )
        metrics = evaluate_rules(
            report,
            [
                {
                    "evidence_id": "facts-E1",
                    "content": "这是一个长度足够并由证据直接支持的事实陈述。",
                    "source": "fixture",
                }
            ],
        )

        self.assertEqual(metrics.total_claims, 1)
        self.assertEqual(metrics.factual_accuracy, 1.0)
        self.assertEqual(metrics.citation_coverage, 1.0)

    def test_frozen_expectations_and_paired_effect_size(self):
        sample = {
            "key_facts": [
                {"required_terms": ["结构化补丁", "回滚"], "minimum_hits": 2},
                {"required_terms": ["语义验证"], "minimum_hits": 1},
            ],
            "forbidden_terms": ["绝对不会失败"],
        }
        metrics = evaluate_expectations("结构化补丁支持回滚，并执行语义验证。", sample)

        self.assertEqual(metrics.key_fact_recall, 1.0)
        self.assertEqual(metrics.forbidden_claim_rate, 0.0)
        self.assertGreater(
            paired_cohens_d([0.3, 0.4, 0.5], [0.4, 0.6, 0.6]),
            0,
        )

    def test_key_fact_aliases_preserve_rule_and_semantic_scores(self):
        sample = {
            "key_facts": [
                {
                    "required_terms": ["不同组", "独立消费"],
                    "minimum_hits": 2,
                }
            ]
        }
        metrics = evaluate_expectations(
            "不同 consumer group 可以独立消费同一主题。",
            sample,
        )

        self.assertEqual(metrics.rule_key_fact_recall, 0.0)
        self.assertEqual(metrics.semantic_key_fact_recall, 1.0)
        self.assertEqual(metrics.key_fact_recall, 1.0)

    def test_adversarial_fault_injection_has_gold_ceiling(self):
        frozen = load_frozen_dataset(EVALUATION_DIR / "datasets" / "researchbench_frozen_v1.0.json")
        case = inject_faults(frozen["samples"][0])

        corrupted = evaluate_adversarial_report(case, case.corrupted_report)
        oracle = evaluate_adversarial_report(case, case.clean_report)

        self.assertEqual(len(case.faults), 4)
        self.assertEqual(corrupted.fault_repair_rate, 0.0)
        self.assertEqual(oracle.fault_repair_rate, 1.0)
        self.assertEqual(oracle.clean_fact_retention_rate, 1.0)
        self.assertEqual(oracle.citation_validity, 1.0)
        self.assertEqual(oracle.citation_preservation_rate, 1.0)

    def test_invalid_patches_are_fully_rolled_back(self):
        guard = evaluate_rollback_guard()

        self.assertEqual(guard["attack_count"], 4)
        self.assertEqual(guard["rejected_count"], 4)
        self.assertEqual(guard["applied_count"], 0)
        self.assertTrue(guard["report_unchanged"])
        self.assertEqual(guard["rollback_correctness"], 1.0)

    def test_semantic_fault_matcher_ignores_chinese_punctuation(self):
        frozen = load_frozen_dataset(EVALUATION_DIR / "datasets" / "researchbench_frozen_v1.0.json")
        case = inject_faults(frozen["samples"][0])
        review = ReviewResult(
            False,
            structured_issues=[
                ReviewIssue(
                    "R-semantic",
                    "factuality",
                    "证据没有提供公平性保证",
                    severity=3,
                    action=RepairAction.DELETE,
                    target="保证公平调度。 [facts-E1]",
                )
            ],
        )

        metrics = evaluate_adversarial_report(case, case.corrupted_report, review=review)

        self.assertIn("F-UNSUPPORTED-CLAIM", metrics.detected_fault_ids)
