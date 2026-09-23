# ruff: noqa: F403, F405

from tests._support import *


class VerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_ledger_uses_semantic_entailment_judgment(self):
        async def verifier_llm(prompt):
            self.assertIn("Claim–Evidence", prompt)
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": "C1::facts-E1",
                            "verdict": "SUPPORTED",
                            "confidence": 0.94,
                            "rationale": "证据直接说明该能力",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        verifier = ClaimEvidenceVerifier(verifier_llm)
        ledger = await verifier.build_ledger(
            "# 报告\n\nPython SDK 支持异步客户端调用。 [facts-E1]\n\n## 局限\n本测试证据范围有限。",
            [
                Evidence(
                    "facts-E1",
                    "facts",
                    "The SDK exposes an asynchronous client.",
                    "official-doc",
                )
            ],
        )

        self.assertEqual(ledger.verification_mode, "semantic")
        self.assertEqual(ledger.claims[0].verdict, SupportVerdict.SUPPORTED)
        self.assertEqual(ledger.metrics["claim_support_rate"], 1.0)

    async def test_claim_verifier_caches_unchanged_pairs_only(self):
        calls = 0

        async def verifier_llm(prompt):
            nonlocal calls
            calls += 1
            pair_id = "C1::facts-E1"
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": pair_id,
                            "verdict": "SUPPORTED",
                            "confidence": 0.95,
                            "rationale": "直接支持",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        verifier = ClaimEvidenceVerifier(verifier_llm)
        evidence = Evidence(
            "facts-E1", "facts", "该客户端明确支持异步搜索和结构化返回。", "official"
        )
        report = "该客户端明确支持异步搜索和结构化返回。 [facts-E1]"
        first = await verifier.build_ledger(report, [evidence])
        second = await verifier.build_ledger(report, [evidence])

        self.assertEqual(calls, 1)
        self.assertEqual(first.metrics["semantic_cache_miss_count"], 1.0)
        self.assertEqual(second.metrics["semantic_cache_hit_count"], 1.0)
        self.assertEqual(second.verification_mode, "semantic")

        await verifier.build_ledger(
            "该客户端明确支持同步搜索和结构化返回。 [facts-E1]",
            [evidence],
        )
        self.assertEqual(calls, 2)

    async def test_claim_verifier_reuses_supported_unchanged_claims_from_ledger(self):
        prompts = []

        async def verifier_llm(prompt):
            prompts.append(prompt)
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": "C1::facts-E1",
                            "verdict": "SUPPORTED",
                            "confidence": 0.95,
                            "rationale": "直接支持",
                        },
                        {
                            "pair_id": "C2::facts-E1",
                            "verdict": "SUPPORTED",
                            "confidence": 0.95,
                            "rationale": "直接支持",
                        },
                    ]
                },
                ensure_ascii=False,
            )

        verifier = ClaimEvidenceVerifier(verifier_llm)
        evidence = Evidence(
            "facts-E1",
            "facts",
            "系统支持异步搜索、结构化返回，并提供结果排序。",
            "official",
        )
        first = await verifier.build_ledger(
            "该系统客户端明确支持异步搜索请求。 [facts-E1]\n"
            "该系统客户端明确提供结构化结果返回。 [facts-E1]",
            [evidence],
        )
        second = await verifier.build_ledger(
            "该系统客户端明确支持异步搜索请求。 [facts-E1]\n"
            "该系统客户端明确提供结果排序能力。 [facts-E1]",
            [evidence],
            previous_ledger=first,
        )

        self.assertEqual(len(prompts), 2)
        self.assertNotIn("支持异步搜索请求", prompts[1])
        self.assertIn("提供结果排序能力", prompts[1])
        self.assertEqual(second.metrics["reused_unchanged_claim_count"], 1.0)

    async def test_rrf_gap_metadata_and_bold_labels_share_one_claim_gate(self):
        report = (
            "# RRF 报告\n\n## 执行摘要\n"
            "RRF 将多个排序列表合并。 [facts-E1]\n\n"
            "## 关键发现\n**1. RRF 的核心公式**\n"
            "融合分数是各列表中 1/(k + rank(d)) 的总和。 [facts-E1]\n"
            "**2. 对分数尺度的要求**\n"
            "RRF 不要求原始分数处于同一尺度。 [facts-E1]\n\n"
            "## 反方证据/局限\n"
            "证据来源为单一来源 frozen://rrf/reference。 [facts-E1]\n"
            "- 未提供参数 k 的典型取值范围；\n"
            "- 未提供与其他融合方法的对比。\n\n"
            "## 结论与建议\n"
            "基于当前证据，本节有 2 项内容无法确认，已合并为待验证问题："
            "参数取值；对比实验。"
        )
        evidence = Evidence(
            "facts-E1",
            "facts",
            "RRF 将多个排序列表合并；融合分数是各列表中 "
            "1/(k + rank(d)) 的总和；不要求原始分数处于同一尺度。",
            "frozen://rrf/reference",
        )
        serialized = [
            {
                "evidence_id": evidence.evidence_id,
                "content": evidence.content,
                "source": evidence.source,
            }
        ]

        rules = evaluate_rules(report, serialized)
        red = await RedTeamReviewer().review("RRF 是什么？", report, [evidence])
        ledger = await ClaimEvidenceVerifier().build_ledger(report, [evidence])

        self.assertEqual(rules.total_claims, 3)
        self.assertEqual(rules.factual_accuracy, 1.0)
        self.assertEqual(rules.citation_coverage, 1.0)
        self.assertEqual(red.metrics["claim_count"], 3.0)
        self.assertEqual(red.metrics["citation_coverage"], 1.0)
        self.assertEqual(ledger.metrics["reviewable_claim_count"], 3.0)
        self.assertEqual(ledger.metrics["claim_support_rate"], 1.0)

    def test_dynamic_final_guard_selects_quality_not_cost(self):
        dynamic = ReportQualityCandidate(
            "dynamic",
            ClaimLedger(metrics={"claim_support_rate": 1.0}),
            ReviewResult(
                False,
                metrics={"citation_coverage": 0.70},
                total_score=0.90,
            ),
            "dynamic_swarm",
        )
        fixed = ReportQualityCandidate(
            "fixed",
            ClaimLedger(metrics={"claim_support_rate": 0.98}),
            ReviewResult(
                True,
                metrics={"citation_coverage": 0.95},
                total_score=0.88,
            ),
            "fixed_harness_fallback",
        )

        selected, reason = select_quality_candidate(
            dynamic,
            fixed,
            min_claim_support_rate=0.8,
            min_citation_coverage=0.8,
            max_pass_issue_severity=1,
            support_drop_tolerance=0.02,
        )

        self.assertIs(selected, fixed)
        self.assertEqual(reason, "fallback_repairs_hard_metrics")

    def test_dynamic_final_guard_never_selects_candidate_below_claim_floor(self):
        dynamic = ReportQualityCandidate(
            "dynamic",
            ClaimLedger(metrics={"claim_support_rate": 1.0}),
            ReviewResult(
                False,
                structured_issues=[
                    ReviewIssue(
                        "R1",
                        "logic",
                        "存在推断越界",
                        severity=2,
                        category="inference_overreach",
                    )
                ],
                metrics={"citation_coverage": 1.0},
                total_score=0.8,
            ),
            "dynamic_swarm",
        )
        fallback = ReportQualityCandidate(
            "fallback",
            ClaimLedger(metrics={"claim_support_rate": 0.71}),
            ReviewResult(
                True,
                metrics={"citation_coverage": 1.0},
                total_score=0.9,
            ),
            "fixed_harness_fallback",
        )

        selected, reason = select_quality_candidate(
            dynamic,
            fallback,
            min_claim_support_rate=0.8,
            min_citation_coverage=0.8,
            max_pass_issue_severity=1,
            support_drop_tolerance=0.02,
        )

        self.assertIs(selected, dynamic)
        self.assertEqual(reason, "fallback_rejected_by_hard_metrics")

    def test_claim_repair_queries_merge_similar_gaps(self):
        ledger = ClaimLedger(
            claims=[
                ClaimRecord(
                    "C1",
                    "Kafka consumer group 中不同组可以独立消费同一主题",
                    verdict=SupportVerdict.INSUFFICIENT,
                ),
                ClaimRecord(
                    "C2",
                    "Kafka consumer group 中不同组能够独立读取同一主题",
                    verdict=SupportVerdict.UNCITED,
                ),
            ]
        )

        queries = ClaimEvidenceVerifier.repair_queries(ledger, limit=4)

        self.assertEqual(len(queries), 1)
        self.assertIn("相关陈述", queries[0])

    async def test_claim_verifier_records_partial_support_and_typed_conflicts(self):
        async def verifier_llm(_prompt):
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": "C1::facts-E1",
                            "verdict": "PARTIALLY_SUPPORTED",
                            "confidence": 0.91,
                            "conflict_types": ["NUMBER"],
                            "supported_aspects": ["支持增长趋势"],
                            "unsupported_aspects": ["未支持 50% 数值"],
                        },
                        {
                            "pair_id": "C2::facts-E2",
                            "verdict": "CONTRADICTED",
                            "confidence": 0.95,
                            "conflict_types": ["TIME", "ENTITY"],
                            "supported_aspects": [],
                            "unsupported_aspects": ["年份和主体均不一致"],
                        },
                    ]
                },
                ensure_ascii=False,
            )

        ledger = await ClaimEvidenceVerifier(verifier_llm).build_ledger(
            "该产品年度增长率已经达到 50% 的明确水平。 [facts-E1]\n"
            "甲公司已经在 2025 年正式发布该行业标准。 [facts-E2]",
            [
                Evidence("facts-E1", "facts", "指标保持增长。", "source-1"),
                Evidence("facts-E2", "facts", "乙公司在 2024 年发布标准。", "source-2"),
            ],
            align_uncited_candidates=False,
        )

        self.assertEqual(ledger.claims[0].verdict, SupportVerdict.PARTIALLY_SUPPORTED)
        self.assertEqual(ledger.claims[1].verdict, SupportVerdict.CONTRADICTED)
        self.assertEqual(ledger.metrics["partially_supported_claim_count"], 1.0)
        self.assertEqual(ledger.metrics["number_conflict_count"], 1.0)
        self.assertEqual(ledger.metrics["time_conflict_count"], 1.0)
        self.assertEqual(ledger.metrics["entity_conflict_count"], 1.0)

    async def test_blue_deterministically_adds_weakens_and_deletes_claims(self):
        report = (
            "第一条事实获得新证据支持。\n"
            "第二条事实包含尚未完全证实的限定。 [facts-E2]\n"
            "第三条事实中的主体和时间均错误。 [facts-E3]\n"
            "第四条事实在检索后仍然没有直接证据。"
        )
        ledger = ClaimLedger(
            claims=[
                ClaimRecord(
                    "C1",
                    "第一条事实获得新证据支持。",
                    source_text="第一条事实获得新证据支持。",
                    verdict=SupportVerdict.UNCITED,
                    links=[
                        ClaimEvidenceLink(
                            "facts-E1",
                            SupportVerdict.SUPPORTED,
                            0.93,
                            cited=False,
                        )
                    ],
                ),
                ClaimRecord(
                    "C2",
                    "第二条事实包含尚未完全证实的限定。",
                    source_text="第二条事实包含尚未完全证实的限定。 [facts-E2]",
                    citations=["facts-E2"],
                    verdict=SupportVerdict.PARTIALLY_SUPPORTED,
                    links=[
                        ClaimEvidenceLink(
                            "facts-E2",
                            SupportVerdict.PARTIALLY_SUPPORTED,
                            0.88,
                        )
                    ],
                ),
                ClaimRecord(
                    "C3",
                    "第三条事实中的主体和时间均错误。",
                    source_text="第三条事实中的主体和时间均错误。 [facts-E3]",
                    citations=["facts-E3"],
                    verdict=SupportVerdict.CONTRADICTED,
                    links=[
                        ClaimEvidenceLink(
                            "facts-E3",
                            SupportVerdict.CONTRADICTED,
                            0.97,
                            conflict_types=["time", "entity"],
                        )
                    ],
                ),
                ClaimRecord(
                    "C4",
                    "第四条事实在检索后仍然没有直接证据。",
                    source_text="第四条事实在检索后仍然没有直接证据。",
                    verdict=SupportVerdict.UNCITED,
                ),
            ]
        )
        evidences = [
            Evidence(f"facts-E{index}", "facts", f"证据 {index}", "source") for index in range(1, 4)
        ]

        result = BlueTeamRepairer(llm=None).repair_claim_gaps(report, evidences, ledger)

        self.assertEqual(
            [patch.action for patch in result.applied],
            [
                RepairAction.MODIFY,
                RepairAction.MODIFY,
                RepairAction.DELETE,
                RepairAction.MODIFY,
            ],
        )
        self.assertIn("[facts-E1]", result.report)
        self.assertIn("仅获得部分支持", result.report)
        self.assertNotIn("主体和时间均错误", result.report)
        self.assertIn("已降级为待验证问题", result.report)
        repaired_ledger = await ClaimEvidenceVerifier().build_ledger(result.report, evidences)
        weakened = [claim for claim in repaired_ledger.claims if "仅获得部分支持" in claim.text]
        self.assertTrue(weakened)
        self.assertTrue(all(claim.verdict is SupportVerdict.NOT_APPLICABLE for claim in weakened))

    def test_blue_never_deletes_a_supported_key_fact(self):
        supported_line = "Kafka 不同 consumer group 可以独立消费。 [facts-E1]"
        unsupported_line = "Kafka 保证消费者数量增加时吞吐无限增长。"
        report = f"{supported_line}\n{unsupported_line}"
        ledger = ClaimLedger(
            claims=[
                ClaimRecord(
                    "C1",
                    "Kafka 不同 consumer group 可以独立消费。",
                    source_text=supported_line,
                    citations=["facts-E1"],
                    verdict=SupportVerdict.SUPPORTED,
                    links=[ClaimEvidenceLink("facts-E1", SupportVerdict.SUPPORTED, 0.99)],
                ),
                ClaimRecord(
                    "C2",
                    "Kafka 保证消费者数量增加时吞吐无限增长。",
                    source_text=unsupported_line,
                    verdict=SupportVerdict.UNCITED,
                ),
            ]
        )

        result = BlueTeamRepairer(llm=None).repair_claim_gaps(
            report,
            [Evidence("facts-E1", "facts", "不同组可独立消费。", "source")],
            ledger,
        )

        self.assertIn(supported_line, result.report)
        self.assertFalse(
            any(
                patch.action is RepairAction.DELETE and patch.target == supported_line
                for patch in result.applied
            )
        )

    def test_blue_compacts_multiple_evidence_gaps_per_section(self):
        lines = [
            "第一条尚未得到证据支持的事实陈述。",
            "第二条尚未得到证据支持的事实陈述。",
            "第三条尚未得到证据支持的事实陈述。",
        ]
        report = "# 报告\n\n## 发现\n" + "\n".join(lines)
        ledger = ClaimLedger(
            claims=[
                ClaimRecord(
                    f"C{index}",
                    line,
                    section="发现",
                    source_text=line,
                    verdict=SupportVerdict.UNCITED,
                )
                for index, line in enumerate(lines, start=1)
            ]
        )

        result = BlueTeamRepairer(llm=None).repair_claim_gaps(report, [], ledger)

        self.assertEqual(result.report.count("已合并为待验证问题"), 1)
        self.assertIn("本节有 3 项内容无法确认", result.report)
        self.assertTrue(any(patch.patch_id == "AUTO-gap-compaction" for patch in result.applied))

    async def test_blue_deletes_exact_material_inference_overreach(self):
        target = "字段存在，因此该接口保证支持全部筛选任务。 [facts-E1]"
        review = ReviewResult(
            False,
            structured_issues=[
                ReviewIssue(
                    "R1",
                    "logic",
                    "字段存在不能推出接口支持全部筛选任务",
                    severity=2,
                    action=RepairAction.MODIFY,
                    target=target,
                    category="inference_overreach",
                )
            ],
        )

        report = f"保留这条由证据直接支持的事实。 [facts-E1]\n{target}"
        result = await BlueTeamRepairer(llm=lambda _: "{}").repair(
            "问题",
            report,
            [Evidence("facts-E1", "facts", "记录包含该字段。", "source")],
            review,
        )

        self.assertNotIn("保证支持全部筛选任务", result.report)
        self.assertEqual(result.applied[0].action, RepairAction.DELETE)

    async def test_targeted_retrieval_aligns_cross_language_candidate(self):
        claim_text = "该客户端提供异步搜索接口并返回结构化结果。"

        async def verifier_llm(_prompt):
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": "C1::verify_r1_1-E1",
                            "verdict": "SUPPORTED",
                            "confidence": 0.96,
                            "conflict_types": [],
                        }
                    ]
                }
            )

        ledger = await ClaimEvidenceVerifier(verifier_llm).build_ledger(
            claim_text,
            [
                Evidence(
                    "verify_r1_1-E1",
                    "verify_r1_1",
                    "The client exposes asynchronous search and structured results.",
                    "official",
                    metadata={"retrieval_query": (f"核验以下陈述并寻找直接一手证据：{claim_text}")},
                )
            ],
        )

        self.assertEqual(ledger.claims[0].verdict, SupportVerdict.UNCITED)
        self.assertEqual(ledger.claims[0].links[0].verdict, SupportVerdict.SUPPORTED)
        self.assertFalse(ledger.claims[0].links[0].cited)

    async def test_partial_patch_keeps_supported_aspect_and_markdown_list(self):
        report = "- **该方案保证所有请求始终成功。** [facts-E1]"
        extracted = await ClaimEvidenceVerifier().build_ledger(
            report,
            [Evidence("facts-E1", "facts", "该方案支持请求重试。", "source")],
        )
        self.assertTrue(extracted.claims[0].source_text.startswith("- **"))
        ledger = ClaimLedger(
            claims=[
                ClaimRecord(
                    "C1",
                    "该方案保证所有请求始终成功。",
                    source_text=report,
                    citations=["facts-E1"],
                    verdict=SupportVerdict.PARTIALLY_SUPPORTED,
                    links=[
                        ClaimEvidenceLink(
                            "facts-E1",
                            SupportVerdict.PARTIALLY_SUPPORTED,
                            0.9,
                            supported_aspects=["该方案支持请求重试"],
                            unsupported_aspects=["无法保证始终成功"],
                        )
                    ],
                )
            ]
        )

        result = BlueTeamRepairer(llm=None).repair_claim_gaps(
            report,
            [Evidence("facts-E1", "facts", "该方案支持请求重试。", "source")],
            ledger,
        )

        self.assertEqual(result.report, "- 该方案支持请求重试。 [facts-E1]")
        self.assertNotIn("**", result.report)

    async def test_blue_rolls_back_semantically_unsupported_patch(self):
        async def blue_llm(prompt):
            return json.dumps(
                {
                    "patches": [
                        {
                            "patch_id": "P1",
                            "issue_ids": ["R1"],
                            "action": "MODIFY",
                            "target": "客户端能力仍需核验。",
                            "replacement": "客户端保证搜索结果绝对正确。 [facts-E1]",
                            "evidence_ids": ["facts-E1"],
                        }
                    ]
                },
                ensure_ascii=False,
            )

        async def verifier_llm(prompt):
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": "P-C1::facts-E1",
                            "verdict": "INSUFFICIENT",
                            "confidence": 0.99,
                            "rationale": "证据只说明客户端存在，没有准确性保证",
                        }
                    ]
                }
            )

        report = "# 报告\n\n客户端能力仍需核验。"
        review = ReviewResult(
            False,
            structured_issues=[
                ReviewIssue("R1", "factuality", "需要修复", action=RepairAction.MODIFY)
            ],
        )
        repairer = BlueTeamRepairer(
            blue_llm,
            max_generation_attempts=1,
            verifier=ClaimEvidenceVerifier(verifier_llm),
        )
        result = await repairer.repair(
            "客户端是否可靠？",
            report,
            [Evidence("facts-E1", "facts", "客户端支持异步调用", "official")],
            review,
        )

        self.assertFalse(result.changed)
        self.assertEqual(result.report, report)
        self.assertIn("claim_evidence_validation_failed", result.rejected[0].reason)

    async def test_blue_retries_once_after_semantic_rejection(self):
        calls = 0

        async def blue_llm(prompt):
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps(
                    {
                        "patches": [
                            {
                                "patch_id": "P1",
                                "issue_ids": ["R1"],
                                "action": "MODIFY",
                                "target": "这是一条无法证实的陈述。",
                                "replacement": "这是带错配引用的新陈述。 [facts-E1]",
                                "evidence_ids": ["facts-E1"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            self.assertIn("claim_evidence_validation_failed", prompt)
            return json.dumps(
                {
                    "patches": [
                        {
                            "patch_id": "P2",
                            "issue_ids": ["R1"],
                            "action": "DELETE",
                            "target": "这是一条无法证实的陈述。",
                            "replacement": "",
                            "evidence_ids": [],
                        }
                    ]
                },
                ensure_ascii=False,
            )

        async def verifier_llm(prompt):
            return json.dumps(
                {
                    "judgments": [
                        {
                            "pair_id": "P-C1::facts-E1",
                            "verdict": "INSUFFICIENT",
                            "confidence": 0.9,
                        }
                    ]
                }
            )

        report = "# 报告\n\n这是一条无法证实的陈述。\n\n## 局限\n证据不足。"
        review = ReviewResult(
            False,
            structured_issues=[
                ReviewIssue("R1", "factuality", "删除无依据内容", action=RepairAction.DELETE)
            ],
        )
        result = await BlueTeamRepairer(
            blue_llm,
            max_generation_attempts=2,
            verifier=ClaimEvidenceVerifier(verifier_llm),
        ).repair(
            "问题",
            report,
            [Evidence("facts-E1", "facts", "其他事实", "official")],
            review,
        )

        self.assertTrue(result.changed)
        self.assertEqual(calls, 2)
        self.assertEqual(result.requested_count, 2)
        self.assertNotIn("无法证实的陈述", result.report)
        self.assertTrue(any(item.patch_id == "P1" for item in result.rejected))

    async def test_actionable_issue_does_not_false_converge(self):
        issue = ReviewIssue("rule-citation-coverage", "citation_quality", "覆盖不足")
        convergence = ReviewConvergence(min_improvement=0.1)
        first = ReviewResult(False, structured_issues=[issue], total_score=0.8)
        second = ReviewResult(False, structured_issues=[issue], total_score=0.8)
        convergence.update("draft one", first)
        convergence.update("draft two", second)
        self.assertFalse(second.converged)

    async def test_code_and_source_index_are_not_counted_as_claims(self):
        from research_engine.schemas import Evidence

        evidence = Evidence("facts-E1", "facts", "可验证事实。", "source")
        report = (
            "# 报告\n\n## 发现\n这是一条有引用的可验证事实。 [facts-E1]\n"
            "```python\nclient.search('query')\n```\n"
            "## 来源与证据索引\n| ID | URL |\n|---|---|\n| facts-E1 | https://example.com |"
        )
        review = await RedTeamReviewer().review("问题", report, [evidence])
        self.assertEqual(review.metrics["claim_count"], 1)
        self.assertEqual(review.metrics["citation_coverage"], 1)

    async def test_bold_fact_labels_and_report_metadata_are_not_claims(self):
        from research_engine.schemas import Evidence

        evidence = Evidence("facts-E1", "facts", "可验证事实。", "source")
        report = (
            "# 报告\n"
            "**研究问题**：测试问题\n"
            "**研究对象**：Python SDK\n"
            "**证据完整性声明**：证据范围有限\n"
            "## 发现\n**事实 1：异步客户端存在。**\n"
            "异步客户端确实存在并受官方支持。 [facts-E1]\n"
            "## 局限\n证据存在局限。 [facts-E1]"
        )
        review = await RedTeamReviewer().review("问题", report, [evidence])
        self.assertEqual(review.metrics["claim_count"], 2)
        self.assertEqual(review.metrics["citation_coverage"], 1)

    async def test_markdown_table_header_is_not_a_claim(self):
        from research_engine.schemas import Evidence

        evidence = Evidence("facts-E1", "facts", "字段定义。", "source")
        report = (
            "# 报告\n\n## 字段\n"
            "| 字段 | 说明 |\n|---|---|\n"
            "| `url` | 结果地址 [facts-E1] |\n"
            "## 局限\n证据存在局限。 [facts-E1]"
        )
        review = await RedTeamReviewer().review("问题", report, [evidence])
        self.assertEqual(review.metrics["claim_count"], 2)
        self.assertEqual(review.metrics["citation_coverage"], 1)
