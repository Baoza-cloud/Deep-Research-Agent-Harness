import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research_engine.adversarial import BlueTeamRepairer
from research_engine.comparison import (
    build_comparison_coverage,
    enforce_comparison_coverage,
    parse_comparison_spec,
)
from research_engine.llm_backends import LLMBackendConfig, LLMProvider, OpenAICompatibleLLM
from research_engine.schemas import (
    Evidence,
    PatchPosition,
    RepairAction,
    ReportPatch,
    ReviewResult,
)
from research_engine.versioning import engine_version, source_commit


QUESTION = (
    "对比 Kubernetes Ingress 与 Gateway API。请基于官方资料，分析 API 成熟度、"
    "核心资源模型、跨命名空间安全与实现兼容性；给出建议。"
)


class ComparisonCoverageGateTests(unittest.TestCase):
    def test_comparison_parser_builds_two_sided_dimension_matrix(self):
        spec = parse_comparison_spec(QUESTION)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.subjects, ("Kubernetes Ingress", "Gateway API"))
        self.assertEqual(
            spec.dimensions,
            ("API 成熟度", "核心资源模型", "跨命名空间安全", "实现兼容性"),
        )

    def test_missing_side_is_prioritized_as_evidence_gap(self):
        report = (
            "# 报告\n\n"
            "### API 成熟度\n"
            "Gateway API 已进入稳定阶段。 [gateway-E1]\n"
            "**证据缺口：** 当前没有 Ingress API 的直接证据。\n\n"
            "### 核心资源模型\n"
            "Gateway API 使用 Gateway 资源。 [gateway-E2]\n"
            "Ingress 使用 Ingress 资源。 [ingress-E1]\n\n"
            "### 跨命名空间安全\n"
            "Gateway API 使用 ReferenceGrant。 [gateway-E3]\n"
            "Ingress 的跨命名空间机制缺乏直接证据。\n\n"
            "### 实现兼容性\n"
            "Gateway API 有一致性报告。 [gateway-E4]\n"
            "Ingress 控制器存在实现差异。 [ingress-E2]"
        )

        coverage = build_comparison_coverage(QUESTION, report)
        review = enforce_comparison_coverage(ReviewResult(True), coverage)

        self.assertFalse(coverage.passed)
        self.assertEqual(len(coverage.missing), 2)
        self.assertTrue(all(item.subject == "Kubernetes Ingress" for item in coverage.missing))
        self.assertIn("Kubernetes Ingress 官方文档", coverage.repair_queries()[0])
        self.assertFalse(review.passed)
        self.assertTrue(all(issue.category == "evidence_gap" for issue in review.structured_issues))
        self.assertTrue(all(issue.action is RepairAction.ADD for issue in review.structured_issues))


class PatchIntegrityGateTests(unittest.TestCase):
    def setUp(self):
        self.repairer = BlueTeamRepairer(max_growth_chars=2_000)
        self.evidence = [Evidence("facts-E1", "facts", "直接支持事实。", "official")]

    def test_patch_that_introduces_duplicate_claim_is_rolled_back(self):
        report = "# 报告\n\n这是一个由证据直接支持的完整事实陈述。 [facts-E1]"
        patch_row = ReportPatch(
            "P-duplicate",
            RepairAction.ADD,
            target="这是一个由证据直接支持的完整事实陈述。 [facts-E1]",
            replacement="这是一个由证据直接支持的完整事实陈述。 [facts-E1]",
            position=PatchPosition.AFTER,
            evidence_ids=["facts-E1"],
        )

        result = self.repairer.apply_patches(report, [patch_row], self.evidence)

        self.assertFalse(result.changed)
        self.assertEqual(result.report, report)
        self.assertIn("duplicate_claim", result.rejected[0].reason)

    def test_patch_that_introduces_enumeration_conflict_is_rolled_back(self):
        report = "Gateway API 具有三种稳定的 API 类别。 [facts-E1]"
        patch_row = ReportPatch(
            "P-count",
            RepairAction.ADD,
            target=report,
            replacement=(
                "Gateway API 具有稳定的 API 类别"
                "（GatewayClass、Gateway、HTTPRoute、GRPCRoute）。 [facts-E1]"
            ),
            evidence_ids=["facts-E1"],
        )

        result = self.repairer.apply_patches(report, [patch_row], self.evidence)

        self.assertFalse(result.changed)
        self.assertIn("enumeration_conflict", result.rejected[0].reason)

    def test_patch_boundary_formatter_separates_citation_and_next_sentence(self):
        report = "原始陈述。 [facts-E1]"
        patch_row = ReportPatch(
            "P-format",
            RepairAction.MODIFY,
            target=report,
            replacement="修复后的陈述。 [facts-E1]下一条说明。",
            evidence_ids=["facts-E1"],
        )

        result = self.repairer.apply_patches(report, [patch_row], self.evidence)

        self.assertTrue(result.changed)
        self.assertIn("[facts-E1] 下一条说明", result.report)

    def test_summary_and_conclusion_may_restate_the_same_supported_claim(self):
        from research_engine.text_quality import report_integrity_issues

        report = (
            "## 执行摘要\nGateway API 是基于 CRD 的 API，需要安装 CRD。 [facts-E1]\n\n"
            "## 结论\nGateway API 是基于 CRD 的 API，需要安装 CRD。 [facts-E1]"
        )

        self.assertFalse(
            any(issue.startswith("duplicate_claim:") for issue in report_integrity_issues(report))
        )

    def test_integrity_gate_detects_truncated_gap_markup(self):
        from research_engine.text_quality import report_integrity_issues

        malformed = (
            "## 跨命名空间安全\n"
            "基于当前证据，以下内容无法确认：“Gateway API 使用 ReferenceGrant "
            "[gateway-cross-names。**"
        )

        issues = report_integrity_issues(malformed)

        self.assertIn("dangling_citation_token", issues)
        self.assertIn("unbalanced_chinese_quotes", issues)
        self.assertIn("unbalanced_markdown_bold", issues)


class ReleaseTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unconfigured_pricing_is_unknown_instead_of_zero(self):
        llm = OpenAICompatibleLLM(
            LLMBackendConfig(
                provider=LLMProvider.DEEPSEEK,
                model="test-model",
                api_key="test-key",
            )
        )

        class FakeCompletions:
            def create(self, **_kwargs):
                return SimpleNamespace(
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                )

        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        await llm("hello")
        usage = llm.usage_snapshot()

        self.assertIsNone(usage["cost_usd"])
        self.assertEqual(usage["cost_status"], "unknown")

    def test_engine_identity_ignores_dev_marker_and_records_git_sha(self):
        with patch.dict(os.environ, {"RESEARCH_ENGINE_VERSION": "dev"}, clear=False):
            self.assertEqual(engine_version(), "v1.2.2")
        resolved = source_commit()
        self.assertIsNotNone(resolved)
        self.assertRegex(resolved, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
