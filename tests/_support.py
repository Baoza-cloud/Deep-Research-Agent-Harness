# ruff: noqa: F401

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

from research_engine import (
    AgentSpec,
    BudgetController,
    BudgetResource,
    ComplexityLevel,
    DeepResearchAgent,
    ClaimEvidenceVerifier,
    ClaimEvidenceLink,
    ClaimLedger,
    ClaimRecord,
    Evidence,
    HeuristicPlanner,
    HeuristicSwarmPolicy,
    LegacyAgentAdapter,
    LLMPlanner,
    LocalAgentRuntime,
    ResearchConfig,
    ResearchPlan,
    ResearchSubtask,
    ResearchWorker,
    RetrievalToolAdapter,
    ReportPatch,
    SharedMemory,
    RunContext,
    Synthesizer,
    TaskStatus,
    TavilySearchBackend,
    ToolRegistry,
    SupportVerdict,
    sanitize_untrusted_content,
)
from research_engine.web_backends import CompositeSearchBackend
from research_engine.parsing import parse_json_payload
from research_engine.llm_backends import LLMProvider, backend_config
from research_engine.env import load_env_files
from research_engine.state import ResearchRunState, StateTransitionError
from research_engine.adversarial import RedTeamReviewer
from research_engine import BlueTeamRepairer, RepairAction, ReviewIssue, ReviewResult
from research_engine.adversarial import ReviewConvergence
from research_engine.orchestrator import (
    ReportQualityCandidate,
    adjudicate_red_with_claim_ledger,
    select_quality_candidate,
)
from research_engine.text_quality import (
    normalize_evidence_bound_language,
    remove_omission_markers,
)

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))
from evaluate_web_search import evaluate_sample
from research_evaluation import (
    evaluate_expectations,
    evaluate_rules,
    paired_cohens_d,
)
from run_ablation_experiments import (
    InstrumentedLLM,
    RUN_CONTEXT,
    VARIANT_DESCRIPTIONS,
    _reprice_row,
    _resume_row_is_compatible,
    load_frozen_dataset,
)
from evaluate_live_swarm import load_dataset as load_live_swarm_dataset
from calibrate_retrieval_thresholds import (
    _aggregate as aggregate_threshold_rows,
    _aggregate_gold,
    _proxy_relevance,
    recommend_gold_thresholds,
)
from adversarial_evaluation import (
    evaluate_adversarial_report,
    evaluate_rollback_guard,
    inject_faults,
)
from analyze_review_issue_distribution import analyze as analyze_review_issues


class FakeBackend:
    async def search(self, query, limit=5):
        return [{"text": f"与任务相关的可验证事实：{query}", "source": "fixture.txt"}]


class SlowBackend:
    async def search(self, query, limit=5):
        await asyncio.sleep(0.1)
        return [{"text": "late evidence", "source": "slow"}]


class ReplanBackend:
    async def search(self, query, limit=5):
        if "换用更窄" not in query:
            raise RuntimeError("primary backend unavailable")
        return [{"text": "替代检索得到的证据", "source": "fallback"}]


class FakeTavilyClient:
    def __init__(self):
        self.kwargs = None

    async def search(self, **kwargs):
        self.kwargs = kwargs
        return {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com/article",
                    "content": "short snippet",
                    "raw_content": "full normalized article content",
                    "score": 0.91,
                }
            ]
        }


class FixedBackend:
    def __init__(self, rows):
        self.rows = rows

    async def search(self, query, limit=5):
        return self.rows[:limit]


class RepairingSynthesizer:
    def __init__(self):
        self.calls = 0

    async def synthesize(self, question, plan, evidences, memory, draft=None, **kwargs):
        self.calls += 1
        long_claim = "这是需要证据支持的长事实陈述" + "内容" * 450
        citation = " [scope-E1]" if draft is not None else ""
        return f"# 报告\n\n## 发现\n{long_claim}。{citation}\n\n## 局限\n存在局限。"


class AlwaysFailReviewer:
    async def review(self, question, report, evidences):
        return ReviewResult(
            False,
            issues=["blocking issue"],
            metrics={"citation_coverage": 0.5},
            total_score=0.5,
        )


class AlwaysPassReviewer:
    async def review(self, question, report, evidences):
        return ReviewResult(
            True,
            metrics={"citation_coverage": 1.0},
            total_score=0.95,
        )


class UnsupportedClaimSynthesizer:
    llm = None

    async def synthesize(self, question, plan, evidences, memory, **kwargs):
        return (
            "# 报告\n\n这是一条没有任何引用且长度足够的事实性结论。\n\n"
            "## 局限\n本报告证据范围有限。"
        )


class OmissionSynthesizer:
    llm = None

    async def synthesize(self, question, plan, evidences, memory, **kwargs):
        return (
            "# 报告\n\n事实一。 [...] 事实二。 [……] 事实三。 [scope-E1]\n\n"
            "## 局限\n证据有限。 [scope-E1]"
        )


class EvidenceGapSynthesizer:
    llm = None

    async def synthesize(self, question, plan, evidences, memory, **kwargs):
        return (
            "# 报告\n\n## 结论\n"
            "基于当前证据，以下内容无法确认，已降级为待验证问题："
            "“该产品在所有地区都具有完全相同的行为。”\n\n"
            "## 局限与不确定性\n当前证据范围有限。"
        )


class QualityFallbackSynthesizer:
    llm = None

    async def synthesize(
        self,
        question,
        plan,
        evidences,
        memory,
        draft=None,
        repair_instructions=None,
        **kwargs,
    ):
        if any("FIXED_HARNESS_QUALITY_FALLBACK" in item for item in (repair_instructions or [])):
            return (
                "# fixed fallback\n\n## 发现\n"
                "系统支持异步搜索。 [scope-E1]\n\n"
                "## 局限与不确定性\n证据范围有限。"
            )
        return (
            "# dynamic draft\n\n## 发现\n"
            "系统支持异步搜索。 [scope-E1]\n\n"
            "## 局限与不确定性\n证据范围有限。"
        )


class QualityFallbackReviewer:
    async def review(self, question, report, evidences):
        fixed = report.startswith("# fixed fallback")
        return ReviewResult(
            fixed,
            metrics={"citation_coverage": 1.0 if fixed else 0.5},
            total_score=0.9,
        )
