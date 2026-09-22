"""Deterministic fault injection and metrics for Red/Blue evaluation."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence

from research_engine import (
    BlueTeamRepairer,
    Evidence,
    PatchApplicationResult,
    RepairAction,
    ReportPatch,
    ReviewResult,
)

from research_evaluation import evaluate_expectations, evaluate_rules


class FaultType(str, Enum):
    UNKNOWN_CITATION = "unknown_citation"
    UNCITED_CLAIM = "uncited_claim"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_LIMITATION = "missing_limitation"


@dataclass(frozen=True)
class InjectedFault:
    fault_id: str
    fault_type: FaultType
    target: str
    expected_issue_id: str | None = None


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    sample: dict[str, Any]
    clean_report: str
    corrupted_report: str
    faults: tuple[InjectedFault, ...]


@dataclass(frozen=True)
class AdversarialMetrics:
    fault_repair_rate: float
    red_detection_recall: float
    clean_fact_retention_rate: float
    collateral_damage_rate: float
    rule_factual_accuracy: float
    rule_hallucination_rate: float
    citation_validity: float
    citation_coverage: float
    citation_preservation_rate: float
    applied_patch_count: int
    rejected_patch_count: int
    patch_acceptance_rate: float
    repaired_fault_ids: tuple[str, ...]
    detected_fault_ids: tuple[str, ...]


SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*")
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+-E\d+)\]")
LIMITATION_PATTERN = re.compile(
    r"(?m)^#{2,6}\s+.*(?:局限|不确定)|(?:证据不足|证据范围).{0,40}(?:有限|限制|未覆盖)"
)


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in SENTENCE_SPLIT.split(text) if item.strip()]


def _ensure_terminal(text: str) -> str:
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else text + "。"


def build_clean_report(sample: dict[str, Any]) -> str:
    """Create a citation-complete report from frozen evidence without an LLM."""

    findings: list[str] = []
    for evidence in sample["evidence"]:
        evidence_id = str(evidence["evidence_id"])
        for sentence in _sentences(str(evidence["content"])):
            findings.append(f"{_ensure_terminal(sentence)} [{evidence_id}]")
    return (
        f"# {sample['question']}\n\n"
        "## 关键发现\n"
        + "\n".join(findings)
        + "\n\n## 局限与不确定性\n"
        "不确定性说明。"
    )


def inject_faults(sample: dict[str, Any]) -> AdversarialCase:
    """Inject four traceable defects while preserving the clean reference report."""

    clean = build_clean_report(sample)
    evidence_id = str(sample["evidence"][0]["evidence_id"])
    ghost_citation = "[ghost-E99]"
    corrupted = clean.replace(f"[{evidence_id}]", ghost_citation, 1)

    first_sentence = _sentences(str(sample["evidence"][0]["content"]))[0]
    uncited_target = "\n".join(
        f"补充说明{index}：{_ensure_terminal(first_sentence)}"
        for index in range(1, 4)
    )
    unsupported_target = _ensure_terminal(
        str(sample.get("forbidden_terms", ["该结论保证绝对成立"])[0])
    ) + f" [{evidence_id}]"
    insertion = f"{uncited_target}\n{unsupported_target}\n\n"
    corrupted = corrupted.replace("## 局限与不确定性", insertion + "## 局限与不确定性", 1)

    limitation_block = (
        "## 局限与不确定性\n"
        "不确定性说明。"
    )
    corrupted = corrupted.replace(limitation_block, "")
    faults = (
        InjectedFault(
            "F-UNKNOWN-CITATION",
            FaultType.UNKNOWN_CITATION,
            ghost_citation,
            "rule-unknown-citations",
        ),
        InjectedFault(
            "F-UNCITED-CLAIM",
            FaultType.UNCITED_CLAIM,
            uncited_target,
            "rule-citation-coverage",
        ),
        InjectedFault(
            "F-UNSUPPORTED-CLAIM",
            FaultType.UNSUPPORTED_CLAIM,
            unsupported_target,
        ),
        InjectedFault(
            "F-MISSING-LIMITATION",
            FaultType.MISSING_LIMITATION,
            "## 局限与不确定性",
            "rule-uncertainty",
        ),
    )
    return AdversarialCase(
        case_id=f"adv-{sample['id']}",
        sample=sample,
        clean_report=clean,
        corrupted_report=corrupted.strip(),
        faults=faults,
    )


def evidence_objects(case: AdversarialCase) -> list[Evidence]:
    return [
        Evidence(
            evidence_id=str(row["evidence_id"]),
            subtask_id=str(row.get("subtask_id", "facts")),
            content=str(row["content"]),
            source=str(row.get("source") or row.get("url") or "frozen"),
            title=row.get("title"),
            url=row.get("url"),
            source_type=str(row.get("source_type", "frozen_reference")),
        )
        for row in case.sample["evidence"]
    ]


def _fault_repaired(fault: InjectedFault, report: str, valid_ids: set[str]) -> bool:
    if fault.fault_type is FaultType.UNKNOWN_CITATION:
        return fault.target not in report
    if fault.fault_type is FaultType.UNSUPPORTED_CLAIM:
        return fault.target not in report
    if fault.fault_type is FaultType.MISSING_LIMITATION:
        return bool(LIMITATION_PATTERN.search(report))
    if fault.target not in report:
        return True
    for line in report.splitlines():
        if fault.target in line:
            return bool(set(CITATION_PATTERN.findall(line)) & valid_ids)
    return False


def _fault_detected(fault: InjectedFault, review: ReviewResult | None) -> bool:
    if review is None:
        return False
    if fault.expected_issue_id and any(
        item.issue_id == fault.expected_issue_id for item in review.structured_issues
    ):
        return True
    if fault.fault_type is FaultType.UNSUPPORTED_CLAIM:
        normalize = lambda value: re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())
        target = normalize(fault.target)
        return any(
            item.dimension == "factuality"
            and (
                target[:12] in normalize(item.target or "")
                or target[:12] in normalize(item.description)
            )
            for item in review.structured_issues
        )
    return False


def evaluate_adversarial_report(
    case: AdversarialCase,
    report: str,
    review: ReviewResult | None = None,
    patch_result: PatchApplicationResult | None = None,
) -> AdversarialMetrics:
    """Measure repair and collateral damage against injected fault ground truth."""

    valid_ids = {str(item["evidence_id"]) for item in case.sample["evidence"]}
    repaired = tuple(
        fault.fault_id for fault in case.faults if _fault_repaired(fault, report, valid_ids)
    )
    detected = tuple(
        fault.fault_id for fault in case.faults if _fault_detected(fault, review)
    )
    before = evaluate_expectations(case.clean_report, case.sample)
    after = evaluate_expectations(report, case.sample)
    retention = (
        after.key_facts_hit / before.key_facts_hit if before.key_facts_hit else 1.0
    )
    retention = min(1.0, retention)
    rules = evaluate_rules(report, case.sample["evidence"])
    clean_citations = set(CITATION_PATTERN.findall(case.clean_report))
    final_citations = set(CITATION_PATTERN.findall(report))
    applied = len(patch_result.applied) if patch_result else 0
    rejected = len(patch_result.rejected) if patch_result else 0
    requested = applied + rejected
    return AdversarialMetrics(
        fault_repair_rate=len(repaired) / len(case.faults),
        red_detection_recall=len(detected) / len(case.faults),
        clean_fact_retention_rate=retention,
        collateral_damage_rate=1.0 - retention,
        rule_factual_accuracy=rules.factual_accuracy,
        rule_hallucination_rate=rules.hallucination_rate,
        citation_validity=rules.citation_validity,
        citation_coverage=rules.citation_coverage,
        citation_preservation_rate=(
            len(clean_citations & final_citations) / len(clean_citations)
            if clean_citations
            else 1.0
        ),
        applied_patch_count=applied,
        rejected_patch_count=rejected,
        patch_acceptance_rate=applied / requested if requested else 0.0,
        repaired_fault_ids=repaired,
        detected_fault_ids=detected,
    )


def aggregate_adversarial_metrics(
    rows: Sequence[AdversarialMetrics],
) -> dict[str, float]:
    if not rows:
        raise ValueError("At least one adversarial result is required")
    fields = (
        "fault_repair_rate",
        "red_detection_recall",
        "clean_fact_retention_rate",
        "collateral_damage_rate",
        "rule_factual_accuracy",
        "rule_hallucination_rate",
        "citation_validity",
        "citation_coverage",
        "citation_preservation_rate",
        "patch_acceptance_rate",
    )
    return {
        name: statistics.fmean(float(getattr(row, name)) for row in rows)
        for name in fields
    } | {
        "applied_patch_count": float(sum(row.applied_patch_count for row in rows)),
        "rejected_patch_count": float(sum(row.rejected_patch_count for row in rows)),
    }


def serialize_case(case: AdversarialCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "sample_id": case.sample["id"],
        "domain": case.sample["domain"],
        "question": case.sample["question"],
        "clean_report": case.clean_report,
        "corrupted_report": case.corrupted_report,
        "faults": [
            {**asdict(item), "fault_type": item.fault_type.value}
            for item in case.faults
        ],
    }


def evaluate_rollback_guard() -> dict[str, Any]:
    """Attack the deterministic patch executor and verify full rollback."""

    report = "重复锚点。\n\n重复锚点。\n\n合法事实。 [facts-E1]"
    evidence = Evidence("facts-E1", "facts", "合法事实。", "fixture")
    attacks = [
        ReportPatch(
            "A-UNKNOWN-CITATION",
            RepairAction.MODIFY,
            "合法事实。 [facts-E1]",
            "伪造事实。 [ghost-E99]",
            evidence_ids=["ghost-E99"],
        ),
        ReportPatch("A-MISSING-TARGET", RepairAction.DELETE, "不存在的唯一锚点"),
        ReportPatch("A-AMBIGUOUS-TARGET", RepairAction.DELETE, "重复锚点。"),
        ReportPatch(
            "A-DELETE-WITH-REPLACEMENT",
            RepairAction.DELETE,
            "合法事实。 [facts-E1]",
            "不允许的替换",
        ),
    ]
    result = BlueTeamRepairer(llm=None).apply_patches(report, attacks, [evidence])
    rejected_ids = {item.patch_id for item in result.rejected}
    protected = [
        patch.patch_id
        for patch in attacks
        if patch.patch_id in rejected_ids and patch.patch_id not in {
            item.patch_id for item in result.applied
        }
    ]
    return {
        "attack_count": len(attacks),
        "rejected_count": len(result.rejected),
        "applied_count": len(result.applied),
        "report_unchanged": result.report == report,
        "rollback_correctness": len(protected) / len(attacks),
        "rejection_reasons": {
            item.patch_id: item.reason for item in result.rejected
        },
    }
