"""Deterministic evidence coverage gates for A/B comparison research."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .schemas import RepairAction, ReviewIssue, ReviewResult


_CITATION = re.compile(r"\[[A-Za-z0-9_-]+-E\d+\]")
_GAP_LANGUAGE = re.compile(
    r"证据缺口|缺乏(?:直接)?证据|未包含|未提供|无法(?:确认|比较|判断)|"
    r"证据不足|待验证|not enough evidence|evidence gap",
    re.IGNORECASE,
)
_HEADING = re.compile(r"(?m)^#{2,4}\s+(?:\d+[.)、]\s*)?(?P<title>[^\n]+)\s*$")
_CHINESE_COMPARISON = re.compile(
    r"(?:对比|比较)\s*(?P<left>[^，。；:：!?！？]{2,60}?)\s*"
    r"(?:与|和|及|vs\.?|versus)\s*"
    r"(?P<right>[^，。；:：!?！？]{2,60})",
    re.IGNORECASE,
)
_ENGLISH_COMPARISON = re.compile(
    r"(?:compare|comparison of)\s+(?P<left>[^,.;:!?]{2,60}?)\s+"
    r"(?:and|with|vs\.?|versus)\s+(?P<right>[^,.;:!?]{2,60})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ComparisonSpec:
    subjects: tuple[str, str]
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonCoverageCell:
    subject: str
    dimension: str
    covered: bool
    section: str
    reason: str


@dataclass(frozen=True)
class ComparisonCoverage:
    spec: ComparisonSpec | None
    cells: tuple[ComparisonCoverageCell, ...] = ()

    @property
    def applicable(self) -> bool:
        return self.spec is not None

    @property
    def missing(self) -> tuple[ComparisonCoverageCell, ...]:
        return tuple(cell for cell in self.cells if not cell.covered)

    @property
    def coverage_rate(self) -> float:
        if not self.cells:
            return 1.0
        return sum(cell.covered for cell in self.cells) / len(self.cells)

    @property
    def passed(self) -> bool:
        return not self.missing

    def repair_queries(self, *, limit: int = 2) -> list[str]:
        if self.spec is None or limit <= 0:
            return []
        missing_by_subject: dict[str, list[str]] = {}
        for cell in self.missing:
            dimensions = missing_by_subject.setdefault(cell.subject, [])
            if cell.dimension not in dimensions:
                dimensions.append(cell.dimension)
        queries = [
            f"{subject} 官方文档：{'、'.join(dimensions)}；原始规范 直接证据 生产限制"
            for subject, dimensions in missing_by_subject.items()
        ]
        return queries[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "subjects": list(self.spec.subjects) if self.spec else [],
            "dimensions": list(self.spec.dimensions) if self.spec else [],
            "coverage_rate": self.coverage_rate,
            "passed": self.passed,
            "missing_cell_count": len(self.missing),
            "cells": [
                {
                    "subject": cell.subject,
                    "dimension": cell.dimension,
                    "covered": cell.covered,
                    "section": cell.section,
                    "reason": cell.reason,
                }
                for cell in self.cells
            ],
        }


def _clean_subject(value: str) -> str:
    value = re.sub(r"^(?:截至\s*[^，,]+[，,]\s*)", "", value.strip())
    return re.sub(r"\s+", " ", value).strip(" '\"“”‘’")


def _dimensions(question: str) -> tuple[str, ...]:
    match = re.search(
        r"分析(?P<body>.+?)(?:；|;|。|\.(?:\s|$)|给出|并给出)",
        question,
        re.IGNORECASE,
    )
    if not match:
        return ("核心结论",)
    body = match.group("body")
    # Remove the leading comparison subjects before the first explicit dimension.
    body = re.sub(
        r"^.+?(?:官方资料|官方文档|资料|证据)[，,]?\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"[、,，/；;]|(?:以及|与)(?=\S)|\s+(?:and|as well as)\s+", body)
    dimensions: list[str] = []
    for raw in parts:
        item = re.sub(r"^(?:以及|及|与|和)\s*", "", raw.strip())
        item = re.sub(r"^(?:请|仅|基于).{0,20}?(?:分析|说明)\s*", "", item)
        if 2 <= len(item) <= 32 and item not in dimensions:
            dimensions.append(item)
    return tuple(dimensions[:8]) or ("核心结论",)


def parse_comparison_spec(question: str) -> ComparisonSpec | None:
    match = _CHINESE_COMPARISON.search(question) or _ENGLISH_COMPARISON.search(question)
    if not match:
        return None
    left = _clean_subject(match.group("left"))
    right = _clean_subject(match.group("right"))
    if not left or not right or left.casefold() == right.casefold():
        return None
    return ComparisonSpec((left, right), _dimensions(question))


def _aliases(subject: str) -> tuple[str, ...]:
    aliases = [subject]
    words = subject.split()
    if len(words) > 1:
        aliases.extend((words[-1], " ".join(words[-2:])))
    if subject.lower().startswith("kubernetes "):
        aliases.append(subject[len("kubernetes ") :])
    return tuple(dict.fromkeys(item.casefold() for item in aliases if len(item) >= 3))


def _sections(report: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(report))
    if not matches:
        return [("全文", report)]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        sections.append((match.group("title").strip(), report[match.end() : end]))
    return sections


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.casefold())


def _best_section(dimension: str, sections: list[tuple[str, str]]) -> tuple[str, str] | None:
    wanted = _normalize(dimension)
    exact = [section for section in sections if wanted and wanted in _normalize(section[0])]
    if exact:
        return exact[0]
    tokens = [token for token in re.split(r"\s+|的", wanted) if len(token) >= 2]
    scored = [
        (sum(token in _normalize(title) for token in tokens), title, body)
        for title, body in sections
    ]
    score, title, body = max(scored, default=(0, "", ""))
    return (title, body) if score > 0 else None


def build_comparison_coverage(question: str, report: str) -> ComparisonCoverage:
    spec = parse_comparison_spec(question)
    if spec is None:
        return ComparisonCoverage(None)
    sections = _sections(report)
    cells: list[ComparisonCoverageCell] = []
    for dimension in spec.dimensions:
        matched = _best_section(dimension, sections)
        for subject in spec.subjects:
            if matched is None:
                cells.append(
                    ComparisonCoverageCell(subject, dimension, False, "", "section_missing")
                )
                continue
            title, body = matched
            aliases = _aliases(subject)
            relevant_lines = [
                line.strip()
                for line in body.splitlines()
                if any(alias in line.casefold() for alias in aliases)
            ]
            supported = [
                line
                for line in relevant_lines
                if _CITATION.search(line) and not _GAP_LANGUAGE.search(line)
            ]
            covered = bool(supported)
            reason = (
                "direct_citation"
                if covered
                else (
                    "explicit_evidence_gap"
                    if any(_GAP_LANGUAGE.search(line) for line in relevant_lines)
                    else "no_subject_citation"
                )
            )
            cells.append(ComparisonCoverageCell(subject, dimension, covered, title, reason))
    return ComparisonCoverage(spec, tuple(cells))


def enforce_comparison_coverage(
    review: ReviewResult,
    coverage: ComparisonCoverage,
) -> ReviewResult:
    """Add stable completeness issues without misclassifying gaps as factual errors."""

    review.metrics["comparison_evidence_coverage"] = coverage.coverage_rate
    review.metrics["comparison_missing_cell_count"] = float(len(coverage.missing))
    if not coverage.applicable or coverage.passed:
        return review
    existing_ids = {issue.issue_id for issue in review.structured_issues}
    missing_by_subject: dict[str, list[str]] = {}
    for cell in coverage.missing:
        missing_by_subject.setdefault(cell.subject, []).append(cell.dimension)
    for index, (subject, dimensions) in enumerate(missing_by_subject.items(), start=1):
        issue_id = f"rule-comparison-coverage-{index}"
        if issue_id in existing_ids:
            continue
        unique_dimensions = list(dict.fromkeys(dimensions))
        description = (
            f"比较题的 {subject} 侧缺少直接一手证据：{'、'.join(unique_dimensions)}。"
            "这是研究目标覆盖缺口，不是事实错误。"
        )
        query = f"{subject} 官方文档：{'、'.join(unique_dimensions)}；原始规范 直接证据"
        issue = ReviewIssue(
            issue_id=issue_id,
            dimension="completeness",
            description=description,
            severity=1,
            action=RepairAction.ADD,
            target=f"{subject} 对比维度",
            repair_query=query,
            category="evidence_gap",
            fingerprint=f"comparison:{subject.casefold()}",
        )
        review.structured_issues.append(issue)
        review.issues.append(description)
        review.missing_information.append(description)
        review.repair_queries.append(query)
    review.repair_queries = list(dict.fromkeys(review.repair_queries))
    review.passed = False
    return review
