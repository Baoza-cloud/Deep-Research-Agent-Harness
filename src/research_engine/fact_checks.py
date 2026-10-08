"""High-precision deterministic checks for critical Claim qualifiers."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Sequence

from .schemas import SupportVerdict


_COUNT_WORDS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_COUNT_FACT = re.compile(
    r"(?P<count>\d+|[零一二两三四五六七八九十])\s*"
    r"(?P<classifier>个|种|类|项)\s*"
    r"(?P<label>[^，。；:：!?！？\n]{0,48})"
)
_ENUMERATION = re.compile(
    r"(?P<label>(?:API\s*)?(?:类别|类型|资源|特性|步骤|阶段|角色|渠道|原则|接口|对象))"
    r"[^（()\n]{0,16}[（(](?P<items>[^）)\n]{3,240})[）)]",
    re.IGNORECASE,
)
_COUNT_TOPICS = (
    "类别",
    "类型",
    "资源",
    "特性",
    "步骤",
    "阶段",
    "角色",
    "渠道",
    "原则",
    "接口",
    "对象",
)
_YEAR = re.compile(r"(?<![\w.])(?:19|20)\d{2}(?![\w.])")
_PERCENT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*%")
_ORG = re.compile(
    r"(?:[A-Z][A-Za-z0-9&_.-]*(?:\s+[A-Z][A-Za-z0-9&_.-]*){0,5}|"
    r"[\u3400-\u9fffA-Za-z0-9]{1,30}?)"
    r"(?:公司|大学|委员会|基金会|协会|实验室|研究院|研究所)"
)
_CITATION = re.compile(r"\[[A-Za-z0-9_-]+-E\d+\]")
_MARKDOWN = re.compile(r"[`*_#]+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*|[\u3400-\u9fff]")


@dataclass(frozen=True)
class CountFact:
    value: int
    topic: str
    surface: str
    subject: str = ""


@dataclass(frozen=True)
class DeterministicFactCheck:
    """A conservative override for a semantic or lexical pair judgment."""

    verdict: SupportVerdict | None = None
    conflict_types: tuple[str, ...] = ()
    supported_aspects: tuple[str, ...] = ()
    unsupported_aspects: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKDOWN.sub("", _CITATION.sub("", text))).strip()


def _tokens(text: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(_clean(text))}


def _count_value(raw: str) -> int | None:
    return int(raw) if raw.isdigit() else _COUNT_WORDS.get(raw)


def _count_topic(classifier: str, label: str) -> str:
    compact = re.sub(r"\s+", "", label).casefold()
    return next((topic for topic in _COUNT_TOPICS if topic in compact), classifier)


def _count_subject(text: str, start: int) -> str:
    prefix = re.split(r"[。！？!?；;，,\n]", text[:start])[-1].strip()
    previous = None
    while prefix and prefix != previous:
        previous = prefix
        prefix = re.sub(
            r"(?:具有|包含|包括|提供|支持|共有|共|分为|可分为|存在|有|稳定的|的)\s*$",
            "",
            prefix,
        ).strip()
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", prefix.casefold())


def extract_count_facts(text: str) -> tuple[CountFact, ...]:
    cleaned = _clean(text)
    facts: list[CountFact] = []
    for match in _COUNT_FACT.finditer(cleaned):
        value = _count_value(match.group("count"))
        if value is None:
            continue
        facts.append(
            CountFact(
                value=value,
                topic=_count_topic(match.group("classifier"), match.group("label")),
                surface=match.group(0).strip(),
                subject=_count_subject(cleaned, match.start()),
            )
        )
    for match in _ENUMERATION.finditer(cleaned):
        items = [
            item.strip() for item in re.split(r"[、,，/]", match.group("items")) if item.strip()
        ]
        if len(items) < 2:
            continue
        facts.append(
            CountFact(
                value=len(items),
                topic=_count_topic("类", match.group("label")),
                surface=match.group(0).strip(),
                subject=_count_subject(cleaned, match.start()),
            )
        )
    return tuple(facts)


def _count_context(text: str) -> str:
    value = _ENUMERATION.sub(lambda match: match.group("label"), _clean(text))
    value = _COUNT_FACT.sub(
        lambda match: f"{match.group('classifier')}{match.group('label')}",
        value,
    )
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.casefold())


def find_report_count_conflicts(
    claims: Sequence[tuple[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Find high-similarity Claims that assign different counts to one topic."""

    entries: list[tuple[str, CountFact, str]] = []
    for claim_id, text in claims:
        context = _count_context(text)
        for fact in extract_count_facts(text):
            # Bare ``N 个`` occurs frequently in unrelated prose. Named topics
            # such as roles/resources remain eligible because _count_topic maps them.
            if fact.topic == "个":
                continue
            entries.append((claim_id, fact, context))

    conflicts: dict[str, list[str]] = {}
    generic_named_tokens = {"api", "data", "model", "service", "system"}
    for index, (left_id, left_fact, left_context) in enumerate(entries):
        for right_id, right_fact, right_context in entries[index + 1 :]:
            if left_id == right_id or left_fact.topic != right_fact.topic:
                continue
            if left_fact.value == right_fact.value:
                continue
            same_subject = bool(
                left_fact.subject
                and right_fact.subject
                and (
                    left_fact.subject == right_fact.subject
                    or (
                        min(len(left_fact.subject), len(right_fact.subject)) >= 4
                        and (
                            left_fact.subject in right_fact.subject
                            or right_fact.subject in left_fact.subject
                        )
                    )
                )
            )
            similarity = SequenceMatcher(None, left_context, right_context).ratio()
            shared_named_tokens = {
                token
                for token in _tokens(left_context) & _tokens(right_context)
                if len(token) >= 3 and not token.isdigit() and token not in generic_named_tokens
            }
            if not same_subject and (similarity < 0.82 or not shared_named_tokens):
                continue
            reason = f"报告内部数量不一致：{left_fact.topic}={left_fact.value}/{right_fact.value}"
            conflicts.setdefault(left_id, []).append(reason)
            conflicts.setdefault(right_id, []).append(reason)
    return {claim_id: tuple(dict.fromkeys(reasons)) for claim_id, reasons in conflicts.items()}


def _relevant_evidence_years(claim: str, evidence: str) -> set[str]:
    claim_tokens = _tokens(_YEAR.sub("", claim))
    candidates: set[str] = set()
    for sentence in re.split(r"[。！？!?\n]+", _clean(evidence)):
        years = set(_YEAR.findall(sentence))
        if not years:
            continue
        overlap = claim_tokens & _tokens(_YEAR.sub("", sentence))
        if len(overlap) >= 2:
            candidates.update(years)
    return candidates


def _organizations(text: str) -> set[str]:
    generic = {"该公司", "本公司", "一家公司"}
    organizations = {re.sub(r"^(?:由|与|和|及|而|在)", "", item) for item in _ORG.findall(text)}
    return {item for item in organizations if item and item not in generic}


def _relevant_evidence_orgs(claim: str, evidence: str) -> set[str]:
    claim_tokens = _tokens(_ORG.sub("", claim))
    candidates: set[str] = set()
    for sentence in re.split(r"[。！？!?\n]+", _clean(evidence)):
        organizations = _organizations(sentence)
        if not organizations:
            continue
        overlap = claim_tokens & _tokens(_ORG.sub("", sentence))
        if len(overlap) >= 2:
            candidates.update(organizations)
    return candidates


def _relevant_evidence_percents(claim: str, evidence: str) -> set[str]:
    claim_tokens = _tokens(_PERCENT.sub("", claim))
    candidates: set[str] = set()
    for sentence in re.split(r"[。！？!?\n]+", _clean(evidence)):
        percentages = {re.sub(r"\s+", "", item) for item in _PERCENT.findall(sentence)}
        if not percentages:
            continue
        overlap = claim_tokens & _tokens(_PERCENT.sub("", sentence))
        if len(overlap) >= 2:
            candidates.update(percentages)
    return candidates


def check_critical_qualifiers(claim: str, evidence: str) -> DeterministicFactCheck:
    """Check count, numeric, temporal and organization qualifiers.

    The checker only overrides a pair when it can identify an explicit critical
    qualifier. It intentionally does not attempt open-domain entity extraction.
    """

    claim_clean = _clean(claim)
    evidence_clean = _clean(evidence)
    conflicts: set[str] = set()
    unsupported: list[str] = []
    supported: list[str] = []
    reasons: list[str] = []
    hard_conflict = False

    claim_counts = extract_count_facts(claim_clean)
    evidence_counts = extract_count_facts(evidence_clean)
    for claim_fact in claim_counts:
        if claim_fact.topic == "个":
            continue
        matching = [fact for fact in evidence_counts if fact.topic == claim_fact.topic]
        values = {fact.value for fact in matching}
        if not matching:
            unsupported.append(f"证据未覆盖数量限定：{claim_fact.surface}")
            reasons.append(f"count_missing:{claim_fact.topic}={claim_fact.value}")
            continue
        if len(values) > 1:
            conflicts.add("number")
            unsupported.append(
                f"证据内部对{claim_fact.topic}给出多个数量：{','.join(map(str, sorted(values)))}"
            )
            reasons.append(f"evidence_count_inconsistent:{claim_fact.topic}")
            continue
        evidence_value = next(iter(values))
        if evidence_value != claim_fact.value:
            hard_conflict = True
            conflicts.add("number")
            unsupported.append(
                f"数量冲突：Claim={claim_fact.value}，Evidence={evidence_value}（{claim_fact.topic}）"
            )
            reasons.append(
                f"count_conflict:{claim_fact.topic}:{claim_fact.value}!={evidence_value}"
            )
        else:
            supported.append(f"数量限定得到支持：{claim_fact.surface}")

    claim_percents = {re.sub(r"\s+", "", item) for item in _PERCENT.findall(claim_clean)}
    evidence_percents = {re.sub(r"\s+", "", item) for item in _PERCENT.findall(evidence_clean)}
    missing_percents = claim_percents - evidence_percents
    if missing_percents:
        conflicts.add("number")
        unsupported.append("证据未支持百分比：" + "、".join(sorted(missing_percents)))
        relevant_percents = _relevant_evidence_percents(claim_clean, evidence_clean)
        if relevant_percents:
            hard_conflict = True
            reasons.append(
                "percentage_conflict:"
                + ",".join(sorted(claim_percents))
                + "!="
                + ",".join(sorted(relevant_percents))
            )
        else:
            reasons.append("percentage_missing")

    claim_years = set(_YEAR.findall(claim_clean))
    if claim_years:
        evidence_years = set(_YEAR.findall(evidence_clean))
        missing_years = claim_years - evidence_years
        if missing_years:
            conflicts.add("time")
            relevant_years = _relevant_evidence_years(claim_clean, evidence_clean)
            unsupported.append("证据未支持时间限定：" + "、".join(sorted(missing_years)))
            if relevant_years:
                hard_conflict = True
                reasons.append(
                    "year_conflict:"
                    + ",".join(sorted(claim_years))
                    + "!="
                    + ",".join(sorted(relevant_years))
                )
            else:
                reasons.append("year_missing")
        else:
            supported.append("时间限定得到支持：" + "、".join(sorted(claim_years)))

    claim_orgs = _organizations(claim_clean)
    if claim_orgs:
        evidence_orgs = _organizations(evidence_clean)
        missing_orgs = claim_orgs - evidence_orgs
        if missing_orgs:
            conflicts.add("entity")
            unsupported.append("证据未支持实体：" + "、".join(sorted(missing_orgs)))
            relevant_orgs = _relevant_evidence_orgs(claim_clean, evidence_clean)
            if relevant_orgs:
                hard_conflict = True
                reasons.append(
                    "entity_conflict:"
                    + ",".join(sorted(claim_orgs))
                    + "!="
                    + ",".join(sorted(relevant_orgs))
                )
            else:
                reasons.append("entity_missing")

    if hard_conflict:
        verdict = SupportVerdict.CONTRADICTED
    elif unsupported:
        verdict = SupportVerdict.PARTIALLY_SUPPORTED
    else:
        verdict = None
    return DeterministicFactCheck(
        verdict=verdict,
        conflict_types=tuple(sorted(conflicts)),
        supported_aspects=tuple(dict.fromkeys(supported)),
        unsupported_aspects=tuple(dict.fromkeys(unsupported)),
        reasons=tuple(dict.fromkeys(reasons)),
    )
