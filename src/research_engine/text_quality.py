"""Deterministic cleanup for extraction artifacts in untrusted evidence."""

from __future__ import annotations

import re
from collections import Counter, defaultdict


_BRACKETED_OMISSION = re.compile(r"(?:\[|【)\s*(?:\.{3,}|…+|⋯+|。{3,})\s*(?:\]|】)")


_EVIDENCE_BOUND_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    # In scaled dot-product attention the logits are divided by sqrt(d_k).
    # Calling sqrt(d_k) itself the scaling factor reverses the formula.
    (
        re.compile(
            r"经\s*(?:缩放因子\s*)?(?:√\s*d_k|\\sqrt\s*\{\s*d_k\s*\})\s*"
            r"(?:缩放|调整)后"
        ),
        "除以 √d_k 后",
    ),
    (
        re.compile(r"缩放因子(?:是|为)\s*(?:√\s*d_k|\\sqrt\s*\{\s*d_k\s*\})"),
        "缩放因子为 1/√d_k",
    ),
    # A source that only describes architecture does not establish a causal
    # account of parallelism. Keep the supported architectural observation.
    (
        re.compile(r"架构层面的并行性来源|并行建模序列的架构基础"),
        "架构组成",
    ),
    (
        re.compile(
            r"(?:Transformer\s*)?并行建模序列的(?:直接)?架构依据是"
            r"放弃循环和卷积[、，,]?\s*使用自注意力与前馈网络"
        ),
        "Transformer 可直接确认的架构组成是放弃循环和卷积、使用自注意力与前馈网络",
    ),
    (
        re.compile(
            r"Transformer\s*(?:能够)?(?:进行)?(?:并行)?序列建模的"
            r"(?:直接)?架构基础(?:在于|是)(?:其)?\s*"
            r"放弃循环和卷积[、，,]?\s*采用自注意力与前馈网络"
        ),
        "Transformer 可直接确认的架构组成是放弃循环和卷积、采用自注意力与前馈网络",
    ),
    # Field presence, an observed idle-consumer condition, and a possible
    # attack path must not be strengthened into capabilities or hard bounds.
    (
        re.compile(r"(?:这些字段)?可用于围绕这些维度组织学术产出信息"),
        "这些字段描述相应的学术产出维度",
    ),
    (
        re.compile(r"\*\*扩容上限\*\*(?=\s*[：:])"),
        "**分区数与空闲消费者**",
    ),
    (
        re.compile(r"分区与消费者的独占映射关系"),
        "分区分配的单向独占约束",
    ),
    (
        re.compile(
            r"(?:同一\s*consumer group\s*内)?分区与消费者实例是"
            r"一对一的独占分配关系"
        ),
        "同一 consumer group 内，一个主题分区在任一时刻只分配给一个消费者实例",
    ),
    (
        re.compile(
            r"消费者扩容的(?:有效)?上限由分区数决定[，,]\s*"
            r"消费者数超过分区数时会出现空闲实例"
        ),
        "消费者数超过分区数时会出现空闲实例；现有证据不支持将分区数外推为唯一扩容上限",
    ),
    (
        re.compile(r"这是扩容的(?:直接)?工程上限表现"),
        "这仅说明超出分区数的消费者会处于空闲状态",
    ),
    (
        re.compile(
            r"单组内有效并行消费的上限与可分配分区数直接相关[，,]\s*"
            r"超出部分不会带来该组内的并行消费增益"
        ),
        "消费者数超过分区数时会出现空闲实例；现有证据不支持将其外推为唯一或硬性工程上限",
    ),
    (
        re.compile(r"扩容消费者的(?:直接|有效|工程)?上限与(?:主题)?分区(?:数量|数)相关"),
        "消费者数超过分区数时会出现空闲实例；现有证据不支持将分区数表述为唯一或硬性扩容上限",
    ),
    (
        re.compile(
            r"\*\*(\d+[.)、]\s*)(?:多头并行机制|"
            r"多头注意力在不同表示子空间中并行计算[。.]?)\*\*"
        ),
        r"**\1多头注意力的已证实事实（不等同于序列并行机制）**",
    ),
    (
        re.compile(
            r"(?m)^(#{1,6}\s+\d+[.)、]\s*)"
            r"多头注意力[：:]不同表示子空间的并行计算\s*$"
        ),
        r"\1多头注意力的已证实事实（不等同于序列并行机制）",
    ),
    (
        re.compile(r"\*\*(\d+[.)、]\s*)关于[“\"]并行建模序列[”\"]的直接证据[。.]?\*\*"),
        r"**\1当前证据的覆盖边界**",
    ),
    (
        re.compile(r"不是抽象风险，而是攻击者可能利用的注入路径"),
        "是需要防御的潜在注入风险",
    ),
    # Preserve the operative GDPR wording without upgrading it into a novel
    # analytical framework or a two-element legal test.
    (
        re.compile(
            r"该要求从两个维度约束数据处理[：:]\s*"
            r"一是相关性[，,]\s*二是必要性范围"
        ),
        "该要求同时包含与目的相关、限于实现目的所必需范围两项表述",
    ),
    (
        re.compile(r"这意味着合规不仅要求实质遵守[，,]还要求控制者具备证明合规的能力[。.]?"),
        "",
    ),
)


def remove_omission_markers(text: str) -> tuple[str, int]:
    """Remove standalone extraction placeholders without joining fragments.

    Search/extraction providers commonly insert ``[...]`` or ``[…]`` between
    non-contiguous passages.  A paragraph boundary is safer than an empty
    replacement because it does not imply that the surrounding clauses were
    adjacent in the source.
    """

    cleaned, count = _BRACKETED_OMISSION.subn("\n\n", text)
    if not count:
        return text, 0
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), count


def normalize_evidence_bound_language(text: str) -> tuple[str, int]:
    """Canonicalize known high-risk wording without adding new facts.

    These substitutions preserve the directly evidenced observation while
    removing causal, capability, or certainty upgrades that a model can add
    during synthesis. They also canonicalize a commonly reversed formula
    explanation for scaled dot-product attention.
    """

    normalized = text
    changes = 0
    for pattern, replacement in _EVIDENCE_BOUND_REWRITES:
        normalized, count = pattern.subn(replacement, normalized)
        changes += count
    return normalized, changes


_CITATION_JOIN = re.compile(r"(\[[A-Za-z0-9_-]+-E\d+\])(?=[A-Za-z0-9\u3400-\u9fff])")
_CITATION_TOKEN = re.compile(r"\[[A-Za-z0-9_-]+-E\d+\]")
_COUNT_WORDS = {
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
_EXPLICIT_COUNT = re.compile(
    r"(?P<count>\d+|[一二两三四五六七八九十])\s*(?:个|种|类|项)\s*"
    r"(?P<label>[^，。；：:（）()\n]{2,32})"
)
_PAREN_LIST = re.compile(
    r"(?P<label>[^，。；：:（）()\n]{4,48})[（(](?P<items>[^（）()\n]{3,160})[）)]"
)
_COUNT_TERMS = ("类别", "类型", "资源", "特性", "步骤", "阶段", "角色", "渠道", "原则")


def normalize_patch_boundaries(text: str) -> str:
    """Repair deterministic formatting joins created at patch boundaries."""

    normalized = _CITATION_JOIN.sub(r"\1 ", text)
    normalized = re.sub(r"[ \t]+([，。；：！？,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    return normalized


def _count_value(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    return _COUNT_WORDS.get(raw)


def _count_topic(label: str) -> str:
    compact = re.sub(r"\s+", "", label)
    term = next((item for item in _COUNT_TERMS if item in compact), "")
    acronyms = re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", label)
    if term:
        return f"{acronyms[-1] + ':' if acronyms else ''}{term}"
    return compact[-12:]


def report_integrity_issues(report: str) -> set[str]:
    """Return stable issue codes for duplicate, join, and enumeration defects."""

    issues: set[str] = set()
    if _CITATION_JOIN.search(report):
        issues.add("citation_boundary_join")

    clauses: list[str] = []
    for raw in re.split(r"[。！？!?；;\n]+", report):
        semantic = _CITATION_TOKEN.sub("", raw)
        semantic = re.sub(r"^[#>*\-+\d.)、\s]+", "", semantic)
        semantic = re.sub(r"\s+", "", semantic).strip("，,:：。；;*_")
        if len(semantic) < 12 or "证据缺口" in semantic or "基于当前证据" in semantic:
            continue
        clauses.append(semantic.casefold())
    for clause, count in Counter(clauses).items():
        if count > 1:
            issues.add(f"duplicate_claim:{clause[:24]}")

    counts: dict[str, set[int]] = defaultdict(set)
    for match in _EXPLICIT_COUNT.finditer(report):
        value = _count_value(match.group("count"))
        if value is not None:
            counts[_count_topic(match.group("label"))].add(value)
    for match in _PAREN_LIST.finditer(report):
        items = [item.strip() for item in re.split(r"[、,，/]", match.group("items"))]
        items = [item for item in items if item]
        if len(items) >= 2:
            counts[_count_topic(match.group("label"))].add(len(items))
    for topic, values in counts.items():
        if len(values) > 1:
            issues.add(f"enumeration_conflict:{topic}:{','.join(map(str, sorted(values)))}")
    return issues
