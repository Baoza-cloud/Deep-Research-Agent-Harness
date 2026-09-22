"""Security boundaries for treating retrieved content as untrusted data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?i)(?:ignore|disregard|override).{0,50}"
            r"(?:previous|system|developer).{0,40}(?:instruction|prompt)s?"
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"(?i)(?:reveal|print|return|leak|expose).{0,50}"
            r"(?:api[-_ ]?keys?|secret|token|system prompt)"
        ),
    ),
    (
        "chinese_instruction_override",
        re.compile(r"忽略.{0,25}(?:之前|以上|系统|开发者).{0,25}(?:指令|提示)"),
    ),
    (
        "chinese_secret_exfiltration",
        re.compile(r"(?:输出|泄露|显示|返回).{0,25}(?:密钥|API.?Key|令牌|系统提示)"),
    ),
    (
        "tool_execution_request",
        re.compile(r"(?:执行|运行|调用).{0,20}(?:系统命令|shell|终端命令|工具调用)"),
    ),
)


@dataclass
class SecurityScanResult:
    content: str
    detected: bool = False
    signals: list[str] = field(default_factory=list)
    replacements: int = 0


def sanitize_untrusted_content(content: str) -> SecurityScanResult:
    """Neutralize common instruction/exfiltration payloads without executing them."""

    sanitized = content
    signals: list[str] = []
    replacements = 0
    for name, pattern in INJECTION_PATTERNS:
        sanitized, count = pattern.subn("[UNTRUSTED_INSTRUCTION_REDACTED]", sanitized)
        if count:
            signals.append(name)
            replacements += count
    return SecurityScanResult(
        content=sanitized,
        detected=bool(signals),
        signals=signals,
        replacements=replacements,
    )
