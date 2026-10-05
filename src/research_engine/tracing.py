"""Structured local tracing primitives with no telemetry dependency."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


TRACE_SCHEMA_VERSION = "1.0"


def new_trace_id() -> str:
    return f"tr-{uuid.uuid4().hex}"


def stable_span_id(trace_id: str, kind: str, key: str) -> str:
    value = f"{trace_id}:{kind}:{key}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_value(value: Any, *, preview_chars: int = 160) -> dict[str, Any]:
    """Return a bounded, secret-safe summary suitable for a Trace event."""

    if value is None:
        return {"type": "none", "count": 0}
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value).strip()
        return {
            "type": "text",
            "characters": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "preview": normalized[:preview_chars],
            "truncated": len(normalized) > preview_chars,
        }
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "keys": sorted(str(key) for key in value)[:32],
            "key_count": len(value),
            "sha256": _digest(value),
        }
    if is_dataclass(value):
        payload = asdict(value)
        summary = summarize_value(payload, preview_chars=preview_chars)
        summary["type"] = type(value).__name__
        for key in ("subtask_id", "question", "kind", "status"):
            if key in payload:
                item = payload[key]
                summary[key] = item.value if isinstance(item, Enum) else item
        if "question" in summary:
            summary["question"] = str(summary["question"])[:preview_chars]
        return summary
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        evidence_ids = [
            str(getattr(item, "evidence_id"))
            for item in items
            if getattr(item, "evidence_id", None)
        ]
        sources = {
            str(getattr(item, "url", None) or getattr(item, "source", ""))
            for item in items
            if getattr(item, "url", None) or getattr(item, "source", None)
        }
        content_chars = sum(len(str(getattr(item, "content", ""))) for item in items)
        return {
            "type": "sequence",
            "count": len(items),
            "evidence_ids": evidence_ids[:32],
            "source_count": len(sources),
            "content_characters": content_chars,
            "sha256": _digest(items),
        }
    return {
        "type": type(value).__name__,
        "preview": str(value)[:preview_chars],
        "sha256": _digest(str(value)),
    }


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = len(re.sub(r"[\s\u3400-\u9fff]", "", text))
    return cjk + math.ceil(non_cjk / 4)


EMPTY_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
    "retry_count": 0,
    "call_count": 0,
}


def usage_snapshot(provider: Any) -> dict[str, Any]:
    getter = getattr(provider, "usage_snapshot", None)
    if not callable(getter):
        return {**EMPTY_USAGE, "measurement": "unavailable"}
    snapshot = dict(getter())
    for key, default in EMPTY_USAGE.items():
        snapshot.setdefault(key, default)
    snapshot.setdefault("measurement", "provider")
    return snapshot


def usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in EMPTY_USAGE:
        value = float(after.get(key, 0)) - float(before.get(key, 0))
        result[key] = round(value, 9) if key == "cost_usd" else int(value)
    result["measurement"] = after.get("measurement", before.get("measurement", "unavailable"))
    return result


def duration_ms(started_at: str, finished_at: str) -> float:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        return round(max(0.0, (finish - start).total_seconds() * 1000), 3)
    except (TypeError, ValueError):
        return 0.0


def _category(event: str) -> str:
    if event.startswith("agent_") or event.startswith("component_"):
        return "agent"
    if event.startswith("task_") or event in {"plan_created", "dynamic_replan"}:
        return "dag"
    if event.startswith("retrieval_") or event == "role_strategy_applied":
        return "retrieval"
    if "review" in event or event.startswith("red_") or event.startswith("blue_"):
        return "quality"
    if "claim" in event or "evidence" in event:
        return "verification"
    if "budget" in event or event in {"swarm_stop", "swarm_configured"}:
        return "control"
    if "fallback" in event or "guardrail" in event:
        return "guardrail"
    if event.startswith("run_") or event.startswith("intermediate_report"):
        return "run"
    return "system"


def _phase(event: str) -> str:
    if event in {"plan_created", "swarm_configured"}:
        return "planning"
    if event.startswith("task_") or event.startswith("agent_") or event.startswith("retrieval_"):
        return "execution"
    if "review" in event or event.startswith("red_"):
        return "review"
    if event.startswith("blue_") or "patch" in event:
        return "repair"
    if "claim" in event or "evidence" in event:
        return "verification"
    if "final" in event or event == "run_finalized":
        return "finalization"
    return "control"


class TraceEventFactory:
    """Add a stable event envelope while retaining legacy top-level payloads."""

    def __init__(self, trace_id: str, initial: Sequence[Mapping[str, Any]] = ()):
        self.trace_id = trace_id
        self.sequence = max((int(item.get("sequence", 0)) for item in initial), default=0)

    def normalize(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if (
            event.get("trace_schema_version") == TRACE_SCHEMA_VERSION
            and event.get("trace_id")
            and event.get("sequence")
        ):
            return dict(event)
        self.sequence += 1
        record = dict(event)
        name = str(record.get("event") or "unknown")
        record.setdefault("at", datetime.now().astimezone().isoformat())
        record.setdefault("trace_schema_version", TRACE_SCHEMA_VERSION)
        record.setdefault("trace_id", self.trace_id)
        record.setdefault("sequence", self.sequence)
        record.setdefault("category", _category(name))
        record.setdefault("phase", _phase(name))
        identity = {
            "trace_id": self.trace_id,
            "sequence": record["sequence"],
            "event": name,
            "at": record["at"],
        }
        record.setdefault("event_id", hashlib.sha256(_digest(identity).encode()).hexdigest()[:24])
        return record


UsageProvider = Callable[[], Mapping[str, Any]]
