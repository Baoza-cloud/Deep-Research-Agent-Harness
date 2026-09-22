"""Defensive LLM JSON parsing with three progressively looser fallbacks."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


class StructuredOutputError(ValueError):
    pass


def _balanced_payload(text: str) -> str | None:
    starts = [(text.find("{"), "{", "}"), (text.find("["), "[", "]")]
    starts = [item for item in starts if item[0] >= 0]
    if not starts:
        return None
    start, opening, closing = min(starts, key=lambda item: item[0])
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_payload(text: str) -> Any:
    """Parse direct JSON, fenced JSON, then a balanced Python/JSON literal."""

    text = text.strip()
    errors: list[str] = []

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        errors.append(f"direct: {exc}")

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError as exc:
            errors.append(f"fenced: {exc}")

    candidate = _balanced_payload(text)
    if candidate:
        for parser_name, parser in (("balanced-json", json.loads), ("literal", ast.literal_eval)):
            try:
                value = parser(candidate)
                if isinstance(value, (dict, list)):
                    return value
            except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                errors.append(f"{parser_name}: {exc}")

    raise StructuredOutputError("Unable to parse structured output; " + " | ".join(errors))
