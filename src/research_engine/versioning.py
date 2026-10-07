"""Resolve reproducible engine identity without trusting stale local markers."""

from __future__ import annotations

import os
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


_SHA = re.compile(r"^[0-9a-f]{40}$")


def engine_version() -> str:
    override = os.getenv("RESEARCH_ENGINE_VERSION", "").strip()
    if override.casefold() not in {"", "dev", "development", "unknown"}:
        return override
    try:
        resolved = version("deep-research-agent-harness")
    except PackageNotFoundError:
        return "unknown"
    return resolved if resolved.startswith("v") else f"v{resolved}"


def source_commit() -> str | None:
    override = os.getenv("RESEARCH_ENGINE_GIT_SHA", "").strip().lower()
    if _SHA.fullmatch(override):
        return override
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    resolved = completed.stdout.strip().lower()
    return resolved if _SHA.fullmatch(resolved) else None
