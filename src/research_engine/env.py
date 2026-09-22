"""Minimal local .env loader with no third-party dependency."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_env_files(paths: Iterable[str | Path], *, override: bool = False) -> list[Path]:
    loaded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if name and (override or name not in os.environ):
                os.environ[name] = value
        loaded.append(path)
    return loaded
