"""Run a zero-key, zero-network CLI smoke test and validate its JSON result."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SECRET_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "MIMO_API_KEY",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "VLLM_API_KEY",
)


def main() -> int:
    env = os.environ.copy()
    for name in SECRET_ENV_VARS:
        env.pop(name, None)

    with tempfile.TemporaryDirectory(prefix="research-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        result_path = temp_path / "result.json"
        command = [
            sys.executable,
            "-m",
            "research_engine",
            "验证 Harness 的规划、检索、合成和引用链路",
            "--offline",
            "--search-provider",
            "fixture",
            "--memory",
            str(temp_path / "memory.sqlite3"),
            "--output",
            str(result_path),
            "--task-timeout",
            "5",
            "--global-timeout",
            "30",
        ]
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stdout, file=sys.stderr)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode or 1

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") not in {
            "completed",
            "completed_with_evidence_gaps",
        }:
            raise RuntimeError(f"unexpected completion status: {payload.get('status')}")
        if len(payload.get("evidences", [])) < 4:
            raise RuntimeError("smoke run did not produce evidence for every DAG branch")
        if payload.get("metrics", {}).get("task_status_counts", {}).get("failed", 0):
            raise RuntimeError("smoke run contains failed tasks")

        print(
            f"offline smoke: PASS status={payload['status']} evidences={len(payload['evidences'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
