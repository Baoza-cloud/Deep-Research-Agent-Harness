"""Offline integrity and completeness checks for the public Web showcase bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_EVENTS = {
    "plan_created",
    "swarm_configured",
    "role_strategy_applied",
    "retrieval_filter_applied",
    "red_review",
    "claim_evidence_alignment",
    "review_verification",
    "blue_repair",
    "claim_ledger_built",
}


def validate(bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if bundle.get("schema_version") != "web-showcase-v1":
        errors.append("unexpected schema_version")
    if len((bundle.get("dag") or {}).get("subtasks", [])) < 3:
        errors.append("DAG has fewer than three tasks")
    swarm = ((bundle.get("run") or {}).get("run_metadata") or {}).get("swarm") or {}
    if not swarm.get("agents"):
        errors.append("missing swarm role selection")
    trace = bundle.get("trace") or []
    events = {item.get("event") for item in trace}
    missing_events = REQUIRED_EVENTS - events
    if missing_events:
        errors.append(f"missing trace events: {sorted(missing_events)}")
    metrics = (bundle.get("run") or {}).get("metrics") or {}
    if metrics.get("dynamic_quality_fallback_triggered") and (
        "dynamic_swarm_final_quality_guard" not in events
    ):
        errors.append("quality fallback was triggered without a final guard trace")
    if not (bundle.get("claim_ledger") or {}).get("claims"):
        errors.append("missing Claim ledger")
    if not any(item.get("applied_patches") for item in trace if item.get("event") == "blue_repair"):
        errors.append("missing applied Blue patches")
    if not str(bundle.get("final_report") or "").strip():
        errors.append("missing final report")
    if any("content" in item for item in bundle.get("evidence_catalog") or []):
        errors.append("raw evidence content leaked into public catalog")
    if metrics.get("claim_support_rate", 0) < 0.8:
        errors.append("final Claim support rate is below the documented gate")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    errors = validate(payload)
    print(json.dumps({"passed": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
