"""Build a reviewable Web-demo bundle without redistributing raw page bodies."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_catalog(evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep provenance and retrieval signals while omitting fetched page text."""

    fields = (
        "evidence_id",
        "subtask_id",
        "source",
        "title",
        "url",
        "source_type",
        "score",
        "published_at",
        "retrieved_at",
        "metadata",
    )
    return [{field: item.get(field) for field in fields} for item in evidences]


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload["trace"]
    filters = [item for item in trace if item.get("event") == "retrieval_filter_applied"]
    repairs = [item for item in trace if item.get("event") == "blue_repair"]
    alignments = [item for item in trace if item.get("event") == "claim_evidence_alignment"]
    removal_counts: Counter[str] = Counter()
    source_classes: Counter[str] = Counter()
    patch_actions: Counter[str] = Counter()
    for item in filters:
        removal_counts.update(item.get("removal_counts", {}))
        source_classes.update(
            row.get("source_class", "unknown") for row in item.get("selected", [])
        )
    for item in repairs:
        patch_actions.update(
            patch.get("action", "unknown") for patch in item.get("applied_patches", [])
        )

    return {
        "status": payload["status"],
        "duration_seconds": round(
            float(payload["metrics"]["budget"]["elapsed_seconds"]),
            3,
        ),
        "dag_task_count": len(payload["plan"]["subtasks"]),
        "swarm": payload["run_metadata"]["swarm"],
        "retrieval": {
            "batches": len(filters),
            "candidate_count": sum(item["candidate_count"] for item in filters),
            "selected_count": sum(item["selected_count"] for item in filters),
            "filtered_count": sum(item["filtered_count"] for item in filters),
            "removal_reasons": dict(removal_counts),
            "selected_source_classes": dict(source_classes),
        },
        "claim_verification": payload["claim_ledger"]["metrics"],
        "red_blue": {
            "review_rounds": payload["metrics"]["review_rounds"],
            "repair_events": len(repairs),
            "applied_patch_count": sum(len(item.get("applied_patches", [])) for item in repairs),
            "rejected_patch_count": sum(len(item.get("rejected_patches", [])) for item in repairs),
            "patch_actions": dict(patch_actions),
            "rewrite_fallbacks": payload["metrics"]["blue_rewrite_fallbacks"],
        },
        "incremental_verification": {
            "alignment_events": len(alignments),
            "final_reused_unchanged_claim_count": payload["claim_ledger"]["metrics"].get(
                "reused_unchanged_claim_count", 0
            ),
        },
        "quality_guard_triggered": payload["metrics"]["dynamic_quality_fallback_triggered"],
        "quality_guard": payload["metrics"]["dynamic_quality_candidates"],
        "quality_guard_selected_fixed": payload["metrics"]["dynamic_quality_fallback_selected"],
        "prompt_injection_evidence_count": payload["metrics"]["prompt_injection_evidence_count"],
        "completion_issue_reasons": payload["metrics"]["completion_issue_reasons"],
        "event_counts": dict(Counter(item["event"] for item in trace)),
    }


def write_report(payload: dict[str, Any], target: Path) -> None:
    lines = [payload["answer"].rstrip(), "", "## 证据索引", ""]
    for item in payload["evidences"]:
        evidence_id = item["evidence_id"]
        title = str(item.get("title") or item.get("source") or "Untitled").replace("\n", " ")
        url = item.get("url") or item.get("source")
        lines.append(f"- `{evidence_id}` [{title}]({url})")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    payload = json.loads(args.raw_run.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "trace_bundle.json"
    report_path = args.output_dir / "final_report.md"
    summary = build_summary(payload)
    bundle = {
        "schema_version": "web-showcase-v1",
        "source_commit": args.source_commit,
        "raw_run_sha256": sha256(args.raw_run),
        "raw_content_policy": (
            "Fetched page bodies are intentionally omitted; URLs, content hashes, retrieval "
            "scores, the complete execution event trace, Claim ledger, reviews, patches, and "
            "final report are retained."
        ),
        "run": {
            "run_id": payload["run_id"],
            "question": payload["question"],
            "status": payload["status"],
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "run_metadata": payload["run_metadata"],
            "metrics": payload["metrics"],
        },
        "summary": summary,
        "dag": payload["plan"],
        "evidence_catalog": evidence_catalog(payload["evidences"]),
        "claim_ledger": payload["claim_ledger"],
        "final_review": payload["review"],
        "trace": payload["trace"],
        "final_report": payload["answer"],
    }
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(payload, report_path)

    manifest = {
        "source_commit": args.source_commit,
        "raw_run_sha256": sha256(args.raw_run),
        "trace_bundle_sha256": sha256(bundle_path),
        "final_report_sha256": sha256(report_path),
        "summary": summary,
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
