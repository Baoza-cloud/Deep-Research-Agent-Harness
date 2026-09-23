"""Run controlled Red/Blue fault-injection experiments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from research_engine import (  # noqa: E402
    BlueTeamRepairer,
    ClaimEvidenceVerifier,
    RedTeamReviewer,
    build_llm,
    load_env_files,
)
from research_engine.planner import call_llm  # noqa: E402

from adversarial_evaluation import (  # noqa: E402
    AdversarialCase,
    AdversarialMetrics,
    aggregate_adversarial_metrics,
    evidence_objects,
    evaluate_adversarial_report,
    evaluate_rollback_guard,
    inject_faults,
    serialize_case,
)
from research_evaluation import bootstrap_ci  # noqa: E402
from run_ablation_experiments import load_frozen_dataset  # noqa: E402


VARIANTS = (
    "corrupted_no_repair",
    "rewrite_blue",
    "structured_patch_blue",
    "oracle_clean",
)


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


async def _rewrite_report(
    llm: Any,
    case: AdversarialCase,
    issues: Sequence[Any],
) -> str:
    evidence = case.sample["evidence"]
    prompt = f"""
你是 Blue Agent。请根据冻结证据和 Red issues 重写整篇报告。
删除无证据结论和未知引用，补齐合法引用与局限说明。不得编造证据 ID。
只输出修复后的 Markdown 报告，不要解释过程。
问题：{case.sample["question"]}
合法证据：{json.dumps(evidence, ensure_ascii=False)}
Red issues：{json.dumps(_normalize([asdict(item) for item in issues]), ensure_ascii=False)}
待修复报告：
{case.corrupted_report}
""".strip()
    return await call_llm(llm, prompt)


async def _run_case(
    variant: str,
    case: AdversarialCase,
    llm: Any | None,
    reviewer: RedTeamReviewer,
    pre_review: Any,
) -> dict[str, Any]:
    evidences = evidence_objects(case)
    patch_result = None
    if variant == "corrupted_no_repair":
        final_report = case.corrupted_report
    elif variant == "oracle_clean":
        final_report = case.clean_report
    elif variant == "rewrite_blue":
        if llm is None:
            final_report = case.corrupted_report
        else:
            final_report = await _rewrite_report(llm, case, pre_review.structured_issues)
    elif variant == "structured_patch_blue":
        repairer = BlueTeamRepairer(
            llm=llm,
            max_generation_attempts=2,
            verifier=ClaimEvidenceVerifier(llm),
        )
        patch_result = await repairer.repair(
            str(case.sample["question"]),
            case.corrupted_report,
            evidences,
            pre_review,
        )
        final_report = patch_result.report
    else:
        raise ValueError(f"Unknown adversarial variant: {variant}")

    post_review = (
        pre_review
        if variant == "corrupted_no_repair"
        else await reviewer.review(str(case.sample["question"]), final_report, evidences)
    )
    metrics = evaluate_adversarial_report(
        case,
        final_report,
        review=pre_review,
        patch_result=patch_result,
    )
    return {
        "case_id": case.case_id,
        "sample_id": case.sample["id"],
        "domain": case.sample["domain"],
        "metrics": asdict(metrics),
        "pre_review": _normalize(asdict(pre_review)),
        "post_review": _normalize(asdict(post_review)),
        "report": final_report,
        "patch_result": _normalize(asdict(patch_result)) if patch_result else None,
    }


def _aggregate(rows: Sequence[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    successful = [row for row in rows if "error" not in row]
    metrics = [AdversarialMetrics(**row["metrics"]) for row in successful]
    aggregate = aggregate_adversarial_metrics(metrics) if metrics else {}
    intervals: dict[str, list[float]] = {}
    for name in (
        "fault_repair_rate",
        "red_detection_recall",
        "clean_fact_retention_rate",
        "collateral_damage_rate",
        "rule_factual_accuracy",
        "rule_hallucination_rate",
        "citation_validity",
        "citation_coverage",
        "citation_preservation_rate",
        "patch_acceptance_rate",
    ):
        values = [float(getattr(item, name)) for item in metrics]
        if values:
            grouped: dict[str, list[float]] = {}
            for row, value in zip(successful, values):
                grouped.setdefault(str(row["case_id"]), []).append(value)
            question_means = [statistics.fmean(items) for items in grouped.values()]
            intervals[name] = list(bootstrap_ci(question_means, samples=bootstrap_samples))
    return {
        "total": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "mean": aggregate,
        "bootstrap_95_ci": intervals,
    }


async def run(args: argparse.Namespace) -> Path:
    load_env_files((BASE_DIR / ".env", EVALUATION_DIR / ".env"))
    dataset_path = Path(args.dataset)
    dataset = load_frozen_dataset(dataset_path)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    protocol_path = EVALUATION_DIR / "datasets" / "researchbench_adversarial_v0.1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("parent_dataset_sha256") != dataset_sha256:
        raise ValueError("Adversarial protocol parent dataset SHA-256 mismatch")
    samples = list(dataset["samples"])
    if args.sample_id:
        requested_ids = set(args.sample_id)
        available_ids = {str(item["id"]) for item in samples}
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            raise ValueError("Unknown --sample-id: " + ", ".join(missing_ids))
        samples = [item for item in samples if str(item["id"]) in requested_ids]
    cases = [inject_faults(item) for item in samples[: args.limit or None]]
    if not args.limit and not args.sample_id and len(cases) != int(protocol["expected_case_count"]):
        raise ValueError("Adversarial protocol case count mismatch")
    resume_payload: dict[str, Any] | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        loaded: Any = json.loads(resume_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Resume result must be a JSON object")
        resume_payload = loaded
        checks = {
            "source_dataset_sha256": dataset_sha256,
            "provider": args.provider,
            "model": args.model,
            "repeats": args.repeats,
        }
        for key, expected_value in checks.items():
            if resume_payload.get(key) != expected_value:
                raise ValueError(f"Resume result has incompatible {key}")
    llm = None if args.provider == "offline" else build_llm(args.provider, args.model)
    semaphore = asyncio.Semaphore(args.experiment_concurrency)
    prior_success: dict[str, dict[str, dict[str, Any]]] = {
        variant: {
            str(row["pair_key"]): row
            for row in (
                (resume_payload or {}).get("variants", {}).get(variant, {}).get("samples", [])
            )
            if "error" not in row
        }
        for variant in args.variants
    }
    new_rows: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in args.variants}

    async def execute_bundle(
        repeat: int,
        case: AdversarialCase,
        pending_variants: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        async with semaphore:
            reviewer = RedTeamReviewer(llm=llm, min_citation_coverage=0.80)
            evidences = evidence_objects(case)
            try:
                pre_review = await reviewer.review(
                    str(case.sample["question"]),
                    case.corrupted_report,
                    evidences,
                )
            except Exception as exc:
                return {
                    variant: {
                        "case_id": case.case_id,
                        "sample_id": case.sample["id"],
                        "domain": case.sample["domain"],
                        "repeat": repeat,
                        "pair_key": f"{case.case_id}#r{repeat}",
                        "error": f"pre_review:{type(exc).__name__}: {exc}",
                    }
                    for variant in pending_variants
                }

            bundle: dict[str, dict[str, Any]] = {}
            for variant in pending_variants:
                try:
                    row = await _run_case(variant, case, llm, reviewer, pre_review)
                    row["repeat"] = repeat
                    row["pair_key"] = f"{case.case_id}#r{repeat}"
                    bundle[variant] = row
                except Exception as exc:
                    bundle[variant] = {
                        "case_id": case.case_id,
                        "sample_id": case.sample["id"],
                        "domain": case.sample["domain"],
                        "repeat": repeat,
                        "pair_key": f"{case.case_id}#r{repeat}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            return bundle

    jobs = []
    expected: list[tuple[int, AdversarialCase, str]] = []
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            pair_key = f"{case.case_id}#r{repeat}"
            expected.append((repeat, case, pair_key))
            pending_variants = [
                variant for variant in args.variants if pair_key not in prior_success[variant]
            ]
            if pending_variants:
                jobs.append(execute_bundle(repeat, case, pending_variants))
    bundles = await asyncio.gather(*jobs)
    for bundle in bundles:
        for variant, row in bundle.items():
            new_rows[variant][str(row["pair_key"])] = row

    variant_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in args.variants:
        combined = {**new_rows[variant], **prior_success[variant]}
        variant_rows[variant] = [combined[pair_key] for _, _, pair_key in expected]

    created_at = datetime.now(timezone.utc)
    payload = {
        "run_id": created_at.strftime("adversarial-%Y%m%dT%H%M%S%fZ"),
        "created_at": created_at.isoformat(),
        "benchmark": "ResearchBench-Adversarial",
        "version": "0.1.0",
        "source_dataset": str(dataset_path.resolve()),
        "source_dataset_sha256": dataset_sha256,
        "protocol_manifest": str(protocol_path.resolve()),
        "protocol": protocol,
        "provider": args.provider,
        "model": args.model,
        "fault_types": [
            "unknown_citation",
            "uncited_claim",
            "unsupported_claim",
            "missing_limitation",
        ],
        "case_count": len(cases),
        "repeats": args.repeats,
        "planned_runs_per_variant": len(cases) * args.repeats,
        "resumed_from": (str(Path(args.resume_from).resolve()) if args.resume_from else None),
        "rollback_guard": evaluate_rollback_guard(),
        "cases": [serialize_case(case) for case in cases],
        "variants": {
            variant: {
                "aggregate": _aggregate(rows, args.bootstrap_samples),
                "samples": rows,
            }
            for variant, rows in variant_rows.items()
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{payload['run_id']}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "rollback_guard": payload["rollback_guard"],
                "summary": {
                    name: value["aggregate"] for name, value in payload["variants"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"result_file={output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled fault-injection Red/Blue evaluation"
    )
    parser.add_argument(
        "--dataset",
        default=str(EVALUATION_DIR / "datasets" / "researchbench_frozen_v1.0.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_DIR / "results" / "adversarial"),
    )
    parser.add_argument(
        "--provider",
        choices=("offline", "deepseek", "mimo", "vllm", "openai"),
        default="offline",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--resume-from",
        help="Reuse successful pair_key rows from a compatible earlier result",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Run only a frozen sample ID; repeat to select multiple samples",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--experiment-concurrency", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    try:
        args = parser.parse_args()
        if args.limit < 0:
            raise ValueError("--limit must be >= 0")
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        if args.experiment_concurrency < 1:
            raise ValueError("--experiment-concurrency must be >= 1")
        if args.bootstrap_samples < 1:
            raise ValueError("--bootstrap-samples must be >= 1")
        asyncio.run(run(args))
    except (RuntimeError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
