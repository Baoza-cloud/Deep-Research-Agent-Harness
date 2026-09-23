"""Live Tavily retrieval evaluation with timestamped, non-destructive results."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from research_engine.web_backends import TavilySearchBackend
from research_engine.env import load_env_files


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _domain_matches(host: str, expected: str) -> bool:
    expected = expected.lower().strip(".")
    return host == expected or host.endswith("." + expected)


def evaluate_sample(
    sample: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    latency_seconds: float,
) -> dict[str, Any]:
    hosts = sorted({_host(str(row.get("url", ""))) for row in rows if row.get("url")})
    searchable = " ".join(
        f"{row.get('title', '')} {row.get('content') or row.get('text') or ''}" for row in rows
    ).lower()
    required_domains = [str(item) for item in sample.get("required_domains", [])]
    required_terms = [str(item) for item in sample.get("required_terms", [])]
    domain_hits = {
        domain: any(_domain_matches(host, domain) for host in hosts) for domain in required_domains
    }
    term_hits = {term: term.lower() in searchable for term in required_terms}
    domain_recall = sum(domain_hits.values()) / len(domain_hits) if domain_hits else 1.0
    term_recall = sum(term_hits.values()) / len(term_hits) if term_hits else 1.0
    minimum_met = len(rows) >= int(sample.get("min_results", 1))
    return {
        "id": sample["id"],
        "question": sample["question"],
        "as_of_date": sample.get("as_of_date"),
        "result_count": len(rows),
        "hosts": hosts,
        "domain_hits": domain_hits,
        "term_hits": term_hits,
        "domain_recall": domain_recall,
        "term_recall": term_recall,
        "minimum_results_met": minimum_met,
        "latency_seconds": latency_seconds,
        "passed": minimum_met and domain_recall == 1.0 and term_recall >= 0.5,
        "results": list(rows),
    }


async def run(args: argparse.Namespace) -> Path:
    load_env_files((BASE_DIR / ".env", EVALUATION_DIR / ".env"))
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is not configured in this terminal. "
            "Export it before running the live evaluation."
        )
    dataset_payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if isinstance(dataset_payload, dict):
        if dataset_payload.get("track") != "live":
            raise ValueError("Live evaluator requires a dataset with track='live'")
        samples = dataset_payload.get("samples", [])
        dataset_name = str(dataset_payload.get("name", "ResearchBench-Live"))
        dataset_version = str(dataset_payload.get("version", "unknown"))
    elif isinstance(dataset_payload, list):
        samples = dataset_payload
        dataset_name = "legacy-web-eval"
        dataset_version = "legacy"
    else:
        raise TypeError("Live dataset must be an object or list")
    results = []
    for sample in samples:
        backend = TavilySearchBackend(
            search_depth=args.search_depth,
            include_domains=(
                sample.get("required_domains", []) if args.enforce_required_domains else None
            ),
        )
        started = time.perf_counter()
        try:
            rows = await backend.search(sample["question"], limit=args.limit)
            evaluation = evaluate_sample(sample, rows, time.perf_counter() - started)
        except Exception as exc:
            evaluation = {
                "id": sample["id"],
                "question": sample["question"],
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": time.perf_counter() - started,
                "results": [],
            }
        results.append(evaluation)

    latencies = [float(item["latency_seconds"]) for item in results]
    passed = sum(bool(item["passed"]) for item in results)
    payload = {
        "run_id": datetime.now(timezone.utc).strftime("tavily-%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "tavily",
        "track": "live",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset": str(Path(args.dataset).resolve()),
        "search_depth": args.search_depth,
        "enforce_required_domains": args.enforce_required_domains,
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['run_id']}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("run_id", "total", "passed", "pass_rate", "mean_latency_seconds")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"result_file={output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate live Tavily Web retrieval")
    parser.add_argument(
        "--dataset",
        default=str(EVALUATION_DIR / "datasets" / "researchbench_live_v1.0.json"),
    )
    parser.add_argument("--output-dir", default=str(EVALUATION_DIR / "results"))
    parser.add_argument("--search-depth", choices=("basic", "advanced"), default="advanced")
    parser.add_argument(
        "--enforce-required-domains",
        action="store_true",
        help="Restrict each query to the benchmark's required primary-source domains",
    )
    parser.add_argument("--limit", type=int, default=5)
    try:
        asyncio.run(run(parser.parse_args()))
    except (RuntimeError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
