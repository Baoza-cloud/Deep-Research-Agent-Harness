"""Validate Dynamic Swarm composition on ResearchBench-Live.

This evaluator intentionally separates policy validation from retrieval quality.
It can use the deterministic planner without network access, or an LLM planner
to measure whether real planning variance changes the selected swarm.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from research_engine import (  # noqa: E402
    AgentSpec,
    HeuristicPlanner,
    HeuristicSwarmPolicy,
    LLMPlanner,
    ResearchConfig,
    ResearchSubtask,
    auto_provider,
    build_llm,
    load_env_files,
)


LEVELS = ("simple", "standard", "complex")
PROBE_TASKS = (
    ResearchSubtask("scope", "定义概念与边界", "scope"),
    ResearchSubtask("facts", "搜集一手事实证据", "facts"),
    ResearchSubtask("counter", "查找反例和限制", "counter"),
    ResearchSubtask("implications", "分析风险、影响与建议", "impact"),
    ResearchSubtask("verify_r1_1", "核验争议事实", "verify"),
)


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("track") != "live":
        raise ValueError("Live Swarm evaluator requires track='live'")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Live dataset must contain a non-empty samples list")
    ids = [str(sample.get("id", "")) for sample in samples]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Live sample IDs must be non-empty and unique")

    distribution: Counter[str] = Counter()
    for sample in samples:
        expected = sample.get("expected_swarm")
        if not isinstance(expected, dict):
            raise ValueError(f"Sample {sample.get('id')} needs expected_swarm")
        level = str(expected.get("level", ""))
        roles = expected.get("roles")
        if level not in LEVELS:
            raise ValueError(f"Sample {sample.get('id')} has invalid level {level!r}")
        if not isinstance(roles, list) or not roles:
            raise ValueError(f"Sample {sample.get('id')} needs expected roles")
        if int(expected.get("agent_count", 0)) != len(roles):
            raise ValueError(f"Sample {sample.get('id')} agent_count must match role count")
        distribution[level] += 1

    declared_count = payload.get("expected_sample_count")
    if declared_count is not None and int(declared_count) != len(samples):
        raise ValueError("Live dataset expected_sample_count does not match samples")
    declared_distribution = payload.get("expected_complexity_distribution")
    if declared_distribution is not None and {
        key: int(value) for key, value in declared_distribution.items()
    } != dict(distribution):
        raise ValueError("Declared complexity distribution does not match samples")
    if set(distribution) != set(LEVELS):
        raise ValueError("Live dataset must cover simple, standard and complex")
    return payload


def _planner(provider: str, model: str | None) -> tuple[Any, str]:
    if provider == "offline":
        return HeuristicPlanner(), "heuristic"
    resolved: Any = auto_provider() if provider == "auto" else provider
    if resolved is None:
        raise RuntimeError("No configured LLM provider")
    return LLMPlanner(build_llm(resolved, model)), "llm"


def _role_names(swarm: Any) -> list[str]:
    return [str(agent.role) for agent in swarm.agents]


async def evaluate_one(
    sample: dict[str, Any],
    repeat: int,
    planner: Any,
    policy: HeuristicSwarmPolicy,
    config: ResearchConfig,
    worker: AgentSpec,
) -> dict[str, Any]:
    started = time.perf_counter()
    plan = await planner.plan(str(sample["question"]))
    swarm = policy.decide(str(sample["question"]), plan, config, worker)
    expected = sample["expected_swarm"]
    actual_roles = _role_names(swarm)
    checks = {
        "level": swarm.level.value == str(expected["level"]),
        "agent_count": len(actual_roles) == int(expected["agent_count"]),
        "roles": actual_roles == [str(item) for item in expected["roles"]],
    }
    routed = [swarm.select_agent(task, worker).role for task in plan.subtasks]
    probe_routes = {task.subtask_id: swarm.select_agent(task, worker).role for task in PROBE_TASKS}
    return {
        "sample_id": sample["id"],
        "repeat": repeat,
        "domain": sample.get("domain"),
        "question": sample["question"],
        "expected": expected,
        "actual": {
            **swarm.to_dict(),
            "policy_version": policy.VERSION,
            "plan_task_count": len(plan.subtasks),
            "routed_roles": dict(sorted(Counter(routed).items())),
            "probe_routes": probe_routes,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "latency_seconds": time.perf_counter() - started,
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_levels = Counter(str(row["expected"]["level"]) for row in rows)
    actual_levels = Counter(str(row["actual"]["level"]) for row in rows)
    agent_counts = Counter(int(row["actual"]["agent_count"]) for row in rows)
    role_combinations = Counter(
        " + ".join(str(agent["role"]) for agent in row["actual"]["agents"]) for row in rows
    )
    confusion = Counter(f"{row['expected']['level']}->{row['actual']['level']}" for row in rows)
    latencies = [float(row["latency_seconds"]) for row in rows]
    return {
        "total": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "expected_complexity_distribution": dict(sorted(expected_levels.items())),
        "actual_complexity_distribution": dict(sorted(actual_levels.items())),
        "agent_count_distribution": {
            str(key): value for key, value in sorted(agent_counts.items())
        },
        "role_combination_distribution": dict(sorted(role_combinations.items())),
        "complexity_confusion": dict(sorted(confusion.items())),
        "mean_planning_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
    }


def render_markdown(payload: dict[str, Any], source: Path) -> str:
    aggregate_payload = payload["aggregate"]
    lines = [
        "# ResearchBench-Live Dynamic Swarm 验证",
        "",
        f"- 运行 ID：`{payload['run_id']}`",
        f"- Planner：`{payload['planner_mode']}`；Provider：`{payload['provider']}`",
        f"- 数据集：`{payload['dataset_name']} {payload['dataset_version']}`",
        f"- 原始结果：`{source.resolve()}`",
        f"- 通过：{aggregate_payload['passed']}/{aggregate_payload['total']}",
        "",
        "## 三档编组",
        "",
        "| 复杂度 | 运行数 | Agent 数 | 角色组合 |",
        "|---|---:|---:|---|",
    ]
    representative: dict[str, dict[str, Any]] = {}
    for row in payload["results"]:
        representative.setdefault(str(row["actual"]["level"]), row)
    for level in LEVELS:
        row = representative.get(level)
        if row is None:
            lines.append(f"| {level} | 0 | N/A | N/A |")
            continue
        count = aggregate_payload["actual_complexity_distribution"].get(level, 0)
        roles = ", ".join(agent["role"] for agent in row["actual"]["agents"])
        lines.append(f"| {level} | {count} | {row['actual']['agent_count']} | {roles} |")
    lines.extend(
        [
            "",
            "## 逐题结果",
            "",
            "| 样本 | 期望 | 实际 | 分数 | Agent | 并发 | 结果 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            f"| {row['sample_id']}#r{row['repeat']} | {row['expected']['level']} | "
            f"{row['actual']['level']} | {row['actual']['complexity_score']:.2f} | "
            f"{row['actual']['agent_count']} | {row['actual']['max_concurrency']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "> 本报告验证 Planner 输出经 SwarmPolicy 后的 Agent 数量、角色组合、并发和预算配置。",
            "> 检索召回率、报告质量和真实角色执行效果应使用端到端 Live 实验单独评估。",
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> Path:
    load_env_files((BASE_DIR / ".env", EVALUATION_DIR / ".env"))
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    samples = dataset["samples"][: args.limit or None]
    planner, planner_mode = _planner(args.provider, args.model)
    policy = HeuristicSwarmPolicy()
    config = ResearchConfig(
        max_concurrency=args.max_concurrency,
        max_swarm_agents=args.max_swarm_agents,
        enable_dynamic_swarm=True,
    )
    worker = AgentSpec(
        agent_id="research-worker",
        role="researcher",
        description="ResearchBench-Live policy probe",
        tool_names=("retrieval",),
    )
    semaphore = asyncio.Semaphore(args.experiment_concurrency)

    async def guarded(sample: dict[str, Any], repeat: int) -> dict[str, Any]:
        async with semaphore:
            try:
                return await evaluate_one(sample, repeat, planner, policy, config, worker)
            except Exception as exc:
                return {
                    "sample_id": sample["id"],
                    "repeat": repeat,
                    "question": sample["question"],
                    "expected": sample["expected_swarm"],
                    "actual": {"level": "error", "agent_count": 0, "agents": []},
                    "checks": {},
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_seconds": 0.0,
                }

    rows = await asyncio.gather(
        *(guarded(sample, repeat) for repeat in range(1, args.repeats + 1) for sample in samples)
    )
    run_id = datetime.now(timezone.utc).strftime("live-swarm-%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "live",
        "dataset_name": dataset["name"],
        "dataset_version": dataset["version"],
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "provider": args.provider,
        "model": args.model,
        "planner_mode": planner_mode,
        "swarm_policy": {
            "name": type(policy).__name__,
            "version": policy.VERSION,
            "simple_threshold": policy.SIMPLE_THRESHOLD,
            "complex_threshold": policy.COMPLEX_THRESHOLD,
        },
        "repeats": args.repeats,
        "aggregate": aggregate(rows),
        "results": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        render_markdown(payload, output_path), encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"result_file={output_path}")
    if args.strict and payload["aggregate"]["failed"]:
        raise RuntimeError("Live Swarm policy validation failed")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate three-tier Dynamic Swarm composition on ResearchBench-Live"
    )
    parser.add_argument(
        "--dataset",
        default=str(EVALUATION_DIR / "datasets" / "researchbench_live_v1.0.json"),
    )
    parser.add_argument(
        "--provider",
        choices=("offline", "auto", "deepseek", "mimo", "vllm", "openai"),
        default="offline",
        help="offline uses the deterministic planner; other providers use LLMPlanner",
    )
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experiment-concurrency", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-swarm-agents", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_DIR / "results" / "live_swarm"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.experiment_concurrency < 1:
        parser.error("repeats and experiment-concurrency must be positive")
    try:
        asyncio.run(run(args))
    except (RuntimeError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
