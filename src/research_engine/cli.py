"""Command-line entry point for the repository's local RAG-backed agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .agents import LocalHybridSearchBackend, ResearchWorker, Synthesizer
from .config import ResearchConfig
from .env import load_env_files
from .memory import SharedMemory
from .llm_backends import auto_provider, build_llm
from .orchestrator import DeepResearchAgent, default_memory_path
from .planner import HeuristicPlanner, LLMPlanner
from .web_backends import CompositeSearchBackend, TavilySearchBackend


class OfflineFixtureSearchBackend:
    """Deterministic evidence backend for CI and installation smoke tests."""

    async def search(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        rows = [
            {
                "title": "Offline fixture: scope and definitions",
                "content": (
                    f"该离线固定证据用于验证研究流程。检索问题为：{query}。"
                    "它覆盖定义、范围、边界、术语和适用条件。"
                ),
                "source": "offline-fixture://scope",
                "source_type": "official_documentation",
                "score": 1.0,
            },
            {
                "title": "Offline fixture: evidence and verification",
                "content": (
                    f"该离线固定证据用于验证研究流程。检索问题为：{query}。"
                    "它提供事实、数据、标准、直接证据和验证要求。"
                ),
                "source": "offline-fixture://evidence",
                "source_type": "standard",
                "score": 0.98,
            },
            {
                "title": "Offline fixture: limitations and impact",
                "content": (
                    f"该离线固定证据用于验证研究流程。检索问题为：{query}。"
                    "它说明限制、风险、失败案例、成本、影响和实施建议。"
                ),
                "source": "offline-fixture://limitations",
                "source_type": "official_reference",
                "score": 0.96,
            },
        ]
        return rows[:limit]


def build_search_backend(args: argparse.Namespace):
    if args.search_provider == "fixture":
        return OfflineFixtureSearchBackend()
    local = LocalHybridSearchBackend()
    if args.search_provider == "local":
        return local
    web = TavilySearchBackend(
        search_depth=args.web_search_depth,
        include_domains=args.web_include_domain,
        exclude_domains=args.web_exclude_domain,
        exclude_url_fragments=args.web_exclude_url_fragment,
    )
    if args.search_provider == "tavily":
        return web
    return CompositeSearchBackend(
        {"local": local, "tavily": web},
        weights={"local": args.local_weight, "tavily": args.web_weight},
    )


async def async_main(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    is_fixture_smoke = args.offline and args.search_provider == "fixture"
    if not is_fixture_smoke:
        load_env_files((project_root / ".env", project_root / "evaluation" / ".env"))
    selected_provider = (
        None if args.offline else (auto_provider() if args.provider == "auto" else args.provider)
    )
    llm = build_llm(selected_provider, args.model) if selected_provider else None
    planner = LLMPlanner(llm) if llm else HeuristicPlanner()
    memory = SharedMemory(args.memory or default_memory_path(project_root))
    agent = DeepResearchAgent(
        planner=planner,
        worker=ResearchWorker(build_search_backend(args)),
        synthesizer=Synthesizer(llm),
        memory=memory,
        config=ResearchConfig(
            max_concurrency=args.concurrency,
            task_timeout_seconds=args.task_timeout,
            global_timeout_seconds=args.global_timeout,
            max_review_rounds=args.max_review_rounds,
            min_citation_coverage=args.min_citation_coverage,
            max_pass_issue_severity=args.max_pass_issue_severity,
            max_verification_queries_per_round=args.max_verification_queries_per_round,
            max_blue_patches_per_round=args.max_blue_patches_per_round,
            max_patch_growth_chars=args.max_patch_growth_chars,
            max_blue_patch_generation_attempts=args.max_blue_patch_generation_attempts,
            enable_semantic_claim_verification=(not args.disable_semantic_claim_verification),
            min_claim_support_rate=args.min_claim_support_rate,
            enable_dynamic_swarm=not args.disable_dynamic_swarm,
            max_swarm_agents=args.max_swarm_agents,
            max_worker_invocations=args.max_worker_invocations,
            max_stagnant_review_rounds=args.max_stagnant_review_rounds,
        ),
    )
    try:
        result = await agent.run(args.question)
    finally:
        memory.close()
    output = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Run the multi-agent deep research engine",
    )
    parser.add_argument("question")
    parser.add_argument("--offline", action="store_true", help="Do not call an LLM")
    parser.add_argument(
        "--provider",
        choices=("auto", "deepseek", "mimo", "vllm", "openai"),
        default="auto",
        help="LLM backend; auto selects from environment variables",
    )
    parser.add_argument("--model", help="Override the provider model")
    parser.add_argument(
        "--search-provider",
        choices=("local", "tavily", "hybrid-web", "fixture"),
        default="local",
        help="Evidence retrieval source; fixture is deterministic and only for smoke tests",
    )
    parser.add_argument(
        "--web-search-depth",
        choices=("basic", "advanced"),
        default="advanced",
    )
    parser.add_argument(
        "--web-include-domain",
        action="append",
        default=[],
        help="Restrict Tavily to a domain; repeat for multiple domains",
    )
    parser.add_argument(
        "--web-exclude-domain",
        action="append",
        default=[],
        help="Exclude a Tavily domain; repeat for multiple domains",
    )
    parser.add_argument(
        "--web-exclude-url-fragment",
        action="append",
        default=[],
        help="Discard results whose URL contains this fragment; repeatable",
    )
    parser.add_argument("--local-weight", type=float, default=1.2)
    parser.add_argument("--web-weight", type=float, default=1.0)
    parser.add_argument("--output", help="Optional JSON result path")
    parser.add_argument("--memory", help="SQLite memory path")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--task-timeout", type=float, default=45.0)
    parser.add_argument("--global-timeout", type=float, default=240.0)
    parser.add_argument("--max-review-rounds", type=int, default=4)
    parser.add_argument("--min-citation-coverage", type=float, default=0.80)
    parser.add_argument("--max-pass-issue-severity", type=int, choices=range(4), default=1)
    parser.add_argument("--max-verification-queries-per-round", type=int, default=4)
    parser.add_argument("--max-blue-patches-per-round", type=int, default=20)
    parser.add_argument("--max-patch-growth-chars", type=int, default=4000)
    parser.add_argument("--max-blue-patch-generation-attempts", type=int, default=2)
    parser.add_argument("--min-claim-support-rate", type=float, default=0.80)
    parser.add_argument(
        "--disable-dynamic-swarm",
        action="store_true",
        help="Use fixed configured concurrency and budgets",
    )
    parser.add_argument("--max-swarm-agents", type=int, default=5)
    parser.add_argument("--max-worker-invocations", type=int, default=16)
    parser.add_argument("--max-stagnant-review-rounds", type=int, default=2)
    parser.add_argument(
        "--disable-semantic-claim-verification",
        action="store_true",
        help="Use deterministic lexical claim verification only",
    )
    args = parser.parse_args()
    if args.search_provider == "fixture" and not args.offline:
        parser.error("--search-provider fixture requires --offline")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
