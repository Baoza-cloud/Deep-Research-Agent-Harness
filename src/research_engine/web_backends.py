"""Real Web search backends and cross-source rank fusion."""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Any, Mapping, Sequence
from urllib.parse import urldefrag, urlparse

from .harness import RetrievalTool


def _domain_matches(url: str, allowed_domains: Sequence[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().strip(".")
    return any(
        host == domain.lower().strip(".")
        or host.endswith("." + domain.lower().strip("."))
        for domain in allowed_domains
    )


class TavilySearchBackend:
    """Async Tavily adapter that emits the project's normalized evidence rows."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        search_depth: str = "advanced",
        include_raw_content: bool | str = "markdown",
        prefer_raw_content: bool = False,
        max_content_chars: int = 6000,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        exclude_url_fragments: Sequence[str] | None = None,
    ):
        if search_depth not in {"basic", "advanced"}:
            raise ValueError("search_depth must be 'basic' or 'advanced'")
        if client is None:
            try:
                from tavily import AsyncTavilyClient  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "Tavily Web search requires: pip install tavily-python"
                ) from exc
            resolved_key = api_key or os.getenv("TAVILY_API_KEY")
            if not resolved_key:
                raise ValueError("Missing TAVILY_API_KEY")
            client = AsyncTavilyClient(resolved_key)
        self.client = client
        self.search_depth = search_depth
        self.include_raw_content = include_raw_content
        self.prefer_raw_content = prefer_raw_content
        self.max_content_chars = max(256, max_content_chars)
        self.include_domains = list(include_domains or [])
        self.exclude_domains = list(exclude_domains or [])
        self.exclude_url_fragments = [
            item.lower() for item in (exclude_url_fragments or []) if item
        ]

    async def search(self, query: str, limit: int = 5) -> Sequence[dict[str, Any]]:
        search_options: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(limit, 20)),
            "search_depth": self.search_depth,
            "include_raw_content": self.include_raw_content,
            "include_answer": False,
        }
        if self.include_domains:
            search_options["include_domains"] = self.include_domains
        if self.exclude_domains:
            search_options["exclude_domains"] = self.exclude_domains
        response = await self.client.search(
            **search_options,
        )
        rows: list[dict[str, Any]] = []
        for item in response.get("results", []):
            focused_content = str(item.get("content") or "").strip()
            raw_content = str(item.get("raw_content") or "").strip()
            if self.prefer_raw_content:
                content = raw_content or focused_content
            else:
                content = focused_content or raw_content
            content = content[: self.max_content_chars].strip()
            url = str(item.get("url") or "").strip()
            if self.include_domains and not _domain_matches(url, self.include_domains):
                continue
            if any(fragment in url.lower() for fragment in self.exclude_url_fragments):
                continue
            if not content or not url:
                continue
            score = item.get("score")
            rows.append(
                {
                    "content": content,
                    "title": item.get("title"),
                    "url": url,
                    "source": url,
                    "source_type": "web",
                    "score": float(score) if isinstance(score, (int, float)) else None,
                    "metadata": {
                        "search_provider": "tavily",
                        "favicon": item.get("favicon"),
                        "raw_content_available": bool(raw_content),
                        "raw_content_chars": len(raw_content),
                        "content_truncated": len(content) >= self.max_content_chars,
                    },
                }
            )
        return rows


def _canonical_key(row: Mapping[str, Any]) -> str:
    url = str(row.get("url") or "").strip()
    if url:
        canonical, _ = urldefrag(url)
        return canonical.rstrip("/").lower()
    payload = f"{row.get('title', '')}|{row.get('content') or row.get('text') or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CompositeSearchBackend:
    """Run local/Web/paper backends concurrently and fuse their ranks with RRF."""

    def __init__(
        self,
        backends: Mapping[str, RetrievalTool],
        *,
        weights: Mapping[str, float] | None = None,
        rrf_k: int = 60,
    ):
        if not backends:
            raise ValueError("CompositeSearchBackend requires at least one backend")
        self.backends = dict(backends)
        self.weights = dict(weights or {})
        self.rrf_k = rrf_k

    async def search(self, query: str, limit: int = 5) -> Sequence[dict[str, Any]]:
        names = list(self.backends)
        responses = await asyncio.gather(
            *(self.backends[name].search(query, limit=limit) for name in names),
            return_exceptions=True,
        )
        fused: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for name, response in zip(names, responses):
            if isinstance(response, BaseException):
                failures.append(f"{name}: {type(response).__name__}")
                continue
            weight = self.weights.get(name, 1.0)
            for rank, original in enumerate(response, start=1):
                row = dict(original)
                key = _canonical_key(row)
                contribution = weight / (self.rrf_k + rank)
                if key not in fused:
                    row["rrf_score"] = contribution
                    row["retrieval_backends"] = [name]
                    row["backend_rank"] = {name: rank}
                    fused[key] = row
                else:
                    current = fused[key]
                    current["rrf_score"] += contribution
                    current["retrieval_backends"].append(name)
                    current["backend_rank"][name] = rank
                    existing_content = str(current.get("content") or current.get("text") or "")
                    candidate_content = str(row.get("content") or row.get("text") or "")
                    if len(candidate_content) > len(existing_content):
                        current["content"] = candidate_content

        if not fused and failures:
            raise RuntimeError("All search backends failed: " + ", ".join(failures))
        ranked = sorted(fused.values(), key=lambda row: row["rrf_score"], reverse=True)
        return ranked[:limit]
