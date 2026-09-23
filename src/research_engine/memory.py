"""SQLite-backed cross-agent memory with semantic filtering and compression."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .schemas import Evidence, utc_now


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;\.])\s*")
CONTRADICTION_PAIRS = (
    ("允许", "禁止"),
    ("支持", "不支持"),
    ("必须", "无需"),
    ("增加", "减少"),
    ("enabled", "disabled"),
    ("required", "optional"),
    ("increase", "decrease"),
    ("true", "false"),
)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HashingEmbedder:
    """Small deterministic embedding fallback; replaceable by BGE/OpenAI vectors."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def textrank_compress(text: str, max_sentences: int = 3) -> str:
    """L2 extractive compression using a compact TextRank implementation."""

    sentences = [item.strip() for item in SENTENCE_PATTERN.split(text) if item.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    token_sets = [set(_tokens(sentence)) for sentence in sentences]
    graph = [[0.0] * len(sentences) for _ in sentences]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            overlap = len(token_sets[i] & token_sets[j])
            denominator = math.log(len(token_sets[i]) + 1) + math.log(len(token_sets[j]) + 1)
            weight = overlap / denominator if denominator else 0.0
            graph[i][j] = graph[j][i] = weight
    scores = [1.0] * len(sentences)
    for _ in range(20):
        updated = []
        for i in range(len(sentences)):
            incoming = 0.0
            for j in range(len(sentences)):
                total = sum(graph[j])
                if total:
                    incoming += graph[j][i] / total * scores[j]
            updated.append(0.15 + 0.85 * incoming)
        scores = updated
    selected = sorted(
        sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)[:max_sentences]
    )
    return " ".join(sentences[index] for index in selected)


@dataclass
class MemoryHit:
    memory_id: str
    content: str
    source: str
    score: float
    metadata: dict


class SharedMemory:
    """Persistent memory shared by planner, workers, reviewers and synthesizer."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder: Callable[[str], list[float]] | None = None,
        dedupe_threshold: float = 0.96,
        contradiction_threshold: float = 0.72,
        conflict_strategy: str = "keep_both",
    ):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.dedupe_threshold = dedupe_threshold
        self.contradiction_threshold = contradiction_threshold
        self.conflict_strategy = conflict_strategy
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                vector TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                superseded_by TEXT
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def _heuristic_contradiction(left: str, right: str) -> bool:
        left_lower, right_lower = left.lower(), right.lower()
        return any(
            (positive in left_lower and negative in right_lower)
            or (negative in left_lower and positive in right_lower)
            for positive, negative in CONTRADICTION_PAIRS
        )

    def add(
        self,
        content: str,
        source: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(memory_id, outcome)`` where outcome describes resolution."""

        content = content.strip()
        if not content:
            raise ValueError("Cannot store empty memory")
        vector = self.embedder(content)
        metadata = dict(metadata or {})
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memories WHERE superseded_by IS NULL"
            ).fetchall()
            nearest = None
            nearest_score = -1.0
            for row in rows:
                score = cosine(vector, json.loads(row["vector"]))
                if score > nearest_score:
                    nearest, nearest_score = row, score
            if nearest is not None and nearest_score >= self.dedupe_threshold:
                return str(nearest["memory_id"]), "duplicate"

            conflict_id = None
            if (
                nearest is not None
                and nearest_score >= self.contradiction_threshold
                and self._heuristic_contradiction(content, str(nearest["content"]))
            ):
                conflict_id = str(nearest["memory_id"])
                metadata["contradicts"] = conflict_id
                if self.conflict_strategy == "prefer_existing":
                    return conflict_id, "conflict_kept_existing"

            memory_id = memory_id or uuid.uuid4().hex
            collision = self._connection.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if collision:
                memory_id = f"{memory_id}-{uuid.uuid4().hex[:8]}"
            self._connection.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (
                    memory_id,
                    content,
                    source,
                    json.dumps(vector),
                    json.dumps(metadata, ensure_ascii=False),
                    utc_now(),
                ),
            )
            if conflict_id and self.conflict_strategy == "prefer_newest":
                self._connection.execute(
                    "UPDATE memories SET superseded_by = ? WHERE memory_id = ?",
                    (memory_id, conflict_id),
                )
            self._connection.commit()
            return memory_id, "conflict" if conflict_id else "inserted"

    def add_evidence(self, evidence: Evidence) -> tuple[str, str]:
        return self.add(
            evidence.content,
            evidence.source,
            {
                **evidence.metadata,
                "evidence_id": evidence.evidence_id,
                "subtask_id": evidence.subtask_id,
                "url": evidence.url,
                "title": evidence.title,
            },
            memory_id=evidence.evidence_id,
        )

    def search(self, query: str, limit: int = 8) -> list[MemoryHit]:
        """L1 embedding coarse filter over all active memories."""

        vector = self.embedder(query)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memories WHERE superseded_by IS NULL"
            ).fetchall()
        hits = [
            MemoryHit(
                str(row["memory_id"]),
                str(row["content"]),
                str(row["source"]),
                cosine(vector, json.loads(row["vector"])),
                json.loads(row["metadata"]),
            )
            for row in rows
        ]
        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]

    def build_context(
        self,
        query: str,
        max_chars: int = 12_000,
        coarse_limit: int = 12,
        preserve_original: int = 2,
    ) -> str:
        """L1 retrieval -> L2 TextRank -> L3 top-evidence original retention."""

        hits = self.search(query, limit=coarse_limit)
        blocks: list[str] = []
        used = 0
        for index, hit in enumerate(hits):
            content = hit.content if index < preserve_original else textrank_compress(hit.content)
            block = f"[{hit.memory_id}] {content}\n来源：{hit.source}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
