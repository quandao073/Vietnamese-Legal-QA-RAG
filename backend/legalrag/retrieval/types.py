"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["hybrid", "dense", "bm25"]


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def rerank_score(self) -> float | None:
        v = self.debug.get("rerank_score")
        return float(v) if v is not None else None


@dataclass
class RetrievalOverrides:
    top_k: int | None = None
    use_rerank: bool | None = None
    use_rewrite: bool | None = None
    strategy: str | None = None
    mode: Mode | None = None
    top_n_dense: int | None = None
    top_n_bm25: int | None = None


@dataclass
class RetrievalResult:
    query_understanding: Any
    candidates: list[Hit]
    reranked: list[Hit]
    timings: dict[str, float] = field(default_factory=dict)
    strategy: str = ""
    mode: str = "hybrid"
    stage_rankings: dict[str, list[str]] = field(default_factory=dict)
