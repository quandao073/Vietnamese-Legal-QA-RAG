"""Hybrid retriever (spec §3.5): queries → dense + BM25 → RRF (+title boost) → rerank → top_k."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from legalrag.config import Settings
from legalrag.retrieval.embedder import EmbedderProtocol
from legalrag.retrieval.fusion import rrf, title_boost
from legalrag.retrieval.query_understanding import QueryUnderstanding
from legalrag.retrieval.reranker import RerankerProtocol, rerank
from legalrag.retrieval.types import Hit, RetrievalOverrides, RetrievalResult

logger = logging.getLogger(__name__)
RERANK_POOL = 30


class _DenseLike(Protocol):
    def search(
        self, query_vec: Any, top_n: int, filters: dict[str, Any] | None = None
    ) -> list[Hit]: ...


class _BM25Like(Protocol):
    def search(self, query: str, top_n: int) -> list[Hit]: ...


class _ChunkRepoLike(Protocol):
    def get_many(self, chunk_ids: list[str]) -> dict[str, Any]: ...


def _payload_of(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    keys = (
        "chunk_id",
        "doc_id",
        "title",
        "text",
        "article",
        "article_title",
        "clause",
        "point",
        "seq",
        "extra",
    )
    return {k: getattr(obj, k, None) for k in keys}


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        dense_by_strategy: Callable[[str], _DenseLike],
        bm25_by_strategy: Callable[[str], _BM25Like],
        reranker: RerankerProtocol | None,
        chunk_repo: _ChunkRepoLike | None,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.dense_by_strategy = dense_by_strategy
        self.bm25_by_strategy = bm25_by_strategy
        self.reranker = reranker
        self.chunk_repo = chunk_repo
        self.settings = settings

    async def retrieve(
        self,
        question: str,
        qu: QueryUnderstanding | None = None,
        overrides: RetrievalOverrides | None = None,
    ) -> RetrievalResult:
        s, o = self.settings, overrides or RetrievalOverrides()
        strategy = o.strategy or s.chunk_strategy
        mode = o.mode or "hybrid"
        top_k = o.top_k or s.retrieval_top_k
        use_rerank = s.rerank_enabled if o.use_rerank is None else o.use_rerank
        use_rerank = use_rerank and self.reranker is not None
        n_dense = o.top_n_dense or s.retrieval_top_n_dense
        n_bm25 = o.top_n_bm25 or s.retrieval_top_n_bm25
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        if qu is not None and qu.intent != "legal_question":
            return RetrievalResult(qu, [], [], {"total_ms": 0.0}, strategy=strategy, mode=mode)
        queries = qu.queries() if qu is not None else [question]
        if not queries:
            queries = [question]
        if o.use_rewrite is False:
            queries = [qu.standalone_question if qu else question]

        ranked: dict[str, list[Hit]] = {}
        t0 = time.perf_counter()
        if mode in ("hybrid", "dense"):
            dense = self.dense_by_strategy(strategy)
            vecs = await asyncio.gather(
                *(asyncio.to_thread(self.embedder.encode_query, q) for q in queries)
            )
            lists = await asyncio.gather(
                *(asyncio.to_thread(dense.search, v, n_dense, None) for v in vecs)
            )
            for i, hits in enumerate(lists):
                ranked[f"dense{i}" if i else "dense"] = hits
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if mode in ("hybrid", "bm25"):
            bm25 = self.bm25_by_strategy(strategy)
            lists = await asyncio.gather(
                *(asyncio.to_thread(bm25.search, q, n_bm25) for q in queries)
            )
            for i, hits in enumerate(lists):
                ranked[f"bm25{i}" if i else "bm25"] = hits
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pool = [h for hits in ranked.values() for h in hits]
        boosts = title_boost(pool, qu.entities.law_names if qu else [])
        candidates = rrf(ranked, k=s.retrieval_rrf_k, boosts=boosts)
        candidates = self._collapse(candidates, strategy)
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if use_rerank and candidates and self.reranker is not None:
            rerank_query = qu.standalone_question if qu else question
            reranked = await asyncio.to_thread(
                rerank, rerank_query, candidates[:RERANK_POOL], self.reranker, top_k
            )
        else:
            reranked = candidates[:top_k]
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        stage_rankings = {
            "dense": [h.chunk_id for h in ranked.get("dense", [])],
            "bm25": [h.chunk_id for h in ranked.get("bm25", [])],
            "hybrid": [h.chunk_id for h in candidates],
            "hybrid_rerank": [h.chunk_id for h in reranked],
        }
        return RetrievalResult(
            qu,
            candidates,
            reranked,
            timings,
            strategy=strategy,
            mode=mode,
            stage_rankings=stage_rankings,
        )

    # ---- strategy-specific post-processing ----------------------------------------------------
    def _collapse(self, hits: list[Hit], strategy: str) -> list[Hit]:
        base = strategy.split("__", 1)[0]
        if base == "parent_child":
            return self._to_parents(hits)
        if base == "clause":
            seen: set[tuple[str, str]] = set()
            out: list[Hit] = []
            for h in hits:
                key = (str(h.payload.get("doc_id")), str(h.payload.get("article")))
                if key in seen and h.payload.get("article"):
                    continue
                seen.add(key)
                out.append(h)
            return out
        return hits

    def _to_parents(self, hits: list[Hit]) -> list[Hit]:
        parent_ids: list[str] = []
        best: dict[str, Hit] = {}
        for h in hits:
            extra = h.payload.get("extra") or {}
            pid = extra.get("parent_chunk_id") if extra.get("role") == "child" else h.chunk_id
            pid = str(pid or h.chunk_id)
            if pid not in best:
                best[pid] = h
                parent_ids.append(pid)
        missing = [p for p in parent_ids if p not in {h.chunk_id for h in hits}]
        fetched = self.chunk_repo.get_many(missing) if (missing and self.chunk_repo) else {}
        out: list[Hit] = []
        for pid in parent_ids:
            child = best[pid]
            if pid == child.chunk_id:
                out.append(child)
                continue
            parent = fetched.get(pid)
            if parent is None:
                out.append(child)
                continue
            payload = _payload_of(parent)
            out.append(
                Hit(
                    chunk_id=pid,
                    score=child.score,
                    payload=payload,
                    debug=dict(child.debug, child_chunk_id=child.chunk_id),
                )
            )
        return out
