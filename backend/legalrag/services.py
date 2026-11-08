"""Process-wide service wiring (singletons) shared by the API, CLIs and eval."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from legalrag.cache import get_cache
from legalrag.config import Settings, get_settings
from legalrag.db.repositories import ChunkRepo, QueryLogRepo
from legalrag.db.session import session_scope
from legalrag.generation.answerer import Answerer
from legalrag.generation.llm_client import LLMClient, get_llm
from legalrag.retrieval.dense import DenseStore, get_qdrant_client
from legalrag.retrieval.embedder import get_embedder
from legalrag.retrieval.lexical import BM25Index
from legalrag.retrieval.pipeline import Retriever
from legalrag.retrieval.reranker import get_reranker
from legalrag.schemas.chat import QueryLogCreate
from legalrag.tokens import get_token_counter

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_bm25: dict[str, BM25Index] = {}
_dense: dict[str, DenseStore] = {}
_retriever: Retriever | None = None
_answerer: Answerer | None = None


class _DbChunkRepo:
    """Session-per-call adapter so the retriever never holds a DB session."""

    def get_many(self, chunk_ids: list[str]) -> dict[str, Any]:
        with session_scope() as s:
            rows = ChunkRepo(s).get_many(chunk_ids)
            return {
                k: {
                    "chunk_id": v.chunk_id,
                    "doc_id": v.doc_id,
                    "title": v.title,
                    "text": v.text,
                    "article": v.article,
                    "article_title": v.article_title,
                    "clause": v.clause,
                    "point": v.point,
                    "seq": v.seq,
                    "extra": v.extra,
                }
                for k, v in rows.items()
            }


class _DbQueryLogRepo:
    def insert(self, log: QueryLogCreate) -> None:
        with session_scope() as s:
            QueryLogRepo(s).insert(log)


def bm25_dir(settings: Settings, strategy: str) -> Path:
    return Path(settings.data_dir) / "index" / "bm25" / strategy


def get_bm25(strategy: str, settings: Settings | None = None) -> BM25Index:
    s = settings or get_settings()
    with _lock:
        if strategy not in _bm25:
            path = bm25_dir(s, strategy)
            if not (path / "ids.json").exists():
                raise FileNotFoundError(f"BM25 index for strategy '{strategy}' not found at {path}")
            logger.info("loading BM25 index %s", path)
            _bm25[strategy] = BM25Index.load(path)
        return _bm25[strategy]


def get_dense(strategy: str, settings: Settings | None = None) -> DenseStore:
    s = settings or get_settings()
    with _lock:
        if strategy not in _dense:
            _dense[strategy] = DenseStore(
                get_qdrant_client(), s.collection_name(strategy), dim=get_embedder().dim
            )
        return _dense[strategy]


def bm25_loaded(strategy: str) -> bool:
    return strategy in _bm25


def available_strategies(settings: Settings | None = None) -> list[str]:
    s = settings or get_settings()
    root = Path(s.data_dir) / "index" / "bm25"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "ids.json").exists())


def get_retriever(settings: Settings | None = None) -> Retriever:
    global _retriever
    s = settings or get_settings()
    if _retriever is None:
        _retriever = Retriever(
            embedder=get_embedder(),
            dense_by_strategy=lambda st: get_dense(st, s),
            bm25_by_strategy=lambda st: get_bm25(st, s),
            reranker=get_reranker() if s.rerank_enabled else None,
            chunk_repo=_DbChunkRepo(),
            settings=s,
        )
    return _retriever


def get_answerer(settings: Settings | None = None, llm: LLMClient | None = None) -> Answerer:
    global _answerer
    s = settings or get_settings()
    if _answerer is None:
        _answerer = Answerer(
            retriever=get_retriever(s),
            llm=llm or get_llm(),
            settings=s,
            counter=get_token_counter(s.embedding_model),
            log_repo=_DbQueryLogRepo(),
        )
    return _answerer


def reload_indexes(strategy: str) -> None:
    """Drop cached BM25/Qdrant handles for a strategy after a (re)ingest; next call reloads."""
    with _lock:
        _bm25.pop(strategy, None)
        _dense.pop(strategy, None)
    logger.info("indexes for strategy %s will be reloaded on next use", strategy)


def warm_up(settings: Settings | None = None) -> dict[str, Any]:
    """Load embedder, reranker and the default BM25 index; run one dummy retrieval."""
    s = settings or get_settings()
    get_cache()
    emb = get_embedder()
    if s.rerank_enabled:
        get_reranker()
    info: dict[str, Any] = {
        "embedder": emb.model_name,
        "dim": emb.dim,
        "strategy": s.chunk_strategy,
    }
    try:
        info["bm25_size"] = len(get_bm25(s.chunk_strategy, s))
    except FileNotFoundError as exc:
        logger.warning("%s — run `make ingest-tamir` first", exc)
        info["bm25_size"] = 0
    try:
        info["qdrant_points"] = get_dense(s.chunk_strategy, s).count()
    except Exception as exc:  # pragma: no cover - infra
        logger.warning("qdrant unavailable during warm-up: %s", exc)
        info["qdrant_points"] = None
    emb.encode_query("khởi động hệ thống")
    return info
