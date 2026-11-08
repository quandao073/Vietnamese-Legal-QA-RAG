"""FastAPI dependencies: services, health probe, rate limiting, admin auth."""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Protocol

from fastapi import Header, Request

from legalrag.api.errors import ApiError
from legalrag.config import Settings, get_settings
from legalrag.retrieval.types import Hit

logger = logging.getLogger(__name__)


class AnswererLike(Protocol):
    def answer(
        self,
        question: str,
        *,
        session_id: str | None,
        history: Any,
        overrides: Any,
        request_id: str | None = None,
    ) -> Any: ...

    async def answer_sync(
        self,
        question: str,
        *,
        session_id: str | None,
        history: Any,
        overrides: Any,
        request_id: str | None = None,
    ) -> Any: ...


class SearcherLike(Protocol):
    async def search(
        self, q: str, top_k: int, mode: str, strategy: str | None = None
    ) -> list[Hit]: ...


class AdminStoreLike(Protocol):
    def message_exists(self, message_id: str) -> bool: ...

    def add_feedback(
        self, *, message_id: str, rating: int, reason: str | None, comment: str | None
    ) -> int: ...

    def stats(self) -> dict[str, Any]: ...


class _DbAdminStore:
    def message_exists(self, message_id: str) -> bool:
        from legalrag.db.repositories import QueryLogRepo
        from legalrag.db.session import session_scope

        with session_scope() as s:
            return QueryLogRepo(s).exists(message_id)

    def add_feedback(
        self, *, message_id: str, rating: int, reason: str | None, comment: str | None
    ) -> int:
        from legalrag.db.repositories import FeedbackRepo
        from legalrag.db.session import session_scope

        with session_scope() as s:
            return FeedbackRepo(s).insert(
                message_id=message_id, rating=rating, reason=reason, comment=comment
            )

    def stats(self) -> dict[str, Any]:
        from legalrag.db.repositories import FeedbackRepo, IngestRunRepo, StatsRepo
        from legalrag.db.session import session_scope

        with session_scope() as s:
            data = StatsRepo(s).summary()
            data["recent_negative_feedback"] = FeedbackRepo(s).latest_negative(20)
            data["ingest_runs"] = [
                {
                    "run_id": r.run_id,
                    "source": r.source,
                    "strategy": r.strategy,
                    "status": r.status,
                    "n_docs": r.n_docs,
                    "n_chunks": r.n_chunks,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                }
                for r in IngestRunRepo(s).latest(10)
            ]
            return data


def get_admin_store() -> AdminStoreLike:
    return _DbAdminStore()


def get_settings_dep() -> Settings:
    return get_settings()


def get_answerer_dep() -> AnswererLike:
    from legalrag.services import get_answerer

    return get_answerer()


class _RetrieverSearcher:
    """Retrieval-only facade used by /search (no LLM, no query rewrite)."""

    async def search(self, q: str, top_k: int, mode: str, strategy: str | None = None) -> list[Hit]:
        from legalrag.retrieval.types import RetrievalOverrides
        from legalrag.services import get_retriever

        res = await get_retriever().retrieve(
            q,
            qu=None,
            overrides=RetrievalOverrides(top_k=top_k, mode=mode, strategy=strategy),  # type: ignore[arg-type]
        )
        return res.reranked


def get_searcher_dep() -> SearcherLike:
    return _RetrieverSearcher()


def health_probe() -> dict[str, Any]:
    """Live checks of the backing services (cheap; used by /health*)."""
    from legalrag import services
    from legalrag.cache import RedisCache
    from legalrag.db.session import ping as pg_ping
    from legalrag.retrieval.dense import get_qdrant_client, qdrant_ping

    s = get_settings()
    out: dict[str, Any] = {
        "qdrant": qdrant_ping(get_qdrant_client()),
        "postgres": pg_ping(),
        "redis": RedisCache(s.redis_url).ping(),
        "bm25_loaded": services.bm25_loaded(s.chunk_strategy),
        "embedder_loaded": services._retriever is not None,
        "collection": s.collection_name(s.chunk_strategy),
        "n_chunks": None,
    }
    if out["qdrant"]:
        try:
            out["n_chunks"] = services.get_dense(s.chunk_strategy, s).count()
        except Exception as exc:  # pragma: no cover - infra
            logger.debug("count failed: %s", exc)
    return out


# ---- rate limiting (Redis fixed window, 30 req/min/IP by default) --------------------------------
_local_buckets: dict[str, tuple[int, int]] = {}


def rate_limit(request: Request) -> None:
    s = get_settings()
    limit = s.rate_limit_per_minute
    if limit <= 0:
        return
    ip = (
        (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?"))
        .split(",")[0]
        .strip()
    )
    minute = int(time.time() // 60)
    key = f"rl:{ip}:{minute}"
    count: int | None = None
    try:
        import redis

        r = redis.Redis.from_url(s.redis_url, socket_connect_timeout=0.5, socket_timeout=0.5)
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        count = int(pipe.execute()[0])
    except Exception:
        # Redis down: fall back to an in-process window so the API keeps working
        prev_minute, prev = _local_buckets.get(ip, (minute, 0))
        count = prev + 1 if prev_minute == minute else 1
        _local_buckets[ip] = (minute, count)
    if count is not None and count > limit:
        raise ApiError(
            429, "rate_limited", "Bạn đã gửi quá nhiều yêu cầu. Vui lòng thử lại sau một phút."
        )


def require_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
    expected = get_settings().admin_token.get_secret_value()
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise ApiError(401, "unauthorized", "Mã quản trị (X-Admin-Token) không hợp lệ.")
