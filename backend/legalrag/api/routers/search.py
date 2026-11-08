from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query

from legalrag.api import deps
from legalrag.generation.context_builder import display_source_of

router = APIRouter(tags=["search"])


def hit_to_dict(h: Any) -> dict[str, Any]:
    p = h.payload
    return {
        "chunk_id": h.chunk_id,
        "doc_id": p.get("doc_id"),
        "title": p.get("title"),
        "display_source": display_source_of(p),
        "article": p.get("article"),
        "clause": p.get("clause"),
        "url": p.get("url"),
        "text": p.get("text", ""),
        "score": h.score,
        "rerank_score": h.rerank_score,
        "debug": {k: v for k, v in h.debug.items() if k != "rerank_score"},
    }


@router.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=2000),
    top_k: int = Query(default=10, ge=1, le=50),
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    strategy: str | None = Query(default=None, max_length=64),
    searcher: deps.SearcherLike = Depends(deps.get_searcher_dep),
) -> dict[str, Any]:
    if not q.strip():
        from legalrag.api.errors import ApiError

        raise ApiError(422, "validation_error", "Từ khóa tìm kiếm không được để trống.")
    hits = await searcher.search(q, top_k, mode, strategy)
    return {"q": q, "mode": mode, "strategy": strategy, "results": [hit_to_dict(h) for h in hits]}
