from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from legalrag.api import deps
from legalrag.api.errors import ApiError
from legalrag.config import Settings
from legalrag.db.repositories import ChunkRepo, DocumentRepo
from legalrag.db.session import session_scope
from legalrag.generation.context_builder import display_source_of

router = APIRouter(tags=["documents"])


def _chunk_dict(c: Any) -> dict[str, Any]:
    payload = {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "seq": c.seq,
        "title": c.title,
        "text": c.text,
        "chapter": c.chapter,
        "chapter_title": c.chapter_title,
        "article": c.article,
        "article_title": c.article_title,
        "clause": c.clause,
        "point": c.point,
        "source_ids": c.source_ids,
        "token_count": c.token_count,
        "chunk_strategy": c.chunk_strategy,
        "extra": c.extra,
    }
    payload["display_source"] = display_source_of(payload)
    return payload


def _get_document(doc_id: str, strategy: str) -> dict[str, Any] | None:
    with session_scope() as s:
        d = DocumentRepo(s).get(doc_id)
        if d is None:
            return None
        return {
            "doc_id": d.doc_id,
            "source": d.source,
            "title": d.title,
            "doc_number": d.doc_number,
            "doc_type": d.doc_type,
            "issuing_body": d.issuing_body,
            "issued_date": d.issued_date.isoformat() if d.issued_date else None,
            "effective_status": d.effective_status,
            "url": d.url,
            "articles": DocumentRepo(s).articles(doc_id, strategy),
        }


def _get_chunk(chunk_id: str) -> dict[str, Any] | None:
    with session_scope() as s:
        prev, cur, nxt = ChunkRepo(s).neighbors(chunk_id)
        if cur is None:
            return None
        doc = DocumentRepo(s).get(cur.doc_id)
        return {
            "chunk": _chunk_dict(cur),
            "prev": _chunk_dict(prev) if prev else None,
            "next": _chunk_dict(nxt) if nxt else None,
            "url": doc.url if doc else None,
        }


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    strategy: str | None = Query(default=None, max_length=64),
    settings: Settings = Depends(deps.get_settings_dep),
) -> dict[str, Any]:
    doc = await run_in_threadpool(_get_document, doc_id, strategy or settings.chunk_strategy)
    if doc is None:
        raise ApiError(404, "not_found", "Không tìm thấy văn bản.")
    return doc


@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    chunk = await run_in_threadpool(_get_chunk, chunk_id)
    if chunk is None:
        raise ApiError(404, "not_found", "Không tìm thấy đoạn văn bản.")
    return chunk
