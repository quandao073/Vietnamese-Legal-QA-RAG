from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from legalrag.api import deps
from legalrag.retrieval.types import RetrievalOverrides
from legalrag.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


def _overrides(req: ChatRequest) -> RetrievalOverrides | None:
    if req.options is None:
        return None
    o = req.options
    return RetrievalOverrides(
        top_k=o.top_k,
        use_rerank=o.use_rerank,
        use_rewrite=o.use_rewrite,
        strategy=o.strategy,
        mode=o.mode,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", dependencies=[Depends(deps.rate_limit)])
async def chat(
    req: ChatRequest, request: Request, answerer: deps.AnswererLike = Depends(deps.get_answerer_dep)
) -> StreamingResponse:
    request_id = getattr(request.state, "request_id", None)

    async def gen() -> AsyncIterator[str]:
        async for ev in answerer.answer(
            req.question,
            session_id=req.session_id,
            history=req.history,
            overrides=_overrides(req),
            request_id=request_id,
        ):
            yield _sse(ev.event, ev.data)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if request_id:
        headers["X-Request-ID"] = request_id
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@router.post("/chat/sync", response_model=ChatResponse, dependencies=[Depends(deps.rate_limit)])
async def chat_sync(
    req: ChatRequest, request: Request, answerer: deps.AnswererLike = Depends(deps.get_answerer_dep)
) -> ChatResponse:
    return await answerer.answer_sync(
        req.question,
        session_id=req.session_id,
        history=req.history,
        overrides=_overrides(req),
        request_id=getattr(request.state, "request_id", None),
    )
