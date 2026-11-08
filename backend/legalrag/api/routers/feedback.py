from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from legalrag.api import deps
from legalrag.api.errors import ApiError
from legalrag.schemas.chat import FeedbackIn

router = APIRouter(tags=["feedback"])


@router.post("/feedback", status_code=201)
async def post_feedback(
    body: FeedbackIn, store: deps.AdminStoreLike = Depends(deps.get_admin_store)
) -> dict[str, Any]:
    if not await run_in_threadpool(store.message_exists, body.message_id):
        raise ApiError(
            404, "not_found", "Không tìm thấy câu trả lời tương ứng để ghi nhận phản hồi."
        )
    fb_id = await run_in_threadpool(
        store.add_feedback,
        message_id=body.message_id,
        rating=body.rating,
        reason=body.reason,
        comment=body.comment,
    )
    return {"id": fb_id, "message": "Cảm ơn bạn đã phản hồi."}
