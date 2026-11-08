from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from legalrag import jobs
from legalrag.api import deps
from legalrag.api.errors import ApiError

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(deps.require_admin)])


class IngestRequest(BaseModel):
    source: Literal["tamir", "yuitc", "all"]
    strategy: str = Field(default="article", max_length=64)
    max_docs: int | None = Field(default=None, ge=0)


@router.get("/stats")
async def stats(store: deps.AdminStoreLike = Depends(deps.get_admin_store)) -> dict[str, Any]:
    data = await run_in_threadpool(store.stats)
    data["jobs"] = jobs.list_jobs()
    return data


@router.post("/ingest", status_code=202)
async def start_ingest(body: IngestRequest) -> dict[str, Any]:
    try:
        job = await jobs.start_ingest(body.source, body.strategy, body.max_docs)
    except RuntimeError as exc:
        raise ApiError(409, "ingest_busy", str(exc)) from exc
    return {"run_id": job.run_id, "status": job.status}


@router.get("/ingest/{run_id}")
async def ingest_status(run_id: str) -> dict[str, Any]:
    job = jobs.get_job(run_id)
    if job is None:
        raise ApiError(404, "not_found", "Không tìm thấy tiến trình nạp dữ liệu.")
    return job.as_dict()
