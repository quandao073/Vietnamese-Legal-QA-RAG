from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from legalrag.api import deps
from legalrag.config import Settings

router = APIRouter(tags=["health"])


def _payload(probe: dict[str, Any], ready: bool) -> tuple[int, dict[str, Any]]:
    healthy = bool(probe.get("qdrant")) and bool(probe.get("postgres"))
    status = "ok" if (healthy and ready) else ("starting" if not ready else "degraded")
    return (200 if status == "ok" else 503), {"status": status, **probe}


@router.get("/health")
def health(request: Request, probe: dict[str, Any] = Depends(deps.health_probe)) -> JSONResponse:
    code, body = _payload(probe, bool(getattr(request.app.state, "ready", False)))
    return JSONResponse(status_code=code, content=body)


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
def ready(request: Request, probe: dict[str, Any] = Depends(deps.health_probe)) -> JSONResponse:
    code, body = _payload(probe, bool(getattr(request.app.state, "ready", False)))
    return JSONResponse(status_code=code, content=body)


@router.get("/config/public")
def public_config(settings: Settings = Depends(deps.get_settings_dep)) -> dict[str, Any]:
    from legalrag.services import available_strategies

    return {
        "default_strategy": settings.chunk_strategy,
        "strategies": available_strategies(settings),
        "default_top_k": settings.retrieval_top_k,
        "top_k_range": [3, 10],
        "rerank_enabled": settings.rerank_enabled,
        "query_rewrite_enabled": settings.query_rewrite_enabled,
        "modes": ["hybrid", "dense", "bm25"],
        "max_question_chars": 2000,
    }
