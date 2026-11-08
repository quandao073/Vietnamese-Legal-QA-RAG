"""FastAPI factory: `uvicorn legalrag.api.app:create_app --factory`."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from legalrag import __version__
from legalrag.api.errors import install_error_handlers
from legalrag.api.routers import admin, chat, documents, feedback, health, search
from legalrag.config import get_settings
from legalrag.logging import request_id_var, setup_logging

logger = logging.getLogger(__name__)
API_PREFIX = "/api/v1"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.ready = False
    if os.environ.get("LEGALRAG_SKIP_WARMUP") != "1":
        from legalrag.services import warm_up

        info = await run_in_threadpool(warm_up, settings)
        logger.info("warm-up complete: %s", info)
    app.state.ready = True
    yield
    logger.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Vietnamese Legal RAG QA",
        version=__version__,
        lifespan=_lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def _request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    install_error_handlers(app)
    for r in (
        health.router,
        chat.router,
        search.router,
        documents.router,
        feedback.router,
        admin.router,
    ):
        app.include_router(r, prefix=API_PREFIX)
    return app
