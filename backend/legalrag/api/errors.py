"""Uniform JSON errors with Vietnamese messages; never leak stack traces (spec §4.4)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from legalrag.generation.llm_client import LLMError
from legalrag.schemas.chat import MAX_QUESTION_CHARS

logger = logging.getLogger(__name__)

_STATUS_MESSAGES = {
    401: "Không có quyền truy cập. Vui lòng kiểm tra mã xác thực.",
    403: "Không có quyền truy cập.",
    404: "Không tìm thấy tài nguyên yêu cầu.",
    405: "Phương thức không được hỗ trợ.",
    429: "Bạn đã gửi quá nhiều yêu cầu. Vui lòng thử lại sau một phút.",
    503: "Hệ thống chưa sẵn sàng. Vui lòng thử lại sau.",
}


class ApiError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message


def _json(status: int, code: str, message: str, request: Request) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "request_id": rid},
        headers={"X-Request-ID": rid} if rid else None,
    )


def _validation_message(exc: RequestValidationError) -> str:
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
        typ = err.get("type", "")
        msg = str(err.get("msg", ""))
        if loc.endswith("question") and typ == "string_too_long":
            return f"Câu hỏi quá dài (tối đa {MAX_QUESTION_CHARS} ký tự)."
        if loc.endswith("question") and typ in ("string_too_short", "missing"):
            return "Câu hỏi không được để trống."
        if "Câu hỏi không được để trống" in msg:
            return "Câu hỏi không được để trống."
        if loc:
            return f"Dữ liệu không hợp lệ ở trường '{loc}': {msg}"
    return "Dữ liệu gửi lên không hợp lệ."


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _json(exc.status_code, exc.code, exc.message, request)

    @app.exception_handler(LLMError)
    async def _llm_error(request: Request, exc: LLMError) -> JSONResponse:
        logger.warning("llm error surfaced to API: %s", exc)
        return _json(502, exc.code, exc.message_vi, request)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json(422, "validation_error", _validation_message(exc), request)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) and exc.detail else None
        return _json(
            exc.status_code,
            f"http_{exc.status_code}",
            message or _STATUS_MESSAGES.get(exc.status_code, "Yêu cầu không thực hiện được."),
            request,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _json(500, "internal_error", "Đã xảy ra lỗi nội bộ. Vui lòng thử lại sau.", request)
