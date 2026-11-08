"""Anthropic Messages API wrapper (spec §3.8): retry, usage/cost, response cache, JSON output.

Model notes (ADR-001): Sonnet 5.5 / Haiku 5.5 reject ``temperature`` and forced ``tool_choice``,
so neither is sent; JSON comes from ``output_config.format`` (structured outputs).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from legalrag.cache import Cache
from legalrag.config import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 5
BACKOFF_SECONDS = (1, 2, 4, 8, 16)
REQUEST_TIMEOUT = 60.0
RESPONSE_CACHE_TTL = 30 * 24 * 3600

ErrorCode = Literal[
    "auth", "rate_limit", "overloaded", "timeout", "connection", "bad_request", "unknown"
]

_MESSAGES_VI: dict[str, str] = {
    "auth": "Không thể xác thực với dịch vụ Anthropic API. Vui lòng kiểm tra khóa API (ANTHROPIC_API_KEY).",
    "rate_limit": "Dịch vụ ngôn ngữ đang quá tải hoặc đã đạt giới hạn truy cập. Vui lòng thử lại sau ít phút.",
    "overloaded": "Dịch vụ ngôn ngữ đang quá tải. Vui lòng thử lại sau ít phút.",
    "timeout": "Dịch vụ ngôn ngữ phản hồi quá lâu. Vui lòng thử lại.",
    "connection": "Không kết nối được tới dịch vụ ngôn ngữ. Vui lòng kiểm tra mạng và thử lại.",
    "bad_request": "Yêu cầu gửi tới dịch vụ ngôn ngữ không hợp lệ. Vui lòng thử lại với câu hỏi khác.",
    "unknown": "Đã xảy ra lỗi khi sinh câu trả lời. Vui lòng thử lại sau.",
}


class LLMError(Exception):
    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        self.message_vi = _MESSAGES_VI[code]
        super().__init__(f"{code}: {detail}")


class LLMJsonError(Exception):
    """The model returned text that is not valid for the requested schema."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
            self.cost_usd + other.cost_usd,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResult:
    text: str
    usage: Usage
    model: str
    stop_reason: str | None = None
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def _classify(exc: BaseException) -> tuple[ErrorCode, bool]:
    """Map an SDK exception to (code, retryable)."""
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return "auth", False
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit", True
    if isinstance(exc, anthropic.APITimeoutError):
        return "timeout", True
    if isinstance(exc, anthropic.APIConnectionError):
        return "connection", True
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status == 529:
            return "overloaded", True
        if status >= 500:
            return "unknown", True
        if status in (400, 404, 413, 422):
            return "bad_request", False
    return "unknown", False


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        cache: Cache | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._client = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value() or None,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,  # we implement our own backoff
        )
        self._sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

    # ---- helpers --------------------------------------------------------------------------
    def _usage(self, u: Any) -> Usage:
        inp = int(getattr(u, "input_tokens", 0) or 0)
        out = int(getattr(u, "output_tokens", 0) or 0)
        cr = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cc = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        cost = (
            inp + cr + cc
        ) / 1e6 * self.settings.price_input_per_m + out / 1e6 * self.settings.price_output_per_m
        return Usage(inp, out, cr, cc, cost)

    @staticmethod
    def _system_blocks(
        system: str | list[dict[str, Any]], cache_system: bool
    ) -> list[dict[str, Any]]:
        blocks = (
            [{"type": "text", "text": system}]
            if isinstance(system, str)
            else [dict(b) for b in system]
        )
        if cache_system and blocks:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def _cache_key(self, kind: str, **params: Any) -> str:
        raw = json.dumps({"kind": kind, **params}, sort_keys=True, ensure_ascii=False, default=str)
        return "llm:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _use_cache(self, use_cache: bool | None) -> bool:
        if self.cache is None:
            return False
        return self.settings.llm_cache_enabled if use_cache is None else use_cache

    async def _with_retry(self, fn: Any) -> Any:
        last: BaseException | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                async with self._sem:
                    return await fn()
            except Exception as exc:
                code, retryable = _classify(exc)
                last = exc
                if not retryable or attempt == MAX_ATTEMPTS - 1:
                    logger.warning("llm error (%s, attempt %d): %s", code, attempt + 1, exc)
                    raise LLMError(code, str(exc)) from exc
                delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                logger.info("llm %s, retrying in %ss (attempt %d)", code, delay, attempt + 1)
                await asyncio.sleep(delay)
        raise LLMError("unknown", str(last))  # pragma: no cover - loop always returns/raises

    @staticmethod
    def _text_of(message: Any) -> str:
        return "".join(
            getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text"
        )

    # ---- public API ------------------------------------------------------------------------
    async def complete(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
        cache_system: bool = True,
        effort: str | None = None,
        use_cache: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResult:
        params: dict[str, Any] = {
            "model": model,
            "system": self._system_blocks(system, cache_system),
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if effort:
            params["output_config"] = {"effort": effort}
        if extra:
            params.update(extra)
        key = self._cache_key("complete", **params)
        if self._use_cache(use_cache) and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                d = json.loads(hit.decode("utf-8"))
                return LLMResult(
                    d["text"], Usage(**d["usage"]), d["model"], d.get("stop_reason"), cached=True
                )

        msg = await self._with_retry(lambda: self._client.messages.create(**params))
        res = LLMResult(
            text=self._text_of(msg),
            usage=self._usage(msg.usage),
            model=str(getattr(msg, "model", model)),
            stop_reason=getattr(msg, "stop_reason", None),
        )
        if self._use_cache(use_cache) and self.cache is not None:
            payload = {
                "text": res.text,
                "usage": res.usage.__dict__,
                "model": res.model,
                "stop_reason": res.stop_reason,
            }
            self.cache.set(
                key, json.dumps(payload, ensure_ascii=False).encode("utf-8"), ttl=RESPONSE_CACHE_TTL
            )
        return res

    async def complete_json(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        schema: type[T],
        max_tokens: int,
        effort: str | None = None,
        use_cache: bool | None = None,
    ) -> tuple[T, Usage]:
        json_schema = _strict_schema(schema.model_json_schema())
        res = await self.complete(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            effort=effort,
            use_cache=use_cache,
            extra={
                "output_config": {
                    "format": {"type": "json_schema", "schema": json_schema},
                    **({"effort": effort} if effort else {}),
                }
            },
        )
        try:
            return schema.model_validate_json(res.text), res.usage
        except ValidationError as exc:
            raise LLMJsonError(f"invalid JSON for {schema.__name__}: {exc}") from exc

    async def stream(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
        cache_system: bool = True,
        effort: str | None = None,
    ) -> AsyncIterator[str | LLMResult]:
        params: dict[str, Any] = {
            "model": model,
            "system": self._system_blocks(system, cache_system),
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if effort:
            params["output_config"] = {"effort": effort}

        async def _open() -> Any:
            return self._client.messages.stream(**params)

        stream_cm = await self._with_retry(_open)
        async with stream_cm as stream:
            try:
                async for delta in stream.text_stream:
                    yield delta
                final = await stream.get_final_message()
            except Exception as exc:
                code, _ = _classify(exc)
                raise LLMError(code, str(exc)) from exc
        yield LLMResult(
            text=self._text_of(final),
            usage=self._usage(final.usage),
            model=str(getattr(final, "model", model)),
            stop_reason=getattr(final, "stop_reason", None),
        )


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic schema → structured-output schema: additionalProperties=false everywhere."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                props = node.get("properties", {})
                node["required"] = list(props.keys())
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(json.loads(json.dumps(schema)))


@lru_cache
def get_llm() -> LLMClient:
    from legalrag.cache import get_cache
    from legalrag.config import get_settings

    return LLMClient(get_settings(), cache=get_cache())
