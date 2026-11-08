"""Byte cache abstraction: Redis in production, in-memory for tests / when Redis is down."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)


class Cache(Protocol):
    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes, ttl: int) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self._d: dict[str, tuple[float, bytes]] = {}

    def get(self, key: str) -> bytes | None:
        item = self._d.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.time():
            del self._d[key]
            return None
        return value

    def set(self, key: str, value: bytes, ttl: int) -> None:
        self._d[key] = (time.time() + ttl, value)


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=2)

    def get(self, key: str) -> bytes | None:
        try:
            val = self._r.get(key)
        except Exception as exc:  # pragma: no cover - infra
            logger.warning("redis get failed: %s", exc)
            return None
        return val if isinstance(val, bytes) else None

    def set(self, key: str, value: bytes, ttl: int) -> None:
        try:
            self._r.set(key, value, ex=ttl)
        except Exception as exc:  # pragma: no cover - infra
            logger.warning("redis set failed: %s", exc)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


@lru_cache
def get_cache() -> Cache:
    from legalrag.config import get_settings

    cache = RedisCache(get_settings().redis_url)
    if cache.ping():
        return cache
    logger.warning("Redis unreachable; falling back to in-process cache")
    return MemoryCache()
