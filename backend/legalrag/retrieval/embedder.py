"""Embedding models (spec §3.2): sentence-transformers backend, fake backend, query cache."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from legalrag.cache import Cache

logger = logging.getLogger(__name__)
_QUERY_CACHE_TTL = 7 * 24 * 3600


class EmbedderProtocol(Protocol):
    model_name: str

    @property
    def dim(self) -> int: ...

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", *, max_seq_length: int = 1024) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device, **kwargs)
        self._model.max_seq_length = max_seq_length
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self.device = device

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(out, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_passages([text], batch_size=1)[0]


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder for tests (no model download)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model_name = f"fake-{dim}"
        self.query_calls = 0
        self.passage_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        return _l2(v) if np.any(v) else v

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.passage_calls += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return self._vec(text)


class CachedQueryEmbedder:
    """Wraps an embedder; caches query vectors (key = sha1(model + text), TTL 7 days)."""

    def __init__(
        self,
        inner: EmbedderProtocol,
        cache: Cache | None,
        *,
        model_name: str | None = None,
        ttl_seconds: int = _QUERY_CACHE_TTL,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.model_name = model_name or inner.model_name
        self.ttl = ttl_seconds

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.inner.encode_passages(texts, batch_size=batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        key = "emb:" + hashlib.sha1(f"{self.model_name}\x00{text}".encode()).hexdigest()
        if self.cache is not None:
            raw = self.cache.get(key)
            if raw is not None:
                return np.frombuffer(raw, dtype=np.float32).copy()
        vec = np.asarray(self.inner.encode_query(text), dtype=np.float32)
        if self.cache is not None:
            self.cache.set(key, vec.tobytes(), ttl=self.ttl)
        return vec


_lock = threading.Lock()


@lru_cache
def get_embedder() -> CachedQueryEmbedder:
    """Process-wide singleton (lazy, thread-safe)."""
    from legalrag.cache import get_cache
    from legalrag.config import get_settings

    s = get_settings()
    with _lock:
        logger.info("loading embedder %s on %s", s.embedding_model, s.device)
        inner = SentenceTransformerEmbedder(s.embedding_model, s.device)
        return CachedQueryEmbedder(inner, get_cache())
