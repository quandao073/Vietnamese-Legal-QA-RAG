"""Cross-encoder reranking (spec §3.5, §6.3)."""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from typing import Protocol

from legalrag.retrieval.types import Hit

logger = logging.getLogger(__name__)
RERANK_MAX_LENGTH = 512
RERANK_BATCH = 16


class RerankerProtocol(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        kwargs: dict = {}
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        self._m = CrossEncoder(model_name, device=device, max_length=RERANK_MAX_LENGTH, **kwargs)
        self.model_name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._m.predict(
            [(query, t) for t in texts], batch_size=RERANK_BATCH, show_progress_bar=False
        )
        # CrossEncoder.predict already applies the model's sigmoid → probabilities in [0, 1]
        return [min(1.0, max(0.0, float(x))) for x in raw]


class FakeReranker:
    """Token-overlap scorer for tests: |q ∩ t| / |q| mapped into (0, 1)."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(re.findall(r"\w+", query.lower()))
        out = []
        for t in texts:
            words = set(re.findall(r"\w+", t.lower()))
            overlap = len(q & words) / max(1, len(q))
            out.append(0.05 + 0.9 * overlap)
        return out


def rerank(query: str, hits: list[Hit], reranker: RerankerProtocol, top_k: int) -> list[Hit]:
    if not hits:
        return []
    scores = reranker.score(query, [h.text for h in hits])
    for rank, (h, s) in enumerate(zip(hits, scores, strict=True), start=1):
        h.debug["rerank_score"] = s
        h.debug["pre_rerank_rank"] = rank
    ordered = sorted(hits, key=lambda h: -(h.rerank_score or 0.0))
    return ordered[:top_k]


_lock = threading.Lock()


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    from legalrag.config import get_settings

    s = get_settings()
    with _lock:
        logger.info("loading reranker %s on %s", s.reranker_model, s.device)
        return CrossEncoderReranker(s.reranker_model, s.device)
