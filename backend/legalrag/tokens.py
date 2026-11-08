"""Token counting with the embedding model's tokenizer (bge-m3) and a heuristic fallback."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_WORD_RATIO = 1.3  # bge-m3 tokens per whitespace word for Vietnamese (empirical)


class TokenCounter:
    def __init__(self, model_name: str | None, *, allow_download: bool = True) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self._tok: Any = None
        self._lock = threading.Lock()
        self.backend = "heuristic" if model_name is None else "pending"

    def _ensure(self) -> None:
        if self.backend != "pending":
            return
        with self._lock:
            if self.backend != "pending":
                return
            try:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=not self.allow_download
                )
                self.backend = "hf"
            except Exception as exc:  # pragma: no cover - depends on network/cache
                logger.warning(
                    "tokenizer %s unavailable (%s); using heuristic", self.model_name, exc
                )
                self.backend = "heuristic"

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        self._ensure()
        if self.backend == "hf":
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, round(len(text.split()) * _WORD_RATIO))


@lru_cache
def get_token_counter(model_name: str | None = None) -> TokenCounter:
    if model_name is None:
        from legalrag.config import get_settings

        model_name = get_settings().embedding_model
    return TokenCounter(model_name)
