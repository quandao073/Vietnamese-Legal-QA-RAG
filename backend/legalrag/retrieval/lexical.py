"""BM25 index with pyvi tokenization (spec §2.7, §3.3)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import bm25s
import numpy as np

from legalrag.ingest.normalize import tokenize_for_bm25
from legalrag.retrieval.types import Hit
from legalrag.schemas.document import LegalChunk

_PAYLOAD_KEYS = ("doc_id", "title", "article", "article_title", "clause", "point", "seq")


class BM25Index:
    def __init__(self, retriever: Any, ids: list[str], payloads: list[dict[str, Any]]) -> None:
        self._r = retriever
        self.ids = ids
        self.payloads = payloads

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, chunks: list[LegalChunk], out_dir: Path) -> BM25Index:
        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        corpus_tokens = [c.text_for_bm25.split() for c in chunks]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        ids = [c.chunk_id for c in chunks]
        payloads = [
            {k: getattr(c, k) for k in _PAYLOAD_KEYS if hasattr(c, k)} | {"text": c.text}
            for c in chunks
        ]
        retriever.save(str(out_dir / "bm25s"))
        (out_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "payloads.json").write_text(
            json.dumps(payloads, ensure_ascii=False), encoding="utf-8"
        )
        return cls(retriever, ids, payloads)

    @classmethod
    def load(cls, out_dir: Path) -> BM25Index:
        out_dir = Path(out_dir)
        retriever = bm25s.BM25.load(str(out_dir / "bm25s"), load_corpus=False)
        ids = json.loads((out_dir / "ids.json").read_text(encoding="utf-8"))
        payloads = json.loads((out_dir / "payloads.json").read_text(encoding="utf-8"))
        return cls(retriever, ids, payloads)

    def search(self, query: str, top_n: int) -> list[Hit]:
        tokens = tokenize_for_bm25(query).split()
        if not tokens or top_n <= 0 or not self.ids:
            return []
        k = min(top_n, len(self.ids))
        docs, scores = self._r.retrieve([tokens], k=k, show_progress=False)
        out: list[Hit] = []
        for idx, score in zip(
            np.asarray(docs[0]).tolist(), np.asarray(scores[0]).tolist(), strict=True
        ):
            if score <= 0:
                continue
            out.append(
                Hit(chunk_id=self.ids[idx], score=float(score), payload=dict(self.payloads[idx]))
            )
        return out
