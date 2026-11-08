"""Loader for ``YuITC/Vietnamese-Legal-Documents`` (corpus.parquet + test split)."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from legalrag.ingest.loaders.base import BaseLoader, hf_download
from legalrag.schemas.document import LegalDocument
from legalrag.schemas.eval import EvalPair

REPO = "YuITC/Vietnamese-Legal-Documents"
_LAW_LINE_RE = re.compile(
    r"^(Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Pháp lệnh|Hiến pháp"
    r"|Văn bản hợp nhất)\b",
    re.IGNORECASE,
)
_DOC_TYPE_RE = re.compile(
    r"^(Luật|Bộ luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Nghị quyết|Pháp lệnh"
    r"|Hiến pháp)",
    re.IGNORECASE,
)
_DOC_NUMBER_RE = re.compile(
    r"\b(\d{1,4}/\d{4}/[A-ZĐ]{1,6}(?:-[A-ZĐ]{1,6})*\d*|\d{1,4}/[A-ZĐ]{1,6}-[A-ZĐ]{1,6})"
)
_MAX_TITLE_LEN = 160


class YuitcLoader(BaseLoader):
    name = "yuitc"

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        *,
        max_docs: int = 20000,
        eval_sample: int = 500,
        seed: int = 42,
    ) -> None:
        super().__init__(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.max_docs = max_docs
        self.eval_sample = eval_sample
        self.seed = seed
        self._selected: list[int] | None = None
        self._eval_rows: pd.DataFrame | None = None

    def download(self) -> None:  # pragma: no cover - network
        hf_download(REPO, "corpus.parquet", self.raw_dir / "corpus.parquet")
        hf_download(REPO, "test.parquet", self.raw_dir / "test.parquet")

    # ---- heuristics -----------------------------------------------------------------------
    @staticmethod
    def split_title(text: str) -> tuple[str | None, str]:
        """Return (title, body): the first line is a title only if it names a legal document."""
        text = text.lstrip('"“” ').rstrip()
        first, _, rest = text.partition("\n")
        first = first.strip()
        if rest and len(first) <= _MAX_TITLE_LEN and _LAW_LINE_RE.match(first):
            return first, rest.lstrip()
        return None, text

    @staticmethod
    def select_cids(
        corpus_cids: list[int], needed: set[int], max_docs: int, seed: int
    ) -> list[int]:
        """Deterministic selection: every needed cid first, then random fill up to max_docs."""
        if max_docs <= 0 or max_docs >= len(corpus_cids):
            return list(corpus_cids)
        chosen = [c for c in corpus_cids if c in needed]
        if len(chosen) >= max_docs:
            return chosen[:max_docs]
        rest = [c for c in corpus_cids if c not in needed]
        random.Random(seed).shuffle(rest)
        return chosen + rest[: max_docs - len(chosen)]

    # ---- data access ---------------------------------------------------------------------
    def _eval_sample(self) -> pd.DataFrame:
        if self._eval_rows is None:
            test = pd.read_parquet(self.raw_dir / "test.parquet")
            n = min(self.eval_sample, len(test)) if self.eval_sample > 0 else len(test)
            self._eval_rows = test.sample(n=n, random_state=self.seed).reset_index(drop=True)
        return self._eval_rows

    def selected_cids(self) -> list[int]:
        if self._selected is None:
            corpus = pd.read_parquet(self.raw_dir / "corpus.parquet", columns=["cid"])
            cids = [int(c) for c in corpus["cid"].tolist()]
            needed = {int(c) for row in self._eval_sample()["cid"].tolist() for c in row}
            self._selected = self.select_cids(cids, needed, self.max_docs, self.seed)
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            (self.processed_dir / "yuitc_selected_cids.json").write_text(
                json.dumps(self._selected), encoding="utf-8"
            )
        return self._selected

    def iter_documents(self) -> Iterator[LegalDocument]:
        selected = set(self.selected_cids())
        corpus = pd.read_parquet(self.raw_dir / "corpus.parquet")
        corpus = corpus[corpus["cid"].isin(selected)]
        records: list[dict[Any, Any]] = corpus.to_dict("records")
        for row in records:
            cid = int(row["cid"])
            title, body = self.split_title(str(row["text"]))
            doc_number = doc_type = None
            if title:
                m = _DOC_NUMBER_RE.search(title)
                doc_number = m.group(1) if m else None
                mt = _DOC_TYPE_RE.match(title)
                doc_type = mt.group(1) if mt else None
            yield LegalDocument(
                doc_id=f"yuitc:cid:{cid}",
                source="yuitc",
                title=title or f"Văn bản pháp luật (cid {cid})",
                doc_number=doc_number,
                doc_type=doc_type,
                raw_text=body,
                extra={"cid": cid, "source_id": f"cid:{cid}"},
            )

    def iter_eval_pairs(self) -> Iterator[EvalPair]:
        records: list[dict[Any, Any]] = self._eval_sample().to_dict("records")
        for row in records:
            yield EvalPair(
                id=f"yuitc_test:{int(row['qid'])}",
                question=str(row["question"]),
                gold_source_ids=[f"cid:{int(c)}" for c in row["cid"]],
                gold_answer=None,
                category="legal",
                split="test",
            )
