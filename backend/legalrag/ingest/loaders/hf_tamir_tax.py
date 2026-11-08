"""Loader for ``Tamir39/vietnam-tax-qa`` (knowledge_base + qa). See ADR-004."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path

from legalrag.ingest.loaders.base import BaseLoader, hf_download
from legalrag.schemas.document import LegalDocument
from legalrag.schemas.eval import EvalPair

REPO = "Tamir39/vietnam-tax-qa"

# "Chương I - Title. Điều 5 - Title. Khoản 26, điểm a: body"
_PASSAGE_RE = re.compile(
    r"^(?:(?P<chapter>Chương\s+[IVXLCDM]+)\s*-\s*(?P<chapter_title>[^.]*?)\.\s*)?"
    r"(?P<article>Điều\s+\d+[a-z]?)\s*-\s*(?P<article_title>[^.:]*?)"
    r"(?:\.\s*Khoản\s+(?P<clause>\d+)(?P<clause_rest>[^:]*))?"
    r"\s*:\s*(?P<body>.*)$",
    re.DOTALL,
)
_POINT_RE = re.compile(r"điểm\s+([a-zđ]{1,2})\b")
_DOC_NUMBER_RE = re.compile(r"(\d{1,4}/(?:\d{4}/)?[A-ZĐ]{1,6}(?:-[A-ZĐ]{1,6})*\d*)")
_DOC_TYPES = ("VBHN", "Luật", "Nghị định", "Thông tư", "Quyết định", "Nghị quyết")


class TamirLoader(BaseLoader):
    name = "tamir"

    def download(self) -> None:  # pragma: no cover - network
        hf_download(REPO, "knowledge_base/knowledge_base.csv", self.kb_path)
        hf_download(REPO, "qa/train_qa.jsonl", self.raw_dir / "qa" / "train_qa.jsonl")
        hf_download(REPO, "qa/test_qa.jsonl", self.raw_dir / "qa" / "test_qa.jsonl")

    @property
    def kb_path(self) -> Path:
        return self.raw_dir / "knowledge_base" / "knowledge_base.csv"

    # ---- parsing -------------------------------------------------------------------------
    @staticmethod
    def parse_passage(text: str) -> dict[str, str | None]:
        m = _PASSAGE_RE.match(text.strip())
        if not m:
            return {
                "chapter": None,
                "chapter_title": None,
                "article": None,
                "article_title": None,
                "clause": None,
                "point": None,
                "body": text.strip(),
            }
        point = None
        if m.group("clause_rest"):
            pm = _POINT_RE.search(m.group("clause_rest"))
            point = pm.group(1) if pm else None
        return {
            "chapter": m.group("chapter"),
            "chapter_title": (m.group("chapter_title") or "").strip() or None,
            "article": m.group("article"),
            "article_title": (m.group("article_title") or "").strip() or None,
            "clause": m.group("clause"),
            "point": point,
            "body": m.group("body").strip(),
        }

    @staticmethod
    def passages_to_raw_text(rows: list[dict]) -> tuple[str, dict[str, tuple[int, int]]]:
        """Rebuild line-based legal text; return it with passage_id → body span."""
        lines: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        pos = 0
        cur_chapter = cur_article = cur_clause = None

        def emit(line: str) -> int:
            nonlocal pos
            start = pos
            lines.append(line)
            pos += len(line) + 1
            return start

        for r in rows:
            p = TamirLoader.parse_passage(r["passage_text"])
            if p["chapter"] and p["chapter"] != cur_chapter:
                emit(
                    f"{p['chapter']}. {p['chapter_title']}"
                    if p["chapter_title"]
                    else f"{p['chapter']}."
                )
                cur_chapter, cur_article, cur_clause = p["chapter"], None, None
            if p["article"] and p["article"] != cur_article:
                emit(
                    f"{p['article']}. {p['article_title']}"
                    if p["article_title"]
                    else f"{p['article']}."
                )
                cur_article, cur_clause = p["article"], None
            body = p["body"] or ""
            if p["clause"]:
                if p["point"]:
                    if p["clause"] != cur_clause:
                        emit(f"{p['clause']}.")
                        cur_clause = p["clause"]
                    start = emit(f"{p['point']}) {body}") + len(p["point"]) + 2
                else:
                    cur_clause = p["clause"]
                    start = emit(f"{p['clause']}. {body}") + len(p["clause"]) + 2
            else:
                start = emit(body)
            spans[r["passage_id"]] = (start, start + len(body))
        return "\n".join(lines), spans

    @staticmethod
    def parse_title(title: str) -> tuple[str | None, str | None]:
        doc_type = next((t for t in _DOC_TYPES if t.lower() in title.lower()), None)
        m = _DOC_NUMBER_RE.search(title)
        return (m.group(1) if m else None), doc_type

    # ---- loader API ---------------------------------------------------------------------
    def _rows(self) -> list[dict]:
        with open(self.kb_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: r["passage_id"])
        return rows

    def iter_documents(self) -> Iterator[LegalDocument]:
        by_doc: dict[str, list[dict]] = {}
        for r in self._rows():
            by_doc.setdefault(r["doc_id"], []).append(r)
        for doc_id, rows in by_doc.items():
            text, spans = self.passages_to_raw_text(rows)
            doc_number, doc_type = self.parse_title(rows[0]["title"])
            yield LegalDocument(
                doc_id=f"tamir:{doc_id}",
                source="tamir",
                title=rows[0]["title"],
                doc_number=doc_number,
                doc_type=doc_type,
                issuing_body="Quốc hội" if doc_type in ("Luật", "VBHN") else None,
                url=rows[0]["url"] or None,
                raw_text=text,
                extra={"span_map": spans, "n_passages": len(rows)},
            )

    def iter_eval_pairs(self) -> Iterator[EvalPair]:
        for split in ("train", "test"):
            path = self.raw_dir / "qa" / f"{split}_qa.jsonl"
            if not path.exists():
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                q = json.loads(line)
                yield EvalPair(
                    id=f"tamir_{split}:{i}",
                    question=q["question"],
                    gold_source_ids=[q["passage_id"]] if q.get("passage_id") else [],
                    gold_answer=q.get("answer"),
                    category="legal",
                    split=split,
                )
