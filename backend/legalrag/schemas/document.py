"""Domain model: normalized legal documents and chunks (spec §2.2)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class LegalDocument(BaseModel):
    doc_id: str
    source: Literal["yuitc", "tamir", "custom"]
    title: str
    doc_number: str | None = None
    doc_type: str | None = None
    issuing_body: str | None = None
    issued_date: date | None = None
    effective_status: Literal["unknown", "in_force", "amended", "expired"] = "unknown"
    url: str | None = None
    raw_text: str
    extra: dict[str, Any] = Field(default_factory=dict)


class LegalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    seq: int
    title: str = ""  # document title, denormalized for display/filters
    text: str
    text_for_bm25: str
    chapter: str | None = None
    chapter_title: str | None = None
    article: str | None = None
    article_title: str | None = None
    clause: str | None = None
    point: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    token_count: int = 0
    char_start: int | None = None
    char_end: int | None = None
    chunk_strategy: str
    embedding_model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def display_source(self, title: str) -> str:
        if not self.article:
            return f"{title} — đoạn {self.seq}"
        head = self.article
        if self.article_title:
            head = f"{self.article}. {self.article_title}"
        parts = [head]
        if self.clause:
            parts.append(self.clause)
        if self.point:
            parts.append(self.point)
        return f"{title} — {', '.join(parts)}"


def make_chunk_id(doc_id: str, seq: int, strategy: str) -> str:
    """``{doc_id}#{strategy}#{seq:05d}`` — the strategy label keeps ids unique across indexes."""
    return f"{doc_id}#{strategy}#{seq:05d}"
