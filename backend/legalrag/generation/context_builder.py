"""Build the [Nguồn #i] context block under a token budget (spec §3.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from legalrag.retrieval.types import Hit
from legalrag.tokens import TokenCounter

BUDGET_SAFETY = 1.1  # bge-m3 tokens ≈ Claude tokens × ~0.9; keep a margin
SNIPPET_CHARS = 240


@dataclass
class SourceRef:
    idx: int
    chunk_id: str
    doc_id: str
    display_source: str
    url: str | None
    snippet: str
    rerank_score: float | None
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "display_source": self.display_source,
            "url": self.url,
            "snippet": self.snippet,
            "rerank_score": self.rerank_score,
        }


@dataclass
class ContextBundle:
    text: str
    sources: list[SourceRef] = field(default_factory=list)
    dropped: int = 0


def display_source_of(payload: dict[str, Any]) -> str:
    title = str(payload.get("title") or "Văn bản pháp luật")
    article = payload.get("article")
    if not article:
        return f"{title} — đoạn {payload.get('seq', 0)}"
    head = (
        f"{article}. {payload['article_title']}" if payload.get("article_title") else str(article)
    )
    parts = [head]
    if payload.get("clause"):
        parts.append(str(payload["clause"]))
    if payload.get("point"):
        parts.append(str(payload["point"]))
    return f"{title} — {', '.join(parts)}"


def _body_text(hit: Hit) -> str:
    """Chunk text without the contextual header line (title › chapter › article)."""
    text = hit.text
    first, nl, rest = text.partition("\n")
    if nl and " › " in first or (nl and first == str(hit.payload.get("title"))):
        return rest.strip() or text
    return text


def _grouped(hits: list[Hit]) -> list[Hit]:
    """Order by rerank score desc, but keep chunks of the same document adjacent."""
    ordered = sorted(
        hits, key=lambda h: -(h.rerank_score if h.rerank_score is not None else h.score)
    )
    out: list[Hit] = []
    for h in ordered:
        doc = str(h.payload.get("doc_id"))
        art = str(h.payload.get("article"))
        same_article = [
            i
            for i, o in enumerate(out)
            if str(o.payload.get("doc_id")) == doc and str(o.payload.get("article")) == art
        ]
        same_doc = [i for i, o in enumerate(out) if str(o.payload.get("doc_id")) == doc]
        if same_article:
            out.insert(max(same_article) + 1, h)  # clauses of one article read as a block
        elif same_doc:
            out.insert(max(same_doc) + 1, h)
        else:
            out.append(h)
    return out


def build_context(hits: list[Hit], budget_tokens: int, counter: TokenCounter) -> ContextBundle:
    sources: list[SourceRef] = []
    blocks: list[str] = []
    used = 0
    dropped = 0
    for h in _grouped(hits):
        idx = len(sources) + 1
        display = display_source_of(h.payload)
        body = _body_text(h)
        block = f"[Nguồn #{idx}] {display}\n{body}"
        cost = int(counter.count(block) * BUDGET_SAFETY)
        if used + cost > budget_tokens:
            dropped += 1
            continue
        used += cost
        blocks.append(block)
        sources.append(
            SourceRef(
                idx=idx,
                chunk_id=h.chunk_id,
                doc_id=str(h.payload.get("doc_id", "")),
                display_source=display,
                url=h.payload.get("url"),
                snippet=body[:SNIPPET_CHARS],
                rerank_score=h.rerank_score,
                text=body,
            )
        )
    return ContextBundle(text="\n\n".join(blocks), sources=sources, dropped=dropped)
