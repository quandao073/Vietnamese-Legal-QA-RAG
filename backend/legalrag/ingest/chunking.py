"""Chunking strategies (spec §2.6): article, clause, fixed_{size}_{overlap}, parent_child."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from legalrag.ingest.normalize import normalize_text, strip_boilerplate, tokenize_for_bm25
from legalrag.ingest.structure import Node, iter_articles, iter_clauses, parse_structure
from legalrag.schemas.document import LegalChunk, LegalDocument, make_chunk_id
from legalrag.tokens import TokenCounter

STRATEGIES = ["article", "clause", "fixed_512_64", "fixed_256_32", "parent_child"]
PSEUDO_ARTICLE = "Điều (không rõ số)"
_FIXED_RE = re.compile(r"^fixed_(\d+)_(\d+)$")
_HEADING_MAX_LEN = 160
_WORD_RE = re.compile(r"\S+")


class Chunker(Protocol):
    name: str

    def chunk(self, doc: LegalDocument, tree: Node, text: str) -> list[LegalChunk]: ...


@dataclass
class _Ctx:
    """Per-document helper shared by the strategies."""

    doc: LegalDocument
    text: str
    counter: TokenCounter
    strategy: str
    seq: int = 0

    def source_ids(self, start: int | None, end: int | None) -> list[str]:
        spans: dict[str, list[int]] | None = self.doc.extra.get("span_map")
        if spans and start is not None and end is not None:
            ids = [pid for pid, (s, e) in spans.items() if s < end and e > start]
            if ids:
                return ids
        if "source_id" in self.doc.extra:
            return [str(self.doc.extra["source_id"])]
        return [self.doc.doc_id]

    def make(
        self,
        body: str,
        *,
        header: str = "",
        start: int | None,
        end: int | None,
        chapter: Node | None = None,
        article: Node | None = None,
        clause: Node | None = None,
        point: Node | None = None,
        article_label: str | None = None,
        article_title: str | None = None,
        extra: dict | None = None,
    ) -> LegalChunk:
        text = (header + body).strip()
        chunk = LegalChunk(
            chunk_id=make_chunk_id(self.doc.doc_id, self.seq),
            doc_id=self.doc.doc_id,
            seq=self.seq,
            title=self.doc.title,
            text=text,
            text_for_bm25=tokenize_for_bm25(text),
            chapter=chapter.label if chapter else None,
            chapter_title=chapter.title if chapter else None,
            article=article.label if article else article_label,
            article_title=article.title if article else article_title,
            clause=clause.label if clause else None,
            point=point.label if point else None,
            source_ids=self.source_ids(start, end),
            token_count=self.counter.count(text),
            char_start=start,
            char_end=end,
            chunk_strategy=self.strategy,
            extra=extra or {},
        )
        self.seq += 1
        return chunk


def prepare_text(doc: LegalDocument) -> str:
    return normalize_text(strip_boilerplate(doc.raw_text))


def build_chunks(doc: LegalDocument, chunker: Chunker) -> list[LegalChunk]:
    text = prepare_text(doc)
    return chunker.chunk(doc, parse_structure(text), text)


# ---- helpers ------------------------------------------------------------------------------


def _header(doc: LegalDocument, chapter: Node | None, article: Node | None) -> str:
    parts = [doc.title]
    if chapter is not None:
        parts.append(f"{chapter.label}. {chapter.title}" if chapter.title else str(chapter.label))
    if article is not None:
        parts.append(f"{article.label}. {article.title}" if article.title else str(article.label))
    return " › ".join(parts) + "\n"


def _body_after_heading(text: str, node: Node) -> tuple[str, int]:
    """Article body without its heading line; returns (body, body_start)."""
    seg = text[node.char_start : node.char_end]
    nl = seg.find("\n")
    if nl == -1:
        return "", node.char_end
    return seg[nl + 1 :], node.char_start + nl + 1


def _detect_heading(text: str) -> tuple[str | None, str, int]:
    """For documents without any `Điều`: (heading, body, body_start)."""
    first, nl, rest = text.partition("\n")
    first = first.strip()
    looks_like_heading = (
        bool(rest)
        and len(first) <= _HEADING_MAX_LEN
        and not re.match(r"^(\d{1,3}\.|Khoản\s+\d|[a-zđ]{1,2}\)|Chương\s)", first)
        and first[-1:] not in ".;:"
    )
    if looks_like_heading:
        return first, rest, len(first) + 1
    return None, text, 0


def _pack_clauses(
    ctx: _Ctx,
    header: str,
    chapter: Node | None,
    article: Node | None,
    clauses: list[Node],
    max_tokens: int,
) -> list[LegalChunk]:
    """Greedily pack consecutive clauses under max_tokens, repeating the header."""
    out: list[LegalChunk] = []
    group: list[Node] = []

    def flush() -> None:
        if not group:
            return
        start, end = group[0].char_start, group[-1].char_end
        out.append(
            ctx.make(
                ctx.text[start:end],
                header=header,
                start=start,
                end=end,
                chapter=chapter,
                article=article,
                clause=group[0],
                extra={"clause_range": [group[0].label, group[-1].label]}
                if len(group) > 1
                else None,
            )
        )
        group.clear()

    for cl in clauses:
        candidate = (
            header + ctx.text[(group[0].char_start if group else cl.char_start) : cl.char_end]
        )
        if group and ctx.counter.count(candidate) > max_tokens:
            flush()
        group.append(cl)
    flush()
    return out


# ---- strategies ---------------------------------------------------------------------------


class ArticleChunker:
    name = "article"

    def __init__(
        self, *, max_tokens: int, counter: TokenCounter, strategy: str | None = None
    ) -> None:
        self.max_tokens = max_tokens
        self.counter = counter
        self.strategy = strategy or self.name

    def chunk(self, doc: LegalDocument, tree: Node, text: str) -> list[LegalChunk]:
        ctx = _Ctx(doc, text, self.counter, self.strategy)
        return self.chunk_with(ctx, tree)

    def chunk_with(self, ctx: _Ctx, tree: Node) -> list[LegalChunk]:
        text = ctx.text
        articles = list(iter_articles(tree))
        if not articles:
            return self._headingless(ctx, tree)
        out: list[LegalChunk] = []
        for chapter, article in articles:
            header = _header(ctx.doc, chapter, article)
            body, body_start = _body_after_heading(text, article)
            clauses = iter_clauses(article)
            if self.counter.count(header + body) <= self.max_tokens or len(clauses) < 2:
                out.append(
                    ctx.make(
                        body,
                        header=header,
                        start=body_start,
                        end=article.char_end,
                        chapter=chapter,
                        article=article,
                    )
                )
            else:
                out.extend(_pack_clauses(ctx, header, chapter, article, clauses, self.max_tokens))
        return out

    def _headingless(self, ctx: _Ctx, tree: Node) -> list[LegalChunk]:
        heading, body, body_start = _detect_heading(ctx.text)
        label = PSEUDO_ARTICLE if heading else None
        header = f"{ctx.doc.title} › {label}. {heading}\n" if heading else f"{ctx.doc.title}\n"
        clauses = [c for c in tree.children if c.kind == "clause"]
        if self.counter.count(header + body) <= self.max_tokens or len(clauses) < 2:
            return [
                ctx.make(
                    body,
                    header=header,
                    start=body_start,
                    end=len(ctx.text),
                    article_label=label,
                    article_title=heading,
                )
            ]
        chunks = _pack_clauses(ctx, header, None, None, clauses, self.max_tokens)
        for c in chunks:
            c.article, c.article_title = label, heading
        return chunks


class ClauseChunker:
    name = "clause"

    def __init__(self, *, counter: TokenCounter, strategy: str | None = None) -> None:
        self.counter = counter
        self.strategy = strategy or self.name

    def chunk(self, doc: LegalDocument, tree: Node, text: str) -> list[LegalChunk]:
        ctx = _Ctx(doc, text, self.counter, self.strategy)
        return self.chunk_with(ctx, tree, parent_ids=None)

    def chunk_with(
        self, ctx: _Ctx, tree: Node, parent_ids: dict[str, str] | None
    ) -> list[LegalChunk]:
        text = ctx.text
        out: list[LegalChunk] = []
        articles = list(iter_articles(tree))
        if not articles:
            heading, body, body_start = _detect_heading(text)
            label = PSEUDO_ARTICLE if heading else None
            header = f"{ctx.doc.title} › {label}. {heading}\n" if heading else f"{ctx.doc.title}\n"
            clauses = [c for c in tree.children if c.kind == "clause"]
            if not clauses:
                return [
                    ctx.make(
                        body,
                        header=header,
                        start=body_start,
                        end=len(text),
                        article_label=label,
                        article_title=heading,
                    )
                ]
            for cl in clauses:
                out.append(
                    ctx.make(
                        text[cl.char_start : cl.char_end],
                        header=header,
                        start=cl.char_start,
                        end=cl.char_end,
                        clause=cl,
                        article_label=label,
                        article_title=heading,
                    )
                )
            return out
        for chapter, article in articles:
            header = _header(ctx.doc, chapter, article)
            extra = (
                {"role": "child", "parent_chunk_id": parent_ids[article.label or ""]}
                if parent_ids
                else None
            )
            clauses = iter_clauses(article)
            if not clauses:
                body, body_start = _body_after_heading(text, article)
                out.append(
                    ctx.make(
                        body,
                        header=header,
                        start=body_start,
                        end=article.char_end,
                        chapter=chapter,
                        article=article,
                        extra=dict(extra) if extra else None,
                    )
                )
                continue
            for cl in clauses:
                out.append(
                    ctx.make(
                        text[cl.char_start : cl.char_end],
                        header=header,
                        start=cl.char_start,
                        end=cl.char_end,
                        chapter=chapter,
                        article=article,
                        clause=cl,
                        extra=dict(extra) if extra else None,
                    )
                )
        return out


class FixedChunker:
    def __init__(
        self, *, size: int, overlap: int, counter: TokenCounter, strategy: str | None = None
    ) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap, self.counter = size, overlap, counter
        self.name = strategy or f"fixed_{size}_{overlap}"

    def chunk(self, doc: LegalDocument, tree: Node, text: str) -> list[LegalChunk]:
        ctx = _Ctx(doc, text, self.counter, self.name)
        words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        if not words:
            return []
        ratio = max(self.counter.count(text) / len(words), 0.5)
        win = max(8, int(self.size / ratio))
        step = max(1, win - int(self.overlap / ratio))
        header = f"{doc.title}\n"
        out: list[LegalChunk] = []
        i = 0
        while i < len(words):
            group = words[i : i + win]
            start, end = group[0][0], group[-1][1]
            out.append(ctx.make(text[start:end], header=header, start=start, end=end))
            if i + win >= len(words):
                break
            i += step
        return out


class ParentChildChunker:
    name = "parent_child"

    def __init__(self, *, counter: TokenCounter, strategy: str | None = None) -> None:
        self.counter = counter
        self.name = strategy or "parent_child"

    def chunk(self, doc: LegalDocument, tree: Node, text: str) -> list[LegalChunk]:
        ctx = _Ctx(doc, text, self.counter, self.name)
        parents = ArticleChunker(
            max_tokens=10**9, counter=self.counter, strategy=self.name
        ).chunk_with(ctx, tree)
        for p in parents:
            p.extra["role"] = "parent"
        parent_ids = {p.article or "": p.chunk_id for p in parents}
        children = ClauseChunker(counter=self.counter, strategy=self.name).chunk_with(
            ctx, tree, parent_ids
        )
        for c in children:
            c.extra.setdefault("role", "child")
            c.extra.setdefault(
                "parent_chunk_id", parent_ids.get(c.article or "", parents[0].chunk_id)
            )
        return parents + children


def get_chunker(name: str, *, counter: TokenCounter, max_tokens: int) -> Chunker:
    """Resolve a strategy label. A label may carry a tag after ``__`` (``article__v2``): the part
    before the tag selects the algorithm, the full label is stored as ``chunk_strategy``."""
    base = name.split("__", 1)[0]
    if base == "article":
        return ArticleChunker(max_tokens=max_tokens, counter=counter, strategy=name)
    if base == "clause":
        return ClauseChunker(counter=counter, strategy=name)
    if base == "parent_child":
        return ParentChildChunker(counter=counter, strategy=name)
    m = _FIXED_RE.match(base)
    if m:
        return FixedChunker(
            size=int(m.group(1)), overlap=int(m.group(2)), counter=counter, strategy=name
        )
    raise ValueError(f"unknown chunk strategy: {name}")
