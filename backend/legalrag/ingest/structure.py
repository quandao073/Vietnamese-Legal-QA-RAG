"""Regex-based legal structure parser: Document → Chapter* → Article* → Clause* → Point*."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

CHAPTER_RE = re.compile(r"^(Chương\s+[IVXLCDM]+)\s*[\.:]?\s*(.*)$")
ARTICLE_RE = re.compile(r"^(Điều\s+\d+[a-z]?)\s*[\.:]?\s*(.*)$")
CLAUSE_RE = re.compile(r"^(?:Khoản\s+)?(\d{1,3})\.(?:\s+(.*))?$")
POINT_RE = re.compile(r"^([a-zđ]{1,2})\)\s+(.*)$")

Kind = Literal["document", "chapter", "article", "clause", "point"]
_RANK: dict[Kind, int] = {"document": 0, "chapter": 1, "article": 2, "clause": 3, "point": 4}


@dataclass
class Node:
    kind: Kind
    label: str | None
    title: str | None
    char_start: int
    char_end: int
    children: list[Node] = field(default_factory=list)
    preamble: str | None = None  # document only: text before the first marker

    def text(self, source: str) -> str:
        return source[self.char_start : self.char_end]


def _match(line: str) -> tuple[Kind, str, str] | None:
    m = CHAPTER_RE.match(line)
    if m:
        return "chapter", m.group(1), m.group(2).strip()
    m = ARTICLE_RE.match(line)
    if m:
        return "article", m.group(1), m.group(2).strip()
    m = CLAUSE_RE.match(line)
    if m:
        return "clause", f"Khoản {m.group(1)}", (m.group(2) or "").strip()
    m = POINT_RE.match(line)
    if m:
        return "point", f"Điểm {m.group(1)}", m.group(2).strip()
    return None


def parse_structure(text: str) -> Node:
    """Parse normalized text. Lines that match no marker extend the current deepest node."""
    root = Node("document", None, None, 0, len(text))
    stack: list[Node] = [root]
    pos = 0
    preamble_parts: list[str] = []

    for raw_line in text.split("\n"):
        start, end = pos, pos + len(raw_line)
        pos = end + 1
        line = raw_line.strip()
        if not line:
            continue
        matched = _match(line)
        if matched is None:
            if len(stack) == 1:
                preamble_parts.append(line)
            continue  # continuation line: parent's char_end is extended lazily below
        kind, label, title = matched
        # close every node with rank >= new node's rank
        while _RANK[stack[-1].kind] >= _RANK[kind]:
            closed = stack.pop()
            closed.char_end = _trim_end(text, closed.char_start, start)
        node = Node(kind, label, title or None, start, end)
        stack[-1].children.append(node)
        stack.append(node)

    while len(stack) > 1:
        closed = stack.pop()
        closed.char_end = _trim_end(text, closed.char_start, len(text))

    root.preamble = "\n".join(preamble_parts) or None
    return root


def _trim_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in "\n ":
        end -= 1
    return end


def iter_articles(root: Node) -> Iterator[tuple[Node | None, Node]]:
    """Yield (chapter_or_None, article) in document order."""
    for child in root.children:
        if child.kind == "chapter":
            for art in child.children:
                if art.kind == "article":
                    yield child, art
        elif child.kind == "article":
            yield None, child


def iter_clauses(article: Node) -> list[Node]:
    return [c for c in article.children if c.kind == "clause"]
