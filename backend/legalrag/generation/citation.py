"""Citation parsing and lightweight verification (spec §3.10)."""

from __future__ import annotations

import re

from legalrag.generation.context_builder import SourceRef
from legalrag.generation.prompts import DISCLAIMER
from legalrag.schemas.chat import CitationReport

CITATION_RE = re.compile(r"\[#?(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
WEAK_THRESHOLD = 0.25


def parse_citations(text: str) -> list[int]:
    seen: list[int] = []
    for m in CITATION_RE.finditer(text):
        i = int(m.group(1))
        if i not in seen:
            seen.append(i)
    return sorted(seen)


def clean_invalid(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove citations whose index is not in ``valid``; normalize ``[3]`` → ``[#3]``."""
    invalid: list[int] = []

    def repl(m: re.Match[str]) -> str:
        i = int(m.group(1))
        if i in valid:
            return f"[#{i}]"
        if i not in invalid:
            invalid.append(i)
        return ""

    out = CITATION_RE.sub(repl, text)
    out = re.sub(r" +([.,;:!?])", r"\1", out)  # "B ." -> "B."
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip(), invalid


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def rouge_l_recall(reference: str, candidate: str) -> float:
    """LCS(candidate, reference) / |reference| on word tokens.

    ``reference`` is the cited sentence, ``candidate`` the chunk text.
    """
    ref, cand = _words(reference), _words(candidate)
    if not ref or not cand:
        return 0.0
    prev = [0] * (len(cand) + 1)
    for r in ref:
        cur = [0] * (len(cand) + 1)
        for j, c in enumerate(cand, start=1):
            cur[j] = prev[j - 1] + 1 if r == c else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(ref)


def verify_citations(answer: str, sources: list[SourceRef]) -> CitationReport:
    """Per-sentence support check: a citation is *weak* if any sentence citing it has
    ROUGE-L recall < 0.25 against the cited chunk. Invalid indices are reported, not removed."""
    by_idx = {s.idx: s for s in sources}
    cited: set[int] = set()
    weak: set[int] = set()
    invalid: list[int] = []
    for sentence in _SENTENCE_RE.split(answer):
        idxs = [int(m) for m in CITATION_RE.findall(sentence)]
        if not idxs:
            continue
        claim = CITATION_RE.sub("", sentence)
        for i in idxs:
            src = by_idx.get(i)
            if src is None:
                if i not in invalid:
                    invalid.append(i)
                continue
            cited.add(i)
            if rouge_l_recall(claim, src.text) < WEAK_THRESHOLD:
                weak.add(i)
    return CitationReport(verified=sorted(cited), weak=sorted(weak), invalid=invalid)


def append_disclaimer(text: str) -> str:
    if DISCLAIMER in text:
        return text
    return text.rstrip() + "\n\n" + DISCLAIMER
