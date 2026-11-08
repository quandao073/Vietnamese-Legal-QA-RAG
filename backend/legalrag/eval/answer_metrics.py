"""Lexical answer metrics (spec §5.3): EM, token-F1, ROUGE-L on the conclusion and full answer."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from legalrag.generation.citation import CITATION_RE
from legalrag.generation.prompts import DISCLAIMER
from legalrag.ingest.normalize import normalize_text

_PUNCT_RE = re.compile(r"[^\w\s%/-]", re.UNICODE)
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*)?\s*kết luận\s*(?:\*\*)?\s*[:：]?\s*", re.IGNORECASE
)
_NEXT_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*]+\*\*\s*[:：]?)", re.IGNORECASE)


def normalize_answer(s: str) -> str:
    s = normalize_text(unicodedata.normalize("NFC", s)).lower()
    s = CITATION_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".").strip()


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    prev = [0] * (len(g) + 1)
    for tok in p:
        cur = [0] * (len(g) + 1)
        for j, gt in enumerate(g, start=1):
            cur[j] = prev[j - 1] + 1 if tok == gt else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(g)
    return 2 * precision * recall / (precision + recall)


def extract_conclusion(answer_md: str) -> str:
    """Text of the 'Kết luận' section (until the next heading); else the first paragraph."""
    text = answer_md.replace(DISCLAIMER, "").strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and "kết luận" in line.lower():
            rest = line[m.end() :].strip()
            body = [rest] if rest else []
            for nxt in lines[i + 1 :]:
                if _NEXT_HEADING_RE.match(nxt):
                    break
                body.append(nxt.strip())
            return " ".join(x for x in body if x).strip()
    first = text.split("\n\n", 1)[0]
    return first.strip()


def lexical_scores(answer_md: str, gold: str) -> dict[str, float]:
    conclusion = extract_conclusion(answer_md)
    full = answer_md.replace(DISCLAIMER, "")
    return {
        "em": exact_match(full, gold),
        "f1": round(token_f1(full, gold), 4),
        "rouge_l": round(rouge_l(full, gold), 4),
        "em_conclusion": exact_match(conclusion, gold),
        "f1_conclusion": round(token_f1(conclusion, gold), 4),
        "rouge_l_conclusion": round(rouge_l(conclusion, gold), 4),
    }
