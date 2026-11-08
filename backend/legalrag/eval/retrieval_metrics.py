"""Retrieval metrics (spec §5.2): Hit@K, Recall@K, MRR@10, nDCG@10 with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Iterable

KS: tuple[int, ...] = (1, 3, 5, 10)


Gold = set[str] | list[set[str]]


def _groups(gold: Gold) -> list[set[str]]:
    """Normalize gold to passage groups: a plain set means one group per chunk id."""
    if isinstance(gold, set):
        return [{c} for c in gold]
    return [set(g) for g in gold if g]


def _relevance(ranked: list[str], gold: Gold, k: int) -> list[int]:
    """Binary relevance per rank: 1 the first time a chunk of a not-yet-seen gold group appears."""
    groups = _groups(gold)
    seen: set[int] = set()
    rel: list[int] = []
    for c in ranked[:k]:
        hit = 0
        for gi, g in enumerate(groups):
            if c in g and gi not in seen:
                seen.add(gi)
                hit = 1
        rel.append(hit)
    return rel


def hit_at_k(ranked: list[str], gold: Gold, k: int) -> float:
    return 1.0 if any(_relevance(ranked, gold, k)) else 0.0


def recall_at_k(ranked: list[str], gold: Gold, k: int) -> float:
    groups = _groups(gold)
    if not groups:
        return 0.0
    return sum(_relevance(ranked, gold, k)) / len(groups)


def mrr(ranked: list[str], gold: Gold, k: int = 10) -> float:
    for i, r in enumerate(_relevance(ranked, gold, k), start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg(ranked: list[str], gold: Gold, k: int = 10) -> float:
    groups = _groups(gold)
    if not groups or not ranked:
        return 0.0
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(_relevance(ranked, gold, k), start=1))
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(groups), k) + 1))
    return dcg / ideal if ideal else 0.0


def per_query_metrics(ranked: list[str], gold: Gold) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in KS:
        out[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
    out["mrr@10"] = mrr(ranked, gold, 10)
    out["ndcg@10"] = ndcg(ranked, gold, 10)
    return out


def aggregate(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if isinstance(rows[0][k], int | float)]
    agg: dict[str, float] = {
        k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 4) for k in keys
    }
    agg["n"] = len(rows)
    return agg
