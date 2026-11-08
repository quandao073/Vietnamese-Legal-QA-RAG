"""Retrieval metrics (spec §5.2): Hit@K, Recall@K, MRR@10, nDCG@10 with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Iterable

KS: tuple[int, ...] = (1, 3, 5, 10)


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(c in gold for c in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k], start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def ndcg(ranked: list[str], gold: set[str], k: int = 10) -> float:
    if not gold or not ranked:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], start=1) if c in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def per_query_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
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
