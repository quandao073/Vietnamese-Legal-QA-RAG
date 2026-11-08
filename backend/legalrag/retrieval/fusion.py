"""Reciprocal Rank Fusion and soft title boosts (spec §3.3, §3.4)."""

from __future__ import annotations

from rapidfuzz import fuzz

from legalrag.retrieval.types import Hit

TITLE_BOOST = 0.1


def rrf(
    ranked_lists: dict[str, list[Hit]], k: int = 60, boosts: dict[str, float] | None = None
) -> list[Hit]:
    """score(c) = Σ_lists 1 / (k + rank_c) (+ boost). Keeps per-list ranks in ``debug``."""
    fused: dict[str, Hit] = {}
    for name, hits in ranked_lists.items():
        for rank, h in enumerate(hits, start=1):
            cur = fused.get(h.chunk_id)
            if cur is None:
                cur = Hit(chunk_id=h.chunk_id, score=0.0, payload=dict(h.payload), debug={})
                fused[h.chunk_id] = cur
            elif not cur.payload and h.payload:
                cur.payload = dict(h.payload)
            cur.score += 1.0 / (k + rank)
            cur.debug[f"{name}_rank"] = rank
            cur.debug[f"{name}_score"] = h.score
    for chunk_id, boost in (boosts or {}).items():
        if chunk_id in fused and boost:
            fused[chunk_id].score += boost
            fused[chunk_id].debug["boost"] = boost
    out = list(fused.values())
    for h in out:
        h.debug["rrf"] = h.score
    out.sort(key=lambda h: -h.score)  # stable: ties keep insertion order
    return out


def title_boost(hits: list[Hit], law_names: list[str], threshold: int = 85) -> dict[str, float]:
    """+0.1 for chunks whose document title fuzzy-matches one of the mentioned law names."""
    if not law_names:
        return {}
    out: dict[str, float] = {}
    for h in hits:
        title = str(h.payload.get("title") or "").lower()
        if not title:
            continue
        for name in law_names:
            if fuzz.partial_ratio(name.lower(), title) >= threshold:
                out[h.chunk_id] = TITLE_BOOST
                break
    return out
