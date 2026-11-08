import pytest

from legalrag.retrieval.fusion import rrf, title_boost
from legalrag.retrieval.types import Hit


def _hits(ids: list[str], **payload) -> list[Hit]:
    return [Hit(chunk_id=i, score=1.0 / (n + 1), payload=dict(payload)) for n, i in enumerate(ids)]


def test_rrf_hand_example():
    fused = rrf({"dense": _hits(["a", "b", "c"]), "bm25": _hits(["b", "a", "d"])}, k=60)
    scores = {h.chunk_id: h.score for h in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63) and scores["d"] == pytest.approx(1 / 63)
    assert [h.chunk_id for h in fused][:2] == ["a", "b"]  # ties broken by first appearance
    a = next(h for h in fused if h.chunk_id == "a")
    assert (
        a.debug["dense_rank"] == 1
        and a.debug["bm25_rank"] == 2
        and a.debug["rrf"] == pytest.approx(scores["a"])
    )


def test_rrf_keeps_payload_and_applies_boosts():
    fused = rrf(
        {"dense": _hits(["a"], doc_id="d1"), "bm25": _hits(["b"], doc_id="d2")},
        k=60,
        boosts={"b": 0.1},
    )
    by = {h.chunk_id: h for h in fused}
    assert by["a"].payload == {"doc_id": "d1"}
    assert by["b"].score == pytest.approx(1 / 61 + 0.1) and by["b"].debug["boost"] == 0.1
    assert fused[0].chunk_id == "b"


def test_rrf_empty():
    assert rrf({}, k=60) == [] and rrf({"dense": []}, k=60) == []


def test_title_boost_fuzzy_match():
    hits = [
        Hit("x", 1.0, {"title": "VBHN 103/VBHN-VPQH năm 2025 - Luật Thuế thu nhập cá nhân"}),
        Hit("y", 1.0, {"title": "Luật Thuế giá trị gia tăng"}),
        Hit("z", 1.0, {}),
    ]
    boosts = title_boost(hits, ["Luật Thuế thu nhập cá nhân"], threshold=85)
    assert boosts == {"x": 0.1}
    assert title_boost(hits, [], threshold=85) == {}
    assert (
        title_boost(hits, ["luật thuế tncn"], threshold=85) == {}
    )  # abbreviation does not fuzzy-match
