import pytest

from legalrag.retrieval.reranker import FakeReranker, rerank
from legalrag.retrieval.types import Hit


def test_fake_reranker_scores_overlap_between_0_and_1():
    r = FakeReranker()
    scores = r.score("thuế thu nhập cá nhân", ["thuế thu nhập cá nhân là gì", "hôn nhân gia đình"])
    assert 0 <= scores[1] < scores[0] <= 1


def test_rerank_sorts_truncates_and_records_scores():
    hits = [
        Hit("a", 0.9, {"text": "hôn nhân và gia đình"}),
        Hit("b", 0.5, {"text": "thu nhập được miễn thuế thu nhập cá nhân"}),
        Hit("c", 0.1, {"text": "thuế thu nhập cá nhân đối với cá nhân cư trú"}),
    ]
    out = rerank("thu nhập miễn thuế thu nhập cá nhân", hits, FakeReranker(), top_k=2)
    assert [h.chunk_id for h in out] == ["b", "c"]
    assert all(h.rerank_score is not None for h in out)
    assert out[0].rerank_score >= out[1].rerank_score
    assert out[0].debug["pre_rerank_rank"] == 2


def test_rerank_empty():
    assert rerank("q", [], FakeReranker(), top_k=5) == []


def test_cross_encoder_scores_are_probabilities_not_double_squashed():
    from legalrag.retrieval.reranker import CrossEncoderReranker

    class FakeModel:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [0.9, 1.6e-05]  # sentence-transformers already applies the sigmoid

    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r._m = FakeModel()
    r.model_name = "fake"
    scores = r.score("q", ["a", "b"])
    assert scores[0] == pytest.approx(0.9) and scores[1] < 0.001  # not 0.71 / 0.5
