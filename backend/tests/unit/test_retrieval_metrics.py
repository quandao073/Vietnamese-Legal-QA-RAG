import math

import pytest

from legalrag.eval.retrieval_metrics import (
    KS,
    aggregate,
    hit_at_k,
    mrr,
    ndcg,
    per_query_metrics,
    recall_at_k,
)


def test_hand_example_single_gold():
    ranked, gold = ["b", "a", "c"], {"a"}
    assert hit_at_k(ranked, gold, 1) == 0.0 and hit_at_k(ranked, gold, 3) == 1.0
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert mrr(ranked, gold) == 0.5
    assert ndcg(ranked, gold) == pytest.approx(1 / math.log2(3))


def test_recall_with_two_golds():
    ranked, gold = ["x", "a", "y", "z", "q", "b"], {"a", "b"}
    assert recall_at_k(ranked, gold, 5) == 0.5 and recall_at_k(ranked, gold, 10) == 1.0
    assert hit_at_k(ranked, gold, 5) == 1.0
    assert mrr(ranked, gold, k=10) == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert ndcg(ranked, gold, k=10) == pytest.approx((1 / math.log2(3) + 1 / math.log2(7)) / ideal)


def test_mrr_cutoff_and_empty():
    assert mrr(["a"] * 11 + ["g"], {"g"}, k=10) == 0.0
    assert hit_at_k([], {"g"}, 5) == 0.0 and ndcg([], {"g"}) == 0.0
    assert recall_at_k(["g"], set(), 5) == 0.0


def test_per_query_and_aggregate():
    rows = [
        per_query_metrics(["a", "b"], {"a"}),
        per_query_metrics(["x", "y", "z", "a"], {"a"}),
    ]
    assert KS == (1, 3, 5, 10)
    assert rows[0]["hit@1"] == 1.0 and rows[1]["hit@3"] == 0.0 and rows[1]["hit@5"] == 1.0
    agg = aggregate(rows)
    assert agg["hit@1"] == 0.5 and agg["recall@5"] == 1.0 and agg["mrr@10"] == pytest.approx(0.625)
    assert agg["n"] == 2


def test_group_level_metrics_treat_any_chunk_of_a_gold_passage_as_relevant():
    # one gold passage split across two fixed windows (c1, c2); retrieving either should count fully
    groups = [{"c1", "c2"}, {"c9"}]
    assert recall_at_k(["c1", "x", "y"], groups, 3) == 0.5
    assert recall_at_k(["c1", "c2", "y"], groups, 3) == 0.5  # same passage twice is still one hit
    assert recall_at_k(["c2", "c9"], groups, 5) == 1.0
    assert hit_at_k(["z", "c2"], groups, 2) == 1.0
    assert mrr(["z", "c2"], groups) == 0.5
    assert ndcg(["c1", "c2", "c9"], groups, k=10) == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    m = per_query_metrics(["c1"], groups)
    assert m["recall@1"] == 0.5 and m["hit@1"] == 1.0
