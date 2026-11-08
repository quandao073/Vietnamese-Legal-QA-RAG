from legalrag.eval.datasets import EvalItem
from legalrag.eval.judge import JudgeScores
from legalrag.eval.runner import _classify_failure, _refusal_metrics


def _judge(faith=5, claims=None, cite=1.0):
    return JudgeScores(
        faithfulness=faith,
        completeness=4,
        relevance=4,
        citation_precision=cite,
        hallucinated_claims=claims or [],
    )


def test_tricky_question_is_ok_when_answer_adds_no_false_claims():
    # a tricky tax question *is* in the tax corpus: answering from sources that the premise
    # does not exist is correct behaviour, so grounded=True must not count as a missed refusal
    it = EvalItem(id="t", question="q", category="tricky")
    assert _classify_failure(it, True, "legal_question", ["c1"], _judge()) == "ok"
    assert (
        _classify_failure(it, True, "legal_question", ["c1"], _judge(claims=["15%"]))
        == "hallucination"
    )
    assert _classify_failure(it, False, "legal_question", [], None) == "ok"


def test_out_of_corpus_requires_refusal():
    it = EvalItem(id="o", question="q", category="out_of_corpus")
    assert _classify_failure(it, True, "legal_question", ["c1"], _judge()) == "missed_refusal"
    assert _classify_failure(it, False, "legal_question", [], None) == "ok"


def test_refusal_metrics_split_tricky_from_out_of_corpus():
    rows = [
        {"category": "out_of_corpus", "grounded": False, "failure": "ok"},
        {"category": "out_of_corpus", "grounded": True, "failure": "missed_refusal"},
        {"category": "out_of_scope", "grounded": False, "failure": "ok"},
        {"category": "tricky", "grounded": True, "failure": "ok"},
        {"category": "tricky", "grounded": True, "failure": "hallucination"},
        {
            "category": "legal",
            "grounded": False,
            "gold_chunk_ids": ["c"],
            "failure": "false_refusal",
        },
        {"category": "legal", "grounded": True, "gold_chunk_ids": ["c"], "failure": "ok"},
    ]
    m = _refusal_metrics(rows)
    assert m["correct_refusal_rate"] == round(2 / 3, 4)  # out_of_corpus + out_of_scope only
    assert m["tricky_safe_rate"] == 0.5
    assert m["hallucination_rate"] == 0.2  # over all 5 refusal-set rows
    assert m["false_refusal_rate"] == 0.5
