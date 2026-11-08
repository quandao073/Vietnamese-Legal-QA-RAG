import pytest

from legalrag.eval.answer_metrics import (
    exact_match,
    extract_conclusion,
    lexical_scores,
    normalize_answer,
    rouge_l,
    token_f1,
)


def test_normalize_answer():
    assert normalize_answer("  Thuế suất là 10%.  ") == "thuế suất là 10%"
    assert normalize_answer("Hoà bình, Việt-Nam!") == "hòa bình việt-nam"


def test_exact_match_after_normalization():
    assert exact_match("Thuế suất 10%.", "thuế suất 10%") == 1.0
    assert exact_match("Thuế suất 10%", "thuế suất 5%") == 0.0


def test_token_f1_hand_example():
    # pred: thuế 10% (2 tokens) ; gold: thuế suất 10% (3 tokens) → P=1, R=2/3, F1=0.8
    assert token_f1("thuế 10%", "thuế suất 10%") == pytest.approx(0.8)
    assert token_f1("", "a") == 0.0


def test_rouge_l_f1():
    assert rouge_l("a b c", "a b c") == 1.0
    assert 0 < rouge_l("a x c", "a b c d") < 1


def test_extract_conclusion_from_markdown():
    answer = (
        "**Kết luận**: Được miễn thuế [#1].\n\n"
        "**Căn cứ pháp lý**:\n- Điều 4 [#1]\n\n**Lưu ý**: không có.\n\nThông tin trên..."
    )
    assert extract_conclusion(answer) == "Được miễn thuế [#1]."
    # fallback: first paragraph when no heading
    assert extract_conclusion("Câu trả lời ngắn.\n\nĐoạn hai.") == "Câu trả lời ngắn."
    assert extract_conclusion("## Kết luận\nCó.\n## Căn cứ\nx") == "Có."


def test_lexical_scores_bundle():
    s = lexical_scores("**Kết luận**: thuế suất 10% [#1].\n\n**Căn cứ**: x", "thuế suất 10%")
    assert set(s) == {"em", "f1", "rouge_l", "em_conclusion", "f1_conclusion", "rouge_l_conclusion"}
    assert s["em_conclusion"] == 1.0 and s["f1_conclusion"] == 1.0 and s["f1"] < 1.0
