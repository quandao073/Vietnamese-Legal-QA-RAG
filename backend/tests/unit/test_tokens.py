from legalrag.tokens import TokenCounter


def test_heuristic_counter_scales_with_words():
    c = TokenCounter(None)
    assert c.count("") == 0
    assert c.count("một hai ba bốn năm sáu bảy tám chín mười") == 13  # 10 words * 1.3
    assert c.backend == "heuristic"


def test_counter_falls_back_when_model_missing():
    c = TokenCounter("this-model/does-not-exist", allow_download=False)
    assert c.count("a b c") >= 1
    assert c.backend == "heuristic"
