from legalrag.generation.citation import (
    append_disclaimer,
    clean_invalid,
    parse_citations,
    rouge_l_recall,
    verify_citations,
)
from legalrag.generation.context_builder import SourceRef
from legalrag.generation.prompts import DISCLAIMER


def _src(idx: int, text: str) -> SourceRef:
    return SourceRef(
        idx=idx,
        chunk_id=f"c{idx}",
        doc_id="d",
        display_source=f"S{idx}",
        url=None,
        snippet=text[:80],
        rerank_score=0.9,
        text=text,
    )


def test_parse_citations_accepts_hash_and_bare_forms():
    assert parse_citations("A [#1]. B [2], C [#1][#3].") == [1, 2, 3]
    assert parse_citations("no cites") == []


def test_clean_invalid_removes_unknown_index_and_normalizes():
    text, invalid = clean_invalid("A [#1]. B [#9]. C [3].", {1, 3})
    assert text == "A [#1]. B. C [#3]."
    assert invalid == [9]


def test_clean_invalid_keeps_text_without_citations():
    assert clean_invalid("Không có trích dẫn.", {1}) == ("Không có trích dẫn.", [])


def test_rouge_l_recall_hand_example():
    # claim "a x c" vs chunk "a b c d": LCS = "a c" -> 2 / len(claim=3)
    assert rouge_l_recall("a x c", "a b c d") == 2 / 3
    assert rouge_l_recall("", "a") == 0.0


def test_verify_marks_supported_weak_and_invalid():
    sources = [
        _src(
            1,
            "Thu nhập từ chuyển nhượng bất động sản giữa vợ với chồng "
            "được miễn thuế thu nhập cá nhân.",
        ),
        _src(2, "Thuế suất thuế giá trị gia tăng là 10% đối với hàng hóa, dịch vụ thông thường."),
    ]
    answer = (
        "Chuyển nhượng bất động sản giữa vợ với chồng được miễn thuế thu nhập cá nhân [#1]. "
        "Sao Hỏa là hành tinh thứ tư trong hệ Mặt Trời [#2]. "
        "Mức thuế suất là 10% [#2]. Câu không có trích dẫn. Nguồn ma [#7]."
    )
    report = verify_citations(answer, sources)
    assert report.verified == [1, 2]
    assert report.weak == [2]  # one sentence citing #2 was unsupported
    assert report.invalid == [7]


def test_append_disclaimer_once():
    out = append_disclaimer("Trả lời.")
    assert out.endswith(DISCLAIMER) and out.count(DISCLAIMER) == 1
    assert append_disclaimer(out).count(DISCLAIMER) == 1
