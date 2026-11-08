from legalrag.schemas.document import LegalChunk, LegalDocument


def _chunk(**kw) -> LegalChunk:
    base = dict(
        chunk_id="tamir:doc_000001#00003",
        doc_id="tamir:doc_000001",
        seq=3,
        text="x",
        text_for_bm25="x",
        source_ids=["passage_000001"],
        token_count=1,
        chunk_strategy="article",
    )
    base.update(kw)
    return LegalChunk(**base)


def test_display_source_with_article_clause_point():
    c = _chunk(
        article="Điều 4", article_title="Thu nhập được miễn thuế", clause="Khoản 1", point="Điểm a"
    )
    assert (
        c.display_source("Luật Thuế TNCN")
        == "Luật Thuế TNCN — Điều 4. Thu nhập được miễn thuế, Khoản 1, Điểm a"
    )


def test_display_source_without_article_uses_seq():
    assert _chunk().display_source("Văn bản X") == "Văn bản X — đoạn 3"


def test_document_defaults():
    d = LegalDocument(doc_id="tamir:doc_000001", source="tamir", title="T", raw_text="r")
    assert d.effective_status == "unknown" and d.extra == {} and d.url is None
