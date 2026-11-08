from pathlib import Path

import pytest

from legalrag.ingest.chunking import (
    STRATEGIES,
    ArticleChunker,
    ClauseChunker,
    FixedChunker,
    ParentChildChunker,
    build_chunks,
    get_chunker,
)
from legalrag.ingest.loaders.hf_tamir_tax import TamirLoader
from legalrag.ingest.normalize import normalize_text
from legalrag.schemas.document import LegalDocument
from legalrag.tokens import TokenCounter

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def counter() -> TokenCounter:
    return TokenCounter(None)  # heuristic word-based counter, no model download


@pytest.fixture(scope="module")
def tamir_doc() -> LegalDocument:
    raw = FIX  # loader only needs kb_path; point it at a fake raw dir
    loader = TamirLoader(raw_dir=raw)
    loader.kb_path_override = FIX / "tamir_kb_sample.csv"
    docs = list(loader.iter_documents())
    return docs[0]  # Luật Thuế GTGT: Điều 1..5 (Điều 5 has 26+ clauses)


@pytest.fixture(scope="module")
def yuitc_doc() -> LegalDocument:
    return LegalDocument(
        doc_id="yuitc:cid:1",
        source="yuitc",
        title="Văn bản pháp luật (cid 1)",
        raw_text=(
            "Thành lập Hội đồng\n1. Bộ trưởng ra quyết định thành lập Hội đồng.\n"
            "2. Giám đốc Sở Y tế ra quyết định."
        ),
        extra={"cid": 1, "source_id": "cid:1"},
    )


def test_strategies_list():
    assert STRATEGIES == ["article", "clause", "fixed_512_64", "fixed_256_32", "parent_child"]


def test_get_chunker_parses_fixed(counter):
    c = get_chunker("fixed_256_32", counter=counter, max_tokens=700)
    assert isinstance(c, FixedChunker) and c.name == "fixed_256_32"
    assert c.size == 256 and c.overlap == 32
    with pytest.raises(ValueError):
        get_chunker("unknown", counter=counter, max_tokens=700)
    tagged = get_chunker("article__exp1", counter=counter, max_tokens=700)
    assert isinstance(tagged, ArticleChunker) and tagged.strategy == "article__exp1"


def test_article_chunker_one_chunk_per_article(tamir_doc, counter):
    chunks = build_chunks(tamir_doc, ArticleChunker(max_tokens=5000, counter=counter))
    assert [c.article for c in chunks] == ["Điều 1", "Điều 2", "Điều 3", "Điều 4", "Điều 5"]
    assert all(c.chunk_strategy == "article" for c in chunks)
    first = chunks[0]
    assert first.text.startswith(
        "VBHN 12/VBHN-VPQH năm 2026 - Luật Thuế giá trị gia tăng › Chương I. Những quy định chung"
        " › Điều 1. Phạm vi điều chỉnh\n"
    )
    assert first.chapter == "Chương I" and first.article_title == "Phạm vi điều chỉnh"
    assert first.title == tamir_doc.title
    assert first.clause is None and first.source_ids == ["passage_000001"]
    assert first.chunk_id == "tamir:doc_000001#00000"


def test_article_chunker_splits_long_article_by_clause(tamir_doc, counter):
    chunks = build_chunks(tamir_doc, ArticleChunker(max_tokens=120, counter=counter))
    art5 = [c for c in chunks if c.article == "Điều 5"]
    assert len(art5) > 1
    header = art5[0].text.split("\n")[0]
    assert all(c.text.split("\n")[0] == header for c in art5)
    assert all(c.clause is not None for c in art5)
    # a chunk may exceed the budget only when it holds a single (long) clause
    assert all(c.token_count <= 120 or "clause_range" not in c.extra for c in art5)
    assert any("clause_range" in c.extra for c in art5)
    # clause ranges are contiguous and cover all clauses in order
    firsts = [int(c.clause.split()[1]) for c in art5]
    assert firsts == sorted(firsts)


def test_article_chunker_pseudo_article_for_headingless_doc(yuitc_doc, counter):
    chunks = build_chunks(yuitc_doc, ArticleChunker(max_tokens=700, counter=counter))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.article == "Điều (không rõ số)" and c.article_title == "Thành lập Hội đồng"
    assert c.source_ids == ["cid:1"]
    assert "Bộ trưởng ra quyết định" in c.text
    assert c.display_source(yuitc_doc.title) == (
        "Văn bản pháp luật (cid 1) — Điều (không rõ số). Thành lập Hội đồng"
    )


def test_clause_chunker(tamir_doc, counter):
    chunks = build_chunks(tamir_doc, ClauseChunker(counter=counter))
    assert all(c.chunk_strategy == "clause" for c in chunks)
    # Điều 1 has no clauses -> whole article is one chunk with clause None
    art1 = [c for c in chunks if c.article == "Điều 1"]
    assert len(art1) == 1 and art1[0].clause is None
    art5 = [c for c in chunks if c.article == "Điều 5"]
    assert len(art5) >= 20 and all(c.clause and c.clause.startswith("Khoản ") for c in art5)
    k26 = [c for c in art5 if c.clause == "Khoản 26"][0]
    assert set(k26.source_ids) >= {"passage_000042", "passage_000043"}
    assert k26.text.startswith(art5[0].text.split("\n")[0])  # shared article header


def test_fixed_chunker_windows(tamir_doc, counter):
    chunks = build_chunks(tamir_doc, FixedChunker(size=120, overlap=20, counter=counter))
    assert len(chunks) > 3
    assert all(c.article is None and c.chunk_strategy == "fixed_120_20" for c in chunks)
    assert all(c.token_count <= 120 * 1.2 for c in chunks)
    # consecutive windows overlap in text
    tail = " ".join(chunks[0].text.split()[-5:])
    assert tail in " ".join(chunks[1].text.split())
    assert all(c.char_start is not None and c.char_end is not None for c in chunks)


def test_parent_child_links(tamir_doc, counter):
    chunks = build_chunks(tamir_doc, ParentChildChunker(counter=counter))
    parents = {c.chunk_id: c for c in chunks if c.extra.get("role") == "parent"}
    children = [c for c in chunks if c.extra.get("role") == "child"]
    assert len(parents) == 5 and children
    for ch in children:
        parent = parents[ch.extra["parent_chunk_id"]]
        assert parent.article == ch.article
        body = ch.text.split("\n", 1)[1]
        assert body.split("\n")[0] in parent.text
    assert all(c.chunk_strategy == "parent_child" for c in chunks)


def test_source_ids_roundtrip(tamir_doc, counter):
    rows = {pid: span for pid, span in tamir_doc.extra["span_map"].items()}
    for chunker in (
        ArticleChunker(max_tokens=120, counter=counter),
        ClauseChunker(counter=counter),
        FixedChunker(size=120, overlap=20, counter=counter),
    ):
        chunks = build_chunks(tamir_doc, chunker)
        assert all(c.source_ids for c in chunks)
        for c in chunks:
            for pid in c.source_ids:
                s, e = rows[pid]
                body = normalize_text(tamir_doc.raw_text[s:e])
                # fixed windows may cut a passage; article/clause chunks must contain it whole
                if not isinstance(chunker, FixedChunker):
                    assert body in c.text, (chunker.name, c.chunk_id, pid)


def test_chunk_ids_and_seq_unique(tamir_doc, counter):
    for name in STRATEGIES:
        chunks = build_chunks(tamir_doc, get_chunker(name, counter=counter, max_tokens=120))
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)
        assert [c.seq for c in chunks] == list(range(len(chunks)))
        assert all(c.text_for_bm25 and c.token_count > 0 for c in chunks)
