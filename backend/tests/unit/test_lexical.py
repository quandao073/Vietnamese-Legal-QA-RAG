from pathlib import Path

import pytest

from legalrag.ingest.chunking import ArticleChunker, build_chunks
from legalrag.ingest.loaders.hf_tamir_tax import TamirLoader
from legalrag.retrieval.lexical import BM25Index
from legalrag.tokens import TokenCounter

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def chunks():
    loader = TamirLoader(raw_dir=FIX)
    loader.kb_path_override = FIX / "tamir_kb_sample.csv"
    chunker = ArticleChunker(max_tokens=150, counter=TokenCounter(None))
    return [c for d in loader.iter_documents() for c in build_chunks(d, chunker)]


def test_bm25_ranks_relevant_article_first(chunks, tmp_path):
    index = BM25Index.build(chunks, tmp_path / "bm25")
    hits = index.search("quà biếu, quà tặng có chịu thuế giá trị gia tăng không?", top_n=5)
    assert len(hits) == 5
    assert hits[0].score >= hits[1].score
    top = next(c for c in chunks if c.chunk_id == hits[0].chunk_id)
    assert top.article == "Điều 5" and top.clause == "Khoản 26"
    assert hits[0].payload["title"] == top.title
    assert hits[0].payload["doc_id"] == top.doc_id and hits[0].debug == {}


def test_bm25_load_roundtrip(chunks, tmp_path):
    out = tmp_path / "bm25"
    BM25Index.build(chunks, out)
    assert (out / "ids.json").exists()
    loaded = BM25Index.load(out)
    assert len(loaded) == len(chunks)
    a = [h.chunk_id for h in loaded.search("thuế thu nhập cá nhân", top_n=3)]
    b = [
        h.chunk_id
        for h in BM25Index.build(chunks, tmp_path / "again").search(
            "thuế thu nhập cá nhân", top_n=3
        )
    ]
    assert a == b


def test_bm25_empty_query_returns_nothing(chunks, tmp_path):
    index = BM25Index.build(chunks, tmp_path / "bm25")
    assert index.search("   ", top_n=5) == []
