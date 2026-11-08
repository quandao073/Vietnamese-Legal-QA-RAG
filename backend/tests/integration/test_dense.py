from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from legalrag.config import get_settings
from legalrag.ingest.chunking import ArticleChunker, build_chunks
from legalrag.ingest.loaders.hf_tamir_tax import TamirLoader
from legalrag.retrieval.dense import DenseStore
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.tokens import TokenCounter

pytestmark = pytest.mark.integration
FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def store():
    client = QdrantClient(url=get_settings().qdrant_url)
    name = f"test_dense_{uuid.uuid4().hex[:6]}"
    s = DenseStore(client, collection=name, dim=32)
    s.ensure_collection()
    yield s
    client.delete_collection(name)


@pytest.fixture(scope="module")
def chunks():
    loader = TamirLoader(raw_dir=FIX)
    loader.kb_path_override = FIX / "tamir_kb_sample.csv"
    chunker = ArticleChunker(max_tokens=150, counter=TokenCounter(None))
    return [c for d in loader.iter_documents() for c in build_chunks(d, chunker)]


def test_upsert_count_existing_and_search(store, chunks):
    emb = FakeEmbedder(dim=32)
    vecs = emb.encode_passages([c.text for c in chunks])
    store.upsert(chunks, vecs, batch=8)
    store.upsert(chunks[:3], vecs[:3])  # idempotent
    assert store.count() == len(chunks)
    ids = [c.chunk_id for c in chunks[:5]] + ["missing#00000"]
    assert store.existing_ids(ids) == set(c.chunk_id for c in chunks[:5])

    hits = store.search(emb.encode_query(chunks[7].text), top_n=3)
    assert hits[0].chunk_id == chunks[7].chunk_id
    assert (
        hits[0].payload["article"] == chunks[7].article and "text_for_bm25" not in hits[0].payload
    )
    assert hits[0].payload["text"] == chunks[7].text


def test_search_filters(store, chunks):
    emb = FakeEmbedder(dim=32)
    q = emb.encode_query("thuế")
    doc_id = chunks[-1].doc_id
    hits = store.search(q, top_n=50, filters={"doc_id": doc_id})
    assert hits and all(h.payload["doc_id"] == doc_id for h in hits)
    hits = store.search(q, top_n=50, filters={"article": "Điều 5"})
    assert hits and all(h.payload["article"] == "Điều 5" for h in hits)
    hits = store.search(q, top_n=50, filters={"title": "thu nhập cá nhân"})
    assert hits and all("thu nhập cá nhân" in h.payload["title"].lower() for h in hits)


def test_point_id_is_uuid5(store):
    assert DenseStore.point_id("a#00001") == str(uuid.uuid5(uuid.NAMESPACE_URL, "a#00001"))
