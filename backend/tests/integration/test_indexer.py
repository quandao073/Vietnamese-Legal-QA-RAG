from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from legalrag.config import Settings
from legalrag.db.repositories import ChunkRepo
from legalrag.db.session import session_scope
from legalrag.ingest import indexer
from legalrag.retrieval.dense import DenseStore, get_qdrant_client
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.retrieval.lexical import BM25Index
from legalrag.tokens import TokenCounter

pytestmark = pytest.mark.integration
FIX = Path(__file__).parent.parent / "fixtures"
STRATEGY = f"article__{uuid.uuid4().hex[:6]}"  # unique strategy name keeps test data isolated


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    (root / "raw" / "tamir" / "knowledge_base").mkdir(parents=True)
    (root / "raw" / "tamir" / "qa").mkdir(parents=True)
    shutil.copy(
        FIX / "tamir_kb_sample.csv",
        root / "raw" / "tamir" / "knowledge_base" / "knowledge_base.csv",
    )
    shutil.copy(
        FIX / "tamir_qa_train_sample.jsonl", root / "raw" / "tamir" / "qa" / "train_qa.jsonl"
    )
    shutil.copy(FIX / "tamir_qa_test_sample.jsonl", root / "raw" / "tamir" / "qa" / "test_qa.jsonl")
    return root


@pytest.fixture(scope="module")
def settings(data_dir) -> Settings:
    return Settings(
        _env_file=None,
        anthropic_api_key="x",
        data_dir=str(data_dir),
        embedding_model="fake-32",
        qdrant_url="http://localhost:6333",
        database_url="postgresql+psycopg://legal:legal@localhost:5432/legalrag",
        chunk_strategy=STRATEGY,
    )


@pytest.fixture(scope="module", autouse=True)
def _cleanup(settings):
    yield
    with session_scope() as s:
        ChunkRepo(s).delete_strategy(STRATEGY)
    client = get_qdrant_client()
    name = settings.collection_name(STRATEGY)
    if client.collection_exists(name):
        client.delete_collection(name)


def test_build_writes_documents_chunks_and_eval_files(settings, data_dir):
    stats = indexer.build(
        ["tamir"], strategy=STRATEGY, settings=settings, counter=TokenCounter(None)
    )
    assert stats.n_docs == 3 and stats.n_chunks > 10
    assert (data_dir / "processed" / "documents.jsonl").exists()
    assert (data_dir / "processed" / f"chunks_{STRATEGY}.jsonl").exists()
    assert (data_dir / "eval" / "tamir_train.jsonl").exists()
    assert (data_dir / "eval" / "tamir_test.jsonl").exists()


def test_index_counts_match_and_resume_skips(settings, data_dir):
    emb = FakeEmbedder(dim=32)
    res = indexer.index(strategy=STRATEGY, settings=settings, embedder=emb, resume=False)
    n = res.n_chunks
    assert n > 10 and emb.passage_calls == n
    with session_scope() as s:
        assert ChunkRepo(s).count(STRATEGY) == n
    store = DenseStore(get_qdrant_client(), settings.collection_name(STRATEGY), dim=32)
    assert store.count() == n
    bm25 = BM25Index.load(data_dir / "index" / "bm25" / STRATEGY)
    assert len(bm25) == n
    state = json.loads((data_dir / "processed" / f"ingest_state_{res.run_id}.json").read_text())
    assert state["status"] == "done" and state["done"] == n

    # resume: nothing new to embed, BM25 rebuilt completely
    emb2 = FakeEmbedder(dim=32)
    res2 = indexer.index(strategy=STRATEGY, settings=settings, embedder=emb2, resume=True)
    assert emb2.passage_calls == 0 and res2.n_skipped == n
    assert len(BM25Index.load(data_dir / "index" / "bm25" / STRATEGY)) == n
    assert store.count() == n


def test_eval_gold_resolvable_and_stats(settings, data_dir):
    rows = [
        json.loads(line)
        for line in (data_dir / "eval" / "tamir_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with session_scope() as s:
        repo = ChunkRepo(s)
        for r in rows:
            assert repo.by_source_ids(r["gold_source_ids"], STRATEGY), r["id"]
    st = indexer.stats(STRATEGY, settings=settings)
    assert st["by_source"]["tamir"]["article_coverage_effective"] >= 0.95
    assert st["by_source"]["tamir"]["median_tokens"] > 0
    assert st["bm25_size"] == st["qdrant_points"] == st["n_chunks"]


def test_add_file_indexes_custom_document(settings, data_dir, tmp_path):
    law = tmp_path / "my_law.txt"
    law.write_text(
        "Điều 1. Phạm vi\nVăn bản thử nghiệm về thuế môn bài.\n"
        "Điều 2. Đối tượng\n1. Hộ kinh doanh.",
        encoding="utf-8",
    )
    emb = FakeEmbedder(dim=32)
    res = indexer.add_file(
        law,
        title="Luật thử nghiệm",
        url="https://example.org",
        strategy=STRATEGY,
        settings=settings,
        embedder=emb,
    )
    assert res.n_chunks == 2
    with session_scope() as s:
        found = ChunkRepo(s).by_source_ids(["custom:luat-thu-nghiem"], STRATEGY)
        assert len(found) == 2
    bm25 = BM25Index.load(data_dir / "index" / "bm25" / STRATEGY)
    assert bm25.search("thuế môn bài", top_n=1)[0].payload["title"] == "Luật thử nghiệm"
