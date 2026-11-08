from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from legalrag.db.repositories import (
    ChunkRepo,
    DocumentRepo,
    EvalRunRepo,
    FeedbackRepo,
    IngestRunRepo,
    QueryLogRepo,
    StatsRepo,
)
from legalrag.db.session import session_scope
from legalrag.ingest.chunking import ArticleChunker, build_chunks
from legalrag.ingest.loaders.hf_tamir_tax import TamirLoader
from legalrag.schemas.chat import QueryLogCreate
from legalrag.tokens import TokenCounter

pytestmark = pytest.mark.integration
FIX = Path(__file__).parent.parent / "fixtures"
STRATEGY = f"test_{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def docs_and_chunks():
    loader = TamirLoader(raw_dir=FIX)
    loader.kb_path_override = FIX / "tamir_kb_sample.csv"
    docs = list(loader.iter_documents())
    chunker = ArticleChunker(max_tokens=300, counter=TokenCounter(None), strategy=STRATEGY)
    chunks = [c for d in docs for c in build_chunks(d, chunker)]
    return docs, chunks


@pytest.fixture(scope="module", autouse=True)
def _cleanup():
    yield
    with session_scope() as s:
        ChunkRepo(s).delete_strategy(STRATEGY)


def test_upsert_documents_and_chunks_is_idempotent(docs_and_chunks):
    docs, chunks = docs_and_chunks
    with session_scope() as s:
        DocumentRepo(s).upsert_many(docs)
        DocumentRepo(s).upsert_many(docs)
        ChunkRepo(s).upsert_many(chunks)
        ChunkRepo(s).upsert_many(chunks)
        assert ChunkRepo(s).count(STRATEGY) == len(chunks)
        assert sorted(ChunkRepo(s).ids(STRATEGY)) == sorted(c.chunk_id for c in chunks)


def test_get_and_neighbors(docs_and_chunks):
    _, chunks = docs_and_chunks
    with session_scope() as s:
        repo = ChunkRepo(s)
        got = repo.get(chunks[1].chunk_id)
        assert got is not None and got.text == chunks[1].text and got.extra == chunks[1].extra
        prev, cur, nxt = repo.neighbors(chunks[1].chunk_id)
        assert cur.chunk_id == chunks[1].chunk_id
        assert prev.chunk_id == chunks[0].chunk_id and nxt.chunk_id == chunks[2].chunk_id
        assert repo.neighbors(chunks[0].chunk_id)[0] is None
        assert repo.get("nope") is None


def test_by_source_ids_finds_overlap(docs_and_chunks):
    _, chunks = docs_and_chunks
    with session_scope() as s:
        found = ChunkRepo(s).by_source_ids(["passage_000042"], STRATEGY)
        assert found and all(cid in {c.chunk_id for c in chunks} for cid in found)
        assert ChunkRepo(s).by_source_ids(["passage_999999"], STRATEGY) == []


def test_documents_lookup(docs_and_chunks):
    docs, _ = docs_and_chunks
    with session_scope() as s:
        d = DocumentRepo(s).get(docs[0].doc_id)
        assert d is not None and d.title == docs[0].title and d.url == docs[0].url
        arts = DocumentRepo(s).articles(docs[0].doc_id, STRATEGY)
        assert arts and arts[0]["article"] == "Điều 1"


def test_ingest_run_lifecycle():
    with session_scope() as s:
        repo = IngestRunRepo(s)
        run_id = repo.start(strategy=STRATEGY, embedding_model="fake", source="tamir")
        repo.finish(run_id, n_docs=3, n_chunks=10, status="done", log="ok")
        run = repo.get(run_id)
        assert run is not None and run.status == "done" and run.n_chunks == 10
        assert run.finished_at is not None


def test_query_log_feedback_and_stats(docs_and_chunks):
    _, chunks = docs_and_chunks
    message_id = uuid.uuid4().hex
    with session_scope() as s:
        QueryLogRepo(s).insert(
            QueryLogCreate(
                message_id=message_id,
                session_id="sess",
                request_id="req",
                question="Q?",
                standalone_question="Q?",
                intent="legal_question",
                retrieved_chunk_ids=[chunks[0].chunk_id],
                cited_chunk_ids=[chunks[0].chunk_id],
                answer_md="A [#1]",
                grounded=True,
                model="m",
                prompt_version="v1",
                usage={"input_tokens": 10, "output_tokens": 5},
                latency={"total_ms": 1200},
            )
        )
        assert QueryLogRepo(s).exists(message_id)
        FeedbackRepo(s).insert(message_id=message_id, rating=-1, reason="incomplete", comment="x")
        neg = FeedbackRepo(s).latest_negative(20)
        assert neg and neg[0]["message_id"] == message_id and neg[0]["question"] == "Q?"
        stats = StatsRepo(s).summary()
        assert stats["queries_24h"] >= 1
        assert 0.0 <= stats["thumbs_up_ratio"] <= 1.0
        assert stats["latency_p50_ms"] is not None
        assert any(c["strategy"] == STRATEGY for c in stats["collections"])


def test_eval_run_repo():
    with session_scope() as s:
        repo = EvalRunRepo(s)
        run_id = repo.insert(
            config_name="E2",
            params={"k": 6},
            dataset="tamir_test",
            metrics={"hit@5": 0.9},
            n=54,
            cost_usd=0.0,
        )
        latest = repo.latest(limit=5)
        assert any(r.run_id == run_id for r in latest)
        assert repo.get(run_id).metrics["hit@5"] == 0.9
