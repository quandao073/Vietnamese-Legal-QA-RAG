"""Retrieval quality on the real Tamir index (requires `make ingest-tamir` or the local run).

Uses the real bge-m3 embedder/reranker when their weights are cached; skipped otherwise.
10 `tamir_train` questions → Hit@6 ≥ 0.8 (spec §3.11).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from legalrag.config import Settings
from legalrag.db.repositories import ChunkRepo
from legalrag.db.session import session_scope

pytestmark = [pytest.mark.integration, pytest.mark.slow]

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
EVAL = DATA_DIR / "eval" / "tamir_train.jsonl"


def _models_cached() -> bool:
    hub = Path(os.environ.get("HF_HOME", DATA_DIR / "models")) / "hub"
    return (hub / "models--BAAI--bge-m3").exists() and (
        hub / "models--BAAI--bge-reranker-v2-m3"
    ).exists()


@pytest.fixture(scope="module")
def retriever():
    if not EVAL.exists() or not _models_cached():
        pytest.skip("Tamir index / models not available locally")
    os.environ.setdefault("HF_HOME", str(DATA_DIR / "models"))
    from legalrag import services

    settings = Settings(
        _env_file=None, anthropic_api_key="x", data_dir=str(DATA_DIR), chunk_strategy="article"
    )
    if not (DATA_DIR / "index" / "bm25" / "article" / "ids.json").exists():
        pytest.skip("BM25 index for 'article' not built")
    return services.get_retriever(settings)


async def test_hit_at_6_on_tamir_train(retriever):
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines()][:10]
    hits = 0
    with session_scope() as s:
        repo = ChunkRepo(s)
        for r in rows:
            gold = set(repo.by_source_ids(r["gold_source_ids"], "article"))
            assert gold, r["id"]
            res = await retriever.retrieve(r["question"], qu=None)
            top = [h.chunk_id for h in res.reranked[:6]]
            hits += bool(gold & set(top))
    assert hits / len(rows) >= 0.8, f"Hit@6 = {hits}/{len(rows)}"
