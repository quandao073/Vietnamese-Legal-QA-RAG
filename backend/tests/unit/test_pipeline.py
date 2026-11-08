from __future__ import annotations

import numpy as np
import pytest

from legalrag.config import Settings
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.retrieval.pipeline import Retriever
from legalrag.retrieval.query_understanding import QueryUnderstanding
from legalrag.retrieval.reranker import FakeReranker
from legalrag.retrieval.types import Hit, RetrievalOverrides


class FakeDense:
    def __init__(self, hits_by_query: dict[str, list[Hit]]):
        self.hits = hits_by_query
        self.calls: list[np.ndarray] = []
        self.default = next(iter(hits_by_query.values())) if hits_by_query else []

    def search(self, vec, top_n, filters=None):
        self.calls.append(vec)
        return list(self.default[:top_n])


class FakeBM25:
    def __init__(self, hits: list[Hit]):
        self.hits, self.calls = hits, []

    def search(self, query, top_n):
        self.calls.append(query)
        return list(self.hits[:top_n])


class FakeChunkRepo:
    def __init__(self, chunks: dict[str, dict]):
        self.chunks = chunks

    def get_many(self, ids):
        return {i: self.chunks[i] for i in ids if i in self.chunks}


def _hit(cid: str, text: str, doc_id="d1", article="Điều 1", **extra) -> Hit:
    return Hit(
        cid,
        1.0,
        {"doc_id": doc_id, "article": article, "text": text, "title": "Luật X", "extra": extra},
    )


def _settings(**kw) -> Settings:
    base: dict = dict(anthropic_api_key="k", rerank_enabled=True, retrieval_top_k=2)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _qu(*rewrites: str) -> QueryUnderstanding:
    return QueryUnderstanding(
        intent="legal_question",
        standalone_question="thuế thu nhập cá nhân",
        rewrites=list(rewrites),
    )


def make_retriever(
    dense: FakeDense, bm25: FakeBM25, settings: Settings, chunk_repo=None
) -> Retriever:
    return Retriever(
        embedder=FakeEmbedder(dim=8),
        dense_by_strategy=lambda s: dense,
        bm25_by_strategy=lambda s: bm25,
        reranker=FakeReranker(),
        chunk_repo=chunk_repo or FakeChunkRepo({}),
        settings=settings,
    )


async def test_hybrid_runs_all_queries_fuses_and_reranks():
    dense = FakeDense({"q": [_hit("a", "thuế thu nhập cá nhân miễn"), _hit("b", "hôn nhân")]})
    bm25 = FakeBM25([_hit("c", "thuế thu nhập cá nhân"), _hit("a", "thuế thu nhập cá nhân miễn")])
    r = make_retriever(dense, bm25, _settings())
    res = await r.retrieve("thuế thu nhập cá nhân", qu=_qu("miễn thuế TNCN", "thu nhập miễn thuế"))
    assert len(dense.calls) == 3 and len(bm25.calls) == 3  # standalone + 2 rewrites
    assert [h.chunk_id for h in res.candidates][0] == "a"  # appears in both lists
    assert len(res.reranked) == 2 and all(h.rerank_score is not None for h in res.reranked)
    assert res.reranked[0].chunk_id in {"a", "c"}
    assert set(res.timings) >= {"dense_ms", "bm25_ms", "fusion_ms", "rerank_ms", "total_ms"}
    assert (
        res.stage_rankings["dense"] and res.stage_rankings["bm25"] and res.stage_rankings["hybrid"]
    )
    assert res.stage_rankings["hybrid_rerank"] == [h.chunk_id for h in res.reranked]


async def test_mode_bm25_skips_dense_and_rerank_off():
    dense = FakeDense({"q": [_hit("a", "x")]})
    bm25 = FakeBM25([_hit("c", "thuế"), _hit("d", "khác")])
    r = make_retriever(dense, bm25, _settings())
    res = await r.retrieve(
        "thuế", qu=None, overrides=RetrievalOverrides(mode="bm25", use_rerank=False, top_k=1)
    )
    assert dense.calls == [] and len(bm25.calls) == 1
    assert [h.chunk_id for h in res.reranked] == ["c"] and res.reranked[0].rerank_score is None


async def test_title_boost_from_entities():
    dense = FakeDense({"q": [_hit("a", "x", doc_id="d1"), _hit("b", "y", doc_id="d2")]})
    bm25 = FakeBM25([])
    hits = dense.default
    hits[1].payload["title"] = "Luật Thuế thu nhập cá nhân"
    r = make_retriever(dense, bm25, _settings(rerank_enabled=False))
    qu = _qu()
    qu.entities.law_names = ["Luật Thuế thu nhập cá nhân"]
    res = await r.retrieve("q", qu=qu)
    assert res.candidates[0].chunk_id == "b" and res.candidates[0].debug["boost"] == 0.1


async def test_parent_child_replaces_children_with_parent():
    children = [
        _hit("p#00001", "khoản 1", role="child", parent_chunk_id="p#00000"),
        _hit("p#00002", "khoản 2", role="child", parent_chunk_id="p#00000"),
    ]
    dense = FakeDense({"q": children})
    bm25 = FakeBM25([])
    repo = FakeChunkRepo(
        {
            "p#00000": {
                "chunk_id": "p#00000",
                "doc_id": "d1",
                "article": "Điều 1",
                "text": "cả điều",
                "title": "Luật X",
                "extra": {"role": "parent"},
            }
        }
    )
    r = make_retriever(
        dense, bm25, _settings(rerank_enabled=False, chunk_strategy="parent_child"), chunk_repo=repo
    )
    res = await r.retrieve("q", qu=None)
    assert [h.chunk_id for h in res.reranked] == ["p#00000"]
    assert res.reranked[0].text == "cả điều"


async def test_clause_strategy_keeps_fusion_order_before_rerank():
    dense = FakeDense(
        {
            "q": [
                _hit("a", "k1", article="Điều 5"),
                _hit("c", "k3", article="Điều 6"),
                _hit("b", "k2", article="Điều 5"),
            ]
        }
    )
    r = make_retriever(
        dense,
        FakeBM25([]),
        _settings(rerank_enabled=False, retrieval_top_k=5, chunk_strategy="clause"),
    )
    res = await r.retrieve("q", qu=None)
    # I4: no reordering/dropping in the fusion stage; grouping happens in the context builder
    assert [h.chunk_id for h in res.reranked] == ["a", "c", "b"]


async def test_non_legal_intent_returns_empty():
    dense = FakeDense({"q": [_hit("a", "x")]})
    r = make_retriever(dense, FakeBM25([]), _settings())
    qu = QueryUnderstanding(intent="out_of_scope", standalone_question="trời mưa?")
    res = await r.retrieve("trời mưa?", qu=qu)
    assert res.candidates == [] and res.reranked == [] and dense.calls == []


@pytest.mark.parametrize("top_k", [1, 3])
async def test_top_k_override(top_k):
    dense = FakeDense({"q": [_hit(str(i), f"t{i}", article=f"Điều {i}") for i in range(5)]})
    r = make_retriever(dense, FakeBM25([]), _settings(rerank_enabled=False))
    res = await r.retrieve("q", qu=None, overrides=RetrievalOverrides(top_k=top_k))
    assert len(res.reranked) == top_k


async def test_rerank_pool_size_is_configurable():
    hits = [_hit(str(i), f"thuế {i}", article=f"Điều {i}") for i in range(40)]
    dense = FakeDense({"q": hits})

    class CountingReranker(FakeReranker):
        seen: list[int] = []

        def score(self, query, texts):
            self.seen.append(len(texts))
            return super().score(query, texts)

    r = make_retriever(dense, FakeBM25([]), _settings(rerank_top_n=12, retrieval_top_n_dense=40))
    r.reranker = CountingReranker()
    await r.retrieve("thuế", qu=None)
    assert CountingReranker.seen == [12]
