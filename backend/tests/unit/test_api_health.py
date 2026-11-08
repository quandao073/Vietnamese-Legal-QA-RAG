from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from legalrag.api import deps
from legalrag.api.app import create_app
from legalrag.retrieval.types import Hit


class FakeHealth:
    def __init__(self, qdrant=True, postgres=True, redis=True):
        self.values = dict(qdrant=qdrant, postgres=postgres, redis=redis)

    def __call__(self):
        return dict(
            self.values,
            bm25_loaded=True,
            embedder_loaded=True,
            collection="legal_chunks__article__baai-bge-m3",
            n_chunks=106,
        )


class FakeSearcher:
    async def search(self, q, top_k, mode, strategy=None):
        return [
            Hit(
                "c1",
                0.5,
                {
                    "doc_id": "d",
                    "title": "Luật X",
                    "article": "Điều 1",
                    "text": "nội dung",
                    "seq": 0,
                },
                {"rerank_score": 0.9},
            )
        ]


@pytest.fixture
async def client():
    app = create_app()
    app.dependency_overrides[deps.health_probe] = FakeHealth()
    app.dependency_overrides[deps.get_searcher_dep] = lambda: FakeSearcher()
    app.state.ready = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health_ok(client):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    d = r.json()
    assert (
        d["status"] == "ok"
        and d["qdrant"] is True
        and d["n_chunks"] == 106
        and d["bm25_loaded"] is True
    )


async def test_health_ready_503_when_qdrant_down():
    app = create_app()
    app.dependency_overrides[deps.health_probe] = FakeHealth(qdrant=False)
    app.state.ready = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/api/v1/health/ready")).status_code == 503
        assert (await c.get("/api/v1/health")).status_code == 503
        assert (await c.get("/api/v1/health/live")).status_code == 200


async def test_ready_503_before_warmup():
    app = create_app()
    app.dependency_overrides[deps.health_probe] = FakeHealth()
    app.state.ready = False
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/health/ready")
        assert r.status_code == 503 and r.json()["status"] == "starting"


async def test_search_endpoint(client):
    r = await client.get("/api/v1/search", params={"q": "thuế", "top_k": 3, "mode": "bm25"})
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "bm25" and d["results"][0]["chunk_id"] == "c1"
    assert (
        d["results"][0]["display_source"] == "Luật X — Điều 1"
        and d["results"][0]["rerank_score"] == 0.9
    )
    assert (await client.get("/api/v1/search", params={"q": "", "mode": "bm25"})).status_code == 422
    assert (
        await client.get("/api/v1/search", params={"q": "x", "mode": "weird"})
    ).status_code == 422


async def test_public_config(client):
    r = await client.get("/api/v1/config/public")
    assert r.status_code == 200
    d = r.json()
    assert d["default_strategy"] == "article" and d["top_k_range"] == [3, 10] and "strategies" in d
