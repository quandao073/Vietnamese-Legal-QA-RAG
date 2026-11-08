from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from legalrag import jobs
from legalrag.api import deps
from legalrag.api.app import create_app

TOKEN = "test-admin-token"


class FakeStore:
    """In-memory stand-in for the DB-backed admin/feedback repositories."""

    def __init__(self):
        self.messages = {"m1": {"question": "Q?", "answer_md": "A"}}
        self.feedback: list[dict] = []
        self.runs: dict[str, dict] = {}

    def message_exists(self, message_id: str) -> bool:
        return message_id in self.messages

    def add_feedback(self, *, message_id, rating, reason, comment) -> int:
        self.feedback.append(
            dict(message_id=message_id, rating=rating, reason=reason, comment=comment)
        )
        return len(self.feedback)

    def stats(self) -> dict:
        neg = [f for f in self.feedback if f["rating"] < 0]
        return {
            "n_documents": 5,
            "collections": [{"strategy": "article", "n_chunks": 106, "n_docs": 5}],
            "queries_24h": 3,
            "grounded_24h": 2,
            "feedback_up": len(self.feedback) - len(neg),
            "feedback_down": len(neg),
            "thumbs_up_ratio": 0.5,
            "latency_p50_ms": 1200.0,
            "latency_p95_ms": 3000.0,
            "tokens_24h": {"input": 10, "output": 5},
            "recent_negative_feedback": [
                {
                    "message_id": f["message_id"],
                    "reason": f["reason"],
                    "comment": f["comment"],
                    "question": "Q?",
                    "answer_md": "A",
                }
                for f in neg
            ],
            "ingest_runs": [],
        }


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
async def client(store, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    from legalrag.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[deps.get_admin_store] = lambda: store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    get_settings.cache_clear()


async def test_admin_requires_token(client):
    assert (await client.get("/api/v1/admin/stats")).status_code == 401
    r = await client.get("/api/v1/admin/stats", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401 and "X-Admin-Token" in r.json()["message"]


async def test_admin_stats_shape(client):
    r = await client.get("/api/v1/admin/stats", headers={"X-Admin-Token": TOKEN})
    assert r.status_code == 200
    d = r.json()
    assert d["collections"][0]["strategy"] == "article" and d["queries_24h"] == 3
    assert "latency_p50_ms" in d and "recent_negative_feedback" in d and "ingest_runs" in d


async def test_feedback_roundtrip(client, store):
    r = await client.post(
        "/api/v1/feedback",
        json={"message_id": "m1", "rating": -1, "reason": "incomplete", "comment": "thiếu"},
    )
    assert r.status_code == 201 and r.json()["id"] == 1
    assert store.feedback[0]["reason"] == "incomplete"
    stats = (await client.get("/api/v1/admin/stats", headers={"X-Admin-Token": TOKEN})).json()
    assert stats["recent_negative_feedback"][0]["message_id"] == "m1"


async def test_feedback_unknown_message_404_and_validation(client):
    r = await client.post("/api/v1/feedback", json={"message_id": "nope", "rating": 1})
    assert r.status_code == 404
    r = await client.post("/api/v1/feedback", json={"message_id": "m1", "rating": 5})
    assert r.status_code == 422


async def test_admin_ingest_starts_background_job(client, monkeypatch):
    calls = []

    def fake_run(source, strategy, max_docs, run_id, progress):
        calls.append((source, strategy, max_docs, run_id))
        progress("built", 1, 1)
        return {"n_docs": 5, "n_chunks": 106}

    reloaded = []
    monkeypatch.setattr(jobs, "_run_ingest", fake_run)
    monkeypatch.setattr(jobs, "_after_success", lambda strategy: reloaded.append(strategy))
    r = await client.post(
        "/api/v1/admin/ingest",
        json={"source": "tamir", "strategy": "article", "max_docs": 100},
        headers={"X-Admin-Token": TOKEN},
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    for _ in range(50):
        st = (
            await client.get(f"/api/v1/admin/ingest/{run_id}", headers={"X-Admin-Token": TOKEN})
        ).json()
        if st["status"] in ("done", "failed"):
            break
        await asyncio.sleep(0.05)
    assert st["status"] == "done" and st["result"]["n_chunks"] == 106 and st["log"]
    assert calls == [("tamir", "article", 100, run_id)] and reloaded == ["article"]
    assert (
        await client.get("/api/v1/admin/ingest/unknown", headers={"X-Admin-Token": TOKEN})
    ).status_code == 404


async def test_admin_ingest_rejects_bad_source(client):
    r = await client.post(
        "/api/v1/admin/ingest",
        json={"source": "web", "strategy": "article"},
        headers={"X-Admin-Token": TOKEN},
    )
    assert r.status_code == 422
