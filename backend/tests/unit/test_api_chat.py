from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from legalrag.api import deps
from legalrag.api.app import create_app
from legalrag.schemas.chat import ChatEvent, QueryLogCreate


class FakeAnswerer:
    def __init__(self, events: list[ChatEvent] | None = None, log_repo=None):
        self.events = events or [
            ChatEvent(event="status", data={"stage": "understanding"}),
            ChatEvent(
                event="sources",
                data={
                    "sources": [
                        {
                            "idx": 1,
                            "chunk_id": "c1",
                            "doc_id": "d",
                            "display_source": "Luật X — Điều 1",
                            "url": None,
                            "snippet": "s",
                            "rerank_score": 0.9,
                        }
                    ]
                },
            ),
            ChatEvent(event="token", data={"text": "Xin chào [#1]."}),
            ChatEvent(event="citations", data={"verified": [1], "weak": [], "invalid": []}),
            ChatEvent(
                event="done",
                data={
                    "message_id": "m1",
                    "answer_md": "Xin chào [#1].",
                    "grounded": True,
                    "intent": "legal_question",
                    "standalone_question": "q",
                    "sources": [],
                    "citations": {"verified": [1], "weak": [], "invalid": []},
                    "usage": {},
                    "latency_ms": {"total_ms": 1.0},
                    "model": "m",
                    "prompt_version": "v1",
                },
            ),
        ]
        self.calls: list[dict] = []
        self.log_repo = log_repo

    async def answer(
        self, question, *, session_id, history, overrides, request_id=None
    ) -> AsyncIterator[ChatEvent]:
        self.calls.append(
            {
                "question": question,
                "session_id": session_id,
                "history": history,
                "overrides": overrides,
                "request_id": request_id,
            }
        )
        for ev in self.events:
            yield ev
        if self.log_repo is not None:
            self.log_repo.insert(
                QueryLogCreate(message_id="m1", question=question, grounded=False, error="auth")
            )

    async def answer_sync(self, question, *, session_id, history, overrides, request_id=None):
        from legalrag.generation.answerer import Answerer

        return await Answerer.answer_sync(
            self,
            question,
            session_id=session_id,
            history=history,
            overrides=overrides,
            request_id=request_id,
        )  # type: ignore[arg-type]


class FakeLogRepo:
    def __init__(self):
        self.rows = []

    def insert(self, log):
        self.rows.append(log)


@pytest.fixture
def fake() -> FakeAnswerer:
    return FakeAnswerer()


@pytest.fixture
async def client(fake):
    app = create_app()
    app.dependency_overrides[deps.get_answerer_dep] = lambda: fake
    app.dependency_overrides[deps.rate_limit] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_chat_rejects_long_question(client):
    r = await client.post("/api/v1/chat/sync", json={"question": "a" * 2001})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error" and "2000" in body["message"]


async def test_chat_rejects_blank(client):
    r = await client.post("/api/v1/chat/sync", json={"question": "   "})
    assert r.status_code == 422 and "trống" in r.json()["message"]


async def test_chat_sync_returns_response(client, fake):
    r = await client.post(
        "/api/v1/chat/sync",
        json={
            "question": "Thuế là gì?",
            "session_id": "s",
            "history": [{"role": "user", "content": "x"}],
            "options": {"top_k": 3, "use_rerank": False},
        },
        headers={"X-Request-ID": "req-123"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["message_id"] == "m1" and d["grounded"] is True and d["sources"][0]["chunk_id"] == "c1"
    assert r.headers["x-request-id"] == "req-123"
    call = fake.calls[0]
    assert (
        call["overrides"].top_k == 3
        and call["overrides"].use_rerank is False
        and call["request_id"] == "req-123"
    )
    assert call["history"][0].content == "x"


async def test_chat_stream_events(client):
    r = await client.post("/api/v1/chat", json={"question": "Thuế là gì?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "event: status\ndata: {" in body and "event: token\n" in body and "event: done\n" in body
    assert body.rstrip().endswith("}")
    assert "x-request-id" in r.headers


async def test_chat_stream_llm_error_logged(fake):
    log = FakeLogRepo()
    err = FakeAnswerer(
        events=[
            ChatEvent(event="status", data={"stage": "understanding"}),
            ChatEvent(
                event="error",
                data={"code": "auth", "message": "Không thể xác thực...", "message_id": "m1"},
            ),
        ],
        log_repo=log,
    )
    app = create_app()
    app.dependency_overrides[deps.get_answerer_dep] = lambda: err
    app.dependency_overrides[deps.rate_limit] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/chat", json={"question": "Thuế là gì?"})
        assert r.status_code == 200 and "event: error" in r.text and "Không thể xác thực" in r.text
        assert len(log.rows) == 1 and log.rows[0].grounded is False
        r2 = await c.post("/api/v1/chat/sync", json={"question": "Thuế là gì?"})
        assert r2.status_code == 502 and r2.json()["code"] == "auth"


async def test_request_id_generated_when_absent(client):
    r = await client.post("/api/v1/chat/sync", json={"question": "Thuế là gì?"})
    assert len(r.headers["x-request-id"]) >= 8


async def test_unknown_route_is_json_vietnamese(client):
    r = await client.get("/api/v1/does-not-exist")
    assert r.status_code == 404 and "message" in r.json()
