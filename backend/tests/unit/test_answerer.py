from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from legalrag.config import Settings
from legalrag.generation.answerer import Answerer
from legalrag.generation.llm_client import LLMError, LLMResult, Usage
from legalrag.generation.prompts import DISCLAIMER, OUT_OF_SCOPE_TEMPLATE
from legalrag.retrieval.query_understanding import QueryUnderstanding
from legalrag.retrieval.types import Hit, RetrievalResult
from legalrag.schemas.chat import Message, QueryLogCreate
from legalrag.tokens import TokenCounter


def _hit(cid: str, text: str, score: float) -> Hit:
    return Hit(
        cid,
        1.0,
        {
            "doc_id": "tamir:doc_000002",
            "title": "Luật Thuế TNCN",
            "article": "Điều 4",
            "text": text,
            "seq": 1,
        },
        {"rerank_score": score},
    )


class FakeRetriever:
    def __init__(self, hits: list[Hit]):
        self.hits, self.calls = hits, []

    async def retrieve(self, question, qu=None, overrides=None):
        self.calls.append((question, qu, overrides))
        return RetrievalResult(
            qu, list(self.hits), list(self.hits), {"total_ms": 12.0}, strategy="article"
        )


class FakeLLM:
    """complete_json → query understanding; stream → scripted answer text."""

    def __init__(self, qu: QueryUnderstanding | Exception, answer: str | Exception):
        self.qu, self.answer, self.stream_calls, self.json_calls = qu, answer, [], []

    async def complete_json(self, **kw):
        self.json_calls.append(kw)
        if isinstance(self.qu, Exception):
            raise self.qu
        return self.qu, Usage(10, 5)

    async def stream(self, **kw) -> AsyncIterator[str | LLMResult]:
        self.stream_calls.append(kw)
        if isinstance(self.answer, Exception):
            raise self.answer
        for piece in self.answer.split(" "):
            yield piece + " "
        yield LLMResult(
            text=self.answer,
            usage=Usage(200, 50),
            model="claude-sonnet-5-5",
            stop_reason="end_turn",
        )


class FakeLogRepo:
    def __init__(self):
        self.rows: list[QueryLogCreate] = []

    def insert(self, log: QueryLogCreate) -> None:
        self.rows.append(log)


def _settings(**kw) -> Settings:
    base: dict = dict(anthropic_api_key="k", min_rerank_score=0.2, llm_max_tokens=300)
    base.update(kw)
    return Settings(_env_file=None, **base)


LEGAL_QU = QueryUnderstanding(
    intent="legal_question",
    standalone_question="Thu nhập nào được miễn thuế TNCN?",
    rewrites=["miễn thuế TNCN"],
)
HITS = [
    _hit(
        "c1",
        "Thu nhập từ chuyển nhượng bất động sản giữa vợ với chồng được miễn thuế thu nhập cá nhân.",
        0.92,
    ),
    _hit(
        "c2",
        "Thu nhập từ tiền lương làm việc ban đêm, làm thêm giờ được trả cao hơn được miễn thuế.",
        0.81,
    ),
    _hit("c3", "Thuế suất thuế giá trị gia tăng là 10%.", 0.35),
]


def make(hits, llm, settings=None, log_repo=None):
    return Answerer(
        retriever=FakeRetriever(hits),
        llm=llm,
        settings=settings or _settings(),
        counter=TokenCounter(None),
        log_repo=log_repo,
    )


async def _collect(ans, question, **kw):
    return [e async for e in ans.answer(question, **kw)]


async def test_happy_path_event_sequence_and_citation_cleanup():
    llm = FakeLLM(
        LEGAL_QU,
        "**Kết luận**: Chuyển nhượng bất động sản giữa vợ với chồng được miễn thuế [#1]. "
        "Thuế GTGT 10% [#9].",
    )
    log = FakeLogRepo()
    ans = make(HITS, llm, log_repo=log)
    events = await _collect(
        ans, "Thu nhập nào được miễn thuế TNCN?", session_id="s1", history=None, overrides=None
    )
    kinds = [e.event for e in events]
    assert kinds[:4] == ["status", "status", "status", "sources"] or kinds[:3] == [
        "status",
        "status",
        "sources",
    ]
    assert kinds.count("token") >= 2 and kinds[-2:] == ["citations", "done"]
    assert [e.data["stage"] for e in events if e.event == "status"][:1] == ["understanding"]
    sources = next(e for e in events if e.event == "sources").data["sources"]
    assert (
        sources[0]["idx"] == 1
        and sources[0]["chunk_id"] == "c1"
        and sources[0]["display_source"].startswith("Luật Thuế TNCN")
    )
    cit = next(e for e in events if e.event == "citations").data
    assert cit["invalid"] == [9] and 1 in cit["verified"]
    done = events[-1].data
    assert (
        done["grounded"] is True and "[#9]" not in done["answer_md"] and "[#1]" in done["answer_md"]
    )
    assert done["answer_md"].endswith(DISCLAIMER)
    assert (
        done["usage"]["output_tokens"] == 55
        and done["message_id"]
        and done["intent"] == "legal_question"
    )
    assert "total_ms" in done["latency_ms"] and done["model"] == "claude-sonnet-5-5"
    # retrieval used the understanding; generation used the main model with the sources
    assert llm.stream_calls[0]["model"] == _settings().llm_model_main
    assert "[Nguồn #1]" in llm.stream_calls[0]["messages"][-1]["content"]
    assert (
        len(log.rows) == 1
        and log.rows[0].grounded is True
        and log.rows[0].cited_chunk_ids == ["c1"]
    )


async def test_out_of_scope_skips_retrieval_and_main_llm():
    qu = QueryUnderstanding(intent="out_of_scope", standalone_question="Hôm nay trời có mưa không?")
    llm = FakeLLM(qu, "should not be used")
    ans = make(HITS, llm)
    events = await _collect(
        ans, "Hôm nay trời có mưa không?", session_id=None, history=None, overrides=None
    )
    assert ans.retriever.calls == [] and llm.stream_calls == []
    done = events[-1].data
    assert done["grounded"] is False and done["intent"] == "out_of_scope"
    assert done["answer_md"].startswith(OUT_OF_SCOPE_TEMPLATE)


async def test_low_rerank_score_returns_no_evidence_template_without_llm():
    weak = [_hit("c3", "Thuế suất thuế giá trị gia tăng là 10%.", 0.12), _hit("c4", "x", 0.05)]
    llm = FakeLLM(LEGAL_QU, "unused")
    ans = make(weak, llm)
    events = await _collect(
        ans, "Điều kiện kết hôn là gì?", session_id=None, history=None, overrides=None
    )
    done = events[-1].data
    assert llm.stream_calls == [] and done["grounded"] is False
    assert "không tìm thấy căn cứ" in done["answer_md"]
    sources = next(e for e in events if e.event == "sources").data["sources"]
    assert [s["chunk_id"] for s in sources] == ["c3", "c4"]  # nearest sources still shown (≤3)


async def test_no_candidates_is_not_grounded():
    llm = FakeLLM(LEGAL_QU, "unused")
    ans = make([], llm)
    events = await _collect(ans, "Câu hỏi lạ?", session_id=None, history=None, overrides=None)
    assert events[-1].data["grounded"] is False and llm.stream_calls == []


async def test_llm_error_emits_error_event_and_logs():
    llm = FakeLLM(LEGAL_QU, LLMError("auth"))
    log = FakeLogRepo()
    ans = make(HITS, llm, log_repo=log)
    events = await _collect(
        ans, "Thu nhập nào được miễn thuế TNCN?", session_id=None, history=None, overrides=None
    )
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "auth" and "khóa api" in events[-1].data["message"].lower()
    assert len(log.rows) == 1 and log.rows[0].error and log.rows[0].grounded is False


async def test_history_passed_to_understanding_and_last_turns_to_generation():
    llm = FakeLLM(LEGAL_QU, "Trả lời [#1].")
    ans = make(HITS, llm)
    history = [
        Message(role="user", content="u1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u2"),
        Message(role="assistant", content="a2"),
        Message(role="user", content="u3"),
        Message(role="assistant", content="a3"),
    ]
    await _collect(ans, "còn vợ chồng?", session_id=None, history=history, overrides=None)
    qu_prompt = llm.json_calls[0]["messages"][0]["content"]
    assert "u3" in qu_prompt and "a3" in qu_prompt and "u1" not in qu_prompt
    gen_messages = llm.stream_calls[0]["messages"]
    assert [m["role"] for m in gen_messages] == ["user", "assistant", "user", "assistant", "user"]
    assert gen_messages[0]["content"] == "u2" and gen_messages[-1]["role"] == "user"


async def test_answer_sync_collects_response():
    llm = FakeLLM(LEGAL_QU, "Kết luận [#1].")
    ans = make(HITS, llm)
    resp = await ans.answer_sync(
        "Thu nhập nào được miễn thuế TNCN?", session_id=None, history=None, overrides=None
    )
    assert resp.grounded is True and resp.citations.verified == [1] and resp.sources[0].idx == 1
    assert resp.answer_md.startswith("Kết luận [#1].")


@pytest.mark.parametrize("intent,template_key", [("chitchat", "chitchat"), ("unsafe", "unsafe")])
async def test_other_intents_use_templates(intent, template_key):
    from legalrag.generation.prompts import INTENT_TEMPLATES

    llm = FakeLLM(QueryUnderstanding(intent=intent, standalone_question="x"), "unused")
    ans = make(HITS, llm)
    events = await _collect(ans, "x", session_id=None, history=None, overrides=None)
    assert events[-1].data["answer_md"].startswith(INTENT_TEMPLATES[template_key])
