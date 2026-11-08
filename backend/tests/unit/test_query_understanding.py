from __future__ import annotations

import pytest

from legalrag.config import Settings
from legalrag.generation.llm_client import LLMError, LLMJsonError, Usage
from legalrag.generation.prompts import PROMPT_VERSION, QUERY_UNDERSTANDING_SYSTEM
from legalrag.retrieval.query_understanding import QueryUnderstanding, understand
from legalrag.schemas.chat import Message


class FakeLLM:
    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.calls = result, error, []

    async def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result, Usage(10, 5)


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, anthropic_api_key="k", **kw)


SAMPLE = QueryUnderstanding(
    intent="legal_question",
    standalone_question="Thu nhập nào được miễn thuế thu nhập cá nhân?",
    rewrites=["các khoản thu nhập được miễn thuế TNCN", "đối tượng miễn thuế thu nhập cá nhân"],
    entities={
        "law_names": ["Luật Thuế thu nhập cá nhân"],
        "doc_numbers": [],
        "articles": ["Điều 4"],
        "topics": ["thuế", "miễn thuế"],
    },
    language="vi",
)


async def test_understand_calls_small_model_with_history_and_returns_result():
    llm = FakeLLM(result=SAMPLE)
    s = _settings()
    qu, usage = await understand(
        "còn với người nước ngoài thì sao?",
        history=[
            Message(role="user", content="Ai phải nộp thuế TNCN?"),
            Message(role="assistant", content="Cá nhân cư trú..."),
        ],
        llm=llm,
        settings=s,
    )
    assert qu == SAMPLE and usage.input_tokens == 10
    call = llm.calls[0]
    assert call["model"] == s.llm_model_small and call["schema"] is QueryUnderstanding
    assert call["system"] == QUERY_UNDERSTANDING_SYSTEM and call["effort"] == "low"
    user_text = call["messages"][0]["content"]
    assert (
        "Ai phải nộp thuế TNCN?" in user_text and "còn với người nước ngoài thì sao?" in user_text
    )


async def test_fallback_when_llm_fails():
    for err in (LLMError("rate_limit"), LLMJsonError("bad")):
        llm = FakeLLM(error=err)
        qu, usage = await understand("Thuế là gì?", history=None, llm=llm, settings=_settings())
        assert qu.intent == "legal_question" and qu.standalone_question == "Thuế là gì?"
        assert qu.rewrites == [] and usage.input_tokens == 0 and qu.fallback is True


async def test_disabled_rewrite_skips_llm():
    llm = FakeLLM(result=SAMPLE)
    qu, _ = await understand(
        "Thuế là gì?", history=None, llm=llm, settings=_settings(query_rewrite_enabled=False)
    )
    assert llm.calls == [] and qu.intent == "legal_question" and qu.rewrites == []


def test_rewrites_capped_at_three_and_prompt_versioned():
    qu = QueryUnderstanding(
        intent="legal_question", standalone_question="q", rewrites=["a", "b", "c", "d"]
    )
    assert qu.rewrites == ["a", "b", "c"]
    assert PROMPT_VERSION == "v1"
    assert "không trả lời" in QUERY_UNDERSTANDING_SYSTEM.lower()


@pytest.mark.parametrize("intent", ["out_of_scope", "chitchat", "unsafe"])
def test_queries_for_retrieval_only_for_legal_questions(intent):
    qu = QueryUnderstanding(intent=intent, standalone_question="x", rewrites=["y"])
    assert qu.queries() == []
    legal = QueryUnderstanding(
        intent="legal_question", standalone_question="x", rewrites=["y", "x", "z"]
    )
    assert legal.queries() == ["x", "y", "z"]  # dedup, standalone first, max 3
