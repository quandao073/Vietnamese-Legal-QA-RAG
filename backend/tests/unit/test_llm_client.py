from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from legalrag.cache import MemoryCache
from legalrag.config import Settings
from legalrag.generation.llm_client import LLMClient, LLMError, LLMJsonError, LLMResult


def _settings(**kw) -> Settings:
    base = dict(anthropic_api_key="k", llm_cache_enabled=False, llm_max_concurrency=2)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _message(text: str, *, cache_read: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
        model="claude-sonnet-5-5",
    )


def _http_error(cls, status: int):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com"))
    return cls("boom", response=resp, body=None)


class FakeStream:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    async def text_stream(self):
        for c in self.chunks:
            yield c

    async def get_final_message(self):
        return _message("".join(self.chunks))


class FakeAnthropic:
    """Minimal stand-in for anthropic.AsyncAnthropic: scripted responses / exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def no_sleep(monkeypatch):
    async def _sleep(_):
        return None

    monkeypatch.setattr("legalrag.generation.llm_client.asyncio.sleep", _sleep)


async def test_complete_returns_text_and_usage_without_temperature():
    fake = FakeAnthropic([_message("Xin chào")])
    llm = LLMClient(
        _settings(price_input_per_m=2.0, price_output_per_m=10.0), cache=None, client=fake
    )
    res = await llm.complete(
        model="claude-sonnet-5-5",
        system="S",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=50,
    )
    assert isinstance(res, LLMResult) and res.text == "Xin chào" and res.cached is False
    assert res.usage.input_tokens == 100 and res.usage.output_tokens == 20
    assert res.usage.cost_usd == pytest.approx(100 / 1e6 * 2.0 + 20 / 1e6 * 10.0)
    call = fake.calls[0]
    assert (
        "temperature" not in call
        and call["max_tokens"] == 50
        and call["model"] == "claude-sonnet-5-5"
    )
    assert call["system"] == [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]


async def test_retries_on_rate_limit_then_succeeds(no_sleep):
    fake = FakeAnthropic(
        [
            _http_error(anthropic.RateLimitError, 429),
            _http_error(anthropic.OverloadedError, 529),
            _message("ok"),
        ]
    )
    llm = LLMClient(_settings(), cache=None, client=fake)
    res = await llm.complete(
        model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
    )
    assert res.text == "ok" and len(fake.calls) == 3


async def test_gives_up_after_max_retries(no_sleep):
    fake = FakeAnthropic([_http_error(anthropic.RateLimitError, 429)] * 6)
    llm = LLMClient(_settings(), cache=None, client=fake)
    with pytest.raises(LLMError) as ei:
        await llm.complete(
            model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    assert ei.value.code == "rate_limit" and len(fake.calls) == 5
    assert "quá tải" in ei.value.message_vi or "giới hạn" in ei.value.message_vi


async def test_auth_error_is_not_retried_and_has_vietnamese_message():
    fake = FakeAnthropic([_http_error(anthropic.AuthenticationError, 401)])
    llm = LLMClient(_settings(), cache=None, client=fake)
    with pytest.raises(LLMError) as ei:
        await llm.complete(
            model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    assert ei.value.code == "auth" and len(fake.calls) == 1
    assert "API" in ei.value.message_vi and "khóa" in ei.value.message_vi.lower()


async def test_response_cache_hits_on_second_call():
    fake = FakeAnthropic([_message("cached answer")])
    llm = LLMClient(_settings(llm_cache_enabled=True), cache=MemoryCache(), client=fake)
    args = dict(model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10)
    first = await llm.complete(**args)
    second = await llm.complete(**args)
    assert first.cached is False and second.cached is True and second.text == "cached answer"
    assert second.usage.input_tokens == 0 and second.usage.cost_usd == 0  # no new tokens spent
    assert len(fake.calls) == 1
    # a different prompt misses the cache
    fake.script.append(_message("other"))
    third = await llm.complete(**{**args, "messages": [{"role": "user", "content": "q2"}]})
    assert third.cached is False and len(fake.calls) == 2


async def test_cache_disabled_by_default_even_if_cache_present():
    fake = FakeAnthropic([_message("a"), _message("b")])
    llm = LLMClient(_settings(llm_cache_enabled=False), cache=MemoryCache(), client=fake)
    args = dict(model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10)
    await llm.complete(**args)
    res = await llm.complete(**args)
    assert res.text == "b" and len(fake.calls) == 2
    # explicit opt-in per call
    fake.script.append(_message("c"))
    r1 = await llm.complete(**args, use_cache=True)
    r2 = await llm.complete(**args, use_cache=True)
    assert r1.text == "c" and r2.cached is True


class Out(BaseModel):
    intent: str
    rewrites: list[str]


async def test_complete_json_uses_structured_output_and_validates():
    payload = {"intent": "legal_question", "rewrites": ["a", "b"]}
    fake = FakeAnthropic([_message(json.dumps(payload))])
    llm = LLMClient(_settings(), cache=None, client=fake)
    obj, usage = await llm.complete_json(
        model="m",
        system="S",
        messages=[{"role": "user", "content": "q"}],
        schema=Out,
        max_tokens=200,
    )
    assert isinstance(obj, Out) and obj.rewrites == ["a", "b"] and usage.output_tokens == 20
    fmt = fake.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert "tool_choice" not in fake.calls[0] and "tools" not in fake.calls[0]


async def test_complete_json_raises_on_invalid_payload():
    fake = FakeAnthropic([_message('{"intent": 1}')])
    llm = LLMClient(_settings(), cache=None, client=fake)
    with pytest.raises(LLMJsonError):
        await llm.complete_json(
            model="m",
            system="S",
            messages=[{"role": "user", "content": "q"}],
            schema=Out,
            max_tokens=200,
        )


async def test_stream_yields_deltas_then_result():
    fake = FakeAnthropic([FakeStream(["Xin ", "chào"])])
    llm = LLMClient(_settings(), cache=None, client=fake)
    items = [
        x
        async for x in llm.stream(
            model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    ]
    assert items[:-1] == ["Xin ", "chào"]
    assert isinstance(items[-1], LLMResult) and items[-1].text == "Xin chào"


async def test_connection_error_maps_to_unknown_after_retries(no_sleep):
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    fake = FakeAnthropic([err] * 5)
    llm = LLMClient(_settings(), cache=None, client=fake)
    with pytest.raises(LLMError) as ei:
        await llm.complete(
            model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    assert ei.value.code == "connection" and len(fake.calls) == 5


class FailingEnterStream(FakeStream):
    """The SDK sends the request inside __aenter__; errors surface there, not while iterating."""

    def __init__(self, exc):
        super().__init__([])
        self.exc = exc

    async def __aenter__(self):
        raise self.exc


async def test_stream_maps_auth_error_raised_on_enter():
    fake = FakeAnthropic([FailingEnterStream(_http_error(anthropic.AuthenticationError, 401))])
    llm = LLMClient(_settings(), cache=None, client=fake)
    with pytest.raises(LLMError) as ei:
        [
            x
            async for x in llm.stream(
                model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
            )
        ]
    assert ei.value.code == "auth" and len(fake.calls) == 1


async def test_stream_retries_rate_limit_on_enter(no_sleep):
    fake = FakeAnthropic(
        [FailingEnterStream(_http_error(anthropic.RateLimitError, 429)), FakeStream(["ok"])]
    )
    llm = LLMClient(_settings(), cache=None, client=fake)
    items = [
        x
        async for x in llm.stream(
            model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    ]
    assert items[0] == "ok" and len(fake.calls) == 2


async def test_stream_uses_response_cache_when_enabled():
    fake = FakeAnthropic([FakeStream(["Xin ", "chào"])])
    llm = LLMClient(_settings(llm_cache_enabled=True), cache=MemoryCache(), client=fake)
    args = dict(model="m", system="S", messages=[{"role": "user", "content": "q"}], max_tokens=10)
    first = [x async for x in llm.stream(**args)]
    second = [x async for x in llm.stream(**args)]
    assert first[-1].cached is False and second[-1].cached is True
    assert second[0] == "Xin chào" and second[-1].text == "Xin chào"
    assert second[-1].usage.output_tokens == 0
    assert len(fake.calls) == 1  # second call never hit the API
