"""API / domain models for chat, logging and feedback (spec §3.9, §4.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 2000


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatOptions(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_rerank: bool | None = None
    use_rewrite: bool | None = None
    strategy: str | None = Field(default=None, max_length=64)
    mode: Literal["hybrid", "dense", "bm25"] | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, max_length=128)
    history: list[Message] = Field(default_factory=list, max_length=8)
    options: ChatOptions | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Câu hỏi không được để trống.")
        return v


class SourceOut(BaseModel):
    idx: int
    chunk_id: str
    doc_id: str
    display_source: str
    url: str | None = None
    snippet: str
    rerank_score: float | None = None


class CitationReport(BaseModel):
    verified: list[int] = Field(default_factory=list)
    weak: list[int] = Field(default_factory=list)
    invalid: list[int] = Field(default_factory=list)


class ChatResponse(BaseModel):
    message_id: str
    answer_md: str
    grounded: bool
    intent: str
    sources: list[SourceOut] = Field(default_factory=list)
    citations: CitationReport = Field(default_factory=CitationReport)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    model: str | None = None
    prompt_version: str = "v1"
    standalone_question: str | None = None


class ChatEvent(BaseModel):
    event: Literal["status", "sources", "token", "citations", "done", "error"]
    data: dict[str, Any]


class QueryLogCreate(BaseModel):
    message_id: str
    session_id: str | None = None
    request_id: str | None = None
    question: str
    standalone_question: str | None = None
    intent: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    answer_md: str | None = None
    grounded: bool = False
    model: str | None = None
    prompt_version: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    latency: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class FeedbackIn(BaseModel):
    message_id: str = Field(max_length=64)
    rating: Literal[1, -1]
    reason: Literal["wrong_citation", "incomplete", "hallucination", "other"] | None = None
    comment: str | None = Field(default=None, max_length=2000)
