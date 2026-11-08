"""Query understanding with the small model (spec §3.4): intent, standalone question, rewrites."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from legalrag.config import Settings
from legalrag.generation.llm_client import LLMClient, LLMError, LLMJsonError, Usage
from legalrag.generation.prompts import QUERY_UNDERSTANDING_SYSTEM
from legalrag.schemas.chat import Message

logger = logging.getLogger(__name__)
Intent = Literal["legal_question", "out_of_scope", "chitchat", "unsafe"]


class Entities(BaseModel):
    law_names: list[str] = Field(default_factory=list)
    doc_numbers: list[str] = Field(default_factory=list)
    articles: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    intent: Intent
    standalone_question: str
    rewrites: list[str] = Field(default_factory=list)
    entities: Entities = Field(default_factory=Entities)
    language: str = "vi"
    fallback: bool = False

    @field_validator("rewrites")
    @classmethod
    def _cap(cls, v: list[str]) -> list[str]:
        return [r.strip() for r in v if r and r.strip()][:3]

    def queries(self) -> list[str]:
        """Standalone question first, then rewrites (deduplicated), at most 3 queries."""
        if self.intent != "legal_question":
            return []
        out: list[str] = []
        for q in [self.standalone_question, *self.rewrites]:
            if q and q not in out:
                out.append(q)
        return out[:3]


def fallback_understanding(question: str) -> QueryUnderstanding:
    return QueryUnderstanding(
        intent="legal_question", standalone_question=question, rewrites=[], fallback=True
    )


def _user_message(question: str, history: list[Message] | None) -> str:
    recent = (history or [])[-4:]  # at most the last 2 user/assistant turns
    if not recent:
        return f"Câu hỏi: {question}"
    lines = ["Lịch sử hội thoại gần nhất:"]
    for m in recent:
        role = "Người dùng" if m.role == "user" else "Trợ lý"
        lines.append(f"- {role}: {m.content[:600]}")
    lines.append(f"\nCâu hỏi mới: {question}")
    return "\n".join(lines)


async def understand(
    question: str,
    *,
    history: list[Message] | None,
    llm: LLMClient,
    settings: Settings,
) -> tuple[QueryUnderstanding, Usage]:
    if not settings.query_rewrite_enabled:
        return fallback_understanding(question), Usage()
    try:
        qu, usage = await llm.complete_json(
            model=settings.llm_model_small,
            system=QUERY_UNDERSTANDING_SYSTEM,
            messages=[{"role": "user", "content": _user_message(question, history)}],
            schema=QueryUnderstanding,
            max_tokens=900,
            effort="low",
        )
        if not qu.standalone_question.strip():
            qu.standalone_question = question
        return qu, usage
    except (LLMError, LLMJsonError) as exc:
        logger.warning("query understanding failed (%s); using raw question", exc)
        return fallback_understanding(question), Usage()
