"""Evaluation data schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalPair(BaseModel):
    """One normalized evaluation example (spec §5.1)."""

    id: str
    question: str
    gold_source_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    category: str = "legal"
    split: str = "test"
