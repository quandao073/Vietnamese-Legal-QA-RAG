"""SQLAlchemy 2 models (spec §2.7, §4.1, §5.4)."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    doc_number: Mapped[str | None] = mapped_column(String(64))
    doc_type: Mapped[str | None] = mapped_column(String(64))
    issuing_body: Mapped[str | None] = mapped_column(String(128))
    issued_date: Mapped[date | None] = mapped_column(Date)
    effective_status: Mapped[str] = mapped_column(String(16), default="unknown", nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_doc_seq", "doc_id", "seq"),
        Index("ix_chunks_strategy", "chunk_strategy"),
        Index("ix_chunks_source_ids", "source_ids", postgresql_using="gin"),
    )

    chunk_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chapter: Mapped[str | None] = mapped_column(String(64))
    chapter_title: Mapped[str | None] = mapped_column(Text)
    article: Mapped[str | None] = mapped_column(String(64))
    article_title: Mapped[str | None] = mapped_column(Text)
    clause: Mapped[str | None] = mapped_column(String(64))
    point: Mapped[str | None] = mapped_column(String(64))
    source_ids: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    chunk_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str | None] = mapped_column(String(32))
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    n_docs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
    log: Mapped[str | None] = mapped_column(Text)


class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = (Index("ix_query_logs_created", "created_at"),)

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(128))
    request_id: Mapped[str | None] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    standalone_question: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(32))
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    cited_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    answer_md: Mapped[str | None] = mapped_column(Text)
    grounded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(16))
    usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    latency: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("query_logs.message_id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(32))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    config_name: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), default="retrieval", nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    n: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
