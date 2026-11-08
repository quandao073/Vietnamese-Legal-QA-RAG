"""Thin repositories over the SQLAlchemy models. All methods take an open Session."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import case, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from legalrag.db.models import Chunk, Document, EvalRun, Feedback, IngestRun, QueryLog
from legalrag.schemas.chat import QueryLogCreate
from legalrag.schemas.document import LegalChunk, LegalDocument

_BATCH = 500


class _Repo:
    def __init__(self, session: Session) -> None:
        self.s = session


class DocumentRepo(_Repo):
    def upsert_many(self, docs: list[LegalDocument]) -> None:
        for i in range(0, len(docs), _BATCH):
            rows = [
                {
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "title": d.title,
                    "doc_number": d.doc_number,
                    "doc_type": d.doc_type,
                    "issuing_body": d.issuing_body,
                    "issued_date": d.issued_date,
                    "effective_status": d.effective_status,
                    "url": d.url,
                    "extra": {k: v for k, v in d.extra.items() if k != "span_map"},
                }
                for d in docs[i : i + _BATCH]
            ]
            stmt = pg_insert(Document).values(rows)
            update_cols = {c: stmt.excluded[c] for c in rows[0] if c != "doc_id"}
            self.s.execute(stmt.on_conflict_do_update(index_elements=["doc_id"], set_=update_cols))

    def get(self, doc_id: str) -> Document | None:
        return self.s.get(Document, doc_id)

    def articles(self, doc_id: str, strategy: str) -> list[dict[str, Any]]:
        rows = self.s.execute(
            select(Chunk.chunk_id, Chunk.seq, Chunk.article, Chunk.article_title, Chunk.clause)
            .where(Chunk.doc_id == doc_id, Chunk.chunk_strategy == strategy)
            .order_by(Chunk.seq)
        ).all()
        return [dict(r._mapping) for r in rows]

    def count(self) -> int:
        return int(self.s.scalar(select(func.count()).select_from(Document)) or 0)


class ChunkRepo(_Repo):
    def upsert_many(self, chunks: list[LegalChunk]) -> None:
        for i in range(0, len(chunks), _BATCH):
            rows = [c.model_dump(exclude={"text_for_bm25"}) for c in chunks[i : i + _BATCH]]
            stmt = pg_insert(Chunk).values(rows)
            update_cols = {c: stmt.excluded[c] for c in rows[0] if c != "chunk_id"}
            self.s.execute(
                stmt.on_conflict_do_update(index_elements=["chunk_id"], set_=update_cols)
            )

    def get(self, chunk_id: str) -> Chunk | None:
        return self.s.get(Chunk, chunk_id)

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        rows = self.s.scalars(select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))).all()
        return {c.chunk_id: c for c in rows}

    def neighbors(self, chunk_id: str) -> tuple[Chunk | None, Chunk | None, Chunk | None]:
        cur = self.get(chunk_id)
        if cur is None:
            return None, None, None
        base = select(Chunk).where(
            Chunk.doc_id == cur.doc_id, Chunk.chunk_strategy == cur.chunk_strategy
        )
        prev = self.s.scalars(
            base.where(Chunk.seq < cur.seq).order_by(Chunk.seq.desc()).limit(1)
        ).first()
        nxt = self.s.scalars(
            base.where(Chunk.seq > cur.seq).order_by(Chunk.seq.asc()).limit(1)
        ).first()
        return prev, cur, nxt

    def by_source_ids(self, source_ids: list[str], strategy: str) -> list[str]:
        if not source_ids:
            return []
        rows = self.s.scalars(
            select(Chunk.chunk_id).where(
                Chunk.chunk_strategy == strategy, Chunk.source_ids.overlap(source_ids)
            )
        ).all()
        return list(rows)

    def count(self, strategy: str) -> int:
        return int(self.s.scalar(select(func.count()).where(Chunk.chunk_strategy == strategy)) or 0)

    def ids(self, strategy: str) -> list[str]:
        return list(
            self.s.scalars(select(Chunk.chunk_id).where(Chunk.chunk_strategy == strategy)).all()
        )

    def delete_strategy(self, strategy: str) -> int:
        res = self.s.execute(delete(Chunk).where(Chunk.chunk_strategy == strategy))
        return int(getattr(res, "rowcount", 0) or 0)

    def strategies(self) -> list[dict[str, Any]]:
        rows = self.s.execute(
            select(Chunk.chunk_strategy, func.count(), func.count(func.distinct(Chunk.doc_id)))
            .group_by(Chunk.chunk_strategy)
            .order_by(Chunk.chunk_strategy)
        ).all()
        return [{"strategy": r[0], "n_chunks": int(r[1]), "n_docs": int(r[2])} for r in rows]

    def stats(self, strategy: str) -> dict[str, Any]:
        """Counts, median tokens and article coverage per source (spec §2.8)."""
        q = (
            select(
                Document.source,
                func.count(),
                func.percentile_cont(0.5).within_group(Chunk.token_count),
                func.sum(case((Chunk.article.is_not(None), 1), else_=0)),
                func.sum(
                    case(
                        (
                            (Chunk.article.is_not(None)) & (Chunk.article != "Điều (không rõ số)"),
                            1,
                        ),
                        else_=0,
                    )
                ),
                func.count(func.distinct(Chunk.doc_id)),
            )
            .join(Document, Document.doc_id == Chunk.doc_id)
            .where(Chunk.chunk_strategy == strategy)
            .group_by(Document.source)
        )
        out: dict[str, Any] = {"strategy": strategy, "by_source": {}}
        total = 0
        for source, n, med, eff, strict, n_docs in self.s.execute(q).all():
            total += int(n)
            out["by_source"][source] = {
                "n_docs": int(n_docs),
                "n_chunks": int(n),
                "median_tokens": float(med or 0),
                "article_coverage_effective": round(int(eff or 0) / int(n), 4) if n else 0.0,
                "article_coverage_strict": round(int(strict or 0) / int(n), 4) if n else 0.0,
            }
        out["n_chunks"] = total
        return out


class IngestRunRepo(_Repo):
    def start(self, *, strategy: str, embedding_model: str | None, source: str | None) -> str:
        run = IngestRun(strategy=strategy, embedding_model=embedding_model, source=source)
        self.s.add(run)
        self.s.flush()
        return run.run_id

    def finish(
        self, run_id: str, *, n_docs: int, n_chunks: int, status: str, log: str | None
    ) -> None:
        run = self.s.get(IngestRun, run_id)
        if run is None:
            return
        run.n_docs, run.n_chunks, run.status, run.log = n_docs, n_chunks, status, log
        run.finished_at = datetime.now(UTC)

    def update_log(self, run_id: str, log: str, *, n_chunks: int | None = None) -> None:
        run = self.s.get(IngestRun, run_id)
        if run is not None:
            run.log = log
            if n_chunks is not None:
                run.n_chunks = n_chunks

    def get(self, run_id: str) -> IngestRun | None:
        return self.s.get(IngestRun, run_id)

    def latest(self, limit: int = 10) -> list[IngestRun]:
        return list(
            self.s.scalars(
                select(IngestRun).order_by(IngestRun.started_at.desc()).limit(limit)
            ).all()
        )


class QueryLogRepo(_Repo):
    def insert(self, log: QueryLogCreate) -> None:
        self.s.add(QueryLog(**log.model_dump()))

    def exists(self, message_id: str) -> bool:
        return self.s.get(QueryLog, message_id) is not None

    def get(self, message_id: str) -> QueryLog | None:
        return self.s.get(QueryLog, message_id)


class FeedbackRepo(_Repo):
    def insert(
        self, *, message_id: str, rating: int, reason: str | None, comment: str | None
    ) -> int:
        fb = Feedback(message_id=message_id, rating=rating, reason=reason, comment=comment)
        self.s.add(fb)
        self.s.flush()
        return fb.id

    def latest_negative(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.s.execute(
            select(Feedback, QueryLog.question, QueryLog.answer_md)
            .join(QueryLog, QueryLog.message_id == Feedback.message_id)
            .where(Feedback.rating < 0)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": fb.id,
                "message_id": fb.message_id,
                "reason": fb.reason,
                "comment": fb.comment,
                "created_at": fb.created_at.isoformat(),
                "question": question,
                "answer_md": answer_md,
            }
            for fb, question, answer_md in rows
        ]


class StatsRepo(_Repo):
    def summary(self) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(hours=24)
        q24 = int(self.s.scalar(select(func.count()).where(QueryLog.created_at >= since)) or 0)
        up = int(self.s.scalar(select(func.count()).where(Feedback.rating > 0)) or 0)
        down = int(self.s.scalar(select(func.count()).where(Feedback.rating < 0)) or 0)
        total_ms = QueryLog.latency["total_ms"].as_float()
        p50, p95 = self.s.execute(
            select(
                func.percentile_cont(0.5).within_group(total_ms),
                func.percentile_cont(0.95).within_group(total_ms),
            ).where(QueryLog.created_at >= since, total_ms.is_not(None))
        ).one()
        tokens = self.s.execute(
            select(
                func.coalesce(func.sum(QueryLog.usage["input_tokens"].as_integer()), 0),
                func.coalesce(func.sum(QueryLog.usage["output_tokens"].as_integer()), 0),
            ).where(QueryLog.created_at >= since)
        ).one()
        grounded = self.s.execute(
            select(func.count()).where(QueryLog.created_at >= since, QueryLog.grounded.is_(True))
        ).scalar()
        return {
            "n_documents": DocumentRepo(self.s).count(),
            "collections": ChunkRepo(self.s).strategies(),
            "queries_24h": q24,
            "grounded_24h": int(grounded or 0),
            "feedback_up": up,
            "feedback_down": down,
            "thumbs_up_ratio": round(up / (up + down), 3) if (up + down) else 0.0,
            "latency_p50_ms": float(p50) if p50 is not None else None,
            "latency_p95_ms": float(p95) if p95 is not None else None,
            "tokens_24h": {"input": int(tokens[0]), "output": int(tokens[1])},
        }


class EvalRunRepo(_Repo):
    def insert(
        self,
        *,
        config_name: str,
        params: dict[str, Any],
        dataset: str,
        metrics: dict[str, Any],
        n: int,
        cost_usd: float,
        kind: str = "retrieval",
        run_id: str | None = None,
    ) -> str:
        run = EvalRun(
            config_name=config_name,
            kind=kind,
            params=params,
            dataset=dataset,
            metrics=metrics,
            n=n,
            cost_usd=cost_usd,
            finished_at=datetime.now(UTC),
        )
        if run_id:
            run.run_id = run_id
        self.s.add(run)
        self.s.flush()
        return run.run_id

    def get(self, run_id: str) -> EvalRun | None:
        return self.s.get(EvalRun, run_id)

    def latest(self, limit: int = 50) -> list[EvalRun]:
        return list(
            self.s.scalars(select(EvalRun).order_by(EvalRun.started_at.desc()).limit(limit)).all()
        )
