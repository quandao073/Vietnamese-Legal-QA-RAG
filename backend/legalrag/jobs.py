"""In-process background ingest jobs triggered from /admin/ingest (spec §4.1, §4.5)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
_jobs: dict[str, IngestJob] = {}
_lock = threading.Lock()
MAX_LOG_LINES = 200


@dataclass
class IngestJob:
    run_id: str
    source: str
    strategy: str
    max_docs: int | None
    status: str = "queued"  # queued | running | done | failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def add_log(self, line: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")
        del self.log[:-MAX_LOG_LINES]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "strategy": self.strategy,
            "max_docs": self.max_docs,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "log": self.log[-50:],
            "result": self.result,
            "error": self.error,
        }


def get_job(run_id: str) -> IngestJob | None:
    return _jobs.get(run_id)


def list_jobs(limit: int = 10) -> list[dict[str, Any]]:
    return [j.as_dict() for j in sorted(_jobs.values(), key=lambda j: -j.started_at)[:limit]]


def _run_ingest(
    source: str, strategy: str, max_docs: int | None, run_id: str, progress: Any
) -> dict[str, Any]:
    """Blocking: build + index (runs in a worker thread). Patched in tests."""
    from legalrag.ingest import indexer

    progress("build", 0, 0)
    bs = indexer.build([source], strategy=strategy, max_docs=max_docs)
    progress("built", bs.n_chunks, bs.n_chunks)
    st = indexer.index(
        strategy=strategy,
        resume=True,
        run_id=run_id,
        source=source,
        progress=lambda done, total: progress("index", done, total),
    )
    return {
        "n_docs": st.n_docs,
        "n_chunks": st.n_chunks,
        "n_skipped": st.n_skipped,
        "seconds": round(bs.seconds + st.seconds, 1),
        "collection": st.collection,
    }


def _after_success(strategy: str) -> None:
    from legalrag.services import reload_indexes

    reload_indexes(strategy)


async def start_ingest(source: str, strategy: str, max_docs: int | None) -> IngestJob:
    with _lock:
        if any(j.status in ("queued", "running") for j in _jobs.values()):
            raise RuntimeError(
                "Đang có một tiến trình nạp dữ liệu khác chạy. Vui lòng chờ hoàn tất."
            )
        job = IngestJob(
            run_id=uuid.uuid4().hex[:12], source=source, strategy=strategy, max_docs=max_docs
        )
        _jobs[job.run_id] = job

    def progress(stage: str, done: int, total: int) -> None:
        job.progress = {"stage": stage, "done": done, "total": total}
        if stage != "index" or done % 640 == 0 or done == total:
            job.add_log(f"{stage}: {done}/{total}")

    async def runner() -> None:
        job.status = "running"
        job.add_log(f"start source={source} strategy={strategy} max_docs={max_docs}")
        try:
            job.result = await asyncio.to_thread(
                _run_ingest, source, strategy, max_docs, job.run_id, progress
            )
            _after_success(strategy)
            job.status = "done"
            job.add_log(f"done: {job.result}")
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)[:1000]
            job.add_log(f"failed: {exc!r}")
            logger.exception("ingest job %s failed", job.run_id)
        finally:
            job.finished_at = time.time()

    asyncio.get_running_loop().create_task(runner())
    return job
