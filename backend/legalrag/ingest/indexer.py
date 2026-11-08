"""Indexer (spec §2.7): documents → chunks → Postgres + Qdrant + BM25, with checkpoint/resume."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from legalrag.config import Settings, get_settings
from legalrag.db.repositories import ChunkRepo, DocumentRepo, IngestRunRepo
from legalrag.db.session import session_scope
from legalrag.ingest.chunking import Chunker, build_chunks, get_chunker
from legalrag.ingest.loaders import SOURCES, get_loader
from legalrag.ingest.loaders.base import write_eval_files
from legalrag.retrieval.dense import DenseStore, get_qdrant_client
from legalrag.retrieval.embedder import EmbedderProtocol, FakeEmbedder
from legalrag.retrieval.lexical import BM25Index
from legalrag.schemas.document import LegalChunk, LegalDocument
from legalrag.tokens import TokenCounter, get_token_counter

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


# ---- jsonl helpers -------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[BaseModel], *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)


# ---- paths -----------------------------------------------------------------------------------


@dataclass
class Paths:
    data_dir: Path

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval(self) -> Path:
        return self.data_dir / "eval"

    def documents(self) -> Path:
        return self.processed / "documents.jsonl"

    def chunks(self, strategy: str) -> Path:
        return self.processed / f"chunks_{strategy}.jsonl"

    def bm25(self, strategy: str) -> Path:
        return self.data_dir / "index" / "bm25" / strategy


# ---- state --------------------------------------------------------------------------------------


@dataclass
class IngestState:
    run_id: str
    strategy: str
    processed_dir: Path
    done: int = 0
    total: int = 0
    status: str = "running"
    last_batch: int = -1

    @property
    def path(self) -> Path:
        return self.processed_dir / f"ingest_state_{self.run_id}.json"

    def update(self, *, done: int, total: int, status: str, last_batch: int) -> None:
        self.done, self.total, self.status, self.last_batch = done, total, status, last_batch
        d = asdict(self)
        d.pop("processed_dir")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d), encoding="utf-8")

    @classmethod
    def load(cls, processed_dir: Path, run_id: str) -> IngestState | None:
        p = processed_dir / f"ingest_state_{run_id}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(processed_dir=processed_dir, **d)


# ---- build ------------------------------------------------------------------------------------


@dataclass
class BuildStats:
    n_docs: int
    n_chunks: int
    strategy: str
    seconds: float


def _sources(sources: list[str]) -> list[str]:
    return list(SOURCES) if "all" in sources else sources


def _source_of(doc_id: str) -> str:
    """Source prefix of a doc id (``tamir:doc_000001`` -> ``tamir``)."""
    return doc_id.split(":", 1)[0]


def build(
    sources: list[str],
    *,
    strategy: str,
    settings: Settings | None = None,
    counter: TokenCounter | None = None,
    max_docs: int | None = None,
) -> BuildStats:
    """Load → normalize → structure → chunk.

    Writes documents.jsonl, chunks_{strategy}.jsonl and the eval jsonl files.
    """
    s = settings or get_settings()
    counter = counter or get_token_counter(s.embedding_model)
    paths = Paths(Path(s.data_dir))
    chunker = get_chunker(strategy, counter=counter, max_tokens=s.max_chunk_tokens)
    t0 = time.time()
    n_docs = n_chunks = 0
    docs_path, chunks_path = paths.documents(), paths.chunks(strategy)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt = _sources(sources)
    # Processed files feed the BM25 build, so documents/chunks of sources that are NOT
    # being rebuilt must survive a partial `build --source X` (review finding C1).
    kept_docs = [
        d
        for d in (read_jsonl(docs_path, LegalDocument) if docs_path.exists() else [])
        if _source_of(d.doc_id) not in rebuilt
    ]
    kept_chunks = [
        c
        for c in (read_jsonl(chunks_path, LegalChunk) if chunks_path.exists() else [])
        if _source_of(c.doc_id) not in rebuilt
    ]
    with (
        open(docs_path, "w", encoding="utf-8") as fd,
        open(chunks_path, "w", encoding="utf-8") as fc,
    ):
        for d in kept_docs:
            fd.write(d.model_dump_json(exclude={"extra": {"span_map"}}) + "\n")
        for c in kept_chunks:
            fc.write(c.model_dump_json() + "\n")
        for source in rebuilt:
            loader = get_loader(
                source,
                Path(s.data_dir),
                max_docs=max_docs if max_docs is not None else s.corpus_max_docs,
                eval_sample=s.eval_yuitc_sample,
                seed=s.random_seed,
            )
            for doc in loader.iter_documents():
                chunks = build_chunks(doc, chunker)
                fd.write(doc.model_dump_json(exclude={"extra": {"span_map"}}) + "\n")
                for c in chunks:
                    fc.write(c.model_dump_json() + "\n")
                n_docs += 1
                n_chunks += len(chunks)
                if n_docs % 1000 == 0:
                    logger.info("built %d docs / %d chunks", n_docs, n_chunks)
            write_eval_files(loader, paths.eval)
    return BuildStats(n_docs=n_docs, n_chunks=n_chunks, strategy=strategy, seconds=time.time() - t0)


# ---- index ------------------------------------------------------------------------------------


@dataclass
class IndexStats:
    run_id: str
    strategy: str
    n_docs: int
    n_chunks: int
    n_skipped: int
    seconds: float
    collection: str


def make_embedder(settings: Settings) -> EmbedderProtocol:
    m = re.match(r"^fake-(\d+)$", settings.embedding_model)
    if m:
        return FakeEmbedder(dim=int(m.group(1)))
    from legalrag.retrieval.embedder import get_embedder

    return get_embedder()


def index(
    *,
    strategy: str,
    settings: Settings | None = None,
    embedder: EmbedderProtocol | None = None,
    resume: bool = False,
    only: set[str] | None = None,
    run_id: str | None = None,
    source: str | None = None,
    docs: list[LegalDocument] | None = None,
    chunks: list[LegalChunk] | None = None,
    progress: Any = None,
) -> IndexStats:
    """Index chunks_{strategy}.jsonl into Postgres, Qdrant and BM25 (all by default)."""
    s = settings or get_settings()
    only = only or {"db", "dense", "bm25"}
    paths = Paths(Path(s.data_dir))
    t0 = time.time()
    run_id = run_id or uuid.uuid4().hex[:12]
    embedder = embedder or make_embedder(s)
    collection = s.collection_name(strategy)

    all_docs = docs if docs is not None else list(read_jsonl(paths.documents(), LegalDocument))
    all_chunks = (
        chunks if chunks is not None else list(read_jsonl(paths.chunks(strategy), LegalChunk))
    )
    for c in all_chunks:
        c.embedding_model = s.embedding_model
    state = IngestState(run_id=run_id, strategy=strategy, processed_dir=paths.processed)
    state.update(done=0, total=len(all_chunks), status="running", last_batch=-1)

    with session_scope() as db:
        IngestRunRepo(db).start(strategy=strategy, embedding_model=s.embedding_model, source=source)
        # reuse run_id as the DB key for traceability
        run = IngestRunRepo(db).latest(1)[0]
        run.run_id = run_id

    n_skipped = 0
    try:
        if "db" in only:
            with session_scope() as db:
                DocumentRepo(db).upsert_many(all_docs)
                ChunkRepo(db).upsert_many(all_chunks)
            logger.info("postgres: %d docs, %d chunks", len(all_docs), len(all_chunks))

        if "dense" in only:
            store = DenseStore(get_qdrant_client(), collection, dim=embedder.dim)
            store.ensure_collection()
            todo = all_chunks
            if resume:
                existing = store.existing_ids([c.chunk_id for c in all_chunks])
                todo = [c for c in all_chunks if c.chunk_id not in existing]
                n_skipped = len(all_chunks) - len(todo)
                logger.info("resume: %d already in Qdrant, %d to embed", n_skipped, len(todo))
            batch = 128 if s.device == "cuda" else 32
            done = n_skipped
            for bi, i in enumerate(range(0, len(todo), batch)):
                part = todo[i : i + batch]
                vecs = embedder.encode_passages([c.text for c in part], batch_size=batch)
                store.upsert(part, vecs, batch=128)
                done += len(part)
                state.update(done=done, total=len(all_chunks), status="running", last_batch=bi)
                if progress:
                    progress(done, len(all_chunks))
                if bi % 20 == 0:
                    logger.info("dense: %d/%d", done, len(all_chunks))

        if "bm25" in only:
            BM25Index.build(all_chunks, paths.bm25(strategy))
            logger.info("bm25: %d chunks -> %s", len(all_chunks), paths.bm25(strategy))

        state.update(
            done=len(all_chunks), total=len(all_chunks), status="done", last_batch=state.last_batch
        )
        with session_scope() as db:
            IngestRunRepo(db).finish(
                run_id, n_docs=len(all_docs), n_chunks=len(all_chunks), status="done", log="ok"
            )
    except BaseException as exc:
        state.update(
            done=state.done, total=state.total, status="failed", last_batch=state.last_batch
        )
        with session_scope() as db:
            IngestRunRepo(db).finish(
                run_id,
                n_docs=len(all_docs),
                n_chunks=state.done,
                status="failed",
                log=repr(exc)[:2000],
            )
        raise
    return IndexStats(
        run_id=run_id,
        strategy=strategy,
        n_docs=len(all_docs),
        n_chunks=len(all_chunks),
        n_skipped=n_skipped,
        seconds=time.time() - t0,
        collection=collection,
    )


# ---- stats / add-file ----------------------------------------------------------------------


def stats(strategy: str, *, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    paths = Paths(Path(s.data_dir))
    with session_scope() as db:
        out = ChunkRepo(db).stats(strategy)
    client = get_qdrant_client()
    collection = s.collection_name(strategy)
    out["collection"] = collection
    out["qdrant_points"] = (
        int(client.count(collection, exact=True).count)
        if client.collection_exists(collection)
        else 0
    )
    ids_path = paths.bm25(strategy) / "ids.json"
    out["bm25_size"] = (
        len(json.loads(ids_path.read_text(encoding="utf-8"))) if ids_path.exists() else 0
    )
    return out


def slugify_title(title: str) -> str:
    import unicodedata

    base = unicodedata.normalize("NFD", title)
    base = (
        "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
        .replace("đ", "d")
        .replace("Đ", "D")
    )
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "custom"


def add_file(
    path: Path,
    *,
    title: str,
    url: str | None,
    strategy: str,
    settings: Settings | None = None,
    embedder: EmbedderProtocol | None = None,
    counter: TokenCounter | None = None,
) -> IndexStats:
    """Add one custom text document: append to documents/chunks jsonl and index its chunks."""
    s = settings or get_settings()
    counter = counter or get_token_counter(s.embedding_model)
    paths = Paths(Path(s.data_dir))
    doc_id = f"custom:{slugify_title(title)}"
    doc = LegalDocument(
        doc_id=doc_id,
        source="custom",
        title=title,
        url=url,
        raw_text=Path(path).read_text(encoding="utf-8"),
        extra={"source_id": doc_id, "path": str(path)},
    )
    chunker: Chunker = get_chunker(strategy, counter=counter, max_tokens=s.max_chunk_tokens)
    chunks = build_chunks(doc, chunker)
    write_jsonl(paths.documents(), [doc], append=paths.documents().exists())
    write_jsonl(paths.chunks(strategy), chunks, append=paths.chunks(strategy).exists())
    # index only the new chunks into DB/Qdrant, then rebuild BM25 over everything
    res = index(
        strategy=strategy,
        settings=s,
        embedder=embedder,
        only={"db", "dense"},
        source="custom",
        docs=[doc],
        chunks=chunks,
    )
    all_chunks = list(read_jsonl(paths.chunks(strategy), LegalChunk))
    BM25Index.build(all_chunks, paths.bm25(strategy))
    return res
