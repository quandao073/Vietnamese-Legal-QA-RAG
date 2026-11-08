"""Qdrant-backed dense index (spec §2.7, §3.3)."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from legalrag.retrieval.types import Hit
from legalrag.schemas.document import LegalChunk

_PAYLOAD_EXCLUDE = {"text_for_bm25"}


class DenseStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def ensure_collection(self) -> None:
        if self.exists():
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            on_disk_payload=True,
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        )
        for field, schema in (
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("article", qm.PayloadSchemaType.KEYWORD),
            ("chunk_strategy", qm.PayloadSchemaType.KEYWORD),
            ("title", qm.PayloadSchemaType.TEXT),
        ):
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)

    def drop(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)

    def upsert(self, chunks: list[LegalChunk], vectors: np.ndarray, batch: int = 128) -> None:
        assert len(chunks) == len(vectors)
        for i in range(0, len(chunks), batch):
            points = [
                qm.PointStruct(
                    id=self.point_id(c.chunk_id),
                    vector=np.asarray(vectors[j], dtype=np.float32).tolist(),
                    payload=c.model_dump(exclude=_PAYLOAD_EXCLUDE),
                )
                for j, c in enumerate(chunks[i : i + batch], start=i)
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        found: set[str] = set()
        for i in range(0, len(chunk_ids), 500):
            batch = chunk_ids[i : i + 500]
            recs = self.client.retrieve(
                self.collection,
                ids=[self.point_id(c) for c in batch],
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            found.update(str(r.payload["chunk_id"]) for r in recs if r.payload)
        return found

    def count(self) -> int:
        if not self.exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    @staticmethod
    def _filter(filters: dict[str, Any] | None) -> qm.Filter | None:
        if not filters:
            return None
        must: list[qm.Condition] = []
        for key in ("doc_id", "article", "chunk_strategy"):
            if filters.get(key):
                must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=filters[key])))
        if filters.get("title"):
            must.append(qm.FieldCondition(key="title", match=qm.MatchText(text=filters["title"])))
        return qm.Filter(must=must) if must else None

    def search(
        self, query_vec: np.ndarray, top_n: int, filters: dict[str, Any] | None = None
    ) -> list[Hit]:
        if top_n <= 0 or not self.exists():
            return []
        res = self.client.query_points(
            self.collection,
            query=np.asarray(query_vec, dtype=np.float32).tolist(),
            limit=top_n,
            query_filter=self._filter(filters),
            with_payload=True,
            search_params=qm.SearchParams(hnsw_ef=128),
        )
        return [
            Hit(chunk_id=str(p.payload["chunk_id"]), score=float(p.score), payload=dict(p.payload))
            for p in res.points
            if p.payload
        ]

    def doc_titles(self) -> dict[str, str]:
        """chunk payload titles are copied by the indexer; used by context builder (fallback)."""
        return {}


@lru_cache
def get_qdrant_client() -> QdrantClient:
    from legalrag.config import get_settings

    return QdrantClient(url=get_settings().qdrant_url, timeout=30)


def qdrant_ping(client: QdrantClient | None = None) -> bool:
    try:
        (client or get_qdrant_client()).get_collections()
        return True
    except Exception:
        return False
