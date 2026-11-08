"""Evaluation datasets (spec §5.1) normalized to EvalItem; gold passages → gold chunk ids."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

DATASETS = ("yuitc_test", "tamir_test", "tamir_train", "refusal_set")
_FILES = {
    "yuitc_test": "yuitc_test.jsonl",
    "tamir_test": "tamir_test.jsonl",
    "tamir_train": "tamir_train.jsonl",
}


class EvalItem(BaseModel):
    id: str
    question: str
    gold_source_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    category: str = "legal"
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_groups: list[list[str]] = Field(default_factory=list)  # one group per gold passage

    def gold_sets(self) -> list[set[str]]:
        return [set(g) for g in self.gold_groups if g]


class _ChunkRepoLike(Protocol):
    def by_source_ids(self, source_ids: list[str], strategy: str) -> list[str]: ...


def _read(path: Path | str, text: str | None = None) -> list[EvalItem]:
    raw = text if text is not None else Path(path).read_text(encoding="utf-8")
    items = []
    for line in raw.splitlines():
        if line.strip():
            d = json.loads(line)
            d.setdefault("category", "legal")
            items.append(EvalItem(**{k: v for k, v in d.items() if k in EvalItem.model_fields}))
    return items


def load_dataset(name: str, data_dir: Path | str, limit: int | None = None) -> list[EvalItem]:
    if name == "refusal_set":
        text = (
            resources.files("legalrag.eval")
            .joinpath("data/refusal_set.jsonl")
            .read_text(encoding="utf-8")
        )
        items = _read("", text)
    elif name in _FILES:
        path = Path(data_dir) / "eval" / _FILES[name]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python -m legalrag.ingest build` first"
            )
        items = _read(path)
    else:
        raise ValueError(f"unknown dataset: {name} (choose from {', '.join(DATASETS)})")
    return items[:limit] if limit else items


def resolve_gold_chunks(
    items: list[EvalItem], repo: _ChunkRepoLike, strategy: str
) -> list[EvalItem]:
    """A gold passage may map to several chunks (fixed windows); any of them counts as a hit."""
    out: list[EvalItem] = []
    for it in items:
        groups = [sorted(set(repo.by_source_ids([sid], strategy))) for sid in it.gold_source_ids]
        groups = [g for g in groups if g]
        ids = sorted({c for g in groups for c in g})
        out.append(it.model_copy(update={"gold_chunk_ids": ids, "gold_groups": groups}))
    return out
