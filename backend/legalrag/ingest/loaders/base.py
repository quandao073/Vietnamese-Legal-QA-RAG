from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from legalrag.schemas.document import LegalDocument
from legalrag.schemas.eval import EvalPair


class BaseLoader(ABC):
    name: str = "base"

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)

    def download(self) -> None:  # noqa: B027  # pragma: no cover - network
        """Fetch raw files into ``raw_dir``. Default: nothing to fetch."""

    @abstractmethod
    def iter_documents(self) -> Iterator[LegalDocument]: ...

    @abstractmethod
    def iter_eval_pairs(self) -> Iterator[EvalPair]: ...


class _EvalSource(Protocol):
    name: str

    def iter_eval_pairs(self) -> Iterator[EvalPair]: ...


def write_eval_files(loader: _EvalSource, out_dir: Path) -> list[Path]:
    """Write ``{name}_{split}.jsonl`` files (spec §2.3) and return the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[Path, list[str]] = {}
    for pair in loader.iter_eval_pairs():
        path = out_dir / f"{loader.name}_{pair.split}.jsonl"
        row = pair.model_dump(exclude={"split"})
        handles.setdefault(path, []).append(json.dumps(row, ensure_ascii=False))
    for path, lines in handles.items():
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list(handles)


def hf_download(repo_id: str, filename: str, dest: Path) -> Path:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    dest.write_bytes(Path(path).read_bytes())
    return dest
