"""Dataset loaders producing LegalDocument / EvalPair streams."""

from __future__ import annotations

from pathlib import Path

from legalrag.ingest.loaders.base import BaseLoader
from legalrag.ingest.loaders.hf_tamir_tax import TamirLoader
from legalrag.ingest.loaders.hf_yuitc_legal import YuitcLoader

SOURCES = ("tamir", "yuitc")


def get_loader(
    source: str, data_dir: Path, *, max_docs: int = 20000, eval_sample: int = 500, seed: int = 42
) -> BaseLoader:
    raw = data_dir / "raw" / source
    if source == "tamir":
        return TamirLoader(raw_dir=raw)
    if source == "yuitc":
        return YuitcLoader(
            raw_dir=raw,
            processed_dir=data_dir / "processed",
            max_docs=max_docs,
            eval_sample=eval_sample,
            seed=seed,
        )
    raise ValueError(f"unknown source: {source}")


__all__ = ["BaseLoader", "TamirLoader", "YuitcLoader", "get_loader", "SOURCES"]
