"""`python -m legalrag.ingest` — download / build / index / stats / add-file / download-models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from legalrag.config import get_settings
from legalrag.logging import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Ingest pipeline CLI")
logger = logging.getLogger("legalrag.ingest")


@app.callback()
def _init() -> None:
    setup_logging(get_settings().log_level)


@app.command()
def download(source: str = typer.Option("all", help="yuitc|tamir|all")) -> None:
    """Download raw datasets from Hugging Face into data/raw/."""
    from legalrag.ingest.indexer import _sources
    from legalrag.ingest.loaders import get_loader

    s = get_settings()
    for src in _sources([source]):
        loader = get_loader(src, Path(s.data_dir))
        logger.info("downloading %s → %s", src, loader.raw_dir)
        loader.download()
    typer.echo("download: done")


@app.command()
def build(
    source: str = typer.Option("all", help="yuitc|tamir|all"),
    strategy: str = typer.Option(None, help="chunk strategy (default: CHUNK_STRATEGY)"),
    max_docs: int = typer.Option(None, help="override CORPUS_MAX_DOCS (0 = all)"),
) -> None:
    """Normalize, parse and chunk documents into data/processed/ and write eval sets."""
    from legalrag.ingest import indexer

    s = get_settings()
    st = indexer.build([source], strategy=strategy or s.chunk_strategy, max_docs=max_docs)
    typer.echo(json.dumps(st.__dict__, ensure_ascii=False))


@app.command()
def index(
    strategy: str = typer.Option(None, help="chunk strategy (default: CHUNK_STRATEGY)"),
    resume: bool = typer.Option(False, "--resume", help="skip chunks already in Qdrant"),
    only: str = typer.Option(None, help="bm25|dense|db (comma-separated); default all"),
) -> None:
    """Write chunks into Postgres, embed into Qdrant and build the BM25 index."""
    from legalrag.ingest import indexer

    s = get_settings()
    parts = {p.strip() for p in only.split(",")} if only else None
    st = indexer.index(strategy=strategy or s.chunk_strategy, resume=resume, only=parts)
    typer.echo(json.dumps(st.__dict__, ensure_ascii=False))


@app.command()
def stats(
    strategy: str = typer.Option(None, help="chunk strategy (default: CHUNK_STRATEGY)"),
) -> None:
    """Document/chunk counts, token medians and article coverage per source."""
    from legalrag.ingest import indexer

    s = get_settings()
    typer.echo(
        json.dumps(indexer.stats(strategy or s.chunk_strategy), ensure_ascii=False, indent=2)
    )


@app.command("add-file")
def add_file(
    path: Path = typer.Option(..., exists=True, readable=True),
    title: str = typer.Option(...),
    url: str = typer.Option(None),
    strategy: str = typer.Option(None),
) -> None:
    """Add a custom plain-text legal document and index it."""
    from legalrag.ingest import indexer

    s = get_settings()
    st = indexer.add_file(path, title=title, url=url, strategy=strategy or s.chunk_strategy)
    typer.echo(json.dumps(st.__dict__, ensure_ascii=False))


@app.command("download-models")
def download_models() -> None:
    """Pre-download the embedder and reranker into HF_HOME (make models)."""
    s = get_settings()
    for name in (s.embedding_model, s.reranker_model):
        logger.info("downloading %s", name)
        typer.echo(f"{name}: {download_model(name)}")


def download_model(name: str) -> str:
    """Snapshot a HF model with safetensors weights, falling back to pytorch_model.bin."""
    from huggingface_hub import snapshot_download

    common = ["*.json", "*.txt", "sentencepiece.bpe.model", "*.model"]
    path = snapshot_download(
        name,
        allow_patterns=[*common, "*.safetensors"],
        ignore_patterns=["onnx/*", "openvino/*", "imgs/*"],
    )
    if not any(Path(path).glob("*.safetensors")):
        path = snapshot_download(name, allow_patterns=[*common, "pytorch_model.bin"])
    return path


def main() -> None:
    app()


if __name__ == "__main__":
    main()
