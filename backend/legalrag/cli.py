"""`python -m legalrag.cli ask "câu hỏi"` — answer one question from the terminal (spec §3.11)."""

from __future__ import annotations

import asyncio
import json
import sys

import typer

from legalrag.config import get_settings
from legalrag.logging import setup_logging
from legalrag.retrieval.types import RetrievalOverrides

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root() -> None:
    """Vietnamese Legal RAG command line."""


@app.command()
def ask(
    question: str = typer.Argument(..., help="Câu hỏi pháp luật bằng tiếng Việt"),
    strategy: str = typer.Option(None, help="chunk strategy (default: CHUNK_STRATEGY)"),
    top_k: int = typer.Option(None, help="number of sources"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite"),
    mode: str = typer.Option("hybrid", help="hybrid|dense|bm25"),
    as_json: bool = typer.Option(False, "--json", help="print the full ChatResponse as JSON"),
) -> None:
    s = get_settings()
    setup_logging("WARNING" if not as_json else "ERROR")
    overrides = RetrievalOverrides(
        top_k=top_k,
        use_rerank=False if no_rerank else None,
        use_rewrite=False if no_rewrite else None,
        strategy=strategy,
        mode=mode,  # type: ignore[arg-type]
    )
    asyncio.run(_run(question, overrides, as_json, s.chunk_strategy))


async def _run(
    question: str, overrides: RetrievalOverrides, as_json: bool, default_strategy: str
) -> None:
    from legalrag.services import get_answerer

    answerer = get_answerer()
    if as_json:
        resp = await answerer.answer_sync(
            question, session_id=None, history=None, overrides=overrides
        )
        typer.echo(resp.model_dump_json(indent=2))
        return
    sources: list[dict] = []
    async for ev in answerer.answer(question, session_id=None, history=None, overrides=overrides):
        if ev.event == "status":
            typer.secho(f"… {ev.data['stage']}", fg="blue", err=True)
        elif ev.event == "sources":
            sources = ev.data["sources"]
        elif ev.event == "token":
            sys.stdout.write(ev.data["text"])
            sys.stdout.flush()
        elif ev.event == "citations":
            typer.echo("\n")
            typer.secho(f"Trích dẫn: {json.dumps(ev.data, ensure_ascii=False)}", fg="yellow")
        elif ev.event == "done":
            typer.secho(
                f"grounded={ev.data['grounded']} intent={ev.data['intent']} "
                f"latency={ev.data['latency_ms'].get('total_ms')}ms usage={ev.data['usage']}",
                fg="green",
            )
            for src in sources:
                score = src.get("rerank_score")
                typer.echo(
                    f"  [#{src['idx']}] {src['display_source']}  ({score:.3f})"
                    if score is not None
                    else f"  [#{src['idx']}] {src['display_source']}"
                )
        elif ev.event == "error":
            typer.secho(f"Lỗi: {ev.data['message']}", fg="red", err=True)
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
