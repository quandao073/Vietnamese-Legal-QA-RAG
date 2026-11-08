"""`python -m legalrag.eval` — retrieval / answer / all / report / errors (spec §5.4)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from legalrag.config import Settings, get_settings
from legalrag.logging import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Evaluation CLI")


def _eval_settings() -> Settings:
    s = get_settings()
    # evaluation defaults: response/judge caching ON so re-runs cost nothing (spec §3.8, §5.6)
    return s.model_copy(update={"llm_cache_enabled": True})


@app.callback()
def _init() -> None:
    setup_logging(get_settings().log_level)


def _print_summary(summary: Any) -> None:
    from legalrag.eval.report import markdown_table

    m = summary.metrics
    typer.secho(
        f"\n== {summary.config_name} · {summary.kind} · {summary.dataset} · n={summary.n} · run={summary.run_id}",
        fg="green",
    )
    if summary.kind == "retrieval":
        rows = [
            [st]
            + [
                m[st].get(k)
                for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")
            ]
            for st in ("dense", "bm25", "hybrid", "hybrid_rerank")
            if st in m
        ]
        typer.echo(
            markdown_table(["stage", "R@1", "R@3", "R@5", "R@10", "MRR@10", "nDCG@10"], rows)
        )
        typer.echo(f"latency: {m.get('latency_ms')}")
    else:
        for key in ("lexical", "judge", "refusal", "failures", "latency_ms", "usage"):
            if key in m:
                typer.echo(f"{key}: {json.dumps(m[key], ensure_ascii=False)}")
        typer.echo(f"cost_usd: {summary.cost_usd:.4f}")


async def _retrieval(configs: list[Any], datasets: list[str], limit: int | None) -> list[Any]:
    from legalrag.eval.runner import run_retrieval
    from legalrag.generation.llm_client import LLMClient
    from legalrag.services import get_retriever

    s = _eval_settings()
    from legalrag.cache import get_cache

    llm = LLMClient(s, cache=get_cache())
    retriever = get_retriever(s)
    out = []
    for cfg in configs:
        for ds in datasets:
            summary = await run_retrieval(
                cfg, ds, retriever=retriever, llm=llm, settings=s, limit=limit
            )
            _print_summary(summary)
            out.append(summary)
    return out


async def _answer(
    configs: list[Any],
    datasets: list[str],
    limit: int | None,
    yes: bool,
    faithfulness_sample: int = 0,
    suffix_by_dataset: dict[str, str] | None = None,
) -> list[Any]:
    from legalrag.cache import get_cache
    from legalrag.eval.datasets import load_dataset
    from legalrag.eval.runner import estimate_cost, run_answer, run_faithfulness_sample
    from legalrag.generation.llm_client import LLMClient
    from legalrag.services import get_answerer

    s = _eval_settings()
    n_items = sum(len(load_dataset(ds, s.data_dir, limit=limit)) for ds in datasets) * len(configs)
    n_items += faithfulness_sample * len(configs)
    est = estimate_cost(n_items, s)
    typer.secho(
        f"Ước lượng: {n_items} câu hỏi × (sinh + chấm) ≈ ${est:.2f} (giá theo PRICE_*_PER_M; cache có thể giảm mạnh)",
        fg="yellow",
    )
    if not yes:
        typer.confirm("Tiếp tục?", abort=True)
    llm = LLMClient(s, cache=get_cache())
    answerer = get_answerer(s, llm=llm)
    out = []
    for cfg in configs:
        for ds in datasets:
            ds_cfg = cfg
            if suffix_by_dataset and ds in suffix_by_dataset:
                ds_cfg = cfg.model_copy(update={"strategy_suffix": suffix_by_dataset[ds]})
            summary = await run_answer(
                ds_cfg, ds, answerer=answerer, llm=llm, settings=s, limit=limit
            )
            _print_summary(summary)
            out.append(summary)
        if faithfulness_sample:
            summary = await run_faithfulness_sample(
                cfg, answerer=answerer, llm=llm, settings=s, n=faithfulness_sample
            )
            _print_summary(summary)
            out.append(summary)
    return out


@app.command()
def retrieval(
    config: str = typer.Option("E2", help="matrix name (E0..E5) or yaml path"),
    dataset: str = typer.Option(
        "tamir_test", help="comma-separated: yuitc_test,tamir_test,tamir_train"
    ),
    limit: int = typer.Option(None, help="only the first N questions"),
) -> None:
    """Retrieval-only experiments (no main LLM call)."""
    from legalrag.eval.experiments import load_suite

    asyncio.run(_retrieval(load_suite(config), dataset.split(","), limit))


@app.command()
def answer(
    config: str = typer.Option("E2-hybrid_rerank", help="config name, e.g. E0 or E2-hybrid_rerank"),
    dataset: str = typer.Option("tamir_test,refusal_set"),
    limit: int = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", help="skip the cost confirmation"),
) -> None:
    """End-to-end answer experiments with LLM-as-judge."""
    from legalrag.eval.experiments import load_suite

    matrix = config.split("-", 1)[0]
    cfgs = [c for c in load_suite(matrix) if c.name == config or config == matrix]
    if not cfgs:
        raise typer.BadParameter(f"unknown config {config}")
    asyncio.run(_answer(cfgs, dataset.split(","), limit, yes))


@app.command("all")
def run_all(
    suite: str = typer.Option("default"),
    yes: bool = typer.Option(False, "--yes"),
    limit: int = typer.Option(None, help="limit questions per dataset (smoke runs)"),
) -> None:
    """Run a composite suite (default = E0+E2+E3 retrieval, E0 vs best answer) and print a summary."""
    from legalrag.eval.experiments import load_suite

    spec = load_suite(suite, full=True)
    summaries: list[Any] = []
    if "retrieval" in spec:
        summaries += asyncio.run(
            _retrieval(spec["retrieval"]["configs"], spec["retrieval"]["datasets"], limit)
        )
    if "answer" in spec:
        fs = int(spec["answer"].get("faithfulness_sample", 0) or 0)
        if limit:
            fs = min(fs, limit)
        summaries += asyncio.run(
            _answer(
                spec["answer"]["configs"],
                spec["answer"]["datasets"],
                limit,
                yes,
                fs,
                spec["answer"].get("strategy_suffix_by_dataset"),
            )
        )
    typer.secho(
        f"\n{len(summaries)} runs completed; run `python -m legalrag.eval report --runs latest`",
        fg="green",
    )


@app.command()
def report(
    runs: str = typer.Option(
        "latest", help="'latest' (newest run per config/dataset) or comma-separated run ids"
    ),
    out: Path = typer.Option(Path("docs/EVAL_REPORT.md")),
    figures: Path = typer.Option(Path("docs/figures")),
) -> None:
    """Generate docs/EVAL_REPORT.md + figures from stored eval_runs."""
    from legalrag.db.repositories import EvalRunRepo
    from legalrag.db.session import session_scope
    from legalrag.eval.report import RunSummary, build_report
    from legalrag.eval.runner import runs_dir
    from legalrag.ingest import indexer

    s = _eval_settings()
    with session_scope() as db:
        rows = EvalRunRepo(db).latest(500)
        selected: list[RunSummary] = []
        if runs == "latest":
            seen: set[tuple[str, str, str]] = set()
            for r in rows:  # newest first
                key = (r.config_name, r.kind, r.dataset)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(
                    RunSummary(
                        r.run_id,
                        r.config_name,
                        r.kind,
                        r.dataset,
                        r.n,
                        r.params,
                        r.metrics,
                        r.cost_usd,
                    )
                )
        else:
            wanted = set(runs.split(","))
            selected = [
                RunSummary(
                    r.run_id, r.config_name, r.kind, r.dataset, r.n, r.params, r.metrics, r.cost_usd
                )
                for r in rows
                if r.run_id in wanted
            ]
    corpus = indexer.stats(s.chunk_strategy, settings=s)
    corpus["n_docs"] = sum(v.get("n_docs", 0) for v in corpus.get("by_source", {}).values())
    build_report(selected, corpus_stats=corpus, out_md=out, fig_dir=figures, runs_dir=runs_dir(s))
    typer.echo(f"wrote {out} ({len(selected)} runs) and figures in {figures}")


@app.command()
def errors(run: str = typer.Option(...), worst: int = typer.Option(20)) -> None:
    """Print the worst N predictions of a run for manual error analysis."""
    from legalrag.eval.runner import worst_examples

    for r in worst_examples(_eval_settings(), run, worst):
        typer.echo(
            json.dumps(
                {
                    k: r.get(k)
                    for k in (
                        "id",
                        "question",
                        "failure",
                        "grounded",
                        "sources",
                        "gold_chunk_ids",
                        "judge",
                        "lexical",
                        "answer_md",
                    )
                },
                ensure_ascii=False,
            )[:1500]
        )
        typer.echo("---")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
