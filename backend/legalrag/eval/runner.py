"""Run retrieval / answer experiments, persist predictions + eval_runs rows (spec §5.4)."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any

from legalrag.config import Settings
from legalrag.db.repositories import ChunkRepo, EvalRunRepo
from legalrag.db.session import session_scope
from legalrag.eval.answer_metrics import lexical_scores
from legalrag.eval.datasets import EvalItem, load_dataset, resolve_gold_chunks
from legalrag.eval.experiments import ExperimentConfig
from legalrag.eval.judge import JudgeScores, judge_answer
from legalrag.eval.report import RunSummary
from legalrag.eval.retrieval_metrics import aggregate, per_query_metrics
from legalrag.generation.llm_client import LLMError, Usage
from legalrag.retrieval.query_understanding import understand

logger = logging.getLogger(__name__)
STAGES = ("dense", "bm25", "hybrid", "hybrid_rerank")
# rough per-question estimate for the cost preview (tokens)
EST_INPUT_TOKENS = 4500
EST_OUTPUT_TOKENS = 700
EST_JUDGE_INPUT = 3500
EST_JUDGE_OUTPUT = 300


def runs_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "eval" / "runs"


def _write_predictions(settings: Settings, run_id: str, rows: list[dict[str, Any]]) -> Path:
    d = runs_dir(settings) / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "predictions.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return p


def _save_run(settings: Settings, summary: RunSummary) -> None:
    with session_scope() as s:
        EvalRunRepo(s).insert(
            run_id=summary.run_id,
            config_name=summary.config_name,
            kind=summary.kind,
            params=summary.params,
            dataset=summary.dataset,
            metrics=summary.metrics,
            n=summary.n,
            cost_usd=summary.cost_usd,
        )


def load_items(name: str, settings: Settings, strategy: str, limit: int | None) -> list[EvalItem]:
    items = load_dataset(name, settings.data_dir, limit=limit)
    with session_scope() as s:
        return resolve_gold_chunks(items, ChunkRepo(s), strategy)


# ---- retrieval ---------------------------------------------------------------------------------


async def run_retrieval(
    cfg: ExperimentConfig,
    dataset: str,
    *,
    retriever: Any,
    llm: Any,
    settings: Settings,
    limit: int | None = None,
) -> RunSummary:
    strategy = cfg.strategy + cfg.strategy_suffix
    items = load_items(dataset, settings, strategy, limit)
    run_id = uuid.uuid4().hex[:12]
    rows: list[dict[str, Any]] = []
    per_stage: dict[str, list[dict[str, float]]] = {st: [] for st in STAGES}
    timings: list[dict[str, float]] = []
    usage = Usage()
    t0 = time.time()
    for it in items:
        qu = None
        if cfg.use_rewrite:
            qu, u = await understand(it.question, history=None, llm=llm, settings=settings)
            usage = usage.add(u)
        res = await retriever.retrieve(it.question, qu=qu, overrides=cfg.overrides())
        gold = it.gold_sets() or set(it.gold_chunk_ids)
        row: dict[str, Any] = {
            "id": it.id,
            "question": it.question,
            "gold_chunk_ids": it.gold_chunk_ids,
            "gold_groups": it.gold_groups,
            "ranked": [h.chunk_id for h in res.reranked],
            "timings": res.timings,
            "stages": {},
        }
        for st in STAGES:
            ranked = res.stage_rankings.get(st) or []
            if st == "hybrid_rerank" and not ranked:
                ranked = [h.chunk_id for h in res.reranked]
            if not ranked and st not in ("hybrid_rerank",):
                continue
            m = per_query_metrics(ranked, gold)
            per_stage[st].append(m)
            row["stages"][st] = m
        timings.append(res.timings)
        rows.append(row)
    metrics: dict[str, Any] = {st: aggregate(v) for st, v in per_stage.items() if v}
    metrics["final"] = metrics.get("hybrid_rerank", {})
    metrics["latency_ms"] = {
        k: round(sum(t.get(k, 0.0) for t in timings) / max(1, len(timings)), 1)
        for k in ("dense_ms", "bm25_ms", "fusion_ms", "rerank_ms", "total_ms")
    }
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["usage"] = usage.as_dict()
    _write_predictions(settings, run_id, rows)
    summary = RunSummary(
        run_id=run_id,
        config_name=cfg.name,
        kind="retrieval",
        dataset=dataset,
        n=len(items),
        params=cfg.params(),
        metrics=metrics,
        cost_usd=usage.cost_usd,
    )
    _save_run(settings, summary)
    return summary


# ---- answer ------------------------------------------------------------------------------------


def estimate_cost(n_items: int, settings: Settings, *, with_judge: bool = True) -> float:
    inp = n_items * (EST_INPUT_TOKENS + (EST_JUDGE_INPUT if with_judge else 0))
    out = n_items * (EST_OUTPUT_TOKENS + (EST_JUDGE_OUTPUT if with_judge else 0))
    return inp / 1e6 * settings.price_input_per_m + out / 1e6 * settings.price_output_per_m


def _classify_failure(
    item: EvalItem, grounded: bool, intent: str, retrieved: list[str], judge: JudgeScores | None
) -> str:
    gold = set(item.gold_chunk_ids)
    if item.category == "out_of_scope":
        return "ok" if intent != "legal_question" or not grounded else "missed_refusal"
    if item.category == "out_of_corpus":
        if not grounded:
            return "ok"
        return "hallucination" if (judge and judge.hallucinated_claims) else "missed_refusal"
    if item.category == "tricky":
        # the topic exists in the corpus; the premise does not. Refusing is fine, and so is an
        # answer that says the premise is unsupported — only invented claims count as failure.
        return "hallucination" if (judge and judge.hallucinated_claims) else "ok"
    # legal question with gold
    if not grounded:
        return "false_refusal" if gold else "ok"
    if gold and not (gold & set(retrieved)):
        return "retrieval_miss"
    if judge and judge.faithfulness <= 2:
        return "hallucination"
    if judge and judge.citation_precision < 0.5:
        return "wrong_citation"
    return "ok"


async def run_answer(
    cfg: ExperimentConfig,
    dataset: str,
    *,
    answerer: Any,
    llm: Any,
    settings: Settings,
    limit: int | None = None,
    judge: bool = True,
) -> RunSummary:
    strategy = cfg.strategy + cfg.strategy_suffix
    items = load_items(dataset, settings, strategy, limit)
    run_id = uuid.uuid4().hex[:12]
    rows: list[dict[str, Any]] = []
    lex_rows: list[dict[str, float]] = []
    judge_rows: list[dict[str, float]] = []
    timings: list[float] = []
    usage = Usage()
    t0 = time.time()
    for it in items:
        try:
            resp = await answerer.answer_sync(
                it.question, session_id=None, history=None, overrides=cfg.overrides()
            )
        except LLMError as exc:
            rows.append(
                {
                    "id": it.id,
                    "question": it.question,
                    "category": it.category,
                    "error": exc.code,
                    "failure": "error",
                }
            )
            continue
        usage = usage.add(
            Usage(**{k: v for k, v in resp.usage.items() if k in Usage.__dataclass_fields__})
        )
        timings.append(float(resp.latency_ms.get("total_ms", 0.0)))
        retrieved = [s.chunk_id for s in resp.sources]
        lex = lexical_scores(resp.answer_md, it.gold_answer) if it.gold_answer else None
        if lex:
            lex_rows.append(lex)
        js: JudgeScores | None = None
        raw_judge = None
        if judge and (resp.grounded or it.category != "out_of_scope"):
            context_text = _context_from_db(retrieved)
            try:
                js, raw_judge, ju = await judge_answer(
                    llm,
                    question=it.question,
                    answer=resp.answer_md,
                    context_text=context_text,
                    gold_answer=it.gold_answer,
                    settings=settings,
                )
                usage = usage.add(ju)
                judge_rows.append(js.as_metrics())
            except Exception as exc:  # judge failures must not abort the run
                logger.warning("judge failed for %s: %s", it.id, exc)
        failure = _classify_failure(it, resp.grounded, resp.intent, retrieved, js)
        rows.append(
            {
                "id": it.id,
                "question": it.question,
                "category": it.category,
                "gold_answer": it.gold_answer,
                "gold_chunk_ids": it.gold_chunk_ids,
                "grounded": resp.grounded,
                "intent": resp.intent,
                "answer_md": resp.answer_md,
                "sources": retrieved,
                "citations": resp.citations.model_dump(),
                "lexical": lex,
                "judge": js.model_dump() if js else None,
                "judge_raw": raw_judge,
                "failure": failure,
                "latency_ms": resp.latency_ms,
                "usage": resp.usage,
            }
        )
    metrics: dict[str, Any] = {}
    if lex_rows:
        metrics["lexical"] = aggregate(lex_rows)
    if judge_rows:
        metrics["judge"] = aggregate(judge_rows)
    metrics["refusal"] = _refusal_metrics(rows)
    metrics["failures"] = {
        k: sum(1 for r in rows if r.get("failure") == k)
        for k in sorted({r.get("failure", "") for r in rows})
    }
    metrics["latency_ms"] = {"total_ms": round(sum(timings) / max(1, len(timings)), 1)}
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["usage"] = usage.as_dict()
    _write_predictions(settings, run_id, rows)
    _write_human_review(settings, run_id, rows)
    summary = RunSummary(
        run_id=run_id,
        config_name=cfg.name,
        kind="answer",
        dataset=dataset,
        n=len(items),
        params=cfg.params(),
        metrics=metrics,
        cost_usd=usage.cost_usd,
    )
    _save_run(settings, summary)
    return summary


def _context_from_db(chunk_ids: list[str]) -> str:
    if not chunk_ids:
        return ""
    with session_scope() as s:
        chunks = ChunkRepo(s).get_many(chunk_ids)
        parts = []
        for i, cid in enumerate(chunk_ids, start=1):
            c = chunks.get(cid)
            if c is not None:
                parts.append(
                    f"[Nguồn #{i}] {c.title} — {c.article or ''} {c.clause or ''}\n{c.text}"
                )
        return "\n\n".join(parts)


def _refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    def rate(pred: Any, pool: list[dict[str, Any]]) -> float:
        return round(sum(1 for r in pool if pred(r)) / len(pool), 4) if pool else 0.0

    refusal_pool = [
        r
        for r in rows
        if r.get("category") in ("out_of_corpus", "out_of_scope") and "grounded" in r
    ]
    tricky = [r for r in rows if r.get("category") == "tricky" and "grounded" in r]
    all_refusal_set = refusal_pool + tricky
    legal = [
        r
        for r in rows
        if r.get("category") == "legal" and r.get("gold_chunk_ids") and "grounded" in r
    ]
    out: dict[str, float] = {}
    if refusal_pool:
        out["correct_refusal_rate"] = rate(lambda r: not r["grounded"], refusal_pool)
    if tricky:
        out["tricky_safe_rate"] = rate(lambda r: r.get("failure") != "hallucination", tricky)
    if all_refusal_set:
        out["hallucination_rate"] = rate(
            lambda r: r.get("failure") == "hallucination", all_refusal_set
        )
    if legal:
        out["false_refusal_rate"] = rate(lambda r: not r["grounded"], legal)
    return out


def _write_human_review(
    settings: Settings, run_id: str, rows: list[dict[str, Any]], n: int = 30
) -> None:
    sample = random.Random(settings.random_seed).sample(rows, min(n, len(rows)))
    p = Path(settings.data_dir) / "eval" / f"human_review_{run_id}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "question",
                "answer_md",
                "sources",
                "gold_answer",
                "judge_faithfulness",
                "judge_correctness",
                "human_score",
                "human_notes",
            ]
        )
        for r in sample:
            j = r.get("judge") or {}
            w.writerow(
                [
                    r.get("id"),
                    r.get("question"),
                    r.get("answer_md"),
                    ";".join(r.get("sources") or []),
                    r.get("gold_answer"),
                    j.get("faithfulness"),
                    j.get("correctness"),
                    "",
                    "",
                ]
            )


# ---- yuitc faithfulness sample ------------------------------------------------------------------


async def run_faithfulness_sample(
    cfg: ExperimentConfig, *, answerer: Any, llm: Any, settings: Settings, n: int
) -> RunSummary:
    """Answer + judge `n` yuitc_test questions (no gold answer → faithfulness/relevance only)."""
    return await run_answer(
        cfg, "yuitc_test", answerer=answerer, llm=llm, settings=settings, limit=n
    )


def worst_examples(settings: Settings, run_id: str, n: int = 20) -> list[dict[str, Any]]:
    p = runs_dir(settings) / run_id / "predictions.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def score(r: dict[str, Any]) -> float:
        if "stages" in r:  # retrieval run → lower mrr is worse
            return float((r["stages"].get("hybrid_rerank") or {}).get("mrr@10", 0.0))
        j = r.get("judge") or {}
        lex = r.get("lexical") or {}
        bad = 1.0 if r.get("failure") not in ("ok", None) else 0.0
        return -bad + float(j.get("faithfulness", 3)) / 5 + float(lex.get("f1_conclusion", 0.0))

    return sorted(rows, key=score)[:n]


async def gather_limited(coros: list[Any], limit: int = 2) -> list[Any]:
    sem = asyncio.Semaphore(limit)

    async def _one(c: Any) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(_one(c) for c in coros))
