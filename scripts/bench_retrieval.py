"""Measure retrieval latency (no LLM) over the first N tamir_train questions.

Usage (host venv):  source scripts/local-env.sh && cd backend && \
    uv run python -X utf8 ../scripts/bench_retrieval.py --n 20 [--strategy article] [--no-rerank]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "bench")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    from legalrag.config import get_settings
    from legalrag.retrieval.types import RetrievalOverrides
    from legalrag.services import get_retriever, warm_up

    s = get_settings()
    warm_up(s)
    retriever = get_retriever(s)
    rows = [
        json.loads(line)
        for line in (Path(s.data_dir) / "eval" / "tamir_train.jsonl").read_text(encoding="utf-8").splitlines()
    ][: args.n]
    ov = RetrievalOverrides(strategy=args.strategy, use_rerank=False if args.no_rerank else None)
    totals, stages = [], {"dense_ms": [], "bm25_ms": [], "rerank_ms": []}
    for r in rows:
        t = time.perf_counter()
        res = await retriever.retrieve(r["question"], qu=None, overrides=ov)
        totals.append((time.perf_counter() - t) * 1000)
        for k in stages:
            stages[k].append(res.timings.get(k, 0.0))
    print(json.dumps({
        "n": len(totals),
        "strategy": args.strategy or s.chunk_strategy,
        "rerank": not args.no_rerank and s.rerank_enabled,
        "device": s.device,
        "p50_ms": round(statistics.median(totals), 1),
        "p95_ms": round(sorted(totals)[int(0.95 * (len(totals) - 1))], 1),
        "mean_stage_ms": {k: round(statistics.mean(v), 1) for k, v in stages.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
