"""Recompute failure labels + refusal metrics of stored answer runs from their predictions.jsonl
(after a change to the failure classifier). Updates eval_runs.metrics in place.

Usage: source scripts/local-env.sh && cd backend && uv run python -X utf8 ../scripts/recompute_refusal.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "x")

from legalrag.config import get_settings
from legalrag.db.models import EvalRun
from legalrag.db.session import session_scope
from legalrag.eval.datasets import EvalItem
from legalrag.eval.judge import JudgeScores
from legalrag.eval.runner import _classify_failure, _refusal_metrics, runs_dir


def main() -> None:
    s = get_settings()
    with session_scope() as db:
        for run in db.query(EvalRun).filter(EvalRun.kind == "answer").all():
            p = runs_dir(s) / run.run_id / "predictions.jsonl"
            if not p.exists():
                continue
            rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            for r in rows:
                if "grounded" not in r:
                    continue
                item = EvalItem(id=r["id"], question=r["question"], category=r.get("category", "legal"), gold_chunk_ids=r.get("gold_chunk_ids") or [])
                judge = JudgeScores(**r["judge"]) if r.get("judge") else None
                r["failure"] = _classify_failure(item, r["grounded"], r.get("intent", ""), r.get("sources") or [], judge)
            metrics = dict(run.metrics)
            metrics["refusal"] = _refusal_metrics(rows)
            metrics["failures"] = {k: sum(1 for r in rows if r.get("failure") == k) for k in sorted({r.get("failure", "") for r in rows})}
            run.metrics = metrics
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            print(run.run_id, run.config_name, run.dataset, metrics["refusal"], metrics["failures"])


if __name__ == "__main__":
    main()
