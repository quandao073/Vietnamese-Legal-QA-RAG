from __future__ import annotations

import json
from pathlib import Path

from legalrag.eval.report import RunSummary, build_report, markdown_table


def test_markdown_table_renders_rows_and_formats_floats():
    md = markdown_table(["cfg", "hit@5"], [["E0", 0.5], ["E2", 0.8333]])
    assert md.splitlines()[0] == "| cfg | hit@5 |" and "| E2 | 0.833 |" in md


def test_build_report_writes_markdown_and_figures(tmp_path: Path):
    runs = [
        RunSummary(
            run_id="r1",
            config_name="E0",
            kind="retrieval",
            dataset="tamir_test",
            n=54,
            params={
                "strategy": "fixed_512_64",
                "mode": "dense",
                "use_rerank": False,
                "use_rewrite": False,
                "top_k": 5,
            },
            metrics={
                "hybrid_rerank": {
                    "recall@1": 0.3,
                    "recall@3": 0.5,
                    "recall@5": 0.6,
                    "recall@10": 0.7,
                    "hit@5": 0.6,
                    "mrr@10": 0.4,
                    "ndcg@10": 0.45,
                },
                "latency_ms": {"total_ms": 120.0},
            },
        ),
        RunSummary(
            run_id="r2",
            config_name="E2-hybrid_rerank",
            kind="retrieval",
            dataset="tamir_test",
            n=54,
            params={
                "strategy": "article",
                "mode": "hybrid",
                "use_rerank": True,
                "use_rewrite": True,
                "top_k": 6,
            },
            metrics={
                "hybrid_rerank": {
                    "recall@1": 0.5,
                    "recall@3": 0.7,
                    "recall@5": 0.8,
                    "recall@10": 0.9,
                    "hit@5": 0.8,
                    "mrr@10": 0.6,
                    "ndcg@10": 0.65,
                },
                "dense": {"recall@5": 0.7},
                "bm25": {"recall@5": 0.65},
                "hybrid": {"recall@5": 0.75},
                "latency_ms": {"total_ms": 900.0},
            },
        ),
        RunSummary(
            run_id="r3",
            config_name="E2-hybrid_rerank",
            kind="answer",
            dataset="tamir_test",
            n=10,
            params={"strategy": "article"},
            cost_usd=0.12,
            metrics={
                "lexical": {"f1_conclusion": 0.41, "em_conclusion": 0.1, "rouge_l": 0.3},
                "judge": {
                    "faithfulness": 4.2,
                    "correctness": 3.9,
                    "citation_precision": 0.9,
                    "completeness": 3.5,
                    "relevance": 4.5,
                },
                "refusal": {
                    "correct_refusal_rate": 0.85,
                    "false_refusal_rate": 0.05,
                    "hallucination_rate": 0.1,
                },
                "latency_ms": {"total_ms": 5000.0},
            },
        ),
    ]
    preds = tmp_path / "runs" / "r3" / "predictions.jsonl"
    preds.parent.mkdir(parents=True)
    preds.write_text(
        "\n".join(
            json.dumps(x, ensure_ascii=False)
            for x in [
                {
                    "id": "a",
                    "question": "q1",
                    "grounded": True,
                    "answer_md": "x",
                    "sources": ["s1"],
                    "judge": {"faithfulness": 1},
                    "lexical": {"f1_conclusion": 0.0},
                    "category": "legal",
                    "failure": "hallucination",
                },
                {
                    "id": "b",
                    "question": "q2",
                    "grounded": False,
                    "answer_md": "y",
                    "sources": [],
                    "judge": None,
                    "lexical": None,
                    "category": "legal",
                    "failure": "false_refusal",
                },
            ]
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "EVAL_REPORT.md"
    fig_dir = tmp_path / "figures"
    build_report(
        runs,
        corpus_stats={
            "n_docs": 20005,
            "n_chunks": 21857,
            "by_source": {"tamir": {"n_chunks": 106}},
        },
        out_md=out_md,
        fig_dir=fig_dir,
        runs_dir=tmp_path / "runs",
    )
    text = out_md.read_text(encoding="utf-8")
    assert "# Báo cáo đánh giá" in text and "20005" in text
    assert "| E0 |" in text and "E2-hybrid_rerank" in text
    assert "correct_refusal_rate" in text or "Tỉ lệ từ chối đúng" in text
    assert "Phân tích lỗi" in text and "q1" in text
    assert (fig_dir / "recall_at_k_by_method.png").exists()
    assert "recall_at_k_by_method.png" in text
