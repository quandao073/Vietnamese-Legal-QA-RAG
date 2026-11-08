"""Generate docs/EVAL_REPORT.md with tables and matplotlib figures (spec §5.5)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

RECALL_COLS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
STAGES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
FAILURE_LABELS = {
    "retrieval_miss": "Truy xuất không tìm được nguồn vàng",
    "rerank_miss": "Nguồn vàng bị xếp hạng thấp sau rerank",
    "hallucination": "Câu trả lời có khẳng định không có trong nguồn",
    "wrong_citation": "Trích dẫn sai nguồn",
    "false_refusal": "Từ chối sai (có căn cứ nhưng không trả lời)",
    "missed_refusal": "Không từ chối câu hỏi ngoài kho",
}


@dataclass
class RunSummary:
    run_id: str
    config_name: str
    kind: str  # retrieval | answer
    dataset: str
    n: int
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return "–"
    return str(v)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def _retrieval_table(runs: list[RunSummary], dataset: str) -> str:
    rows = []
    for r in sorted(runs, key=lambda x: x.config_name):
        if r.kind != "retrieval" or r.dataset != dataset:
            continue
        final = r.metrics.get("hybrid_rerank") or r.metrics.get("final") or {}
        p = r.params
        rows.append(
            [
                r.config_name,
                p.get("strategy"),
                p.get("mode"),
                p.get("use_rerank"),
                p.get("use_rewrite"),
                p.get("top_k"),
            ]
            + [final.get(c) for c in RECALL_COLS]
            + [r.metrics.get("latency_ms", {}).get("total_ms"), r.n]
        )
    if not rows:
        return "_Chưa có run retrieval cho bộ này._"
    return markdown_table(
        ["Cấu hình", "Chunking", "Phương pháp", "Rerank", "Rewrite", "k"]
        + RECALL_COLS
        + ["latency ms", "n"],
        rows,
    )


def _stage_table(run: RunSummary) -> str:
    rows = []
    for st in STAGES:
        m = run.metrics.get(st)
        if m:
            rows.append([st] + [m.get(c) for c in RECALL_COLS])
    return markdown_table(["Giai đoạn"] + RECALL_COLS, rows) if rows else ""


def _answer_table(runs: list[RunSummary]) -> str:
    rows = []
    for r in sorted(runs, key=lambda x: (x.dataset, x.config_name)):
        if r.kind != "answer":
            continue
        lex, judge = r.metrics.get("lexical", {}), r.metrics.get("judge", {})
        rows.append(
            [
                r.config_name,
                r.dataset,
                r.n,
                lex.get("em_conclusion"),
                lex.get("f1_conclusion"),
                lex.get("rouge_l"),
                judge.get("correctness"),
                judge.get("faithfulness"),
                judge.get("citation_precision"),
                judge.get("completeness"),
                judge.get("relevance"),
                r.metrics.get("latency_ms", {}).get("total_ms"),
                r.cost_usd,
            ]
        )
    if not rows:
        return "_Chưa có run answer._"
    return markdown_table(
        [
            "Cấu hình",
            "Bộ dữ liệu",
            "n",
            "EM (KL)",
            "F1 (KL)",
            "ROUGE-L",
            "Correct.",
            "Faithful.",
            "Cite prec.",
            "Complete.",
            "Relev.",
            "latency ms",
            "cost $",
        ],
        rows,
    )


def _refusal_table(runs: list[RunSummary]) -> str:
    rows = []
    for r in runs:
        ref = r.metrics.get("refusal")
        if r.kind == "answer" and ref:
            rows.append(
                [
                    r.config_name,
                    r.dataset,
                    ref.get("correct_refusal_rate"),
                    ref.get("tricky_safe_rate"),
                    ref.get("false_refusal_rate"),
                    ref.get("hallucination_rate"),
                    r.n,
                ]
            )
    if not rows:
        return "_Chưa có số liệu từ chối._"
    return markdown_table(
        [
            "Cấu hình",
            "Bộ dữ liệu",
            "correct_refusal_rate",
            "tricky_safe_rate",
            "false_refusal_rate",
            "hallucination_rate",
            "n",
        ],
        rows,
    )


def _plot_recall(runs: list[RunSummary], fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    ks = [1, 3, 5, 10]
    # Figure 1: recall@K per method (E2-like runs + E0) per dataset
    datasets = sorted({r.dataset for r in runs if r.kind == "retrieval"})
    if datasets:
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
        for ax, ds in zip(axes[0], datasets, strict=True):
            for r in sorted(runs, key=lambda x: x.config_name):
                if (
                    r.kind != "retrieval"
                    or r.dataset != ds
                    or not (r.config_name.startswith(("E0", "E2")))
                ):
                    continue
                final = r.metrics.get("hybrid_rerank") or {}
                ys = [final.get(f"recall@{k}") for k in ks]
                if any(y is not None for y in ys):
                    ax.plot(ks, [y or 0 for y in ys], marker="o", label=r.config_name)
            ax.set_title(f"Recall@K — {ds}")
            ax.set_xlabel("K")
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = fig_dir / "recall_at_k_by_method.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p.name)
    # Figure 2: chunking comparison (E1) recall@5
    e1 = [r for r in runs if r.kind == "retrieval" and r.config_name.startswith("E1")]
    if e1:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r.config_name.split("-", 1)[-1] for r in e1]
        vals = [(r.metrics.get("hybrid_rerank") or {}).get("recall@5", 0) or 0 for r in e1]
        ax.bar(names, vals, color="#4C72B0")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall@5")
        ax.set_title("Chiến lược chunking (E1, hybrid+rerank, k=6)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        fig.tight_layout()
        p = fig_dir / "chunking_comparison.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p.name)
    return written


def _error_analysis(runs: list[RunSummary], runs_dir: Path | None, limit: int = 10) -> str:
    if runs_dir is None:
        return "_Không có predictions để phân tích._"
    examples: list[tuple[str, dict[str, Any]]] = []
    for r in runs:
        if r.kind != "answer":
            continue
        p = Path(runs_dir) / r.run_id / "predictions.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failure") and row["failure"] != "ok":
                examples.append((r.config_name, row))
    if not examples:
        return "_Không phát hiện ví dụ thất bại trong các run answer đã chọn._"
    by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for cfg, row in examples:
        by_type.setdefault(row["failure"], []).append((cfg, row))
    out: list[str] = []
    shown = 0
    for ftype, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        out.append(f"### {FAILURE_LABELS.get(ftype, ftype)} ({len(items)} trường hợp)")
        for cfg, row in items[: max(1, limit // len(by_type))]:
            if shown >= limit:
                break
            shown += 1
            judge = row.get("judge") or {}
            out.append(
                f"- **[{cfg}] {row.get('id')}** — _{row.get('question')}_  \n"
                f"  grounded={row.get('grounded')}, nguồn={len(row.get('sources') or [])}, "
                f"faithfulness={judge.get('faithfulness', '–')}, f1(KL)={(row.get('lexical') or {}).get('f1_conclusion', '–')}  \n"
                f"  Nhận xét tự động: {_remark(ftype)}"
            )
        out.append("")
    return "\n".join(out)


def _remark(ftype: str) -> str:
    return {
        "retrieval_miss": "nguồn vàng không nằm trong top-K; cân nhắc rewrite tốt hơn hoặc tăng top_n trước rerank.",
        "rerank_miss": "nguồn vàng có trong ứng viên nhưng bị reranker đẩy xuống; xem lại độ dài chunk đưa vào reranker.",
        "hallucination": "mô hình bổ sung chi tiết ngoài nguồn; siết prompt hoặc giảm ngữ cảnh lạc đề.",
        "wrong_citation": "chỉ số nguồn không khớp nội dung; kiểm tra ngữ cảnh gom theo văn bản.",
        "false_refusal": "ngưỡng MIN_RERANK_SCORE có thể quá cao hoặc nguồn vàng không được truy xuất.",
        "missed_refusal": "câu hỏi ngoài kho nhưng vẫn có nguồn vượt ngưỡng; cân nhắc tăng ngưỡng hoặc judge-gate.",
    }.get(ftype, "cần người đánh giá xem lại.")


def build_report(
    runs: list[RunSummary],
    *,
    corpus_stats: dict[str, Any],
    out_md: Path,
    fig_dir: Path,
    runs_dir: Path | None = None,
) -> None:
    figures = _plot_recall(runs, fig_dir)
    rel_fig = Path(fig_dir).name
    datasets = sorted({r.dataset for r in runs if r.kind == "retrieval"})
    lines: list[str] = [
        "# Báo cáo đánh giá — Vietnamese Legal RAG QA",
        "",
        f"_Sinh tự động ngày {date.today().isoformat()} bởi `python -m legalrag.eval report`._",
        "",
        "## 1. Kho dữ liệu đã index",
        "",
        f"- Số văn bản: **{corpus_stats.get('n_docs', '–')}**; số chunk (strategy `{corpus_stats.get('strategy', 'article')}`): **{corpus_stats.get('n_chunks', '–')}**",
    ]
    for src, st in (corpus_stats.get("by_source") or {}).items():
        lines.append(
            f"- `{src}`: {st.get('n_docs', '–')} văn bản, {st.get('n_chunks', '–')} chunk, trung vị {st.get('median_tokens', '–')} token/chunk, "
            f"bao phủ Điều (hiệu dụng/chặt): {st.get('article_coverage_effective', '–')} / {st.get('article_coverage_strict', '–')}"
        )
    lines += ["", "## 2. Retrieval (E0–E4)", ""]
    for ds in datasets:
        lines += [f"### Bộ `{ds}`", "", _retrieval_table(runs, ds), ""]
        best = [
            r
            for r in runs
            if r.kind == "retrieval" and r.dataset == ds and r.config_name.endswith("hybrid_rerank")
        ]
        if best:
            lines += ["Theo giai đoạn (cấu hình hybrid+rerank):", "", _stage_table(best[0]), ""]
    for f in figures:
        lines += [f"![{f}]({rel_fig}/{f})", ""]
    lines += [
        "## 3. Chất lượng câu trả lời",
        "",
        _answer_table(runs),
        "",
        "### Từ chối & bịa đặt",
        "",
        _refusal_table(runs),
        "",
    ]
    lines += ["## 4. Độ trễ & chi phí", ""]
    lat_rows = [
        [
            r.config_name,
            r.kind,
            r.dataset,
            r.metrics.get("latency_ms", {}).get("total_ms"),
            r.cost_usd,
            round(r.cost_usd / r.n, 4) if r.n else 0.0,
        ]
        for r in runs
    ]
    lines += [
        markdown_table(
            ["Cấu hình", "Loại", "Bộ dữ liệu", "latency ms/câu", "cost $", "cost $/câu"], lat_rows
        ),
        "",
    ]
    lines += ["## 5. Phân tích lỗi", "", _error_analysis(runs, runs_dir), ""]
    lines += [
        "## 6. Ghi chú phương pháp",
        "",
        "- Seed cố định 42; `yuitc_test` lấy mẫu `EVAL_YUITC_SAMPLE` câu từ split test; `tamir_test` 54 câu; `refusal_set` 50 câu tự tạo (30 ngoài kho, 10 ngoài phạm vi, 10 câu bẫy).",
        "- Retrieval metrics tính trên `gold_chunk_ids` suy ra từ `source_ids` (một passage vàng có thể ứng với nhiều chunk).",
        "- LLM-judge dùng Claude Sonnet, output JSON có cache; điểm judge có thiên lệch (tự đánh giá cùng họ mô hình) và test set nhỏ → khoảng tin cậy rộng.",
        "- Cảnh báo rò rỉ: `bkai-foundation-models/vietnamese-bi-encoder` được huấn luyện trên BKAI-Legal-Retrieval (nguồn của YuITC) nên E5 không so sánh công bằng trên `yuitc_test`.",
        "- Các run E1 chạy trên index Tamir-only (`*__tamir`) để so sánh chunking trong cùng kho nhỏ; các run E0/E2/E3/E4 chạy trên kho đầy đủ (~20k văn bản).",
    ]
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
