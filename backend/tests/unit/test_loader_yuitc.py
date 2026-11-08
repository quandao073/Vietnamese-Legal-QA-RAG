import json
from pathlib import Path

import pandas as pd
import pytest

from legalrag.ingest.loaders.hf_yuitc_legal import YuitcLoader

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def loader(tmp_path: Path) -> YuitcLoader:
    raw = tmp_path / "yuitc"
    raw.mkdir()
    rows = [
        json.loads(line)
        for line in (FIX / "yuitc_sample.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    corpus = [r for r in rows if r["kind"] == "corpus"]
    tests = [r for r in rows if r["kind"] == "test"]
    pd.DataFrame([{"cid": r["cid"], "text": r["text"]} for r in corpus]).to_parquet(
        raw / "corpus.parquet"
    )
    pd.DataFrame(
        [
            {
                "qid": r["qid"],
                "question": r["question"],
                "cid": r["cid"],
                "context_list": r["context_list"],
            }
            for r in tests
        ]
    ).to_parquet(raw / "test.parquet")
    return YuitcLoader(
        raw_dir=raw, processed_dir=tmp_path / "processed", max_docs=10, eval_sample=3, seed=42
    )


def test_split_title_detects_law_name_line():
    title, body = YuitcLoader.split_title(
        "Luật Thuế thu nhập cá nhân\nĐiều 4. Thu nhập được miễn thuế\n1. A"
    )
    assert title == "Luật Thuế thu nhập cá nhân"
    assert body.startswith("Điều 4.")


def test_split_title_none_when_first_line_is_article():
    title, body = YuitcLoader.split_title("Điều 5. Thời hiệu\n1. ...")
    assert title is None and body.startswith("Điều 5.")


def test_split_title_from_long_first_line_is_rejected():
    first = "Áp dụng chế độ tiền lương và phụ cấp quy định tại Nghị định số 26/CP " + "x " * 60
    title, _ = YuitcLoader.split_title(first + "\n1. A")
    assert title is None


def test_select_cids_prefers_test_cids_and_is_deterministic():
    corpus = list(range(100))
    needed = {5, 50, 95}
    a = YuitcLoader.select_cids(corpus, needed, max_docs=10, seed=42)
    b = YuitcLoader.select_cids(corpus, needed, max_docs=10, seed=42)
    assert a == b and len(a) == 10 and needed <= set(a)
    assert YuitcLoader.select_cids(corpus, needed, max_docs=0, seed=1) == corpus


def test_documents_have_stable_ids_and_fallback_title(loader):
    docs = list(loader.iter_documents())
    assert 0 < len(docs) <= 10
    d = docs[0]
    assert d.doc_id == f"yuitc:cid:{d.extra['cid']}" and d.source == "yuitc"
    titles = [d.title for d in docs]
    assert any(t.startswith("Văn bản pháp luật (cid ") for t in titles)
    assert all(d.extra["source_id"] == f"cid:{d.extra['cid']}" for d in docs)


def test_selected_cids_written_and_include_eval_gold(loader):
    docs = list(loader.iter_documents())
    sel = json.loads((loader.processed_dir / "yuitc_selected_cids.json").read_text())
    assert sorted(sel) == sorted(d.extra["cid"] for d in docs)
    pairs = list(loader.iter_eval_pairs())
    assert len(pairs) == 3
    for p in pairs:
        assert p.id.startswith("yuitc_test:") and p.gold_answer is None
        assert all(g.startswith("cid:") and int(g[4:]) in sel for g in p.gold_source_ids)
