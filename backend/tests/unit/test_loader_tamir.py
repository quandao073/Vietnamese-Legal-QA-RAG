import csv
from pathlib import Path

import pytest

from legalrag.ingest.loaders.hf_tamir_tax import TamirLoader
from legalrag.ingest.normalize import normalize_text

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_rows() -> list[dict]:
    with open(FIX / "tamir_kb_sample.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def loader(tmp_path: Path) -> TamirLoader:
    raw = tmp_path / "tamir"
    (raw / "knowledge_base").mkdir(parents=True)
    (raw / "qa").mkdir()
    (raw / "knowledge_base" / "knowledge_base.csv").write_bytes(
        (FIX / "tamir_kb_sample.csv").read_bytes()
    )
    (raw / "qa" / "train_qa.jsonl").write_bytes((FIX / "tamir_qa_train_sample.jsonl").read_bytes())
    (raw / "qa" / "test_qa.jsonl").write_bytes((FIX / "tamir_qa_test_sample.jsonl").read_bytes())
    return TamirLoader(raw_dir=raw)


def test_parse_passage_markers():
    p = TamirLoader.parse_passage(
        "Chương I - Những quy định chung. Điều 5 - Đối tượng không chịu thuế. "
        "Khoản 26, điểm a: Quà tặng cho cơ quan nhà nước"
    )
    assert p == {
        "chapter": "Chương I",
        "chapter_title": "Những quy định chung",
        "article": "Điều 5",
        "article_title": "Đối tượng không chịu thuế",
        "clause": "26",
        "point": "a",
        "body": "Quà tặng cho cơ quan nhà nước",
    }


def test_parse_passage_without_clause():
    p = TamirLoader.parse_passage(
        "Chương I - Chung. Điều 1 - Phạm vi điều chỉnh: Luật này quy định"
    )
    assert p["clause"] is None and p["point"] is None and p["body"] == "Luật này quy định"


def test_parse_passage_odd_clause_label():
    p = TamirLoader.parse_passage(
        "Chương II - Căn cứ tính thuế. Điều 8 - Thuế suất. "
        "Khoản 1, Biểu thuế, phần I, mục 4, điểm a: Loại có dung tích"
    )
    assert p["clause"] == "1" and p["point"] == "a" and p["body"] == "Loại có dung tích"


def test_tamir_passages_to_lines(sample_rows):
    text, spans = TamirLoader.passages_to_raw_text(sample_rows[:5])
    lines = text.splitlines()
    assert lines[0] == "Chương I. Những quy định chung"
    assert lines[1] == "Điều 1. Phạm vi điều chỉnh"
    assert lines[2].startswith("Luật này quy định")
    assert lines[3] == "Điều 2. Thuế giá trị gia tăng"
    s, e = spans["passage_000001"]
    assert text[s:e].startswith("Luật này quy định") and text[s:e].endswith("giá trị gia tăng")
    # chapter header emitted only once
    assert text.count("Chương I. Những quy định chung") == 1


def test_tamir_clause_and_point_lines(sample_rows):
    text, spans = TamirLoader.passages_to_raw_text(sample_rows)
    assert "\n26. \na) Quà tặng" in text or "\n26.\na) Quà tặng" in text
    s, e = spans["passage_000042"]
    assert text[s:e].startswith("Quà tặng cho cơ quan nhà nước")
    # every passage body is a substring of the normalized text
    norm = normalize_text(text)
    for r in sample_rows:
        body = TamirLoader.parse_passage(r["passage_text"])["body"]
        assert normalize_text(body) in norm, r["passage_id"]


def test_tamir_documents_grouped(loader):
    docs = list(loader.iter_documents())
    assert [d.doc_id for d in docs] == ["tamir:doc_000001", "tamir:doc_000002", "tamir:doc_000003"]
    d = docs[0]
    assert d.source == "tamir" and d.url.startswith("https://")
    assert d.title == "VBHN 12/VBHN-VPQH năm 2026 - Luật Thuế giá trị gia tăng"
    assert d.doc_number == "12/VBHN-VPQH" and d.doc_type == "VBHN"
    assert "passage_000001" in d.extra["span_map"]
    assert d.raw_text.splitlines()[0] == "Chương I. Những quy định chung"


def test_tamir_doc_metadata_other_formats(loader):
    docs = {d.doc_id: d for d in loader.iter_documents()}
    assert docs["tamir:doc_000003"].doc_number == "48/2010/QH12"
    assert docs["tamir:doc_000003"].doc_type == "Luật"


def test_tamir_eval_pairs(loader):
    pairs = list(loader.iter_eval_pairs())
    assert len(pairs) == 16
    train = [p for p in pairs if p.split == "train"]
    assert train[0].gold_source_ids == ["passage_000001"]
    assert train[0].gold_answer and train[0].id.startswith("tamir_train:")
    assert {p.split for p in pairs} == {"train", "test"}


def test_tamir_raw_text_is_already_normalized(loader):
    for d in loader.iter_documents():
        assert normalize_text(d.raw_text) == d.raw_text
