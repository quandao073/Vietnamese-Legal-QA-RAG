from pathlib import Path

import pytest

from legalrag.ingest.structure import Node, iter_articles, iter_clauses, parse_structure

FIX = Path(__file__).parent.parent / "fixtures" / "structure"


def _load(name: str) -> str:
    return (FIX / f"{name}.txt").read_text(encoding="utf-8")


def _kinds(node: Node, kind: str) -> list[Node]:
    out = []
    stack = [node]
    while stack:
        n = stack.pop(0)
        if n.kind == kind:
            out.append(n)
        stack.extend(n.children)
    return out


@pytest.mark.parametrize(
    "name,chapters,articles,clauses,points",
    [
        ("tamir_1", 1, 3, 2, 0),
        ("tamir_2", 2, 2, 3, 3),
        ("tamir_3", 1, 2, 4, 0),
        ("tamir_4", 1, 2, 5, 2),
        ("tamir_5", 1, 1, 3, 5),
        ("yuitc_1", 0, 1, 2, 0),
        ("yuitc_2", 0, 0, 2, 0),
        ("yuitc_3", 0, 1, 3, 0),
    ],
)
def test_parse_structure_counts(name, chapters, articles, clauses, points):
    root = parse_structure(_load(name))
    assert root.kind == "document"
    assert len(_kinds(root, "chapter")) == chapters
    assert len(_kinds(root, "article")) == articles
    assert len(_kinds(root, "clause")) == clauses
    assert len(_kinds(root, "point")) == points


def test_labels_and_titles():
    root = parse_structure(_load("tamir_2"))
    chapters = _kinds(root, "chapter")
    assert chapters[0].label == "Chương I"
    assert chapters[0].title == "Những quy định chung"
    art = _kinds(root, "article")[0]
    assert art.label == "Điều 5" and art.title == "Đối tượng không chịu thuế"
    clause26 = [c for c in iter_clauses(art) if c.label == "Khoản 26"][0]
    assert [p.label for p in clause26.children] == ["Điểm a", "Điểm b", "Điểm c"]


def test_char_spans_cover_text():
    text = _load("tamir_1")
    root = parse_structure(text)
    arts = _kinds(root, "article")
    assert text[arts[0].char_start :].startswith("Điều 1. Phạm vi điều chỉnh")
    assert text[arts[0].char_start : arts[0].char_end].rstrip().endswith("giá trị gia tăng.")
    assert arts[0].char_end <= arts[1].char_start
    assert root.char_start == 0 and root.char_end == len(text)


def test_article_without_title_and_wrapped_lines():
    text = "Điều 7.\n1. Câu bị ngắt\ngiữa dòng.\n2. Khoản hai"
    root = parse_structure(text)
    art = root.children[0]
    assert art.kind == "article" and art.label == "Điều 7" and art.title is None
    assert len(art.children) == 2
    first = art.children[0]
    assert text[first.char_start : first.char_end].endswith("giữa dòng.")


def test_wrapped_line_inside_clause_is_merged():
    root = parse_structure(_load("yuitc_2"))
    clause2 = _kinds(root, "clause")[1]
    assert clause2.label == "Khoản 2"
    assert "Sở\nY tế" in _load("yuitc_2")[clause2.char_start : clause2.char_end]


def test_iter_articles_yields_chapter_context():
    root = parse_structure(_load("tamir_2"))
    pairs = list(iter_articles(root))
    assert [(c.label if c else None, a.label) for c, a in pairs] == [
        ("Chương I", "Điều 5"),
        ("Chương II", "Điều 6"),
    ]
    root2 = parse_structure(_load("yuitc_1"))
    assert [(c, a.label) for c, a in iter_articles(root2)] == [(None, "Điều 5")]


def test_preamble_text_before_first_marker_kept_on_document():
    text = _load("yuitc_3")
    root = parse_structure(text)
    assert root.preamble is not None
    assert root.preamble.startswith("Áp dụng chế độ tiền lương")
    # clauses before any article attach to the document root
    assert [c.label for c in root.children if c.kind == "clause"] == ["Khoản 1", "Khoản 2"]
