import json
from pathlib import Path

import pytest

from legalrag.eval.datasets import DATASETS, EvalItem, load_dataset, resolve_gold_chunks


class FakeChunkRepo:
    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping

    def by_source_ids(self, source_ids, strategy):
        out = []
        for sid in source_ids:
            out += self.mapping.get(sid, [])
        return out


def test_load_tamir_dataset_from_data_dir(tmp_path: Path):
    (tmp_path / "eval").mkdir()
    rows = [
        {
            "id": "tamir_test:0",
            "question": "q0",
            "gold_source_ids": ["passage_000001"],
            "gold_answer": "a0",
            "category": "legal",
        },
        {
            "id": "tamir_test:1",
            "question": "q1",
            "gold_source_ids": ["passage_000002"],
            "gold_answer": "a1",
            "category": "legal",
        },
    ]
    (tmp_path / "eval" / "tamir_test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    items = load_dataset("tamir_test", tmp_path)
    assert [i.id for i in items] == ["tamir_test:0", "tamir_test:1"] and items[
        0
    ].gold_answer == "a0"
    assert load_dataset("tamir_test", tmp_path, limit=1)[0].id == "tamir_test:0"
    with pytest.raises(FileNotFoundError):
        load_dataset("yuitc_test", tmp_path)
    with pytest.raises(ValueError):
        load_dataset("unknown", tmp_path)


def test_refusal_set_is_bundled_and_well_formed(tmp_path: Path):
    items = load_dataset("refusal_set", tmp_path)  # bundled with the package, no data dir needed
    assert len(items) == 50
    cats = {i.category for i in items}
    assert cats == {"out_of_corpus", "out_of_scope", "tricky"}
    assert sum(i.category == "out_of_corpus" for i in items) == 30
    assert sum(i.category == "out_of_scope" for i in items) == 10
    assert sum(i.category == "tricky" for i in items) == 10
    assert all(i.gold_source_ids == [] for i in items if i.category != "tricky")
    assert len({i.id for i in items}) == 50


def test_resolve_gold_chunks_maps_sources_to_chunks():
    items = [
        EvalItem(id="a", question="q", gold_source_ids=["p1", "p2"]),
        EvalItem(id="b", question="q", gold_source_ids=["missing"]),
    ]
    repo = FakeChunkRepo({"p1": ["c1", "c2"], "p2": ["c2", "c3"]})
    out = resolve_gold_chunks(items, repo, "article")
    assert out[0].gold_chunk_ids == ["c1", "c2", "c3"]
    assert out[1].gold_chunk_ids == []
    assert set(DATASETS) == {"yuitc_test", "tamir_test", "tamir_train", "refusal_set"}


def test_resolve_gold_chunks_keeps_per_passage_groups():
    items = [EvalItem(id="a", question="q", gold_source_ids=["p1", "p2"])]
    repo = FakeChunkRepo({"p1": ["c1", "c2"], "p2": ["c2", "c3"]})
    (out,) = resolve_gold_chunks(items, repo, "article")
    assert out.gold_groups == [["c1", "c2"], ["c2", "c3"]]
    assert out.gold_sets() == [{"c1", "c2"}, {"c2", "c3"}]
