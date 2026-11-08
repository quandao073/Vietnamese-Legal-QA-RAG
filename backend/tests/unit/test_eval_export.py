import json
from pathlib import Path

from legalrag.ingest.loaders.base import write_eval_files
from legalrag.schemas.eval import EvalPair


class _FakeLoader:
    name = "tamir"

    def iter_eval_pairs(self):
        yield EvalPair(
            id="tamir_train:0",
            question="q1",
            gold_source_ids=["p1"],
            gold_answer="a",
            split="train",
        )
        yield EvalPair(
            id="tamir_test:0", question="q2", gold_source_ids=["p2"], gold_answer="b", split="test"
        )


def test_write_eval_files_splits_by_split(tmp_path: Path):
    written = write_eval_files(_FakeLoader(), tmp_path)
    assert set(written) == {tmp_path / "tamir_train.jsonl", tmp_path / "tamir_test.jsonl"}
    row = json.loads((tmp_path / "tamir_test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row == {
        "id": "tamir_test:0",
        "question": "q2",
        "gold_source_ids": ["p2"],
        "gold_answer": "b",
        "category": "legal",
    }
