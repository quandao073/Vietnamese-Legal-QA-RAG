import json
from pathlib import Path

from legalrag.ingest.indexer import IngestState, read_jsonl, write_jsonl
from legalrag.schemas.document import LegalChunk


def test_state_roundtrip(tmp_path: Path):
    st = IngestState(run_id="r1", strategy="article", processed_dir=tmp_path)
    st.update(done=10, total=100, status="running", last_batch=1)
    data = json.loads((tmp_path / "ingest_state_r1.json").read_text(encoding="utf-8"))
    assert data == {
        "run_id": "r1",
        "strategy": "article",
        "done": 10,
        "total": 100,
        "status": "running",
        "last_batch": 1,
    }
    again = IngestState.load(tmp_path, "r1")
    assert again is not None and again.done == 10 and again.status == "running"
    assert IngestState.load(tmp_path, "missing") is None


def test_jsonl_roundtrip_of_chunks(tmp_path: Path):
    c = LegalChunk(
        chunk_id="d#00000",
        doc_id="d",
        seq=0,
        title="T",
        text="x",
        text_for_bm25="x",
        chunk_strategy="article",
    )
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [c])
    back = list(read_jsonl(path, LegalChunk))
    assert back == [c]
