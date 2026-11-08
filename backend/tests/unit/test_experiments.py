from pathlib import Path

import pytest

from legalrag.eval.experiments import ExperimentConfig, load_suite, suite_names


def test_bundled_suites_expand_to_expected_sizes():
    assert {"E0", "E1", "E2", "E3", "E4", "E5", "default"} <= set(suite_names())
    assert len(load_suite("E0")) == 1
    assert len(load_suite("E1")) == 5
    assert len(load_suite("E2")) == 4
    assert len(load_suite("E3")) == 5
    assert len(load_suite("E4")) == 2
    assert len(load_suite("E5")) == 2


def test_e0_is_the_proposal_baseline():
    (e0,) = load_suite("E0")
    assert e0 == ExperimentConfig(
        name="E0",
        strategy="fixed_512_64",
        mode="dense",
        use_rerank=False,
        use_rewrite=False,
        top_k=5,
    )
    assert e0.overrides().top_k == 5 and e0.overrides().mode == "dense"


def test_e3_varies_top_k_and_names_are_unique():
    cfgs = load_suite("E3")
    assert sorted(c.top_k for c in cfgs) == [2, 4, 6, 8, 10]
    assert len({c.name for c in cfgs}) == 5 and all(c.name.startswith("E3") for c in cfgs)
    assert all(c.strategy == "article" and c.use_rerank and c.mode == "hybrid" for c in cfgs)


def test_e2_methods():
    cfgs = {c.name: c for c in load_suite("E2")}
    assert {c.mode for c in cfgs.values()} == {"dense", "bm25", "hybrid"}
    assert sum(c.use_rerank for c in cfgs.values()) == 1


def test_default_suite_lists_retrieval_and_answer_parts():
    suite = load_suite("default", full=True)
    assert suite["retrieval"]["configs"] and suite["answer"]["configs"]
    assert set(suite["retrieval"]["datasets"]) >= {"yuitc_test", "tamir_test"}
    assert set(suite["answer"]["datasets"]) >= {"tamir_test", "refusal_set"}


def test_load_custom_yaml(tmp_path: Path):
    p = tmp_path / "X.yaml"
    p.write_text(
        "name: X\nbase: {strategy: clause, mode: hybrid, use_rerank: true, use_rewrite: true, top_k: 6}\n"
        "grid: {top_k: [2, 4]}\n",
        encoding="utf-8",
    )
    cfgs = load_suite(str(p))
    assert [c.top_k for c in cfgs] == [2, 4] and cfgs[0].strategy == "clause"
    with pytest.raises(FileNotFoundError):
        load_suite("nope")
