from __future__ import annotations

from legalrag.config import Settings
from legalrag.eval.judge import JUDGE_SYSTEM, PROMPT_VERSION, JudgeScores, judge_answer
from legalrag.generation.llm_client import Usage


class FakeLLM:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def complete_json(self, **kw):
        self.calls.append(kw)
        return self.result, Usage(100, 20)


def _settings() -> Settings:
    return Settings(_env_file=None, anthropic_api_key="k")


def test_scores_model_tolerates_missing_optional_fields():
    s = JudgeScores(faithfulness=4, completeness=3, relevance=5, citation_precision=0.5)
    assert s.correctness is None and s.hallucinated_claims == [] and s.rationale == ""
    assert s.as_metrics() == {
        "faithfulness": 4,
        "completeness": 3,
        "relevance": 5,
        "citation_precision": 0.5,
    }


def test_scores_clamped_to_rubric_range():
    s = JudgeScores(
        correctness=9, faithfulness=0, completeness=3, relevance=3, citation_precision=1.7
    )
    assert s.correctness == 5 and s.faithfulness == 1 and s.citation_precision == 1.0


async def test_judge_answer_uses_main_model_cache_and_includes_gold_only_when_given():
    scores = JudgeScores(
        correctness=4,
        faithfulness=5,
        completeness=4,
        relevance=5,
        citation_precision=1.0,
        rationale="ok",
    )
    llm = FakeLLM(scores)
    out, raw, usage = await judge_answer(
        llm,
        question="q",
        answer="a [#1]",
        context_text="[Nguồn #1] ...",
        gold_answer="gold",
        settings=_settings(),
    )
    assert out == scores and usage.input_tokens == 100 and "ok" in raw
    call = llm.calls[0]
    assert call["model"] == _settings().llm_model_main and call["use_cache"] is True
    assert call["system"] == JUDGE_SYSTEM and "gold" in call["messages"][0]["content"]
    assert PROMPT_VERSION == "v1" and "1-5" in JUDGE_SYSTEM

    llm2 = FakeLLM(scores)
    await judge_answer(
        llm2, question="q", answer="a", context_text="c", gold_answer=None, settings=_settings()
    )
    assert "Câu trả lời tham chiếu" not in llm2.calls[0]["messages"][0]["content"]
