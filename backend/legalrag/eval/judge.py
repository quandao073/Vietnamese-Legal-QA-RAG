"""Claude-as-judge (spec §5.3): rubric 1-5, JSON via structured outputs, cached."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from legalrag.config import Settings
from legalrag.generation.llm_client import Usage

PROMPT_VERSION = "v1"

JUDGE_SYSTEM = """Bạn là giám khảo đánh giá chất lượng câu trả lời của một hệ thống hỏi đáp pháp luật Việt Nam dựa trên truy xuất văn bản (RAG).
Bạn nhận: câu hỏi, các đoạn NGUỒN [Nguồn #i] mà hệ thống đã dùng, câu trả lời của hệ thống và (nếu có) câu trả lời tham chiếu.
Hãy chấm theo rubric, thang 1-5 (5 là tốt nhất), trả về JSON đúng schema, kèm lý do ngắn gọn.

Rubric:
- correctness (chỉ chấm khi có câu trả lời tham chiếu; nếu không có → null): câu trả lời có đúng về nội dung pháp lý so với tham chiếu không?
  5 = đúng và đầy đủ; 3 = đúng một phần/thiếu chi tiết quan trọng; 1 = sai hoặc trái ngược.
- faithfulness: mọi khẳng định pháp lý trong câu trả lời có được NGUỒN hỗ trợ không? KHÔNG dùng kiến thức riêng của bạn.
  5 = mọi khẳng định đều có trong nguồn; 3 = có khẳng định không kiểm chứng được nhưng không sai lệch; 1 = bịa quy định/số liệu/điều khoản không có trong nguồn.
  Nếu hệ thống từ chối ("không tìm thấy căn cứ") và nguồn thật sự không trả lời được → faithfulness = 5.
- citation_precision: tỉ lệ (0-1) các câu có trích dẫn [#i] mà nguồn i thực sự hỗ trợ nội dung câu đó. Không có trích dẫn → 0 nếu câu trả lời khẳng định nội dung pháp lý, 1 nếu câu trả lời là từ chối hợp lý.
- completeness: mức độ bao phủ các điểm cần thiết để trả lời câu hỏi (dựa trên nguồn có sẵn). 5 = đầy đủ; 1 = bỏ sót phần lớn.
- relevance: câu trả lời có đúng trọng tâm câu hỏi không, có lan man không. 5 = đúng trọng tâm; 1 = lạc đề.
- hallucinated_claims: liệt kê (ngắn) các khẳng định pháp lý KHÔNG có trong nguồn (để trống nếu không có).

Ví dụ hiệu chuẩn:
1) Nguồn nói "Thu nhập từ chuyển nhượng bất động sản giữa vợ với chồng... được miễn thuế". Câu trả lời: "Chuyển nhượng BĐS giữa vợ chồng được miễn thuế TNCN [#1]." Tham chiếu tương tự.
   → correctness 5, faithfulness 5, citation_precision 1.0, completeness 5, relevance 5, hallucinated_claims [].
2) Nguồn chỉ nói về đối tượng chịu thuế GTGT. Câu trả lời: "Thuế suất GTGT với phân bón là 5% theo Điều 9 [#1]" (nguồn không nêu).
   → faithfulness 1, citation_precision 0.0, hallucinated_claims ["thuế suất 5% với phân bón theo Điều 9"], relevance 3, completeness 2.

Chỉ trả về JSON hợp lệ theo schema."""


class JudgeScores(BaseModel):
    correctness: int | None = None
    faithfulness: int
    citation_precision: float
    completeness: int
    relevance: int
    hallucinated_claims: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("correctness", "faithfulness", "completeness", "relevance", mode="before")
    @classmethod
    def _clamp_1_5(cls, v: Any) -> Any:
        if v is None:
            return None
        return max(1, min(5, int(v)))

    @field_validator("citation_precision", mode="before")
    @classmethod
    def _clamp_0_1(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))

    def as_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "citation_precision": self.citation_precision,
        }
        if self.correctness is not None:
            out["correctness"] = self.correctness
        return out


def _user_message(question: str, answer: str, context_text: str, gold_answer: str | None) -> str:
    parts = [
        f"Câu hỏi: {question}",
        "",
        "NGUỒN hệ thống đã dùng:",
        context_text or "(không có nguồn)",
        "",
        "Câu trả lời của hệ thống:",
        answer,
    ]
    if gold_answer:
        parts += ["", f"Câu trả lời tham chiếu: {gold_answer}"]
    return "\n".join(parts)


async def judge_answer(
    llm: Any,
    *,
    question: str,
    answer: str,
    context_text: str,
    gold_answer: str | None,
    settings: Settings,
) -> tuple[JudgeScores, str, Usage]:
    scores, usage = await llm.complete_json(
        model=settings.llm_model_main,
        system=JUDGE_SYSTEM,
        messages=[
            {"role": "user", "content": _user_message(question, answer, context_text, gold_answer)}
        ],
        schema=JudgeScores,
        max_tokens=1200,
        effort="low",
        use_cache=True,
    )
    return scores, scores.model_dump_json(), usage
