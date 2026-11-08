"""RAG orchestrator (spec §3.9): understanding → retrieval → context → streaming generation →
citation verification → logging. Emits ChatEvents suitable for SSE."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol

from legalrag.config import Settings
from legalrag.generation.citation import append_disclaimer, clean_invalid, verify_citations
from legalrag.generation.context_builder import ContextBundle, build_context
from legalrag.generation.llm_client import LLMClient, LLMError, LLMResult, Usage
from legalrag.generation.prompts import (
    ANSWER_SYSTEM,
    INTENT_TEMPLATES,
    NO_EVIDENCE_TEMPLATE,
    PROMPT_VERSION,
    format_user_message,
)
from legalrag.retrieval.query_understanding import QueryUnderstanding, understand
from legalrag.retrieval.types import Hit, RetrievalOverrides, RetrievalResult
from legalrag.schemas.chat import (
    ChatEvent,
    ChatResponse,
    CitationReport,
    Message,
    QueryLogCreate,
    SourceOut,
)
from legalrag.tokens import TokenCounter

logger = logging.getLogger(__name__)
NEAREST_SOURCES_WHEN_UNGROUNDED = 3
HISTORY_TURNS_FOR_GENERATION = 2  # user+assistant pairs


class _RetrieverLike(Protocol):
    async def retrieve(
        self,
        question: str,
        qu: QueryUnderstanding | None = None,
        overrides: RetrievalOverrides | None = None,
    ) -> RetrievalResult: ...


class _LogRepoLike(Protocol):
    def insert(self, log: QueryLogCreate) -> None: ...


class Answerer:
    def __init__(
        self,
        *,
        retriever: _RetrieverLike,
        llm: LLMClient,
        settings: Settings,
        counter: TokenCounter,
        log_repo: _LogRepoLike | None = None,
        corpus_hint: str = (
            "các luật về thuế (GTGT, TNCN, TNDN, TTĐB, sử dụng đất phi nông nghiệp) "
            "và văn bản pháp luật tổng hợp"
        ),
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.counter = counter
        self.log_repo = log_repo
        self.corpus_hint = corpus_hint

    # ---- public --------------------------------------------------------------------------------
    async def answer(
        self,
        question: str,
        *,
        session_id: str | None,
        history: list[Message] | None,
        overrides: RetrievalOverrides | None,
        request_id: str | None = None,
    ) -> AsyncIterator[ChatEvent]:
        s = self.settings
        message_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        latency: dict[str, float] = {}
        usage = Usage()
        log = QueryLogCreate(
            message_id=message_id,
            session_id=session_id,
            request_id=request_id,
            question=question,
            prompt_version=PROMPT_VERSION,
        )
        answer_md = ""
        try:
            # 1. understanding
            yield ChatEvent(event="status", data={"stage": "understanding"})
            t = time.perf_counter()
            qu, u = await understand(question, history=history, llm=self.llm, settings=s)
            usage = usage.add(u)
            latency["understanding_ms"] = (time.perf_counter() - t) * 1000
            log.intent, log.standalone_question = qu.intent, qu.standalone_question

            if qu.intent != "legal_question":
                answer_md = INTENT_TEMPLATES[qu.intent]
                yield ChatEvent(event="sources", data={"sources": []})
                yield ChatEvent(event="token", data={"text": answer_md})
                yield ChatEvent(event="citations", data=CitationReport().model_dump())
                latency["total_ms"] = (time.perf_counter() - t0) * 1000
                log.answer_md, log.grounded, log.latency, log.usage = (
                    answer_md,
                    False,
                    latency,
                    usage.as_dict(),
                )
                yield self._done(
                    message_id, answer_md, False, qu, [], CitationReport(), usage, latency, None
                )
                return

            # 2. retrieval
            yield ChatEvent(event="status", data={"stage": "retrieving"})
            t = time.perf_counter()
            result = await self.retriever.retrieve(
                qu.standalone_question, qu=qu, overrides=overrides
            )
            latency["retrieval_ms"] = (time.perf_counter() - t) * 1000
            latency.update({f"retrieval_{k}": v for k, v in result.timings.items()})
            log.retrieved_chunk_ids = [h.chunk_id for h in result.reranked]
            if result.reranked and result.reranked[0].rerank_score is not None:
                yield ChatEvent(event="status", data={"stage": "reranking"})

            # 3. grounding gate
            grounded = self._is_grounded(result.reranked, overrides)
            if not grounded:
                nearest = result.reranked[:NEAREST_SOURCES_WHEN_UNGROUNDED]
                bundle = build_context(nearest, s.context_token_budget, self.counter)
                sources = [SourceOut(**src.as_dict()) for src in bundle.sources]
                yield ChatEvent(
                    event="sources", data={"sources": [x.model_dump() for x in sources]}
                )
                answer_md = append_disclaimer(
                    NO_EVIDENCE_TEMPLATE.format(
                        question=qu.standalone_question, corpus_hint=self.corpus_hint
                    )
                )
                yield ChatEvent(event="token", data={"text": answer_md})
                yield ChatEvent(event="citations", data=CitationReport().model_dump())
                latency["total_ms"] = (time.perf_counter() - t0) * 1000
                log.answer_md, log.grounded, log.latency, log.usage = (
                    answer_md,
                    False,
                    latency,
                    usage.as_dict(),
                )
                yield self._done(
                    message_id,
                    answer_md,
                    False,
                    qu,
                    sources,
                    CitationReport(),
                    usage,
                    latency,
                    None,
                )
                return

            # 4. context + generation
            bundle = build_context(result.reranked, s.context_token_budget, self.counter)
            sources = [SourceOut(**src.as_dict()) for src in bundle.sources]
            yield ChatEvent(event="sources", data={"sources": [x.model_dump() for x in sources]})
            yield ChatEvent(event="status", data={"stage": "generating"})
            t = time.perf_counter()
            final: LLMResult | None = None
            raw_parts: list[str] = []
            async for item in self.llm.stream(
                model=s.llm_model_main,
                system=ANSWER_SYSTEM,
                messages=self._generation_messages(bundle, qu.standalone_question, history),
                max_tokens=s.llm_max_tokens,
            ):
                if isinstance(item, LLMResult):
                    final = item
                else:
                    raw_parts.append(item)
                    yield ChatEvent(event="token", data={"text": item})
            latency["generation_ms"] = (time.perf_counter() - t) * 1000
            raw_answer = final.text if final else "".join(raw_parts)
            if final:
                usage = usage.add(final.usage)

            # 5. post-processing
            valid = {src.idx for src in bundle.sources}
            cleaned, invalid = clean_invalid(raw_answer, valid)
            report = verify_citations(cleaned, bundle.sources)
            report.invalid = sorted(set(report.invalid) | set(invalid))
            answer_md = append_disclaimer(cleaned)
            yield ChatEvent(event="citations", data=report.model_dump())
            latency["total_ms"] = (time.perf_counter() - t0) * 1000
            cited = [src.chunk_id for src in bundle.sources if src.idx in report.verified]
            log.answer_md, log.grounded, log.cited_chunk_ids = answer_md, True, cited
            log.model, log.latency, log.usage = (
                (final.model if final else s.llm_model_main),
                latency,
                usage.as_dict(),
            )
            yield self._done(
                message_id,
                answer_md,
                True,
                qu,
                sources,
                report,
                usage,
                latency,
                final.model if final else None,
            )
        except LLMError as exc:
            latency["total_ms"] = (time.perf_counter() - t0) * 1000
            log.error, log.grounded, log.latency, log.usage = (
                f"{exc.code}: {exc.detail}"[:500],
                False,
                latency,
                usage.as_dict(),
            )
            logger.warning("answer failed: %s", exc)
            yield ChatEvent(
                event="error",
                data={"code": exc.code, "message": exc.message_vi, "message_id": message_id},
            )
        except Exception as exc:  # pragma: no cover - defensive
            latency["total_ms"] = (time.perf_counter() - t0) * 1000
            log.error, log.grounded, log.latency = repr(exc)[:500], False, latency
            logger.exception("answer crashed")
            yield ChatEvent(
                event="error",
                data={
                    "code": "internal",
                    "message": "Đã xảy ra lỗi nội bộ khi xử lý câu hỏi. Vui lòng thử lại.",
                    "message_id": message_id,
                },
            )
        finally:
            self._log(log)

    async def answer_sync(
        self,
        question: str,
        *,
        session_id: str | None,
        history: list[Message] | None,
        overrides: RetrievalOverrides | None,
        request_id: str | None = None,
    ) -> ChatResponse:
        sources: list[SourceOut] = []
        async for ev in self.answer(
            question,
            session_id=session_id,
            history=history,
            overrides=overrides,
            request_id=request_id,
        ):
            if ev.event == "sources":
                sources = [SourceOut(**s) for s in ev.data["sources"]]
            elif ev.event == "done":
                d = ev.data
                return ChatResponse(
                    message_id=d["message_id"],
                    answer_md=d["answer_md"],
                    grounded=d["grounded"],
                    intent=d["intent"],
                    sources=sources,
                    citations=CitationReport(**d["citations"]),
                    usage=d["usage"],
                    latency_ms=d["latency_ms"],
                    model=d.get("model"),
                    prompt_version=PROMPT_VERSION,
                    standalone_question=d.get("standalone_question"),
                )
            elif ev.event == "error":
                raise LLMError(ev.data.get("code", "unknown"), ev.data.get("message", ""))  # type: ignore[arg-type]
        raise LLMError("unknown", "stream ended without done event")

    # ---- helpers --------------------------------------------------------------------------------
    def _is_grounded(self, hits: list[Hit], overrides: RetrievalOverrides | None) -> bool:
        if not hits:
            return False
        top = hits[0].rerank_score
        if top is None:  # rerank disabled → trust fusion ranking
            return True
        return top >= self.settings.min_rerank_score

    def _generation_messages(
        self, bundle: ContextBundle, question: str, history: list[Message] | None
    ) -> list[dict[str, Any]]:
        recent = (history or [])[-2 * HISTORY_TURNS_FOR_GENERATION :]
        # must alternate and start with user; drop a leading assistant turn if any
        while recent and recent[0].role != "user":
            recent = recent[1:]
        msgs: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in recent]
        if msgs and msgs[-1]["role"] == "user":
            msgs.pop()  # the current question replaces a dangling user turn
        msgs.append({"role": "user", "content": format_user_message(bundle.text, question)})
        return msgs

    @staticmethod
    def _done(
        message_id: str,
        answer_md: str,
        grounded: bool,
        qu: QueryUnderstanding,
        sources: list[SourceOut],
        report: CitationReport,
        usage: Usage,
        latency: dict[str, float],
        model: str | None,
    ) -> ChatEvent:
        return ChatEvent(
            event="done",
            data={
                "message_id": message_id,
                "answer_md": answer_md,
                "grounded": grounded,
                "intent": qu.intent,
                "standalone_question": qu.standalone_question,
                "sources": [s.model_dump() for s in sources],
                "citations": report.model_dump(),
                "usage": usage.as_dict(),
                "latency_ms": {k: round(v, 1) for k, v in latency.items()},
                "model": model,
                "prompt_version": PROMPT_VERSION,
            },
        )

    def _log(self, log: QueryLogCreate) -> None:
        if self.log_repo is None:
            return
        try:
            self.log_repo.insert(log)
        except Exception as exc:  # pragma: no cover - logging must never break the stream
            logger.warning("query log failed: %s", exc)
