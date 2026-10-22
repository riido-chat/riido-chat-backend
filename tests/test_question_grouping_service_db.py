"""QuestionGroupingService 를 실제 store·reader 와 로컬 DB 로 끝까지 돌리는 통합 테스트.

- 판별 모델과 임베딩은 가짜다(API 호출 없음). DB 는 로컬 PostgreSQL 과 head 마이그레이션이 필요하고
  연결할 수 없으면 skip 한다.
- 세션은 외부 트랜잭션 안의 savepoint 로 commit 하고, 끝에 외부 트랜잭션을 rollback 한다.

실행: DATABASE_URL=postgresql+asyncpg://riido:riido@localhost:5433/riido \
    .venv/bin/python -m unittest tests.test_question_grouping_service_db -v
"""

import json
import logging
import unittest
from typing import Any, List, Optional

from sqlalchemy import select

from app.chat.log_store import CitationLog
from app.core.model_trace import ModelCallTrace
from app.database.models import (
    AnswerCitation,
    AnswerStatus,
    AttributionSource,
    CacheAttemptOutcome,
    ClassificationDecision,
    ExecutionStatus,
    ModelCall,
    ModelCallPurpose,
    QuestionCacheAttempt,
    QuestionClassification,
    QuestionEmbedding,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    RagRun,
)
from app.question_grouping.constants import (
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER,
    REJECT_CANONICAL_SERVE_FAILED,
)
from app.question_grouping.models import JudgeCall, JudgeFailure, JudgeFailureKind
from app.question_grouping.outline_reader import DocumentOutlineCache, DocumentOutlineReader
from app.question_grouping.service import GroupingTurn, QuestionGroupingService
from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_PROVIDER
from app.retrieval.models import HybridSearchCall, RetrievalChunk, RetrievalResult
from tests.test_question_grouping_readers_db import _vector
from tests.test_question_grouping_store_db import TITLE, _StoreDbTestCase

QUESTION = "구독을 취소하고 싶어요"


def _trace(succeeded: bool = True, error: Optional[str] = None) -> ModelCallTrace:
    return ModelCallTrace(
        provider=JUDGE_PROVIDER,
        model_name=JUDGE_MODEL,
        succeeded=succeeded,
        latency_ms=900,
        input_tokens=4000,
        output_tokens=200,
        cached_input_tokens=3000,
        reasoning_tokens=80,
        prompt_version=JUDGE_PROMPT_VERSION,
        error_message=error,
    )


class ConnectFirstCandidateJudge:
    """제시된 세부 문제 중 첫 key 로 CONNECT 하거나, 지정한 실패를 돌려준다."""

    provider = JUDGE_PROVIDER
    model_name = JUDGE_MODEL
    prompt_version = JUDGE_PROMPT_VERSION

    def __init__(self, failure: Optional[JudgeFailure] = None) -> None:
        self.failure = failure

    async def judge(self, payload, *, before_model_call=None) -> JudgeCall:
        await before_model_call(JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_PROMPT_VERSION)
        if self.failure is not None:
            return JudgeCall(trace=_trace(False, self.failure.safe_message), failure=self.failure)
        candidate = payload["candidates"][0]
        output = {
            "decision": "CONNECT",
            "groupId": candidate["group"]["id"],
            "subproblemId": candidate["subproblem"]["id"],
            "confidence": 0.93,
            "rationaleCode": "SAME_ASK",
            "ambiguityReason": None,
            "matchedCriteria": ["I1"],
            "conflictingCriteria": [],
            "documentDecision": "SUBPROBLEM_DOCUMENT",
            "documentCandidateId": None,
            "documentRationaleCode": None,
        }
        return JudgeCall(trace=_trace(), output_text=json.dumps(output))


class UnusedEmbedder:
    def embed_many_with_usage(self, texts, **kwargs):  # pragma: no cover - 재사용 경로라 부르지 않는다
        raise AssertionError("검색 벡터를 재사용해야 합니다.")


class ServiceDbTest(_StoreDbTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        service_logger = logging.getLogger("app.question_grouping.service")
        service_logger.disabled = True
        self.addCleanup(setattr, service_logger, "disabled", False)
        self.subproblem = await self._billing_subproblem()
        self.canonical = await self.seed.canonical(
            self.subproblem, [(self.billing_chunks[1], self.billing_v1.id)]
        )
        # record_serve_failure 의 rollback 뒤에도 쓰도록 식별자를 먼저 읽어 둔다(만료된 ORM 객체 접근 방지).
        self.canonical_id = self.canonical.id
        self.canonical_markdown = self.canonical.content_markdown
        self.embed_id = self.embed.id

    def _service(self, judge: Any) -> QuestionGroupingService:
        return QuestionGroupingService(
            self.session,
            self.log_store,
            judge,
            UnusedEmbedder(),
            store=self.store,
            outline_reader=DocumentOutlineReader(self.session, cache=DocumentOutlineCache()),
        )

    def _search(self) -> HybridSearchCall:
        def result(chunk_id: int, version_id: int, rank: int) -> RetrievalResult:
            return RetrievalResult(
                chunk=RetrievalChunk(
                    document_id="doc",
                    section_id=f"doc:{chunk_id}",
                    document_title=TITLE,
                    section_path=(TITLE,),
                    source_url="https://docs.riido.io",
                    category=None,
                    content="본문",
                    chunk_id=chunk_id,
                    document_version_id=version_id,
                    index_version_id=self.scope.index_version_id,
                ),
                score=1.0,
                rank=rank,
            )

        results = (
            result(self.billing_chunks[1], self.billing_v1.id, 1),
            result(self.members_chunks[0], self.members_v1.id, 2),
        )
        return HybridSearchCall(
            bm25_results=results,
            vector_results=results,
            retrieval_query=QUESTION,
            query_embedding=tuple(_vector((0, 1.0))),
        )

    async def _start_turn(self, service: QuestionGroupingService):
        """ChatService 순서를 흉내 낸다: 턴 시작 → 분류 실행 → 검색 임베딩 checkpoint."""

        turn = await self._turn()
        await self.session.commit()
        grouping_turn = await service.open_online_run(
            rag_run_id=turn.id,
            document_group_id=self.scope.document_group_id,
            index_version_id=self.scope.index_version_id,
        )
        embedding_call = await self.log_store.start_model_call(
            purpose=ModelCallPurpose.QUERY_EMBEDDING.value,
            provider=OPENAI_EMBEDDING_PROVIDER,
            model_name=OPENAI_EMBEDDING_MODEL,
            rag_run_id=turn.id,
            classification_run_id=grouping_turn.classification_run_id,
        )
        embedding_call_id = embedding_call.id
        await self.session.commit()

        async def finish_retrieval_embedding() -> None:
            await self.log_store.finish_model_call(
                embedding_call_id, status=ExecutionStatus.SUCCESS, input_tokens=10, latency_ms=50
            )

        return grouping_turn, finish_retrieval_embedding

    async def _model_calls(self, turn: GroupingTurn) -> List[Any]:
        return (
            await self.session.execute(
                select(
                    ModelCall.purpose,
                    ModelCall.status,
                    ModelCall.classification_run_id,
                    ModelCall.cached_input_tokens,
                ).where(ModelCall.rag_run_id == turn.rag_run_id)
                .order_by(ModelCall.id)
            )
        ).all()

    async def _rows(self, turn: GroupingTurn):
        classifications = (
            await self.session.execute(
                select(
                    QuestionClassification.id,
                    QuestionClassification.decision,
                    QuestionClassification.attribution_source,
                    QuestionClassification.problem_group_id,
                    QuestionClassification.subproblem_id,
                    QuestionClassification.run_id,
                    QuestionClassification.judgment_input,
                ).where(QuestionClassification.rag_run_id == turn.rag_run_id)
            )
        ).all()
        attempts = (
            await self.session.execute(
                select(
                    QuestionCacheAttempt.classification_id,
                    QuestionCacheAttempt.outcome,
                    QuestionCacheAttempt.canonical_answer_id,
                    QuestionCacheAttempt.rejection_reasons,
                ).where(QuestionCacheAttempt.rag_run_id == turn.rag_run_id)
            )
        ).all()
        embeddings = await self.session.scalar(
            select(QuestionEmbedding.embedding_config_id).where(
                QuestionEmbedding.rag_run_id == turn.rag_run_id
            )
        )
        return classifications, attempts, embeddings

    async def test_served_turn_end_to_end(self) -> None:
        service = self._service(ConnectFirstCandidateJudge())
        turn, finish_retrieval_embedding = await self._start_turn(service)

        prepared = await service.prepare(turn, QUESTION, self._search())
        self.assertTrue(prepared.ready, prepared.failure)
        judged = await service.judge(prepared, before_checkpoint=finish_retrieval_embedding)
        recorded = await service.record_judgment_and_gate(prepared, judged, semantic_cache_enabled=True)
        self.assertTrue(recorded.served)
        await self.log_store.complete_rag_run(
            turn.rag_run_id,
            answer_content=recorded.served_answer_markdown,
            citations=recorded.served_citations,
        )
        await self.session.commit()

        run = await self.session.get(RagRun, turn.rag_run_id)
        await self.session.refresh(run)
        self.assertEqual(AnswerStatus.COMPLETED, run.status)
        self.assertEqual(self.canonical_markdown, run.answer_content)
        citations = (
            await self.session.execute(
                select(AnswerCitation.chunk_id, AnswerCitation.document_version_id, AnswerCitation.citation_order)
                .where(AnswerCitation.rag_run_id == turn.rag_run_id)
            )
        ).all()
        self.assertEqual([(self.billing_chunks[1], self.billing_v1.id, 1)], [tuple(row) for row in citations])

        classifications, attempts, embedding_config = await self._rows(turn)
        self.assertEqual(1, len(classifications))
        row = classifications[0]
        self.assertEqual(
            (ClassificationDecision.CONNECT, AttributionSource.SUBPROBLEM, self.subproblem.problem_group_id, self.subproblem.id, turn.classification_run_id),
            (row.decision, row.attribution_source, row.problem_group_id, row.subproblem_id, row.run_id),
        )
        self.assertEqual("SERVED", row.judgment_input["gate"]["outcome"])
        self.assertEqual(str(self.canonical.id), row.judgment_input["gate"]["canonicalAnswerId"])
        self.assertEqual(str(turn.rag_run_id), row.judgment_input["presentationSeed"])
        self.assertEqual(
            [(row.id, CacheAttemptOutcome.SERVED, self.canonical.id, None)], [tuple(item) for item in attempts]
        )
        self.assertEqual(self.embed.id, embedding_config)
        self.assertEqual(
            [
                (ModelCallPurpose.QUERY_EMBEDDING.value, ExecutionStatus.SUCCESS, turn.classification_run_id, None),
                (ModelCallPurpose.QUESTION_CLASSIFICATION.value, ExecutionStatus.SUCCESS, turn.classification_run_id, 3000),
            ],
            [tuple(call) for call in await self._model_calls(turn)],
        )

    async def test_judge_failure_then_generation_completes_with_citation_attribution(self) -> None:
        failure = JudgeFailure(JudgeFailureKind.API_ERROR, "OpenAI 판별 호출 실패: HTTP 500")
        service = self._service(ConnectFirstCandidateJudge(failure))
        turn, finish_retrieval_embedding = await self._start_turn(service)

        prepared = await service.prepare(turn, QUESTION, self._search())
        judged = await service.judge(prepared, before_checkpoint=finish_retrieval_embedding)
        recorded = await service.record_judgment_and_gate(prepared, judged, semantic_cache_enabled=True)
        await self.session.commit()

        classifications, attempts, _ = await self._rows(turn)
        row = classifications[0]
        self.assertEqual((ClassificationDecision.UNCLASSIFIED, AttributionSource.NONE), (row.decision, row.attribution_source))
        group = await self._problem_group(row.problem_group_id)
        self.assertEqual(QuestionProblemGroupKind.NO_DOCUMENT, group.kind)
        self.assertEqual("API_ERROR", row.judgment_input["failure"]["kind"])
        self.assertEqual([CacheAttemptOutcome.FAILED], [item.outcome for item in attempts])
        self.assertEqual(ExecutionStatus.FAILED, (await self._model_calls(turn))[1].status)

        # 생성이 인용 [1] 로 완료: 턴 끝 귀속을 complete_rag_run 보다 먼저, 같은 트랜잭션에서.
        generation_citations = [CitationLog(self.members_chunks[0], self.members_v1.id, 1)]
        self.assertTrue(await service.finalize_attribution(recorded, generation_citations))
        await self.log_store.complete_rag_run(
            turn.rag_run_id, answer_content="생성 답변 [1]", citations=generation_citations
        )
        await self.session.commit()

        classifications, _, _ = await self._rows(turn)
        row = classifications[0]
        self.assertEqual(AttributionSource.CITATION, row.attribution_source)
        group = await self._problem_group(row.problem_group_id)
        self.assertEqual((QuestionProblemGroupKind.DOCUMENT, self.members.id), (group.kind, group.document_source_id))
        self.assertEqual("FAILED", row.judgment_input["gate"]["outcome"])

    async def test_serve_failure_rewrites_rejected_attempt_and_turn_continues(self) -> None:
        service = self._service(ConnectFirstCandidateJudge())
        turn, finish_retrieval_embedding = await self._start_turn(service)
        prepared = await service.prepare(turn, QUESTION, self._search())
        judged = await service.judge(prepared, before_checkpoint=finish_retrieval_embedding)
        served = await service.record_judgment_and_gate(prepared, judged, semantic_cache_enabled=True)

        # complete_rag_run 이 실패한 상황(인용 없음으로 ValueError)을 흉내 낸다.
        with self.assertRaises(ValueError):
            await self.log_store.complete_rag_run(turn.rag_run_id, answer_content="x", citations=[])
        retried = await service.record_serve_failure(prepared, judged, served)
        await self.session.commit()

        self.assertEqual(CacheAttemptOutcome.REJECTED, retried.gate.outcome)
        classifications, attempts, embedding_config = await self._rows(turn)
        self.assertEqual(1, len(classifications))
        row = classifications[0]
        self.assertEqual((ClassificationDecision.CONNECT, AttributionSource.SUBPROBLEM), (row.decision, row.attribution_source))
        self.assertEqual("REJECTED", row.judgment_input["gate"]["outcome"])
        self.assertEqual(str(self.canonical_id), row.judgment_input["gate"]["canonicalAnswerId"])
        self.assertEqual(
            [(row.id, CacheAttemptOutcome.REJECTED, None, [REJECT_CANONICAL_SERVE_FAILED])],
            [tuple(item) for item in attempts],
        )
        self.assertEqual(self.embed_id, embedding_config)
        run = await self.session.get(RagRun, turn.rag_run_id)
        await self.session.refresh(run)
        self.assertEqual(AnswerStatus.PROCESSING, run.status)
        self.assertEqual(
            [ExecutionStatus.SUCCESS, ExecutionStatus.SUCCESS],
            [call.status for call in await self._model_calls(turn)],
        )

    async def test_catalog_data_error_records_preparation_failure(self) -> None:
        # 같은 문서 그룹의 다른 문서에 같은 key 가 있으면 payload 식별자가 겹친다.
        members_group = await self.store.ensure_document_problem_group(self.members.id)
        await self.seed.subproblem(
            await self.session.get(QuestionProblemGroup, members_group),
            self.subproblem.key,
            embedding=self.embed,
            vector=_vector((1, 1.0)),
        )
        service = self._service(ConnectFirstCandidateJudge())
        turn, finish_retrieval_embedding = await self._start_turn(service)

        prepared = await service.prepare(turn, QUESTION, self._search())
        self.assertFalse(prepared.ready)
        await finish_retrieval_embedding()
        await service.record_preparation_failure(prepared)
        await self.session.commit()

        classifications, attempts, embedding_config = await self._rows(turn)
        self.assertEqual(ClassificationDecision.UNCLASSIFIED, classifications[0].decision)
        self.assertEqual("CANDIDATES", classifications[0].judgment_input["failure"]["stage"])
        self.assertEqual([CacheAttemptOutcome.FAILED], [item.outcome for item in attempts])
        self.assertEqual(self.embed.id, embedding_config)
        self.assertEqual(
            [ModelCallPurpose.QUERY_EMBEDDING.value], [call.purpose for call in await self._model_calls(turn)]
        )


if __name__ == "__main__":
    unittest.main()
