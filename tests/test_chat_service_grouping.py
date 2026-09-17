"""ChatService 와 질문 판별(QuestionGroupingService)의 통합 분기를 가짜 협력자로 고정한다.

판별 서비스 내부(prepare/judge/record_*)는 test_question_grouping_service 가 검증한다. 여기서는
ChatService 가 판별 메서드를 부르는 시점, 트랜잭션 경계(commit/rollback), 검색 로그를 한 번만
기록하는지, SERVED 턴의 마감과 응답, 실패 시 fail-closed 경로를 호출 순서로 확인한다.
"""

import asyncio
import unittest
import uuid
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import ANY, AsyncMock, Mock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from app.answering.service import WITHHELD_RESPONSES, GenerationService
from app.chat.log_store import (
    CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE,
    CitationLog,
    RagLogStore,
)
from app.chat.progress import ProgressStage
from app.chat.query_rewrite import (
    QUERY_REWRITE_PROMPT_VERSION,
    QueryResolution,
    QueryRewriteCall,
    QueryRewriteDecision,
    QueryRewriteService,
)
from app.chat.schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponseStatus,
    ChatWithheldResponse,
)
from app.chat.service import ChatService
from app.core.model_trace import ModelCallTrace
from app.database.models import (
    CacheAttemptOutcome,
    ChatProfileRevisionStatus,
    ContextStrategy,
    ModelCallPurpose,
)
from app.document.document_key import (
    DEFAULT_DOCUMENT_GROUP_KEY,
    build_console_canonical_uri,
    build_upload_document_key,
)
from app.question_grouping.exact_question import exact_question_hash
from app.question_grouping.models import GateResult
from app.question_grouping.service import (
    GroupingTurn,
    ExactQuestionResult,
    QuestionGroupingService,
    RecordedJudgment,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalChunk,
    RetrievalResult,
)

INDEX_VERSION_ID = 91
DOCUMENT_GROUP_ID = 23
PROFILE_REVISION_ID = 11
CLASSIFICATION_RUN_ID = 5
EMBEDDING_CALL_ID = 1

SERVED_ANSWER = "멤버를 초대할 수 있습니다. [1] 업로드 문서도 참고하세요. [2]"
CACHED_SERVED_ANSWER = f"{SERVED_ANSWER} (캐시된 답변)"
SERVED_TITLE = "멤버 관리"
SERVED_NODE_PATH = "멤버 관리 > 워크스페이스 > 멤버 초대"
SERVED_URL = "https://docs.riido.io/member/invite"
CONSOLE_URI = build_console_canonical_uri(
    DEFAULT_DOCUMENT_GROUP_KEY,
    build_upload_document_key("업로드 문서"),
)


def served_citation_logs() -> tuple:
    return (
        CitationLog(
            chunk_id=41,
            document_version_id=7,
            citation_order=1,
            document_title_snapshot=SERVED_TITLE,
            node_path_snapshot=SERVED_NODE_PATH,
            source_uri_snapshot=SERVED_URL,
        ),
        CitationLog(
            chunk_id=42,
            document_version_id=8,
            citation_order=2,
            document_title_snapshot="업로드 문서",
            node_path_snapshot="업로드 문서",
            source_uri_snapshot=CONSOLE_URI,
        ),
    )


def recorded_judgment(outcome: str, *, served: bool = False) -> SimpleNamespace:
    """ChatService 가 읽는 RecordedJudgment 속성만 가진 가짜 기록."""

    return SimpleNamespace(
        outcome=outcome,
        served=served,
        served_answer_markdown=SERVED_ANSWER if served else None,
        served_citations=served_citation_logs() if served else (),
    )


def _chunk(index: int) -> RetrievalChunk:
    return RetrievalChunk(
        document_id=f"document-{index}",
        section_id=f"section-{index}",
        document_title=f"문서 {index}",
        section_path=(f"문서 {index}", f"섹션 {index}"),
        source_url=f"https://docs.riido.io/{index}",
        category="guide",
        content=f"본문 {index}",
        chunk_id=index,
        document_version_id=100 + index,
        index_version_id=INDEX_VERSION_ID,
    )


def search_call() -> HybridSearchCall:
    chunks = [_chunk(1), _chunk(2)]
    return HybridSearchCall(
        bm25_results=(
            RetrievalResult(chunk=chunks[0], score=9.5, rank=1),
            RetrievalResult(chunk=chunks[1], score=4.5, rank=2),
        ),
        vector_results=(RetrievalResult(chunk=chunks[0], score=0.91, rank=1),),
        fused_results=(
            HybridRetrievalResult(
                chunk=chunks[0], rrf_score=0.5, final_rank=1, bm25_rank=1, vector_rank=1
            ),
        ),
        bm25_latency_ms=12,
        vector_latency_ms=34,
        embedding_call=ModelCallTrace(
            provider="openai",
            model_name="text-embedding-3-large",
            succeeded=True,
            latency_ms=30,
            input_tokens=11,
        ),
        retrieval_query="질문",
        query_embedding=(0.1, 0.2),
    )


def generation_trace() -> ModelCallTrace:
    return ModelCallTrace(
        provider="openai",
        model_name="gpt-5.4-mini",
        succeeded=True,
        latency_ms=900,
        input_tokens=1200,
        output_tokens=300,
        prompt_version="v2",
    )


def completed_generation() -> FinalGenerationResult:
    return FinalGenerationResult(
        status=FinalAnswerStatus.COMPLETED,
        answer_markdown="첫 번째 근거입니다. [1]",
        citations=(
            Citation(
                citation_number=1,
                document_title="문서 1",
                section_path=("문서 1", "섹션 1"),
                source_url="https://docs.riido.io/1",
                source_kind=CitationSourceKind.GITBOOK,
                chunk_id=1,
                document_version_id=101,
            ),
        ),
        model_call=generation_trace(),
    )


def expected_served_response(
    conversation_id: uuid.UUID, rag_run_id: uuid.UUID
) -> ChatCompletedResponse:
    return ChatCompletedResponse(
        status=ChatResponseStatus.COMPLETED,
        conversation_id=conversation_id,
        rag_run_id=rag_run_id,
        answer=ChatAnswer(answer_markdown=CACHED_SERVED_ANSWER),
        citations=[
            ChatCitation(
                citation_number=1,
                document_title=SERVED_TITLE,
                section_path=["워크스페이스", "멤버 초대"],
                source_url=SERVED_URL,
                source_kind=CitationSourceKind.GITBOOK,
            ),
            ChatCitation(
                citation_number=2,
                document_title="업로드 문서",
                section_path=[],
                source_url=None,
                source_kind=CitationSourceKind.CONSOLE,
            ),
        ],
    )


class GroupingChatFixture:
    """판별이 붙은 ChatService 와 호출 순서를 남기는 가짜 협력자 묶음.

    start() 로 프로필 해석 patch 를 시작하고 stop() 으로 끝낸다. events 에는 로그 쓰기,
    트랜잭션 경계, 검색·생성·판별 호출이 실제 순서대로 쌓인다.
    """

    def __init__(
        self,
        *,
        grouping_enabled: bool = True,
        inject_grouping: bool = True,
        semantic_cache_enabled: bool = True,
        exact_cache_enabled: Optional[bool] = None,
        turn_no: int = 1,
        index_version_id: Optional[int] = INDEX_VERSION_ID,
    ) -> None:
        self.events: List[str] = []
        self.conversation_id = uuid.uuid4()
        self.rag_run_id = uuid.uuid4()
        self.search = search_call()
        self.generation_result: FinalGenerationResult = completed_generation()
        self.prepared = SimpleNamespace(ready=True, label="prepared")
        self.judged = SimpleNamespace(label="judged")
        self.recorded = recorded_judgment("SHADOW")
        self.serve_failure_recorded = recorded_judgment("REJECTED_SERVE_FAILED")
        revision_kwargs = dict(
            id=PROFILE_REVISION_ID,
            document_group_id=DOCUMENT_GROUP_ID,
            semantic_cache_enabled=semantic_cache_enabled,
        )
        if exact_cache_enabled is not None:
            revision_kwargs["exact_cache_enabled"] = exact_cache_enabled
        self.revision = SimpleNamespace(**revision_kwargs)

        self.retriever = AsyncMock(spec=HybridRetriever)
        self.generation_service = AsyncMock(spec=GenerationService)
        self.query_rewrite_service = AsyncMock(spec=QueryRewriteService)
        self.log_store = AsyncMock(spec=RagLogStore)
        self.session = AsyncMock(spec=AsyncSession)
        self.grouping = AsyncMock(spec=QuestionGroupingService)

        self._model_call_ids = iter(range(1, 100))
        self._wire_log_store(turn_no)
        self._wire_session()
        self._wire_pipeline()
        self._wire_grouping()

        self.service = ChatService(
            retriever=None,
            generation_service=self.generation_service,
            query_rewrite_service=self.query_rewrite_service,
            log_store=self.log_store,
            session=self.session,
            index_version_id=None,
            profile_status=ChatProfileRevisionStatus.PUBLISHED,
            retriever_factory=lambda _group_id: (self.retriever, index_version_id),
            **(
                {
                    "question_grouping": self.grouping,
                    "question_grouping_enabled": grouping_enabled,
                }
                if inject_grouping
                else {}
            ),
        )
        self._patchers = [
            patch(
                "app.chat.service.resolve_chat_profile_revision",
                new=AsyncMock(return_value=self.revision),
            ),
            patch("app.chat.service.validate_runtime_model_configuration", new=Mock()),
        ]

    def start(self) -> None:
        for patcher in self._patchers:
            patcher.start()

    def stop(self) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()

    # ------------------------------------------------------------------

    def _record(self, label: str, result: Any = None, error: Any = None):
        async def effect(*_args, **_kwargs):
            self.events.append(label)
            if error is not None:
                raise error
            return result

        return effect

    def _wire_log_store(self, turn_no: int) -> None:
        store = self.log_store
        store.create_conversation.side_effect = self._record(
            "create_conversation", SimpleNamespace(id=self.conversation_id)
        )
        store.start_rag_run.side_effect = self._record(
            "start_rag_run", SimpleNamespace(id=self.rag_run_id, turn_no=turn_no)
        )

        async def start_model_call(**kwargs):
            owner = kwargs.get("classification_run_id")
            suffix = "" if owner is None else f"@run{owner}"
            self.events.append(f"start_model_call:{kwargs['purpose']}{suffix}")
            return SimpleNamespace(id=next(self._model_call_ids))

        async def finish_model_call(model_call_id, **_kwargs):
            self.events.append(f"finish_model_call:{model_call_id}")

        store.start_model_call.side_effect = start_model_call
        store.finish_model_call.side_effect = finish_model_call
        for name in (
            "record_retrieval_results",
            "complete_rag_run",
            "withhold_rag_run",
            "fail_rag_run",
            "record_query_resolution",
            "fail_processing_model_calls",
            "cancel_rag_run",
        ):
            getattr(store, name).side_effect = self._record(name)
        store.get_query_rewrite_candidates.return_value = []

    def _wire_session(self) -> None:
        self.session.commit.side_effect = self._record("commit")
        self.session.rollback.side_effect = self._record("rollback")

    def _wire_pipeline(self) -> None:
        async def search(_query, *, before_model_call):
            await before_model_call("openai", "text-embedding-3-large", None)
            self.events.append("search")
            return self.search

        async def generate(_query, _results, *, before_model_call, on_progress_stage=None):
            await before_model_call("openai", "gpt-5.4-mini", "v2")
            self.events.append("generate")
            return self.generation_result

        self.retriever.search_with_trace.side_effect = search
        self.generation_service.generate_answer.side_effect = generate

    def _wire_grouping(self) -> None:
        grouping = self.grouping

        async def open_online_run(*, rag_run_id, document_group_id, index_version_id):
            self.events.append("open_online_run")
            return GroupingTurn(
                rag_run_id=rag_run_id,
                document_group_id=document_group_id,
                index_version_id=index_version_id,
                classification_run_id=CLASSIFICATION_RUN_ID,
            )

        async def prepare(_turn, _query, _search):
            self.events.append("prepare")
            return self.prepared

        async def judge(_prepared, *, before_checkpoint=None):
            # 실제 judge 의 checkpoint 트랜잭션을 흉내낸다.
            if before_checkpoint is not None:
                await before_checkpoint()
            await self.log_store.start_model_call(
                purpose=ModelCallPurpose.QUESTION_CLASSIFICATION.value,
                provider="openai",
                model_name="judge",
                rag_run_id=self.rag_run_id,
                classification_run_id=CLASSIFICATION_RUN_ID,
            )
            await self.session.commit()
            self.events.append("judge")
            return self.judged

        async def record_judgment_and_gate(_prepared, _judged, *, semantic_cache_enabled):
            self.events.append("record_judgment_and_gate")
            return self.recorded

        async def record_preparation_failure(_prepared):
            self.events.append("record_preparation_failure")
            return self.recorded

        async def record_serve_failure(_prepared, _judged, _recorded):
            self.events.append("record_serve_failure")
            return self.serve_failure_recorded

        async def finalize_attribution(recorded, _citations):
            outcome = getattr(recorded, "outcome", None)
            if outcome is None:
                outcome = recorded.gate.outcome.value
            self.events.append(f"finalize_attribution:{outcome}")
            return True

        grouping.open_online_run.side_effect = open_online_run
        grouping.prepare.side_effect = prepare
        grouping.judge.side_effect = judge
        grouping.record_judgment_and_gate.side_effect = record_judgment_and_gate
        grouping.record_preparation_failure.side_effect = record_preparation_failure
        grouping.record_serve_failure.side_effect = record_serve_failure
        grouping.finalize_attribution.side_effect = finalize_attribution
        # 기본은 과거 첫 턴 질문 로그 일치가 없는 턴이다.
        grouping.record_exact_question.return_value = None


QUERY_EMBEDDING = ModelCallPurpose.QUERY_EMBEDDING.value
QUESTION_CLASSIFICATION = ModelCallPurpose.QUESTION_CLASSIFICATION.value
ANSWER_GENERATION = ModelCallPurpose.ANSWER_GENERATION.value

TURN_START = ["create_conversation", "start_rag_run", "commit"]
SEARCH_WITH_GROUPING = [
    "open_online_run",
    f"start_model_call:{QUERY_EMBEDDING}@run{CLASSIFICATION_RUN_ID}",
    "commit",
    "search",
    "rollback",
]
JUDGE_CHECKPOINT = [
    "prepare",
    f"finish_model_call:{EMBEDDING_CALL_ID}",
    "record_retrieval_results",
    f"start_model_call:{QUESTION_CLASSIFICATION}@run{CLASSIFICATION_RUN_ID}",
    "commit",
    "judge",
]


class ChatServiceGroupingTest(unittest.IsolatedAsyncioTestCase):
    def _fixture(self, **kwargs) -> GroupingChatFixture:
        fixture = GroupingChatFixture(**kwargs)
        fixture.start()
        self.addCleanup(fixture.stop)
        return fixture

    async def _answer(self, fixture: GroupingChatFixture, question: str = "질문", **kwargs):
        stages: List[ProgressStage] = []

        async def on_progress_stage(stage: ProgressStage) -> None:
            stages.append(stage)

        response = await fixture.service.answer_question(
            question, on_progress_stage=on_progress_stage, **kwargs
        )
        return response, stages

    # ------------------------------------------------------------------
    # SERVED
    # ------------------------------------------------------------------

    async def test_served_turn_completes_with_canonical_answer_without_generation(
        self,
    ) -> None:
        fixture = self._fixture()
        fixture.recorded = recorded_judgment("SERVED", served=True)

        response, stages = await self._answer(fixture)

        self.assertEqual([ProgressStage.RETRIEVING], stages)
        fixture.generation_service.generate_answer.assert_not_awaited()
        self.assertEqual(
            TURN_START
            + SEARCH_WITH_GROUPING
            + JUDGE_CHECKPOINT
            + ["record_judgment_and_gate", "complete_rag_run", "commit"],
            fixture.events,
        )
        fixture.log_store.complete_rag_run.assert_awaited_once_with(
            fixture.rag_run_id,
            answer_content=CACHED_SERVED_ANSWER,
            citations=list(served_citation_logs()),
            total_latency_ms=ANY,
        )
        fixture.grouping.record_judgment_and_gate.assert_awaited_once_with(
            fixture.prepared, fixture.judged, semantic_cache_enabled=True
        )
        fixture.grouping.finalize_attribution.assert_not_awaited()
        fixture.grouping.record_serve_failure.assert_not_awaited()
        self.assertEqual(
            expected_served_response(fixture.conversation_id, fixture.rag_run_id),
            response,
        )

    async def test_open_online_run_receives_turn_scope(self) -> None:
        fixture = self._fixture()

        await self._answer(fixture)

        fixture.grouping.open_online_run.assert_awaited_once_with(
            rag_run_id=fixture.rag_run_id,
            document_group_id=DOCUMENT_GROUP_ID,
            index_version_id=INDEX_VERSION_ID,
        )
        prepare_args = fixture.grouping.prepare.await_args.args
        self.assertEqual(CLASSIFICATION_RUN_ID, prepare_args[0].classification_run_id)
        self.assertEqual("질문", prepare_args[1])
        self.assertIs(fixture.search, prepare_args[2])

    def _exact_result(self, outcome: CacheAttemptOutcome) -> ExactQuestionResult:
        recorded = RecordedJudgment(
            classification_id=1,
            cache_attempt_id=1,
            judgment=SimpleNamespace(),
            gate=GateResult(
                outcome=outcome,
                canonical_answer_id=(uuid.uuid4() if outcome is CacheAttemptOutcome.SERVED else None),
                rejection_reasons=("TEST_REJECTED",) if outcome is CacheAttemptOutcome.REJECTED else (),
            ),
            latency_ms=1,
            served_answer_markdown=(SERVED_ANSWER if outcome is CacheAttemptOutcome.SERVED else None),
            served_citations=(served_citation_logs() if outcome is CacheAttemptOutcome.SERVED else ()),
        )
        return ExactQuestionResult(
            recorded=recorded,
            prepared=SimpleNamespace(),
            judged=SimpleNamespace(),
        )

    async def test_every_turn_start_stores_exact_question_hash(self) -> None:
        for turn_no in (1, 2):
            with self.subTest(turn_no=turn_no):
                fixture = self._fixture(turn_no=turn_no)
                if turn_no == 2:
                    fixture.query_rewrite_service.rewrite.side_effect = self._new_topic_rewrite(
                        "  추천   질문 "
                    )

                await self._answer(fixture, "  추천   질문 ")

                kwargs = fixture.log_store.start_rag_run.await_args.kwargs
                self.assertEqual("  추천   질문 ", kwargs["user_query"])
                self.assertEqual(exact_question_hash("추천 질문"), kwargs["query_hash"])

    async def test_exact_question_served_skips_search_judge_and_generation(self) -> None:
        fixture = self._fixture()
        fixture.grouping.record_exact_question.return_value = self._exact_result(
            CacheAttemptOutcome.SERVED
        )

        response, _ = await self._answer(fixture, "추천 질문")

        self.assertEqual(CACHED_SERVED_ANSWER, response.answer.answer_markdown)
        fixture.grouping.record_exact_question.assert_awaited_once()
        call = fixture.grouping.record_exact_question.await_args
        self.assertEqual("추천 질문", call.args[1])
        self.assertEqual(fixture.rag_run_id, call.args[0].rag_run_id)
        self.assertTrue(call.kwargs["exact_cache_enabled"])
        fixture.retriever.search_with_trace.assert_not_awaited()
        fixture.grouping.prepare.assert_not_awaited()
        fixture.grouping.judge.assert_not_awaited()
        fixture.generation_service.generate_answer.assert_not_awaited()
        fixture.log_store.complete_rag_run.assert_awaited_once()
        self.assertEqual(
            CACHED_SERVED_ANSWER,
            fixture.log_store.complete_rag_run.await_args.kwargs["answer_content"],
        )

    async def test_exact_question_rejection_is_committed_then_falls_back_to_generation(self) -> None:
        fixture = self._fixture()
        fixture.grouping.record_exact_question.return_value = self._exact_result(
            CacheAttemptOutcome.REJECTED
        )

        response, _ = await self._answer(fixture, "추천 질문")

        self.assertIsInstance(response, ChatCompletedResponse)
        # 정확 일치 게이트 기록을 검색 전에 확정해 검색 뒤 rollback 에서 살아남게 한다.
        self.assertEqual(
            TURN_START + ["open_online_run", "commit"],
            fixture.events[: len(TURN_START) + 2],
        )
        fixture.retriever.search_with_trace.assert_awaited_once()
        fixture.grouping.prepare.assert_not_awaited()
        fixture.grouping.judge.assert_not_awaited()
        fixture.grouping.record_judgment_and_gate.assert_not_awaited()
        fixture.generation_service.generate_answer.assert_awaited_once()

    async def test_exact_question_serve_failure_records_rejection_then_generates(self) -> None:
        fixture = self._fixture()
        exact = self._exact_result(CacheAttemptOutcome.SERVED)
        fixture.grouping.record_exact_question.return_value = exact
        fixture.grouping.record_serve_failure.return_value = recorded_judgment(
            "REJECTED_SERVE_FAILED"
        )
        fixture.log_store.complete_rag_run.side_effect = [RuntimeError("write failed"), None]

        response, _ = await self._answer(fixture, "추천 질문")

        self.assertIsInstance(response, ChatCompletedResponse)
        fixture.grouping.record_serve_failure.assert_awaited_once_with(
            exact.prepared, exact.judged, exact.recorded
        )
        fixture.generation_service.generate_answer.assert_awaited_once()
        fixture.retriever.search_with_trace.assert_awaited_once()
        fixture.grouping.judge.assert_not_awaited()

    async def test_exact_question_gate_states_fall_back_without_second_judgment(self) -> None:
        for outcome in (
            CacheAttemptOutcome.SHADOW,
            CacheAttemptOutcome.REJECTED,
            CacheAttemptOutcome.GROUP_DISABLED,
        ):
            with self.subTest(outcome=outcome):
                fixture = self._fixture()
                fixture.grouping.record_exact_question.return_value = self._exact_result(
                    outcome
                )
                response, _ = await self._answer(fixture, "추천 질문")

                self.assertIsInstance(response, ChatCompletedResponse)
                fixture.generation_service.generate_answer.assert_awaited_once()
                fixture.grouping.prepare.assert_not_awaited()
                fixture.grouping.judge.assert_not_awaited()

    async def test_no_exact_question_match_runs_normal_judgment_without_extra_rows(self) -> None:
        # 과거 첫 턴 질문 로그 일치가 없으면 서비스가 None 을 돌려준다.
        fixture = self._fixture()
        fixture.recorded = recorded_judgment("REJECTED")

        response, _ = await self._answer(fixture, "처음 보는 질문")

        self.assertIsInstance(response, ChatCompletedResponse)
        fixture.grouping.record_exact_question.assert_awaited_once()
        self.assertEqual(
            TURN_START + SEARCH_WITH_GROUPING + JUDGE_CHECKPOINT,
            fixture.events[: len(TURN_START) + len(SEARCH_WITH_GROUPING) + len(JUDGE_CHECKPOINT)],
        )
        fixture.grouping.record_judgment_and_gate.assert_awaited_once()
        fixture.grouping.record_serve_failure.assert_not_awaited()
        fixture.generation_service.generate_answer.assert_awaited_once()

    def _new_topic_rewrite(self, resolved_query: str):
        call = QueryRewriteCall(
            trace=ModelCallTrace(
                provider="openai",
                model_name="gpt-5.4-mini",
                succeeded=True,
                latency_ms=1,
            ),
            resolution=QueryResolution(
                decision=QueryRewriteDecision.NEW_TOPIC,
                resolved_query=resolved_query,
                selected_turns=(),
            ),
        )

        async def rewrite(_question, _candidates, *, before_model_call):
            await before_model_call("openai", "gpt-5.4-mini", QUERY_REWRITE_PROMPT_VERSION)
            return call

        return rewrite

    def _counting_rewrite(self, resolved_query: str) -> Tuple[Callable[..., Any], List[str]]:
        rewrite = self._new_topic_rewrite(resolved_query)
        questions: List[str] = []

        async def counted(question, candidates, *, before_model_call):
            questions.append(question)
            return await rewrite(question, candidates, before_model_call=before_model_call)

        return counted, questions

    async def test_follow_up_exact_question_served_skips_rewrite_search_and_judge(self) -> None:
        fixture = self._fixture(turn_no=2)
        fixture.query_rewrite_service.rewrite.side_effect, rewrite_questions = self._counting_rewrite("추천 질문")
        fixture.grouping.record_exact_question.return_value = self._exact_result(
            CacheAttemptOutcome.SERVED
        )

        response, stages = await self._answer(fixture, "  추천 질문 ")

        self.assertEqual(CACHED_SERVED_ANSWER, response.answer.answer_markdown)
        self.assertEqual([ProgressStage.RETRIEVING], stages)
        call = fixture.grouping.record_exact_question.await_args
        self.assertEqual("  추천 질문 ", call.args[1])
        self.assertEqual([], rewrite_questions)
        fixture.query_rewrite_service.rewrite.assert_not_awaited()
        fixture.log_store.get_query_rewrite_candidates.assert_not_awaited()
        fixture.retriever.search_with_trace.assert_not_awaited()
        fixture.grouping.prepare.assert_not_awaited()
        fixture.grouping.judge.assert_not_awaited()
        fixture.generation_service.generate_answer.assert_not_awaited()
        fixture.log_store.start_model_call.assert_not_awaited()
        # Query Rewrite 를 건너뛴 턴은 원문을 NEW_TOPIC·빈 문맥으로 확정한 뒤 같은 트랜잭션에서 완료한다.
        fixture.log_store.record_query_resolution.assert_awaited_once_with(
            fixture.rag_run_id,
            resolved_query="  추천 질문 ",
            context_strategy=ContextStrategy.NEW_TOPIC,
            context_turn_count=0,
            context_snapshot=None,
        )
        self.assertEqual(
            TURN_START + ["open_online_run", "record_query_resolution", "complete_rag_run", "commit"],
            fixture.events,
        )

    async def test_follow_up_without_exact_match_runs_rewrite_and_judge_as_before(self) -> None:
        fixture = self._fixture(turn_no=2)
        fixture.query_rewrite_service.rewrite.side_effect, rewrite_questions = self._counting_rewrite("보충된 질문")
        fixture.recorded = recorded_judgment("REJECTED")

        response, _ = await self._answer(fixture, "그건 어떻게 해?")

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual(["그건 어떻게 해?"], rewrite_questions)
        self.assertEqual(
            ["open_online_run", "record_exact_question"],
            [name for name, _args, _kwargs in fixture.grouping.mock_calls][:2],
        )
        fixture.grouping.prepare.assert_awaited_once()
        self.assertEqual("보충된 질문", fixture.grouping.prepare.await_args.args[1])
        fixture.grouping.judge.assert_awaited_once()
        fixture.grouping.record_judgment_and_gate.assert_awaited_once()
        self.assertEqual("보충된 질문", fixture.retriever.search_with_trace.await_args.args[0])
        fixture.generation_service.generate_answer.assert_awaited_once()

    async def test_follow_up_exact_rejection_rewrites_and_generates_without_second_judgment(self) -> None:
        fixture = self._fixture(turn_no=2)
        fixture.query_rewrite_service.rewrite.side_effect, rewrite_questions = self._counting_rewrite("추천 질문")
        fixture.grouping.record_exact_question.return_value = self._exact_result(
            CacheAttemptOutcome.REJECTED
        )

        response, _ = await self._answer(fixture, "추천 질문")

        self.assertIsInstance(response, ChatCompletedResponse)
        # 게이트 기록을 Query Rewrite·검색 전에 확정한다.
        self.assertEqual(
            TURN_START + ["open_online_run", "commit"],
            fixture.events[: len(TURN_START) + 2],
        )
        self.assertEqual(["추천 질문"], rewrite_questions)
        fixture.log_store.record_query_resolution.assert_awaited_once()
        self.assertEqual(
            ContextStrategy.NEW_TOPIC,
            fixture.log_store.record_query_resolution.await_args.kwargs["context_strategy"],
        )
        fixture.retriever.search_with_trace.assert_awaited_once()
        fixture.grouping.prepare.assert_not_awaited()
        fixture.grouping.judge.assert_not_awaited()
        fixture.grouping.record_judgment_and_gate.assert_not_awaited()
        fixture.generation_service.generate_answer.assert_awaited_once()

    async def test_profile_semantic_cache_flag_reaches_gate(self) -> None:
        fixture = self._fixture(semantic_cache_enabled=False)

        await self._answer(fixture)

        fixture.grouping.record_judgment_and_gate.assert_awaited_once_with(
            fixture.prepared, fixture.judged, semantic_cache_enabled=False
        )

    # ------------------------------------------------------------------
    # 생성으로 진행
    # ------------------------------------------------------------------

    async def test_non_served_outcomes_record_attempt_before_generation_checkpoint(
        self,
    ) -> None:
        for outcome in ("SHADOW", "GROUP_DISABLED", "REJECTED", "FAILED"):
            with self.subTest(outcome=outcome):
                fixture = GroupingChatFixture()
                fixture.start()
                try:
                    fixture.recorded = recorded_judgment(outcome)

                    response, stages = await self._answer(fixture)
                finally:
                    fixture.stop()

                self.assertIsInstance(response, ChatCompletedResponse)
                self.assertEqual(
                    [ProgressStage.RETRIEVING, ProgressStage.GENERATING], stages
                )
                self.assertEqual(
                    TURN_START
                    + SEARCH_WITH_GROUPING
                    + JUDGE_CHECKPOINT
                    + [
                        "record_judgment_and_gate",
                        "commit",
                        f"start_model_call:{ANSWER_GENERATION}",
                        "commit",
                        "generate",
                        "finish_model_call:3",
                        f"finalize_attribution:{outcome}",
                        "complete_rag_run",
                        "commit",
                    ],
                    fixture.events,
                )
                fixture.grouping.finalize_attribution.assert_awaited_once_with(
                    fixture.recorded,
                    [
                        CitationLog(
                            chunk_id=1,
                            document_version_id=101,
                            citation_order=1,
                            document_title_snapshot="문서 1",
                            node_path_snapshot="문서 1 > 섹션 1",
                            source_uri_snapshot="https://docs.riido.io/1",
                        )
                    ],
                )

    async def test_preparation_failure_records_search_logs_in_failure_transaction(
        self,
    ) -> None:
        fixture = self._fixture()
        fixture.prepared = SimpleNamespace(ready=False, label="prepared-failed")
        fixture.recorded = recorded_judgment("FAILED")

        response, stages = await self._answer(fixture)

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual([ProgressStage.RETRIEVING, ProgressStage.GENERATING], stages)
        fixture.grouping.judge.assert_not_awaited()
        fixture.grouping.record_judgment_and_gate.assert_not_awaited()
        fixture.grouping.record_preparation_failure.assert_awaited_once_with(
            fixture.prepared
        )
        self.assertEqual(
            TURN_START
            + SEARCH_WITH_GROUPING
            + [
                "prepare",
                f"finish_model_call:{EMBEDDING_CALL_ID}",
                "record_retrieval_results",
                "record_preparation_failure",
                "commit",
                f"start_model_call:{ANSWER_GENERATION}",
                "commit",
                "generate",
                "finish_model_call:2",
                "finalize_attribution:FAILED",
                "complete_rag_run",
                "commit",
            ],
            fixture.events,
        )

    async def test_generation_withheld_or_error_does_not_finalize_attribution(
        self,
    ) -> None:
        withheld = FinalGenerationResult(
            status=FinalAnswerStatus.WITHHELD,
            answer_markdown=WITHHELD_RESPONSES[FinalWithheldReason.INSUFFICIENT_EVIDENCE],
            citations=(),
            withheld_reason=FinalWithheldReason.INSUFFICIENT_EVIDENCE,
            model_call=generation_trace(),
        )
        error = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="MODEL_OUTPUT_INVALID",
            model_call=generation_trace(),
        )
        for result, closing in ((withheld, "withhold_rag_run"), (error, "fail_rag_run")):
            with self.subTest(status=result.status):
                fixture = GroupingChatFixture()
                fixture.start()
                try:
                    fixture.recorded = recorded_judgment("REJECTED")
                    fixture.generation_result = result

                    await self._answer(fixture)
                finally:
                    fixture.stop()

                fixture.grouping.finalize_attribution.assert_not_awaited()
                self.assertEqual([closing, "commit"], fixture.events[-2:])
                self.assertEqual(1, fixture.events.count("record_retrieval_results"))

    # ------------------------------------------------------------------
    # 검색 로그 중복 방지
    # ------------------------------------------------------------------

    async def test_generation_exception_paths_do_not_repeat_search_logs(self) -> None:
        async def raise_before_checkpoint(_query, _results, *, before_model_call, **_):
            raise RuntimeError("before checkpoint")

        async def raise_after_checkpoint(_query, _results, *, before_model_call, **_):
            await before_model_call("openai", "gpt-5.4-mini", "v2")
            raise RuntimeError("after checkpoint")

        for name, effect in (
            ("before", raise_before_checkpoint),
            ("after", raise_after_checkpoint),
        ):
            with self.subTest(raised=name):
                fixture = GroupingChatFixture()
                fixture.start()
                try:
                    fixture.generation_service.generate_answer.side_effect = effect

                    response, _ = await self._answer(fixture)
                finally:
                    fixture.stop()

                self.assertIsInstance(response, ChatErrorResponse)
                self.assertEqual(1, fixture.events.count("record_retrieval_results"))
                self.assertEqual(
                    1, fixture.events.count(f"finish_model_call:{EMBEDDING_CALL_ID}")
                )
                fixture.log_store.record_retrieval_results.assert_awaited_once()
                fixture.grouping.finalize_attribution.assert_not_awaited()
                fixture.log_store.fail_rag_run.assert_awaited_once_with(
                    fixture.rag_run_id,
                    error_code="INTERNAL_ERROR",
                    total_latency_ms=ANY,
                )

    async def test_completed_generation_does_not_repeat_search_logs(self) -> None:
        fixture = self._fixture()

        await self._answer(fixture)

        fixture.log_store.record_retrieval_results.assert_awaited_once()
        finished_ids = [call.args[0] for call in fixture.log_store.finish_model_call.await_args_list]
        self.assertEqual([EMBEDDING_CALL_ID, 3], finished_ids)

    # ------------------------------------------------------------------
    # 판별을 부르지 않는 경로
    # ------------------------------------------------------------------

    async def test_follow_up_withheld_or_rewrite_error_does_not_judge(
        self,
    ) -> None:
        trace = ModelCallTrace(
            provider="openai",
            model_name="gpt-5.4-mini",
            succeeded=True,
            latency_ms=120,
            prompt_version=QUERY_REWRITE_PROMPT_VERSION,
        )
        rewrite_error = RuntimeError("invalid structured output")
        calls = {
            "withheld": QueryRewriteCall(
                trace=trace,
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_UNRESOLVED,
                    resolved_query=None,
                    selected_turns=(),
                ),
            ),
            "error": QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-5.4-mini",
                    succeeded=False,
                    latency_ms=120,
                    error_message=str(rewrite_error),
                ),
                error_code="MODEL_OUTPUT_INVALID",
                error=rewrite_error,
            ),
        }
        for name, call in calls.items():
            with self.subTest(rewrite=name):
                fixture = GroupingChatFixture(turn_no=2)
                fixture.start()
                try:
                    async def rewrite(_question, _candidates, *, before_model_call, _call=call):
                        await before_model_call(
                            "openai", "gpt-5.4-mini", QUERY_REWRITE_PROMPT_VERSION
                        )
                        return _call

                    fixture.query_rewrite_service.rewrite.side_effect = rewrite

                    response, _ = await self._answer(fixture)
                finally:
                    fixture.stop()

                if name == "withheld":
                    self.assertIsInstance(response, ChatWithheldResponse)
                else:
                    self.assertIsInstance(response, ChatErrorResponse)
                # 정확 일치 조회는 Query Rewrite 전에 하지만 일치가 없으면 판별 행을 쓰지 않는다.
                self.assertEqual(
                    ["open_online_run", "record_exact_question"],
                    [name for name, _args, _kwargs in fixture.grouping.mock_calls],
                )
                fixture.retriever.search_with_trace.assert_not_awaited()

    async def test_search_error_does_not_prepare_or_record_judgment(self) -> None:
        fixture = self._fixture()
        fixture.search = HybridSearchCall(
            embedding_call=ModelCallTrace(
                provider="openai",
                model_name="text-embedding-3-large",
                succeeded=False,
                latency_ms=15,
                error_message="invalid embedding response",
            ),
            error=RuntimeError("invalid embedding response"),
        )

        response, stages = await self._answer(fixture)

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual([ProgressStage.RETRIEVING], stages)
        # 분류 실행은 검색 임베딩 checkpoint 전에 열지만 판별 행·시도는 만들지 않는다(R13).
        self.assertEqual(
            ["open_online_run", "record_exact_question"],
            [name for name, _args, _kwargs in fixture.grouping.mock_calls],
        )
        self.assertEqual(
            TURN_START
            + SEARCH_WITH_GROUPING
            + [
                f"finish_model_call:{EMBEDDING_CALL_ID}",
                "fail_rag_run",
                "commit",
            ],
            fixture.events,
        )

    async def test_switch_off_or_missing_service_matches_turn_without_grouping(
        self,
    ) -> None:
        baseline = self._fixture(inject_grouping=False)
        baseline_response, baseline_stages = await self._answer(baseline)

        variants = {
            "switch_off": dict(grouping_enabled=False),
            "index_version_missing": dict(index_version_id=None),
        }
        for name, kwargs in variants.items():
            with self.subTest(variant=name):
                fixture = GroupingChatFixture(**kwargs)
                fixture.start()
                try:
                    fixture.conversation_id = baseline.conversation_id
                    fixture.rag_run_id = baseline.rag_run_id
                    fixture._wire_log_store(1)
                    response, stages = await self._answer(fixture)
                finally:
                    fixture.stop()

                self.assertEqual([], fixture.grouping.mock_calls)
                self.assertEqual(baseline.events, fixture.events)
                self.assertEqual(baseline_stages, stages)
                self.assertEqual(
                    baseline_response.model_dump(mode="json", by_alias=True),
                    response.model_dump(mode="json", by_alias=True),
                )
                self.assertEqual(
                    baseline.log_store.start_model_call.await_args_list,
                    fixture.log_store.start_model_call.await_args_list,
                )
                for call in fixture.log_store.start_model_call.await_args_list:
                    self.assertNotIn("classification_run_id", call.kwargs)

    async def test_legacy_service_without_profile_never_enables_grouping(self) -> None:
        fixture = self._fixture()
        fixture.service._profile_status = None
        fixture.service._retriever = fixture.retriever
        fixture.service._retriever_factory = None
        fixture.service._index_version_id = INDEX_VERSION_ID

        response, _ = await self._answer(fixture)

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual([], fixture.grouping.mock_calls)

    # ------------------------------------------------------------------
    # 서빙 실패와 fail-closed
    # ------------------------------------------------------------------

    async def test_serve_write_failure_rerecords_and_continues_to_generation(
        self,
    ) -> None:
        for failing in ("complete_rag_run", "commit"):
            with self.subTest(failing=failing):
                fixture = GroupingChatFixture()
                fixture.start()
                try:
                    fixture.recorded = recorded_judgment("SERVED", served=True)
                    self._fail_first_after(fixture, "record_judgment_and_gate", failing)

                    response, stages = await self._answer(fixture)
                finally:
                    fixture.stop()

                self.assertIsInstance(response, ChatCompletedResponse)
                self.assertEqual("첫 번째 근거입니다. [1]", response.answer.answer_markdown)
                self.assertEqual(
                    [ProgressStage.RETRIEVING, ProgressStage.GENERATING], stages
                )
                fixture.grouping.record_serve_failure.assert_awaited_once_with(
                    fixture.prepared, fixture.judged, fixture.recorded
                )
                served_index = fixture.events.index("record_judgment_and_gate")
                self.assertEqual(
                    [
                        "record_judgment_and_gate",
                        "complete_rag_run",
                        *(["commit"] if failing == "commit" else []),
                        "record_serve_failure",
                        "commit",
                        f"start_model_call:{ANSWER_GENERATION}",
                        "commit",
                        "generate",
                        "finish_model_call:3",
                        "finalize_attribution:REJECTED_SERVE_FAILED",
                        "complete_rag_run",
                        "commit",
                    ],
                    fixture.events[served_index:],
                )
                self.assertEqual(1, fixture.events.count("record_retrieval_results"))

    async def test_grouping_write_failures_close_turn_as_error(self) -> None:
        failures = {
            "open_online_run": "open_online_run",
            "record_judgment_and_gate": "record_judgment_and_gate",
            "record_preparation_failure": "record_preparation_failure",
            "judge_checkpoint": "record_retrieval_results",
            "record_serve_failure": "record_serve_failure",
        }
        for name, method in failures.items():
            with self.subTest(failure=name):
                fixture = GroupingChatFixture()
                fixture.start()
                try:
                    if name == "record_preparation_failure":
                        fixture.prepared = SimpleNamespace(ready=False)
                    if name == "record_serve_failure":
                        fixture.recorded = recorded_judgment("SERVED", served=True)
                        fixture.log_store.complete_rag_run.side_effect = fixture._record(
                            "complete_rag_run", error=RuntimeError("serve write")
                        )
                    target = (
                        fixture.log_store
                        if method == "record_retrieval_results"
                        else fixture.grouping
                    )
                    getattr(target, method).side_effect = fixture._record(
                        method, error=RuntimeError("db write failed")
                    )

                    response, _ = await self._answer(fixture)
                finally:
                    fixture.stop()

                self.assertIsInstance(response, ChatErrorResponse)
                self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
                self.assertEqual(fixture.rag_run_id, response.rag_run_id)
                fixture.generation_service.generate_answer.assert_not_awaited()
                self.assertEqual(
                    ["rollback", "fail_processing_model_calls", "fail_rag_run", "commit"],
                    fixture.events[-4:],
                )
                fixture.log_store.fail_rag_run.assert_awaited_once_with(
                    fixture.rag_run_id,
                    error_code="INTERNAL_ERROR",
                    total_latency_ms=ANY,
                )

    async def test_record_turn_failure_after_attribution_fails_quietly(self) -> None:
        for failing in ("finalize_attribution", "complete_rag_run"):
            with self.subTest(failing=failing):
                fixture = GroupingChatFixture()
                fixture.start()
                try:
                    target = (
                        fixture.grouping
                        if failing == "finalize_attribution"
                        else fixture.log_store
                    )
                    getattr(target, failing).side_effect = fixture._record(
                        failing, error=RuntimeError("write failed")
                    )

                    response, _ = await self._answer(fixture)
                finally:
                    fixture.stop()

                self.assertIsInstance(response, ChatErrorResponse)
                self.assertEqual(
                    [
                        failing,
                        "rollback",
                        "fail_processing_model_calls",
                        "fail_rag_run",
                        "commit",
                    ],
                    fixture.events[-5:],
                )
                if failing == "complete_rag_run":
                    fixture.grouping.finalize_attribution.assert_awaited_once()

    async def test_cancellation_during_judgment_cancels_turn(self) -> None:
        fixture = self._fixture()
        judging = asyncio.Event()

        async def hang(_prepared, *, before_checkpoint=None):
            await before_checkpoint()
            judging.set()
            await asyncio.Event().wait()

        fixture.grouping.judge.side_effect = hang
        task = asyncio.create_task(fixture.service.answer_question("질문"))
        await judging.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        fixture.grouping.record_judgment_and_gate.assert_not_awaited()
        fixture.generation_service.generate_answer.assert_not_awaited()
        fixture.log_store.fail_processing_model_calls.assert_awaited_once_with(
            fixture.rag_run_id,
            error_message=CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE,
        )
        fixture.log_store.cancel_rag_run.assert_awaited_once_with(fixture.rag_run_id)
        self.assertEqual(
            ["rollback", "fail_processing_model_calls", "cancel_rag_run", "commit"],
            fixture.events[-4:],
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _fail_first_after(fixture: GroupingChatFixture, after: str, failing: str) -> None:
        """after 이벤트 뒤 처음 오는 failing 호출 한 번만 실패시킨다."""

        target = fixture.session.commit if failing == "commit" else fixture.log_store.complete_rag_run
        state = {"failed": False}

        async def effect(*_args, **_kwargs):
            fixture.events.append(failing)
            if not state["failed"] and after in fixture.events:
                state["failed"] = True
                raise RuntimeError(f"{failing} failed")

        target.side_effect = effect


class StartTurnGroupingFlagsTest(unittest.IsolatedAsyncioTestCase):
    async def test_turn_start_carries_grouping_flags(self) -> None:
        cases = {
            "enabled": (dict(), True, True, True),
            "cache_off": (dict(semantic_cache_enabled=False), True, False, False),
            "exact_cache_off": (dict(exact_cache_enabled=False), True, True, False),
            "exact_cache_on_semantic_cache_off": (
                dict(semantic_cache_enabled=False, exact_cache_enabled=True),
                True,
                False,
                True,
            ),
            "switch_off": (dict(grouping_enabled=False), False, True, True),
            "not_injected": (dict(inject_grouping=False), False, True, True),
            "no_index_version": (dict(index_version_id=None), False, True, True),
            "switch_off_cache_off": (
                dict(grouping_enabled=False, semantic_cache_enabled=False),
                False,
                False,
                False,
            ),
        }
        for name, (kwargs, grouping_enabled, cache_enabled, exact_cache_enabled) in cases.items():
            with self.subTest(case=name):
                fixture = GroupingChatFixture(**kwargs)
                fixture.start()
                try:
                    turn = await fixture.service._start_turn("질문", None)
                finally:
                    fixture.stop()

                self.assertEqual(grouping_enabled, turn.grouping_enabled)
                self.assertEqual(cache_enabled, turn.semantic_cache_enabled)
                self.assertEqual(exact_cache_enabled, turn.exact_cache_enabled)
                self.assertEqual(TURN_START, fixture.events)


if __name__ == "__main__":
    unittest.main()
