"""질문 판별 DB 쓰기(QuestionGroupingStore)와 model_calls.classification_run_id 로컬 DB 통합 테스트.

- 로컬 PostgreSQL 과 head 마이그레이션이 필요하다. 연결할 수 없으면 skip 한다.
- 대부분은 외부 트랜잭션 하나에서 시드하고 마지막에 rollback 한다.
- 동시성 테스트만 독립 connection 두 개로 commit 한 데이터를 쓰고 끝에 지운다.

실행: DATABASE_URL=postgresql+asyncpg://riido:riido@localhost:5433/riido \
    .venv/bin/python -m unittest tests.test_question_grouping_store_db -v
"""

import asyncio
import unittest
import uuid
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.chat.log_store import CitationLog, RagLogStore
from app.core.config import get_settings
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    AnswerCitation,
    AnswerStatus,
    AttributionSource,
    CacheAttemptOutcome,
    ChunkingConfig,
    ClassificationDecision,
    ClassificationRun,
    ClassificationRunKind,
    DocumentGroup,
    DocumentSource,
    EmbeddingConfig,
    ExecutionStatus,
    IndexVersion,
    ModelCall,
    ModelCallPurpose,
    QuestionCacheAttempt,
    QuestionClassification,
    QuestionEmbedding,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    RecommendedQuestion,
    ExactQuestionMatch,
    ExactQuestionMatchSource,
    ExactQuestionMatchState,
    RagRun,
)
from app.question_grouping.attribution import document_attribution, initial_attribution, no_document_attribution
from app.question_grouping.catalog_reader import QuestionCatalogReader
from app.question_grouping.constants import (
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
    REJECT_SUBPROBLEM_UNUSED,
)
from app.question_grouping.decision import failed_turn_judgment
from app.question_grouping.gate import evaluate_cache_gate, gate_judgment_input, resolve_citations
from app.question_grouping.models import (
    GateResult,
    IndexScope,
    JudgeFailure,
    JudgeFailureKind,
    NormalizedJudgment,
    PresentedSubproblem,
    TurnJudgment,
)
from app.question_grouping.store import QuestionGroupingStore, served_citation_logs
from tests.test_question_grouping_readers_db import Section, _available, _Seed, _vector

TITLE = "구독 및 결제"
BILLING_SECTIONS: Sequence[Section] = (
    ("id-intro", "c-intro", "", "머리말"),
    ("id-cancel", "c-cancel", "구독 취소", "취소 본문"),
    ("id-refund", "c-refund", "환불", "환불 본문"),
)


def _gate_input(outcome: str = "FAILED") -> Dict[str, Any]:
    return {"schemaVersion": "v1", "gate": {"outcome": outcome}}


class _StoreDbTestCase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 DB에 연결할 수 없어 통합 테스트를 건너뜁니다.")

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(
            bind=self.connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        self.seed = _Seed(self.session)
        self.log_store = RagLogStore(self.session)
        self.store = QuestionGroupingStore(self.session, self.log_store)

        seed = self.seed
        self.chunking = await seed.chunking()
        self.embed = await seed.embedding()
        self.group = await seed.group()
        self.billing = await seed.source(self.group, "workspaces/plans-and-billing", TITLE)
        self.members = await seed.source(self.group, "workspaces/members", "멤버")
        self.billing_v1 = await seed.version(self.billing, 1)
        self.billing_chunks = await seed.sections(self.billing_v1, TITLE, self.chunking, BILLING_SECTIONS)
        self.members_v1 = await seed.version(self.members, 1)
        self.members_chunks = await seed.sections(
            self.members_v1, "멤버", self.chunking, [("id-invite", "c-invite", "초대", "초대 본문")]
        )
        self.scope = await seed.index(self.group, self.chunking, self.embed, [self.billing_v1, self.members_v1])

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def _turn(self, scope: Optional[IndexScope] = None) -> RagRun:
        conversation = await self.log_store.create_conversation()
        return await self.log_store.start_rag_run(
            conversation.id,
            user_query="구독을 취소하고 싶어요",
            index_version_id=(scope or self.scope).index_version_id,
        )

    async def _run_id(self, scope: Optional[IndexScope] = None) -> int:
        scope = scope or self.scope
        return await self.store.get_or_open_online_run(
            document_group_id=scope.document_group_id,
            index_version_id=scope.index_version_id,
            model=JUDGE_MODEL,
            prompt_version=JUDGE_PROMPT_VERSION,
            actor="test",
        )

    async def _billing_subproblem(self, key: str = "billing.cancel", **kwargs: Any):
        problem_group = await self.store.ensure_document_problem_group(self.billing.id)
        group_row = await self.session.get(QuestionProblemGroup, problem_group)
        return await self.seed.subproblem(
            group_row, key, embedding=self.embed, vector=_vector((0, 1.0)), **kwargs
        )

    @staticmethod
    def _presented(subproblem: Any, source: DocumentSource, canonical_id: Optional[uuid.UUID] = None) -> PresentedSubproblem:
        return PresentedSubproblem(
            key=subproblem.key,
            subproblem_id=subproblem.id,
            subproblem_version=subproblem.current_version,
            problem_group_id=subproblem.problem_group_id,
            document_source_id=source.id,
            document_key=source.document_key,
            canonical_answer_id=canonical_id,
            similarity=0.9,
            retrieval_rank=1,
            presented_order=1,
            inclusion_count=2,
            exclusion_count=1,
        )

    def _connect(self, presented: PresentedSubproblem, confidence: Optional[float] = 0.82) -> TurnJudgment:
        return TurnJudgment(
            decision=ClassificationDecision.CONNECT,
            attribution=initial_attribution(ClassificationDecision.CONNECT, subproblem=presented),
            subproblem=presented,
            normalized=NormalizedJudgment(
                decision=ClassificationDecision.CONNECT, subproblem=presented, confidence=confidence
            ),
        )

    @staticmethod
    def _separate(document_source_id: Optional[int] = None) -> TurnJudgment:
        attribution = (
            no_document_attribution() if document_source_id is None else document_attribution(document_source_id)
        )
        return TurnJudgment(
            decision=ClassificationDecision.SEPARATE,
            attribution=attribution,
            normalized=NormalizedJudgment(decision=ClassificationDecision.SEPARATE),
        )

    @staticmethod
    def _failed() -> TurnJudgment:
        return failed_turn_judgment(JudgeFailure(JudgeFailureKind.API_ERROR, "HTTP 500"))

    async def _classification(self, classification_id: int) -> Any:
        return (
            await self.session.execute(
                select(
                    QuestionClassification.rag_run_id,
                    QuestionClassification.subproblem_id,
                    QuestionClassification.problem_group_id,
                    QuestionClassification.run_id,
                    QuestionClassification.decision,
                    QuestionClassification.subproblem_version,
                    QuestionClassification.attribution_source,
                    QuestionClassification.is_composite,
                    QuestionClassification.confidence,
                    QuestionClassification.judgment_input,
                    QuestionClassification.effective_from,
                    QuestionClassification.effective_to,
                ).where(QuestionClassification.id == classification_id)
            )
        ).one()

    async def _problem_group(self, group_id: uuid.UUID) -> Any:
        return (
            await self.session.execute(
                select(
                    QuestionProblemGroup.kind,
                    QuestionProblemGroup.document_source_id,
                    QuestionProblemGroup.document_group_id,
                ).where(QuestionProblemGroup.id == group_id)
            )
        ).one()


# ---------------------------------------------------------------------------


class OnlineRunDbTest(_StoreDbTestCase):
    async def _open_runs(self, document_group_id: int) -> List[Any]:
        return list(
            (
                await self.session.execute(
                    select(ClassificationRun.id, ClassificationRun.index_version_id, ClassificationRun.prompt_version)
                    .where(
                        ClassificationRun.document_group_id == document_group_id,
                        ClassificationRun.kind == ClassificationRunKind.ONLINE,
                        ClassificationRun.finished_at.is_(None),
                    )
                    .order_by(ClassificationRun.id)
                )
            ).all()
        )

    async def test_reuses_open_run_of_same_config(self) -> None:
        first = await self._run_id()
        second = await self._run_id()

        self.assertEqual(first, second)
        row = (
            await self.session.execute(
                select(
                    ClassificationRun.kind,
                    ClassificationRun.model,
                    ClassificationRun.prompt_version,
                    ClassificationRun.row_count,
                    ClassificationRun.finished_at,
                    ClassificationRun.actor,
                ).where(ClassificationRun.id == first)
            )
        ).one()
        self.assertEqual(
            (ClassificationRunKind.ONLINE, JUDGE_MODEL, JUDGE_PROMPT_VERSION, 0, None, "test"),
            tuple(row),
        )

    async def test_unique_violation_rolls_back_savepoint_and_reselects(self) -> None:
        existing = await self._run_id()
        turn = await self._turn()
        racing_store = QuestionGroupingStore(self.session, self.log_store)
        original = racing_store._find_open_online_run
        calls = []

        async def miss_first(*args: Any) -> Optional[int]:
            calls.append(args)
            if len(calls) == 1:
                return None  # 다른 턴이 먼저 연 실행을 아직 보지 못한 상태
            return await original(*args)

        with patch.object(racing_store, "_find_open_online_run", side_effect=miss_first):
            run_id = await racing_store.get_or_open_online_run(
                document_group_id=self.scope.document_group_id,
                index_version_id=self.scope.index_version_id,
                model=JUDGE_MODEL,
                prompt_version=JUDGE_PROMPT_VERSION,
            )

        self.assertEqual(existing, run_id)
        self.assertEqual(2, len(calls))
        self.assertEqual([existing], [row.id for row in await self._open_runs(self.group.id)])
        # savepoint 만 되돌렸으므로 같은 트랜잭션에서 계속 쓸 수 있다.
        await self.store.insert_question_embedding(
            turn.id, embedding=_vector((0, 1.0)), embedding_config_id=self.embed.id
        )
        self.assertEqual(
            1, await self.session.scalar(select(func.count()).select_from(QuestionEmbedding).where(QuestionEmbedding.rag_run_id == turn.id))
        )

    async def test_new_config_opens_alongside_existing_open_runs(self) -> None:
        """다른 설정의 열린 실행을 닫지 않는다(색인 전환 중 열고 닫기 반복 방지)."""

        other_group = await self.seed.group()
        other_scope = await self.seed.index(other_group, self.chunking, self.embed, [])
        other_run = await self._run_id(other_scope)
        v1_run = await self._run_id()
        new_scope = await self.seed.index(self.group, self.chunking, self.embed, [self.billing_v1])

        v2_run = await self._run_id(new_scope)

        self.assertNotEqual(v1_run, v2_run)
        self.assertEqual(
            [v1_run, v2_run], [row.id for row in await self._open_runs(self.group.id)]
        )
        # 옛 색인 판 턴이 다시 와도 새로 열지 않고 열린 실행을 그대로 쓴다.
        self.assertEqual(v1_run, await self._run_id())
        self.assertEqual(v2_run, await self._run_id(new_scope))
        self.assertEqual([other_run], [row.id for row in await self._open_runs(other_group.id)])

        # 프롬프트 판만 달라도 새 실행을 열고, 기존 실행은 열린 채로 둔다.
        prompt_run = await self.store.get_or_open_online_run(
            document_group_id=self.group.id,
            index_version_id=new_scope.index_version_id,
            model=JUDGE_MODEL,
            prompt_version="question-grouping-v8",
        )
        open_runs = await self._open_runs(self.group.id)
        self.assertEqual(
            [
                (v1_run, JUDGE_PROMPT_VERSION),
                (v2_run, JUDGE_PROMPT_VERSION),
                (prompt_run, "question-grouping-v8"),
            ],
            [(row.id, row.prompt_version) for row in open_runs],
        )
        self.assertEqual(0, await self.session.scalar(select(ClassificationRun.row_count).where(ClassificationRun.id == v2_run)))


class ModelCallOwnerDbTest(_StoreDbTestCase):
    async def test_classification_run_owner_combinations(self) -> None:
        turn = await self._turn()
        run_id = await self._run_id()

        embedding_call = await self.log_store.start_model_call(
            rag_run_id=turn.id,
            classification_run_id=run_id,
            purpose=ModelCallPurpose.QUERY_EMBEDDING.value,
            provider="openai",
            model_name="text-embedding-test",
        )
        judge_call = await self.log_store.start_model_call(
            rag_run_id=turn.id,
            classification_run_id=run_id,
            purpose=ModelCallPurpose.QUESTION_CLASSIFICATION.value,
            provider="openai",
            model_name=JUDGE_MODEL,
            prompt_version=JUDGE_PROMPT_VERSION,
        )
        backfill_call = await self.log_store.start_model_call(
            classification_run_id=run_id,
            purpose=ModelCallPurpose.QUESTION_CLASSIFICATION.value,
            provider="openai",
            model_name=JUDGE_MODEL,
        )
        # 판별 응답이 무효여도 API 는 응답했으므로 SUCCESS 로 마감한다(결정 B).
        finished = await self.log_store.finish_model_call(
            judge_call.id,
            status=ExecutionStatus.SUCCESS,
            input_tokens=900,
            output_tokens=120,
            cached_input_tokens=512,
            reasoning_tokens=64,
            latency_ms=2100,
        )

        rows = (
            await self.session.execute(
                select(ModelCall.id, ModelCall.rag_run_id, ModelCall.classification_run_id, ModelCall.status)
                .where(ModelCall.id.in_([embedding_call.id, judge_call.id, backfill_call.id]))
                .order_by(ModelCall.id)
            )
        ).all()
        self.assertEqual(
            [
                (embedding_call.id, turn.id, run_id, ExecutionStatus.PROCESSING),
                (judge_call.id, turn.id, run_id, ExecutionStatus.SUCCESS),
                (backfill_call.id, None, run_id, ExecutionStatus.PROCESSING),
            ],
            [tuple(row) for row in rows],
        )
        self.assertEqual(64, finished.reasoning_tokens)

        rejected = (
            dict(purpose=ModelCallPurpose.QUESTION_CLASSIFICATION.value, rag_run_id=turn.id),
            dict(purpose=ModelCallPurpose.ANSWER_GENERATION.value, rag_run_id=turn.id, classification_run_id=run_id),
            dict(purpose=ModelCallPurpose.QUERY_EMBEDDING.value, classification_run_id=run_id),
        )
        for kwargs in rejected:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(IntegrityError) as raised:
                    async with self.session.begin_nested():
                        await self.log_store.start_model_call(provider="openai", model_name="m", **kwargs)
                self.assertIn("owner_combination", str(raised.exception))


class ProblemGroupDbTest(_StoreDbTestCase):
    async def test_ensure_is_idempotent_and_returns_existing_rows(self) -> None:
        first = await self.store.ensure_document_problem_group(self.billing.id)
        second = await self.store.ensure_document_problem_group(self.billing.id)
        no_doc = await self.store.ensure_no_document_problem_group(self.group.id)
        no_doc_again = await self.store.ensure_no_document_problem_group(self.group.id)
        seeded = await self.seed.document_group(self.group, self.members)

        self.assertEqual(first, second)
        self.assertEqual(no_doc, no_doc_again)
        self.assertEqual(seeded.id, await self.store.ensure_document_problem_group(self.members.id))
        self.assertEqual(
            (QuestionProblemGroupKind.DOCUMENT, self.billing.id, None), tuple(await self._problem_group(first))
        )
        self.assertEqual(
            (QuestionProblemGroupKind.NO_DOCUMENT, None, self.group.id), tuple(await self._problem_group(no_doc))
        )
        counts = await self.session.scalar(
            select(func.count()).select_from(QuestionProblemGroup).where(
                (QuestionProblemGroup.document_source_id.in_([self.billing.id, self.members.id]))
                | (QuestionProblemGroup.document_group_id == self.group.id)
            )
        )
        self.assertEqual(3, counts)


class RecommendedQuestionLookupDbTest(_StoreDbTestCase):
    async def test_lookup_is_scoped_to_group_and_rejects_mismatched_owner(self) -> None:
        current_problem_group = await self.seed.document_group(self.group, self.billing)
        current_subproblem = await self.seed.subproblem(
            current_problem_group, "current.exact"
        )

        other_group = await self.seed.group()
        other_source = await self.seed.source(other_group, "other-doc", "다른 문서")
        other_problem_group = await self.seed.document_group(other_group, other_source)
        other_subproblem = await self.seed.subproblem(other_problem_group, "other.exact")

        self.session.add_all(
            [
                RecommendedQuestion(
                    document_group_id=self.group.id,
                    subproblem_id=current_subproblem.id,
                    question="추천 질문",
                    normalized_question="추천 질문",
                ),
                RecommendedQuestion(
                    document_group_id=other_group.id,
                    subproblem_id=other_subproblem.id,
                    question="추천 질문",
                    normalized_question="추천 질문",
                ),
                # The foreign keys alone permit this malformed cross-group row.
                RecommendedQuestion(
                    document_group_id=self.group.id,
                    subproblem_id=other_subproblem.id,
                    question="잘못된 매핑",
                    normalized_question="잘못된 매핑",
                ),
            ]
        )
        await self.session.flush()

        current = await self.store.find_recommended_question(self.group.id, "추천 질문")
        repeated_whitespace = await self.store.find_recommended_question(
            self.group.id, "  추천   질문  "
        )
        other = await self.store.find_recommended_question(other_group.id, "추천 질문")
        malformed = await self.store.find_recommended_question(self.group.id, "잘못된 매핑")
        inserted_space = await self.store.find_recommended_question(self.group.id, "추 천 질문")
        removed_space = await self.store.find_recommended_question(self.group.id, "추천질문")
        changed_ending = await self.store.find_recommended_question(self.group.id, "추천 질문입니다")

        self.assertIsNotNone(current)
        self.assertEqual(current_subproblem.id, current.subproblem_id)
        self.assertIsNotNone(repeated_whitespace)
        self.assertEqual(current_subproblem.id, repeated_whitespace.subproblem_id)
        self.assertIsNotNone(other)
        self.assertEqual(other_subproblem.id, other.subproblem_id)
        self.assertIsNone(malformed)
        self.assertIsNone(inserted_space)
        self.assertIsNone(removed_space)
        self.assertIsNone(changed_ending)

    async def test_historical_match_materializes_and_conflicts_are_disabled(self) -> None:
        subproblem = await self._billing_subproblem(current_version=1)
        canonical = await self.seed.canonical(
            subproblem, [(self.billing_chunks[1], self.billing_v1.id)]
        )
        turn = await self._turn()
        inserted = await self.store.materialize_historical_exact(
            document_group_id=self.group.id,
            question="과거 질문",
            subproblem_id=subproblem.id,
            subproblem_version=1,
            canonical_answer_id=canonical.id,
            source_rag_run_id=turn.id,
        )
        self.assertTrue(inserted)
        match = await self.store.find_exact_question_match(self.group.id, "과거 질문")
        self.assertIsNotNone(match)
        self.assertEqual(ExactQuestionMatchSource.HISTORICAL_SERVED, match.source)
        self.assertEqual(ExactQuestionMatchState.ACTIVE, match.state)
        self.assertEqual(canonical.id, match.canonical_answer_id)

        other = await self._billing_subproblem(key="billing.refund", current_version=1)
        other_canonical = await self.seed.canonical(
            other, [(self.billing_chunks[2], self.billing_v1.id)]
        )
        other_turn = await self._turn()
        inserted = await self.store.materialize_historical_exact(
            document_group_id=self.group.id,
            question="과거 질문",
            subproblem_id=other.id,
            subproblem_version=1,
            canonical_answer_id=other_canonical.id,
            source_rag_run_id=other_turn.id,
        )
        self.assertTrue(inserted)
        conflict = await self.store.find_exact_question_match(self.group.id, "과거 질문")
        # The conflict transition is flushed as a real row mutation so the caller
        # can commit it; a later retry is a no-op and must not clear the conflict.
        self.assertEqual(ExactQuestionMatchState.CONFLICT, conflict.state)
        self.assertFalse(
            await self.store.materialize_historical_exact(
                document_group_id=self.group.id,
                question="과거 질문",
                subproblem_id=other.id,
                subproblem_version=1,
                canonical_answer_id=other_canonical.id,
                source_rag_run_id=other_turn.id,
            )
        )

    async def test_recommended_match_is_never_overwritten_by_history(self) -> None:
        subproblem = await self._billing_subproblem(current_version=1)
        other = await self._billing_subproblem(key="billing.refund", current_version=1)
        turn = await self._turn()
        self.session.add(
            RecommendedQuestion(
                document_group_id=self.group.id,
                subproblem_id=subproblem.id,
                question="고정 질문",
                normalized_question="고정 질문",
            )
        )
        await self.session.flush()
        canonical = await self.seed.canonical(
            other, [(self.billing_chunks[2], self.billing_v1.id)]
        )
        self.assertFalse(
            await self.store.materialize_historical_exact(
                document_group_id=self.group.id,
                question="고정 질문",
                subproblem_id=other.id,
                subproblem_version=1,
                canonical_answer_id=canonical.id,
                source_rag_run_id=turn.id,
            )
        )
        match = await self.store.find_exact_question_match(self.group.id, "고정 질문")
        self.assertEqual(subproblem.id, match.subproblem_id)
        self.assertEqual(ExactQuestionMatchSource.RECOMMENDED, match.source)


class ClassificationDbTest(_StoreDbTestCase):
    async def test_connect_row_uses_subproblem_group_and_version(self) -> None:
        subproblem = await self._billing_subproblem(current_version=1)
        turn = await self._turn()
        run_id = await self._run_id()
        judgment = self._connect(self._presented(subproblem, self.billing))
        judgment_input = {"resolvedQuery": "구독 취소", "gate": {"outcome": "SERVED"}}

        classification_id = await self.store.insert_classification(
            turn.id, run_id=run_id, judgment=judgment, judgment_input=judgment_input
        )

        row = await self._classification(classification_id)
        self.assertEqual(turn.id, row.rag_run_id)
        self.assertEqual(subproblem.id, row.subproblem_id)
        self.assertEqual(subproblem.problem_group_id, row.problem_group_id)
        self.assertEqual(run_id, row.run_id)
        self.assertEqual(ClassificationDecision.CONNECT, row.decision)
        self.assertEqual(1, row.subproblem_version)
        self.assertEqual(AttributionSource.SUBPROBLEM, row.attribution_source)
        self.assertFalse(row.is_composite)
        self.assertAlmostEqual(0.82, float(row.confidence))
        self.assertEqual(judgment_input, row.judgment_input)
        self.assertIsNotNone(row.effective_from)
        self.assertIsNone(row.effective_to)

    async def test_document_none_and_failure_rows_ensure_problem_groups(self) -> None:
        run_id = await self._run_id()
        cases = (
            (self._separate(self.members.id), ClassificationDecision.SEPARATE, AttributionSource.DOCUMENT,
             (QuestionProblemGroupKind.DOCUMENT, self.members.id, None)),
            (self._separate(), ClassificationDecision.SEPARATE, AttributionSource.NONE,
             (QuestionProblemGroupKind.NO_DOCUMENT, None, self.group.id)),
            (self._failed(), ClassificationDecision.UNCLASSIFIED, AttributionSource.NONE,
             (QuestionProblemGroupKind.NO_DOCUMENT, None, self.group.id)),
        )
        for judgment, decision, source, group in cases:
            with self.subTest(decision=decision, source=source):
                turn = await self._turn()
                classification_id = await self.store.insert_classification(
                    turn.id, run_id=run_id, judgment=judgment, judgment_input=_gate_input()
                )
                row = await self._classification(classification_id)
                self.assertEqual((decision, source), (row.decision, row.attribution_source))
                self.assertIsNone(row.subproblem_id)
                self.assertIsNone(row.subproblem_version)
                self.assertIsNone(row.confidence)
                self.assertEqual(group, tuple(await self._problem_group(row.problem_group_id)))

    async def test_current_row_is_unique_per_turn(self) -> None:
        turn = await self._turn()
        run_id = await self._run_id()
        await self.store.insert_classification(
            turn.id, run_id=run_id, judgment=self._separate(), judgment_input=_gate_input()
        )

        with self.assertRaises(IntegrityError) as raised:
            async with self.session.begin_nested():
                await self.store.insert_classification(
                    turn.id, run_id=run_id, judgment=self._failed(), judgment_input=_gate_input()
                )
        self.assertIn("uq_question_classifications_rag_run_id_current", str(raised.exception))

    async def test_rejects_missing_gate_and_run_of_other_index(self) -> None:
        turn = await self._turn()
        other_scope = await self.seed.index(self.group, self.chunking, self.embed, [self.billing_v1])
        other_run = await self._run_id(other_scope)

        with self.assertRaisesRegex(ValueError, "gate"):
            await self.store.insert_classification(
                turn.id, run_id=other_run, judgment=self._separate(), judgment_input={"resolvedQuery": "x"}
            )
        with self.assertRaisesRegex(ValueError, "색인 판"):
            await self.store.insert_classification(
                turn.id, run_id=other_run, judgment=self._separate(), judgment_input=_gate_input()
            )


class QuestionEmbeddingDbTest(_StoreDbTestCase):
    async def test_inserts_one_vector_per_turn(self) -> None:
        turn = await self._turn()
        await self.store.insert_question_embedding(
            turn.id, embedding=_vector((3, 0.5)), embedding_config_id=self.embed.id
        )
        row = await self.session.scalar(select(QuestionEmbedding).where(QuestionEmbedding.rag_run_id == turn.id))
        self.assertEqual(self.embed.id, row.embedding_config_id)
        self.assertEqual(EMBEDDING_DIMENSIONS, len(row.embedding))
        self.assertEqual(0.5, float(row.embedding[3]))
        with self.assertRaises(ValueError):
            await self.store.insert_question_embedding(turn.id, embedding=[0.1], embedding_config_id=self.embed.id)


class CacheAttemptDbTest(_StoreDbTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.subproblem = await self._billing_subproblem()
        self.canonical = await self.seed.canonical(self.subproblem, [(self.billing_chunks[1], self.billing_v1.id)])
        self.presented = self._presented(self.subproblem, self.billing, self.canonical.id)
        self.run_id = await self._run_id()

    async def _classified_turn(self, judgment: TurnJudgment):
        turn = await self._turn()
        classification_id = await self.store.insert_classification(
            turn.id, run_id=self.run_id, judgment=judgment, judgment_input=_gate_input()
        )
        return turn, classification_id

    async def test_inserts_every_outcome(self) -> None:
        canonical_id = self.canonical.id
        cases = (
            (self._connect(self.presented), GateResult(CacheAttemptOutcome.SERVED, canonical_answer_id=canonical_id)),
            (self._connect(self.presented), GateResult(CacheAttemptOutcome.SHADOW, canonical_answer_id=canonical_id)),
            (self._connect(self.presented), GateResult(CacheAttemptOutcome.GROUP_DISABLED, canonical_answer_id=canonical_id)),
            (self._connect(self.presented), GateResult(CacheAttemptOutcome.REJECTED, rejection_reasons=(REJECT_SUBPROBLEM_UNUSED,))),
            (self._separate(), GateResult(CacheAttemptOutcome.REJECTED, rejection_reasons=(REJECT_CLASSIFICATION_NOT_CONNECTED,))),
            (self._failed(), GateResult(CacheAttemptOutcome.FAILED)),
        )
        for judgment, gate in cases:
            with self.subTest(outcome=gate.outcome, decision=judgment.decision):
                turn, classification_id = await self._classified_turn(judgment)
                attempt_id = await self.store.insert_cache_attempt(
                    turn.id, classification_id=classification_id, gate=gate, latency_ms=12
                )
                row = (
                    await self.session.execute(
                        select(
                            QuestionCacheAttempt.rag_run_id,
                            QuestionCacheAttempt.classification_id,
                            QuestionCacheAttempt.outcome,
                            QuestionCacheAttempt.canonical_answer_id,
                            QuestionCacheAttempt.rejection_reasons,
                            QuestionCacheAttempt.latency_ms,
                        ).where(QuestionCacheAttempt.id == attempt_id)
                    )
                ).one()
                self.assertEqual(
                    (
                        turn.id,
                        classification_id,
                        gate.outcome,
                        gate.canonical_answer_id,
                        list(gate.rejection_reasons) or None,
                        12,
                    ),
                    tuple(row),
                )

    async def test_rejects_check_violations_before_and_at_database(self) -> None:
        turn, connect_id = await self._classified_turn(self._connect(self.presented))
        separate_turn, separate_id = await self._classified_turn(self._separate())

        invalid = (
            (turn.id, connect_id, GateResult(CacheAttemptOutcome.SERVED)),
            (turn.id, connect_id, GateResult(CacheAttemptOutcome.REJECTED)),
            (turn.id, connect_id, GateResult(CacheAttemptOutcome.FAILED, canonical_answer_id=self.canonical.id)),
            (turn.id, connect_id, GateResult(CacheAttemptOutcome.FAILED, rejection_reasons=("X",))),
            (turn.id, connect_id, GateResult(CacheAttemptOutcome.SKIPPED)),
            # 정본 id 를 담는 결과는 CONNECT 행에만 붙는다.
            (separate_turn.id, separate_id, GateResult(CacheAttemptOutcome.SHADOW, canonical_answer_id=self.canonical.id)),
            # 다른 턴의 판별 행
            (turn.id, separate_id, GateResult(CacheAttemptOutcome.FAILED)),
        )
        for rag_run_id, classification_id, gate in invalid:
            with self.subTest(gate=gate, classification_id=classification_id), self.assertRaises(ValueError):
                await self.store.insert_cache_attempt(rag_run_id, classification_id=classification_id, gate=gate)

        # 저장 계층을 거치지 않은 행은 DB CHECK 가 막는다.
        db_invalid = (
            dict(outcome=CacheAttemptOutcome.SERVED, canonical_answer_id=None, rejection_reasons=None),
            dict(outcome=CacheAttemptOutcome.REJECTED, canonical_answer_id=None, rejection_reasons=[]),
            dict(outcome=CacheAttemptOutcome.FAILED, canonical_answer_id=self.canonical.id, rejection_reasons=None),
        )
        for values in db_invalid:
            with self.subTest(values=values), self.assertRaises(IntegrityError):
                async with self.session.begin_nested():
                    self.session.add(QuestionCacheAttempt(rag_run_id=turn.id, classification_id=connect_id, **values))
                    await self.session.flush()

        await self.store.insert_cache_attempt(
            turn.id, classification_id=connect_id, gate=GateResult(CacheAttemptOutcome.FAILED)
        )
        self.assertEqual(
            1,
            await self.session.scalar(
                select(func.count()).select_from(QuestionCacheAttempt).where(QuestionCacheAttempt.rag_run_id == turn.id)
            ),
        )


class FinalizeCitationAttributionDbTest(_StoreDbTestCase):
    def _citations(self, *documents: str) -> List[CitationLog]:
        chunk_by_document = {
            "billing": (self.billing_chunks[1], self.billing_v1.id),
            "members": (self.members_chunks[0], self.members_v1.id),
        }
        return [
            CitationLog(
                chunk_id=chunk_by_document[name][0],
                document_version_id=chunk_by_document[name][1],
                citation_order=order,
            )
            for order, name in enumerate(documents, 1)
        ]

    async def _row(self, judgment: TurnJudgment, judgment_input: Optional[Dict[str, Any]] = None):
        turn = await self._turn()
        classification_id = await self.store.insert_classification(
            turn.id,
            run_id=await self._run_id(),
            judgment=judgment,
            judgment_input=judgment_input or _gate_input(),
        )
        return turn, classification_id

    async def test_most_cited_document_wins_and_judgment_input_is_unchanged(self) -> None:
        judgment_input = {"gate": {"outcome": "REJECTED"}, "output": {"decision": "SEPARATE"}}
        turn, classification_id = await self._row(self._separate(self.members.id), judgment_input)
        before = await self._classification(classification_id)
        citations = self._citations("members", "billing", "billing")

        updated = await self.store.finalize_citation_attribution(classification_id, citations)
        await self.log_store.complete_rag_run(turn.id, answer_content="답 [1][2][3]", citations=citations)

        self.assertTrue(updated)
        after = await self._classification(classification_id)
        self.assertEqual(AttributionSource.CITATION, after.attribution_source)
        self.assertEqual(
            (QuestionProblemGroupKind.DOCUMENT, self.billing.id, None),
            tuple(await self._problem_group(after.problem_group_id)),
        )
        self.assertEqual(judgment_input, after.judgment_input)
        for field in ("decision", "subproblem_id", "run_id", "confidence", "effective_from", "effective_to", "judgment_input"):
            self.assertEqual(getattr(before, field), getattr(after, field), field)

    async def test_tie_goes_to_document_cited_first(self) -> None:
        cases = (
            (("members", "billing"), self.members.id),
            (("billing", "members"), self.billing.id),
            (("billing", "members", "members"), self.members.id),
        )
        for documents, expected in cases:
            with self.subTest(documents=documents):
                _, classification_id = await self._row(self._failed())
                self.assertTrue(
                    await self.store.finalize_citation_attribution(classification_id, self._citations(*documents))
                )
                row = await self._classification(classification_id)
                self.assertEqual(ClassificationDecision.UNCLASSIFIED, row.decision)
                self.assertEqual(AttributionSource.CITATION, row.attribution_source)
                self.assertEqual(expected, (await self._problem_group(row.problem_group_id)).document_source_id)

    async def test_connect_row_and_empty_citations_are_not_updated(self) -> None:
        subproblem = await self._billing_subproblem()
        _, connect_id = await self._row(self._connect(self._presented(subproblem, self.billing)))
        _, separate_id = await self._row(self._separate())
        groups_before = await self.session.scalar(select(func.count()).select_from(QuestionProblemGroup))

        self.assertFalse(await self.store.finalize_citation_attribution(connect_id, self._citations("members")))
        self.assertFalse(await self.store.finalize_citation_attribution(separate_id, []))

        connect = await self._classification(connect_id)
        self.assertEqual((AttributionSource.SUBPROBLEM, subproblem.problem_group_id), (connect.attribution_source, connect.problem_group_id))
        self.assertEqual(AttributionSource.NONE, (await self._classification(separate_id)).attribution_source)
        # 갱신하지 않으면 문서 문제 그룹도 만들지 않는다.
        self.assertEqual(groups_before, await self.session.scalar(select(func.count()).select_from(QuestionProblemGroup)))


class ServedCitationDbTest(_StoreDbTestCase):
    async def test_served_citations_complete_turn_with_current_chunks(self) -> None:
        subproblem = await self._billing_subproblem()
        canonical = await self.seed.canonical(subproblem, [(self.billing_chunks[1], self.billing_v1.id)])
        # 새 판에서 인용 절은 그대로이고 위치만 바뀌었다(R17 2단계).
        v2 = await self.seed.version(self.billing, 2)
        v2_chunks = await self.seed.sections(
            v2, TITLE, self.chunking, [("id-new", "c-new", "새 절", "새 본문"), *BILLING_SECTIONS]
        )
        scope = await self.seed.index(self.group, self.chunking, self.embed, [v2, self.members_v1])
        turn = await self._turn(scope)
        run_id = await self._run_id(scope)
        presented = self._presented(subproblem, self.billing, canonical.id)
        judgment = self._connect(presented)

        inputs = await QuestionCatalogReader(self.session).load_gate_inputs(subproblem.id, scope)
        resolutions = resolve_citations(inputs.citations, inputs.contexts_by_source_id)
        gate = evaluate_cache_gate(
            judgment,
            subproblem=inputs.subproblem,
            canonical_answer=inputs.canonical_answer,
            citation_resolutions=resolutions,
            semantic_cache_enabled=True,
        )
        self.assertEqual(CacheAttemptOutcome.SERVED, gate.outcome)
        judgment_input = {
            "gate": gate_judgment_input(gate, citation_resolutions=resolutions, canonical_answer_id=canonical.id)
        }

        classification_id = await self.store.insert_classification(
            turn.id, run_id=run_id, judgment=judgment, judgment_input=judgment_input
        )
        await self.store.insert_cache_attempt(turn.id, classification_id=classification_id, gate=gate, latency_ms=40)
        logs = served_citation_logs(gate)
        completed = await self.log_store.complete_rag_run(
            turn.id, answer_content=inputs.canonical_answer.content_markdown, citations=logs
        )

        self.assertEqual(AnswerStatus.COMPLETED, completed.status)
        citations = (
            await self.session.execute(
                select(
                    AnswerCitation.chunk_id,
                    AnswerCitation.document_version_id,
                    AnswerCitation.citation_order,
                    AnswerCitation.document_title_snapshot,
                    AnswerCitation.node_path_snapshot,
                    AnswerCitation.source_uri_snapshot,
                ).where(AnswerCitation.rag_run_id == turn.id)
            )
        ).all()
        self.assertEqual(
            [(v2_chunks[2], v2.id, 1, TITLE, f"{TITLE} > 구독 취소", self.billing.canonical_uri)],
            [tuple(row) for row in citations],
        )
        stored = await self._classification(classification_id)
        self.assertEqual(v2_chunks[2], stored.judgment_input["gate"]["citationResolutions"][0]["chunkId"])
        self.assertEqual(2, stored.judgment_input["gate"]["citationResolutions"][0]["step"])
        # 턴이 끝나면 더 쓰지 않는다.
        with self.assertRaisesRegex(ValueError, "PROCESSING"):
            await self.store.finalize_citation_attribution(classification_id, logs)


class NonProcessingTurnDbTest(_StoreDbTestCase):
    async def test_refuses_writes_for_finished_turn(self) -> None:
        run_id = await self._run_id()
        turn = await self._turn()
        classification_id = await self.store.insert_classification(
            turn.id, run_id=run_id, judgment=self._failed(), judgment_input=_gate_input()
        )
        await self.log_store.withhold_rag_run(turn.id, reason_code="OUT_OF_SCOPE")
        members_citation = [CitationLog(chunk_id=self.members_chunks[0], document_version_id=self.members_v1.id, citation_order=1)]

        writes = (
            self.store.insert_question_embedding(turn.id, embedding=_vector((0, 1.0)), embedding_config_id=self.embed.id),
            self.store.insert_classification(turn.id, run_id=run_id, judgment=self._separate(), judgment_input=_gate_input()),
            self.store.insert_cache_attempt(turn.id, classification_id=classification_id, gate=GateResult(CacheAttemptOutcome.FAILED)),
            self.store.finalize_citation_attribution(classification_id, members_citation),
        )
        for write in writes:
            with self.assertRaisesRegex(ValueError, "PROCESSING"):
                await write

        for model in (QuestionEmbedding, QuestionCacheAttempt):
            self.assertEqual(
                0, await self.session.scalar(select(func.count()).select_from(model).where(model.rag_run_id == turn.id))
            )
        row = await self._classification(classification_id)
        self.assertEqual(AttributionSource.NONE, row.attribution_source)


# ---------------------------------------------------------------------------


class StoreConcurrencyDbTest(unittest.IsolatedAsyncioTestCase):
    """독립 connection 두 개로 문제 그룹 ensure 와 ONLINE 실행 열기 경합을 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 DB에 연결할 수 없어 동시성 테스트를 건너뜁니다.")

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        async with AsyncSession(bind=self.engine, expire_on_commit=False) as session:
            seed = _Seed(session)
            chunking = await seed.chunking()
            embed = await seed.embedding()
            group = await seed.group()
            source = await seed.source(group, "workspaces/plans-and-billing", TITLE)
            scope = await seed.index(group, chunking, embed, [])
            self.ids = {
                "chunking": chunking.id,
                "embedding": embed.id,
                "group": group.id,
                "source": source.id,
                "index": scope.index_version_id,
            }
            await session.commit()

    async def asyncTearDown(self) -> None:
        ids = self.ids
        async with AsyncSession(bind=self.engine) as session:
            await session.execute(delete(ClassificationRun).where(ClassificationRun.document_group_id == ids["group"]))
            await session.execute(
                delete(QuestionProblemGroup).where(
                    (QuestionProblemGroup.document_source_id == ids["source"])
                    | (QuestionProblemGroup.document_group_id == ids["group"])
                )
            )
            await session.execute(delete(DocumentSource).where(DocumentSource.id == ids["source"]))
            await session.execute(delete(IndexVersion).where(IndexVersion.id == ids["index"]))
            await session.execute(delete(ChunkingConfig).where(ChunkingConfig.id == ids["chunking"]))
            await session.execute(delete(EmbeddingConfig).where(EmbeddingConfig.id == ids["embedding"]))
            await session.execute(delete(DocumentGroup).where(DocumentGroup.id == ids["group"]))
            await session.commit()
        await self.engine.dispose()

    async def _race(self, first_call, second_call):
        first_session = AsyncSession(bind=self.engine, expire_on_commit=False)
        second_session = AsyncSession(bind=self.engine, expire_on_commit=False)
        second_task = None
        try:
            first = await first_call(QuestionGroupingStore(first_session))
            second_pid = await second_session.scalar(text("SELECT pg_backend_pid()"))
            second_task = asyncio.create_task(second_call(QuestionGroupingStore(second_session)))
            await self._wait_for_lock(second_pid)
            self.assertFalse(second_task.done())
            await first_session.commit()
            second = await asyncio.wait_for(second_task, timeout=3)
            await second_session.commit()
            return first, second
        finally:
            if second_task is not None and not second_task.done():
                second_task.cancel()
                await asyncio.gather(second_task, return_exceptions=True)
            await first_session.rollback()
            await second_session.rollback()
            await first_session.close()
            await second_session.close()

    async def _wait_for_lock(self, backend_pid: int) -> None:
        deadline = asyncio.get_running_loop().time() + 3
        while True:
            async with AsyncSession(bind=self.engine) as monitor:
                wait_event_type = await monitor.scalar(
                    text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"),
                    {"pid": backend_pid},
                )
            if wait_event_type == "Lock":
                return
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("두 번째 세션이 유니크 인덱스 잠금 대기에 들어가지 않았습니다.")
            await asyncio.sleep(0.01)

    async def test_concurrent_problem_group_ensure_creates_one_row(self) -> None:
        source_id, group_id = self.ids["source"], self.ids["group"]

        async def ensure_both(store: QuestionGroupingStore):
            return (
                await store.ensure_document_problem_group(source_id),
                await store.ensure_no_document_problem_group(group_id),
            )

        first, second = await self._race(ensure_both, ensure_both)

        self.assertEqual(first, second)
        async with AsyncSession(bind=self.engine) as session:
            count = await session.scalar(
                select(func.count()).select_from(QuestionProblemGroup).where(
                    (QuestionProblemGroup.document_source_id == source_id)
                    | (QuestionProblemGroup.document_group_id == group_id)
                )
            )
        self.assertEqual(2, count)

    async def test_concurrent_online_run_open_reuses_winner(self) -> None:
        async def open_run(store: QuestionGroupingStore) -> int:
            return await store.get_or_open_online_run(
                document_group_id=self.ids["group"],
                index_version_id=self.ids["index"],
                model=JUDGE_MODEL,
                prompt_version=JUDGE_PROMPT_VERSION,
            )

        first, second = await self._race(open_run, open_run)

        self.assertEqual(first, second)
        async with AsyncSession(bind=self.engine) as session:
            count = await session.scalar(
                select(func.count()).select_from(ClassificationRun).where(
                    ClassificationRun.document_group_id == self.ids["group"]
                )
            )
        self.assertEqual(1, count)


if __name__ == "__main__":
    unittest.main()
