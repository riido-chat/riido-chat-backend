"""질문 판별과 정본 캐시 서빙을 HTTP → ChatService → PostgreSQL 기록까지 끝까지 보는 수용 테스트.

- 합성 fixture 만 쓴다. 문서 그룹·문서·청크·임베딩·ACTIVE 색인·PUBLISHED 프로필 판을 이 테스트의
  문서 그룹으로 만들고, 세부 문제와 정본은 시드 스크립트(run_seed, 가짜 임베딩)로 적재한다.
- 공유 로컬 DB 에는 다른 ACTIVE 색인(HELP_CHATBOT, HELP_CHATBOT_TEST)과 PUBLISHED 프로필 판이
  이미 있다. 검색기 팩토리가 이 테스트 색인만 돌려주고, 기존 PUBLISHED 판은 외부 트랜잭션 안에서
  RETIRED 로 내렸다가 마지막 rollback 으로 되돌린다.
- 판별 모델, 검색, 생성, 질의 재작성은 가짜다(API 호출 없음). 판별 서비스는 실제 store·reader 다.
- ChatService 가 Python 3.10+ 문법을 쓰므로 Python 3.12 환경에서 실행한다.

실행: DATABASE_URL=postgresql+asyncpg://riido:riido@localhost:5433/riido \\
    python -m unittest tests.test_question_grouping_acceptance_db -v
"""

import asyncio
import json
import logging
import unittest
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalAnswerStatus,
    FinalGenerationResult,
)
from app.answering.service import GenerationService
from app.chat.dependencies import get_chat_service, get_rag_log_store
from app.chat.log_store import RagLogStore
from app.chat.profile import HELP_CHATBOT_PROFILE_KEY
from app.chat.query_rewrite import QUERY_REWRITE_PROMPT_VERSION, QueryRewriteService
from app.chat.service import ChatService
from app.core.config import get_settings
from app.core.model_trace import ModelCallTrace
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    AnswerCitation,
    AnswerStatus,
    AttributionSource,
    CacheAttemptOutcome,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ChunkEmbedding,
    ClassificationDecision,
    ClassificationRun,
    DocumentSource,
    EmbeddingConfig,
    ExecutionStatus,
    IndexVersion,
    IndexVersionStatus,
    ModelCall,
    ModelCallPurpose,
    QuestionCacheAttempt,
    QuestionClassification,
    QuestionEmbedding,
    QuestionProblemGroup,
    QuestionSubproblem,
    QuestionSubproblemServingState,
    RagRun,
)
from app.database.session import get_db_session
from app.main import create_app
from app.ops.seed_question_grouping import CanonicalAction, parse_input, run_seed
from app.question_grouping.constants import (
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
)
from app.question_grouping.models import JudgeCall
from app.question_grouping.outline_reader import DocumentOutlineCache, DocumentOutlineReader
from app.question_grouping.service import QuestionGroupingService
from app.question_grouping.subproblem_search import build_subproblem_embedding_text
from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_PROVIDER
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalChunk,
    RetrievalResult,
)
from tests.test_question_grouping_readers_db import _available, _now, _Seed
from tests.test_seed_question_grouping_db import FakeEmbedder

BILLING_TITLE = "구독 및 결제"
MEMBERS_TITLE = "멤버"
SUBPROBLEM_KEY = "billing.cancel"
SUBPROBLEM_NAME = "구독 취소"
INCLUSION = "유료 구독을 취소하려는 질문"
CANONICAL = "설정에서 구독을 취소합니다 [1]. 멤버 초대는 따로 봅니다 [2]."
QUESTION = "구독을 취소하고 싶어요"
GENERATION_MODEL = "gpt-test"
GENERATION_PROMPT = "v3"

# (신원 해시, 내용 해시, H2 제목, 본문)
BILLING_SECTIONS = [
    ("id-intro", "c-intro", "", "구독과 결제를 안내합니다."),
    ("id-cancel", "c-cancel", "구독 변경 또는 취소", "설정에서 구독을 취소할 수 있습니다."),
    ("id-cycle", "c-cycle", "결제 주기", "월별과 연간 결제를 고를 수 있습니다."),
]
MEMBERS_SECTIONS = [
    ("m-invite", "c-invite", "멤버 초대", "멤버를 초대합니다."),
    ("m-roles", "c-roles", "멤버 권한", "멤버 권한을 바꿉니다."),
]


def _embedding_trace() -> ModelCallTrace:
    return ModelCallTrace(
        provider=OPENAI_EMBEDDING_PROVIDER,
        model_name=OPENAI_EMBEDDING_MODEL,
        succeeded=True,
        latency_ms=15,
        input_tokens=5,
    )


class FakeJudge:
    """판별 모델 대신 payload 를 받아 정한 출력을 돌려준다. 호출한 payload 를 남긴다."""

    provider = JUDGE_PROVIDER
    model_name = JUDGE_MODEL
    prompt_version = JUDGE_PROMPT_VERSION

    def __init__(self, decide: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._decide = decide
        self.payloads: List[Dict[str, Any]] = []

    async def judge(self, payload, *, before_model_call=None) -> JudgeCall:
        self.payloads.append(payload)
        await before_model_call(JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_PROMPT_VERSION)
        return JudgeCall(
            trace=ModelCallTrace(
                provider=JUDGE_PROVIDER,
                model_name=JUDGE_MODEL,
                succeeded=True,
                latency_ms=700,
                input_tokens=3000,
                output_tokens=120,
                cached_input_tokens=2000,
                reasoning_tokens=40,
                prompt_version=JUDGE_PROMPT_VERSION,
            ),
            output_text=json.dumps(self._decide(payload)),
        )


def connect_to(key: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def decide(payload: Dict[str, Any]) -> Dict[str, Any]:
        candidate = next(item for item in payload["candidates"] if item["subproblem"]["id"] == key)
        return {
            "decision": "CONNECT",
            "groupId": candidate["group"]["id"],
            "subproblemId": key,
            "confidence": 0.95,
            "rationaleCode": "SAME_ASK",
            "ambiguityReason": None,
            "matchedCriteria": ["I1"],
            "conflictingCriteria": [],
            "documentDecision": "SUBPROBLEM_DOCUMENT",
            "documentCandidateId": None,
            "documentRationaleCode": None,
        }

    return decide


def separate_matching_document(title: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def decide(payload: Dict[str, Any]) -> Dict[str, Any]:
        document = next(item for item in payload["documentCandidates"] if item["title"] == title)
        return {
            "decision": "SEPARATE",
            "groupId": None,
            "subproblemId": None,
            "confidence": 0.8,
            "rationaleCode": "DIFFERENT_ASK",
            "ambiguityReason": None,
            "matchedCriteria": [],
            "conflictingCriteria": [],
            "documentDecision": "MATCHED",
            "documentCandidateId": document["id"],
            "documentRationaleCode": "TOPIC_ONLY",
        }

    return decide


class UnusedEmbedder:
    def embed_many_with_usage(self, texts, **kwargs):  # pragma: no cover - 검색 벡터를 재사용한다
        raise AssertionError("판별은 검색 벡터를 재사용해야 합니다.")


class QuestionGroupingAcceptanceDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 DB에 연결할 수 없어 질문 판별 수용 테스트를 건너뜁니다.")

    async def asyncSetUp(self) -> None:
        for name in ("app.question_grouping.service", "app.chat.service"):
            target = logging.getLogger(name)
            previous = target.disabled
            target.disabled = True
            self.addCleanup(setattr, target, "disabled", previous)

        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(
            bind=self.connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        self.log_store = RagLogStore(self.session)
        await self._seed_documents()
        await self._seed_profile()
        await self._seed_subproblem()
        await self._reindex_billing()

        self.service: ChatService
        self.app = create_app()

        async def override_chat_service() -> ChatService:
            return self.service

        async def override_db_session():
            yield self.session

        self.app.dependency_overrides[get_chat_service] = override_chat_service
        self.app.dependency_overrides[get_db_session] = override_db_session
        self.app.dependency_overrides[get_rag_log_store] = lambda: self.log_store
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://testserver",
            headers={"Accept": "application/json"},
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.app.dependency_overrides.clear()
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    # ------------------------------------------------------------------
    # 합성 fixture
    # ------------------------------------------------------------------

    async def _seed_documents(self) -> None:
        seed = _Seed(self.session)
        self.seed = seed
        self.chunking = await seed.chunking()
        self.embed = EmbeddingConfig(
            version=seed._name("accept-embed"),
            provider=OPENAI_EMBEDDING_PROVIDER,
            model_name=OPENAI_EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            input_template_version="v1",
            created_at=_now(),
        )
        await seed.add(self.embed)
        self.group = await seed.group()
        self.group_id = self.group.id
        self.group_key = self.group.group_key
        prefix = f"ws{seed.suffix}"
        self.billing_path = f"{prefix}/plans-and-billing.md"
        self.members_path = f"{prefix}/members.md"
        self.billing = await self._source(f"{prefix}/plans-and-billing", BILLING_TITLE)
        self.members = await self._source(f"{prefix}/members", MEMBERS_TITLE)

        self.billing_v1 = await seed.version(self.billing, 1)
        self.billing_v1_chunks = await self._sections(self.billing_v1, BILLING_TITLE, BILLING_SECTIONS)
        self.members_v1 = await seed.version(self.members, 1)
        self.members_chunks = await self._sections(self.members_v1, MEMBERS_TITLE, MEMBERS_SECTIONS)
        scope = await seed.index(self.group, self.chunking, self.embed, [self.billing_v1, self.members_v1])
        await self._activate(scope.index_version_id)
        self.index_version_id = scope.index_version_id

    async def _source(self, key: str, title: str) -> DocumentSource:
        row = DocumentSource(
            document_group_id=self.group.id,
            document_key=key,
            source_type="GITBOOK",
            canonical_uri=f"https://docs.riido.io/{key}.md",
            title=title,
            enabled=True,
            created_at=_now(),
            updated_at=_now(),
        )
        await self.seed.add(row)
        return row

    async def _sections(self, version, title: str, sections) -> List[int]:
        chunk_ids = await self.seed.sections(version, title, self.chunking, sections)
        vectors = FakeEmbedder().embed_many_with_usage([body for _, _, _, body in sections]).embeddings
        await self.seed.add(
            *[
                ChunkEmbedding(
                    chunk_id=chunk_id,
                    embedding_config_id=self.embed.id,
                    embedding=vector,
                    embedding_input_hash=f"input-{chunk_id}",
                    created_at=_now(),
                )
                for chunk_id, vector in zip(chunk_ids, vectors)
            ]
        )
        return chunk_ids

    async def _activate(self, index_version_id: int) -> None:
        await self.session.execute(
            update(IndexVersion)
            .where(
                IndexVersion.document_group_id == self.group.id,
                IndexVersion.status == IndexVersionStatus.ACTIVE,
            )
            .values(status=IndexVersionStatus.INACTIVE)
        )
        await self.session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == index_version_id)
            .values(status=IndexVersionStatus.ACTIVE)
        )
        await self.session.flush()

    async def _seed_profile(self) -> None:
        """HELP_CHATBOT 의 PUBLISHED 판을 이 테스트 문서 그룹으로 바꾼다(외부 트랜잭션 rollback 으로 복구)."""

        profile = await self.session.scalar(
            select(ChatProfile).where(ChatProfile.profile_key == HELP_CHATBOT_PROFILE_KEY)
        )
        if profile is None:
            profile = ChatProfile(profile_key=HELP_CHATBOT_PROFILE_KEY, name="이용가이드 챗봇")
            await self.seed.add(profile)
        await self.session.execute(
            update(ChatProfileRevision)
            .where(
                ChatProfileRevision.profile_id == profile.id,
                ChatProfileRevision.status == ChatProfileRevisionStatus.PUBLISHED,
            )
            .values(status=ChatProfileRevisionStatus.RETIRED)
        )
        version = await self.session.scalar(
            select(func.coalesce(func.max(ChatProfileRevision.version), 0)).where(
                ChatProfileRevision.profile_id == profile.id
            )
        )
        revision = ChatProfileRevision(
            profile_id=profile.id,
            version=version + 1,
            document_group_id=self.group.id,
            status=ChatProfileRevisionStatus.PUBLISHED,
            generation_model_name=GENERATION_MODEL,
            generation_prompt_version=GENERATION_PROMPT,
            query_rewrite_model_name=GENERATION_MODEL,
            query_rewrite_prompt_version=QUERY_REWRITE_PROMPT_VERSION,
            semantic_cache_enabled=True,
        )
        await self.seed.add(revision)
        self.profile_revision_id = revision.id

    def _seed_input(self) -> List[Dict[str, Any]]:
        return [
            {
                "schemaVersion": 1,
                "document": {"path": self.billing_path, "title": BILLING_TITLE, "parentPath": "ws", "contentSha256": None},
                "subproblems": [
                    {
                        "key": SUBPROBLEM_KEY,
                        "legacyId": None,
                        "name": SUBPROBLEM_NAME,
                        "inclusionCriteria": [INCLUSION],
                        "exclusionCriteria": ["환불을 묻는 질문은 제외"],
                        "canonical": {
                            "contentMarkdown": CANONICAL,
                            "applicabilityRules": ["환불은 다루지 않는다"],
                            "citations": [
                                {"order": 1, "documentPath": self.billing_path, "sectionPath": ["구독 변경 또는 취소"], "evidence": None},
                                {"order": 2, "documentPath": self.members_path, "sectionPath": ["멤버 초대"], "evidence": None},
                            ],
                        },
                        "reviewStatus": "approved",
                    }
                ],
                "notCandidates": [],
            }
        ]

    async def _run_seed(self):
        embedder = FakeEmbedder()
        report = await run_seed(
            self.session,
            group_key=self.group_key,
            parsed=parse_input([("subproblems.json", doc) for doc in self._seed_input()], include_draft=False),
            actor="acceptance",
            apply=True,
            serving_state=QuestionSubproblemServingState.SERVING,
            embedder_factory=lambda: embedder,
        )
        self.assertTrue(report.ok, [issue.render() for issue in report.plan.errors])
        await self.session.commit()
        return report

    async def _seed_subproblem(self) -> None:
        report = await self._run_seed()
        (plan,) = report.plan.subproblem_plans
        self.assertEqual(CanonicalAction.CREATE, plan.canonical_action)
        row = (
            await self.session.execute(
                select(QuestionSubproblem.id, QuestionSubproblem.problem_group_id).where(
                    QuestionSubproblem.problem_group_id.in_(
                        select(QuestionProblemGroup.id).where(QuestionProblemGroup.document_source_id == self.billing.id)
                    ),
                    QuestionSubproblem.key == SUBPROBLEM_KEY,
                )
            )
        ).one()
        self.subproblem_id = row.id
        self.subproblem_problem_group_id = row.problem_group_id
        self.canonical_id = await self.session.scalar(
            select(CanonicalAnswer.id).where(
                CanonicalAnswer.subproblem_id == self.subproblem_id,
                CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
            )
        )
        self.question_vector = tuple(
            FakeEmbedder()
            .embed_many_with_usage([build_subproblem_embedding_text(SUBPROBLEM_NAME, [INCLUSION])])
            .embeddings[0]
        )

    async def _reindex_billing(self) -> None:
        """같은 절 내용으로 구독 및 결제 새 판을 만들고 새 색인을 ACTIVE 로 바꾼다. 청크 id 만 바뀐다."""

        self.billing_v2 = await self.seed.version(self.billing, 2)
        self.billing_chunks = await self._sections(self.billing_v2, BILLING_TITLE, BILLING_SECTIONS)
        scope = await self.seed.index(self.group, self.chunking, self.embed, [self.billing_v2, self.members_v1])
        await self._activate(scope.index_version_id)
        self.index_version_id = scope.index_version_id
        await self.session.commit()
        # 재시드는 정본을 두고(해시에 청크 id 가 없다) 쓰지 않는다.
        report = await self._run_seed()
        (plan,) = report.plan.subproblem_plans
        self.assertEqual(CanonicalAction.UNCHANGED, plan.canonical_action)
        self.assertFalse(plan.writes)

    # ------------------------------------------------------------------
    # 가짜 협력자와 서비스
    # ------------------------------------------------------------------

    def _result(self, chunk_id: int, version_id: int, title: str, heading: str, rank: int) -> Tuple[RetrievalResult, HybridRetrievalResult]:
        source = self.billing if title == BILLING_TITLE else self.members
        chunk = RetrievalChunk(
            document_id=source.document_key,
            section_id=f"{source.document_key}:{chunk_id}",
            document_title=title,
            section_path=(title, heading),
            source_url=source.canonical_uri,
            category="ws",
            content=f"{heading} 본문",
            chunk_id=chunk_id,
            document_version_id=version_id,
            index_version_id=self.index_version_id,
        )
        return (
            RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank),
            HybridRetrievalResult(chunk=chunk, rrf_score=1.0 / (60 + rank), final_rank=rank, bm25_rank=rank, vector_rank=rank),
        )

    def _search_call(self, query: str) -> HybridSearchCall:
        pairs = [
            self._result(self.billing_chunks[1], self.billing_v2.id, BILLING_TITLE, "구독 변경 또는 취소", 1),
            self._result(self.members_chunks[0], self.members_v1.id, MEMBERS_TITLE, "멤버 초대", 2),
            self._result(self.members_chunks[1], self.members_v1.id, MEMBERS_TITLE, "멤버 권한", 3),
        ]
        results = tuple(pair[0] for pair in pairs)
        return HybridSearchCall(
            bm25_results=results,
            vector_results=results,
            fused_results=tuple(pair[1] for pair in pairs),
            bm25_latency_ms=5,
            vector_latency_ms=7,
            embedding_call=_embedding_trace(),
            retrieval_query=query,
            query_embedding=self.question_vector,
        )

    def _generation_citation(self, number: int, chunk_id: int, version_id: int, title: str, heading: str) -> Citation:
        source = self.billing if title == BILLING_TITLE else self.members
        return Citation(
            citation_number=number,
            document_title=title,
            section_path=(title, heading),
            source_url=source.canonical_uri,
            source_kind=CitationSourceKind.GITBOOK,
            chunk_id=chunk_id,
            document_version_id=version_id,
        )

    def _generated(self, query: str) -> FinalGenerationResult:
        # 멤버 문서 2회, 구독 문서 1회 인용: 최다 인용 문서는 멤버다.
        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown=f"{query}: 멤버 초대 [1], 권한 [2], 구독 [3].",
            citations=(
                self._generation_citation(1, self.members_chunks[0], self.members_v1.id, MEMBERS_TITLE, "멤버 초대"),
                self._generation_citation(2, self.members_chunks[1], self.members_v1.id, MEMBERS_TITLE, "멤버 권한"),
                self._generation_citation(3, self.billing_chunks[1], self.billing_v2.id, BILLING_TITLE, "구독 변경 또는 취소"),
            ),
            model_call=ModelCallTrace(
                provider="openai",
                model_name=GENERATION_MODEL,
                succeeded=True,
                latency_ms=30,
                input_tokens=20,
                output_tokens=10,
                prompt_version=GENERATION_PROMPT,
            ),
        )

    def _build_service(
        self,
        judge: Optional[FakeJudge],
        *,
        enabled: bool,
    ) -> Tuple[AsyncMock, AsyncMock]:
        retriever = AsyncMock(spec=HybridRetriever)
        generation = AsyncMock(spec=GenerationService)
        generation.model_name = GENERATION_MODEL
        generation.prompt_version = GENERATION_PROMPT
        query_rewrite = AsyncMock(spec=QueryRewriteService)
        query_rewrite.model_name = GENERATION_MODEL
        query_rewrite.prompt_version = QUERY_REWRITE_PROMPT_VERSION

        async def search(query, *, before_model_call):
            await before_model_call(OPENAI_EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL, None)
            return self._search_call(query)

        async def generate(query, _results, *, before_model_call):
            await before_model_call("openai", GENERATION_MODEL, GENERATION_PROMPT)
            return self._generated(query)

        retriever.search_with_trace.side_effect = search
        generation.generate_answer.side_effect = generate

        def retriever_factory(document_group_id: int):
            self.assertEqual(self.group_id, document_group_id)
            return retriever, self.index_version_id

        grouping = None
        if judge is not None:
            grouping = QuestionGroupingService(
                self.session,
                self.log_store,
                judge,
                UnusedEmbedder(),
                outline_reader=DocumentOutlineReader(self.session, cache=DocumentOutlineCache()),
            )
        self.service = ChatService(
            retriever=None,
            generation_service=generation,
            query_rewrite_service=query_rewrite,
            log_store=self.log_store,
            session=self.session,
            index_version_id=None,
            profile_status=ChatProfileRevisionStatus.PUBLISHED,
            retriever_factory=retriever_factory,
            question_grouping=grouping,
            question_grouping_enabled=enabled,
        )
        return retriever, generation

    # ------------------------------------------------------------------
    # 조회 도우미
    # ------------------------------------------------------------------

    async def _ask(self, question: str = QUESTION) -> Dict[str, Any]:
        response = await self.client.post("/api/chat", json={"question": question})
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        polled = await self.client.get(f"/api/chat/{body['ragRunId']}")
        self.assertEqual(200, polled.status_code, polled.text)
        # 결과 조회는 동기 응답과 같은 기록에서 같은 응답을 만든다.
        self.assertEqual(body, polled.json())
        return body

    async def _grouping_rows(self, rag_run_id: uuid.UUID):
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
                ).where(QuestionClassification.rag_run_id == rag_run_id)
            )
        ).all()
        attempts = (
            await self.session.execute(
                select(
                    QuestionCacheAttempt.classification_id,
                    QuestionCacheAttempt.outcome,
                    QuestionCacheAttempt.canonical_answer_id,
                    QuestionCacheAttempt.rejection_reasons,
                ).where(QuestionCacheAttempt.rag_run_id == rag_run_id)
            )
        ).all()
        embeddings = (
            await self.session.execute(
                select(QuestionEmbedding.embedding_config_id).where(QuestionEmbedding.rag_run_id == rag_run_id)
            )
        ).all()
        return classifications, attempts, embeddings

    async def _model_calls(self, rag_run_id: uuid.UUID) -> List[Any]:
        return (
            await self.session.execute(
                select(ModelCall.purpose, ModelCall.status, ModelCall.classification_run_id)
                .where(ModelCall.rag_run_id == rag_run_id)
                .order_by(ModelCall.id)
            )
        ).all()

    async def _answer_citations(self, rag_run_id: uuid.UUID) -> List[Tuple[Any, ...]]:
        rows = (
            await self.session.execute(
                select(
                    AnswerCitation.citation_order,
                    AnswerCitation.chunk_id,
                    AnswerCitation.document_version_id,
                    AnswerCitation.document_title_snapshot,
                    AnswerCitation.node_path_snapshot,
                )
                .where(AnswerCitation.rag_run_id == rag_run_id)
                .order_by(AnswerCitation.citation_order)
            )
        ).all()
        return [tuple(row) for row in rows]

    async def _run(self, rag_run_id: uuid.UUID) -> RagRun:
        run = await self.session.get(RagRun, rag_run_id)
        await self.session.refresh(run)
        return run

    async def _open_run_id(self) -> Optional[int]:
        return await self.session.scalar(
            select(ClassificationRun.id).where(ClassificationRun.document_group_id == self.group_id)
        )

    # ------------------------------------------------------------------
    # 시나리오
    # ------------------------------------------------------------------

    async def test_connect_serves_canonical_answer_without_generation(self) -> None:
        judge = FakeJudge(connect_to(SUBPROBLEM_KEY))
        retriever, generation = self._build_service(judge, enabled=True)

        body = await self._ask()
        rag_run_id = uuid.UUID(body["ragRunId"])

        self.assertEqual("COMPLETED", body["status"])
        self.assertEqual(CANONICAL, body["answer"]["answerMarkdown"])
        self.assertEqual(
            [
                (1, BILLING_TITLE, ["구독 변경 또는 취소"], self.billing.canonical_uri, "GITBOOK"),
                (2, MEMBERS_TITLE, ["멤버 초대"], self.members.canonical_uri, "GITBOOK"),
            ],
            [
                (item["citationNumber"], item["documentTitle"], item["sectionPath"], item["sourceUrl"], item["sourceKind"])
                for item in body["citations"]
            ],
        )
        retriever.search_with_trace.assert_awaited_once()
        generation.generate_answer.assert_not_awaited()
        self.assertEqual(1, len(judge.payloads))
        self.assertEqual([SUBPROBLEM_KEY], [item["subproblem"]["id"] for item in judge.payloads[0]["candidates"]])

        run = await self._run(rag_run_id)
        self.assertEqual((AnswerStatus.COMPLETED, CANONICAL), (run.status, run.answer_content))
        self.assertEqual(self.index_version_id, run.index_version_id)

        run_id = await self._open_run_id()
        self.assertIsNotNone(run_id)
        classifications, attempts, embeddings = await self._grouping_rows(rag_run_id)
        (classification,) = classifications
        self.assertEqual(
            (ClassificationDecision.CONNECT, AttributionSource.SUBPROBLEM, self.subproblem_problem_group_id, self.subproblem_id, run_id),
            (classification.decision, classification.attribution_source, classification.problem_group_id,
             classification.subproblem_id, classification.run_id),
        )
        gate = classification.judgment_input["gate"]
        self.assertEqual(("SERVED", str(self.canonical_id)), (gate["outcome"], gate["canonicalAnswerId"]))
        # 인용 [1] 은 재색인 뒤 새 판의 같은 절(R17 2단계), [2] 는 같은 청크(1단계)다.
        self.assertEqual([2, 1], [item["step"] for item in gate["citationResolutions"]])
        self.assertEqual(
            [(classification.id, CacheAttemptOutcome.SERVED, self.canonical_id, None)],
            [tuple(row) for row in attempts],
        )
        self.assertEqual([(self.embed.id,)], [tuple(row) for row in embeddings])

        # 정본 인용을 현재 색인 청크로 복사한다(정본에 저장된 옛 청크가 아니다).
        self.assertNotEqual(self.billing_v1_chunks[1], self.billing_chunks[1])
        self.assertEqual(
            [
                (1, self.billing_chunks[1], self.billing_v2.id, BILLING_TITLE, f"{BILLING_TITLE} > 구독 변경 또는 취소"),
                (2, self.members_chunks[0], self.members_v1.id, MEMBERS_TITLE, f"{MEMBERS_TITLE} > 멤버 초대"),
            ],
            await self._answer_citations(rag_run_id),
        )
        self.assertEqual(
            [
                (ModelCallPurpose.QUERY_EMBEDDING, ExecutionStatus.SUCCESS, run_id),
                (ModelCallPurpose.QUESTION_CLASSIFICATION, ExecutionStatus.SUCCESS, run_id),
            ],
            [tuple(row) for row in await self._model_calls(rag_run_id)],
        )

    async def test_separate_with_matched_document_generates_and_attributes_by_citation(self) -> None:
        judge = FakeJudge(separate_matching_document(BILLING_TITLE))
        _, generation = self._build_service(judge, enabled=True)

        body = await self._ask()
        rag_run_id = uuid.UUID(body["ragRunId"])

        self.assertEqual("COMPLETED", body["status"])
        self.assertEqual(f"{QUESTION}: 멤버 초대 [1], 권한 [2], 구독 [3].", body["answer"]["answerMarkdown"])
        self.assertEqual(3, len(body["citations"]))
        generation.generate_answer.assert_awaited_once()
        self.assertEqual(1, len(judge.payloads))

        run_id = await self._open_run_id()
        classifications, attempts, embeddings = await self._grouping_rows(rag_run_id)
        (classification,) = classifications
        members_group = await self.session.scalar(
            select(QuestionProblemGroup.id).where(QuestionProblemGroup.document_source_id == self.members.id)
        )
        # 판별 시점 귀속은 MATCHED 문서(구독 및 결제)였고, 턴 끝에 최다 인용 문서(멤버)로 덮어썼다.
        self.assertEqual("MATCHED", classification.judgment_input["output"]["documentDecision"])
        self.assertEqual(
            (ClassificationDecision.SEPARATE, AttributionSource.CITATION, members_group, None, run_id),
            (classification.decision, classification.attribution_source, classification.problem_group_id,
             classification.subproblem_id, classification.run_id),
        )
        self.assertEqual("REJECTED", classification.judgment_input["gate"]["outcome"])
        self.assertEqual(
            [(classification.id, CacheAttemptOutcome.REJECTED, None, [REJECT_CLASSIFICATION_NOT_CONNECTED])],
            [tuple(row) for row in attempts],
        )
        self.assertEqual([(self.embed.id,)], [tuple(row) for row in embeddings])
        self.assertEqual(
            [
                (1, self.members_chunks[0], self.members_v1.id),
                (2, self.members_chunks[1], self.members_v1.id),
                (3, self.billing_chunks[1], self.billing_v2.id),
            ],
            [row[:3] for row in await self._answer_citations(rag_run_id)],
        )
        self.assertEqual(
            [
                (ModelCallPurpose.QUERY_EMBEDDING, ExecutionStatus.SUCCESS, run_id),
                (ModelCallPurpose.QUESTION_CLASSIFICATION, ExecutionStatus.SUCCESS, run_id),
                (ModelCallPurpose.ANSWER_GENERATION, ExecutionStatus.SUCCESS, None),
            ],
            [tuple(row) for row in await self._model_calls(rag_run_id)],
        )
        self.assertEqual(AnswerStatus.COMPLETED, (await self._run(rag_run_id)).status)

    async def test_switch_off_matches_turn_without_grouping(self) -> None:
        judge = FakeJudge(connect_to(SUBPROBLEM_KEY))
        _, disabled_generation = self._build_service(judge, enabled=False)
        disabled = await self._ask()

        _, legacy_generation = self._build_service(None, enabled=False)
        legacy = await self._ask()

        def without_ids(body: Dict[str, Any]) -> Dict[str, Any]:
            return {key: value for key, value in body.items() if key not in ("conversationId", "ragRunId")}

        self.assertEqual("COMPLETED", disabled["status"])
        self.assertEqual(without_ids(legacy), without_ids(disabled))
        disabled_generation.generate_answer.assert_awaited_once()
        legacy_generation.generate_answer.assert_awaited_once()
        self.assertEqual([], judge.payloads)
        self.assertIsNone(await self._open_run_id())

        for body in (disabled, legacy):
            rag_run_id = uuid.UUID(body["ragRunId"])
            self.assertEqual(([], [], []), await self._grouping_rows(rag_run_id))
            self.assertEqual(
                [
                    (ModelCallPurpose.QUERY_EMBEDDING, ExecutionStatus.SUCCESS, None),
                    (ModelCallPurpose.ANSWER_GENERATION, ExecutionStatus.SUCCESS, None),
                ],
                [tuple(row) for row in await self._model_calls(rag_run_id)],
            )
            self.assertEqual(3, len(await self._answer_citations(rag_run_id)))
        disabled_detail = await self.log_store.get_rag_run_detail(uuid.UUID(disabled["ragRunId"]))
        legacy_detail = await self.log_store.get_rag_run_detail(uuid.UUID(legacy["ragRunId"]))
        self.assertEqual(len(legacy_detail.retrieval_results), len(disabled_detail.retrieval_results))
        self.assertGreater(len(disabled_detail.retrieval_results), 0)


if __name__ == "__main__":
    unittest.main()
