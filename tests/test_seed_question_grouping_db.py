"""세부 문제 시드 스크립트 로컬 DB 통합 테스트.

- 로컬 PostgreSQL 과 head 마이그레이션이 필요하다. 연결할 수 없으면 skip 한다.
- 외부 트랜잭션 하나에서 시드하고 마지막에 rollback 해 데이터를 남기지 않는다.
- 입력은 인라인 합성 fixture, 임베딩은 가짜 클라이언트다(API 호출 없음).

실행: DATABASE_URL=postgresql+asyncpg://riido:riido@localhost:5433/riido \\
    .venv/bin/python -m unittest tests.test_seed_question_grouping_db -v
"""

import asyncio
import hashlib
import unittest
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    CacheAttemptOutcome,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    CanonicalAnswerOrigin,
    ClassificationDecision,
    DocumentSource,
    EmbeddingConfig,
    IndexVersion,
    IndexVersionStatus,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    QuestionSubproblemRevision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.ops.seed_question_grouping import (
    ERROR_DOCUMENT_DISABLED,
    ERROR_DUPLICATE_KEY,
    ERROR_KEY_IN_OTHER_PROBLEM_GROUP,
    ERROR_SECTION_NOT_FOUND,
    CanonicalAction,
    EmbeddingAction,
    SeedReport,
    SubproblemAction,
    parse_input,
    render_report,
    run_seed,
)
from app.question_grouping.attribution import initial_attribution
from app.question_grouping.catalog_reader import QuestionCatalogReader
from app.question_grouping.constants import SUBPROBLEM_EMBEDDING_TEXT_VERSION
from app.question_grouping.gate import evaluate_cache_gate, resolve_citations
from app.question_grouping.models import IndexScope, PresentedSubproblem, TurnJudgment
from app.question_grouping.store import served_citation_logs
from app.retrieval.embedding import EmbeddingResponse, OPENAI_EMBEDDING_MODEL
from tests.test_question_grouping_readers_db import _available, _now, _Seed

BILLING_TITLE = "구독 및 결제"
MEMBERS_TITLE = "멤버"


class FakeEmbedder:
    """텍스트 해시로 정한 한 차원에 1.0 을 둔 벡터를 돌려준다."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def embed_many_with_usage(self, texts: Sequence[str]) -> EmbeddingResponse:
        self.calls.append(list(texts))
        vectors = []
        for value in texts:
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIMENSIONS] = 1.0
            vectors.append(vector)
        return EmbeddingResponse(embeddings=vectors, input_tokens=7 * len(texts), retry_count=0)

    @property
    def texts(self) -> List[str]:
        return [text for call in self.calls for text in call]


def _refuse_embedder() -> Any:
    raise AssertionError("임베딩 클라이언트를 만들면 안 됩니다.")


class SeedQuestionGroupingDbTest(unittest.IsolatedAsyncioTestCase):
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
        seed = _Seed(self.session)
        self.seed = seed
        self.chunking = await seed.chunking()
        rechunking = await seed.chunking()
        self.embed = EmbeddingConfig(
            version=seed._name("seed-embed"),
            provider="openai",
            model_name=OPENAI_EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            input_template_version="v1",
            created_at=_now(),
        )
        await seed.add(self.embed)
        self.group = await seed.group()
        prefix = f"ws{seed.suffix}"
        self.billing_path = f"{prefix}/plans-and-billing.md"
        self.members_path = f"{prefix}/members.md"
        self.old_path = f"{prefix}/old.md"
        self.billing = await self._source(f"{prefix}/plans-and-billing", BILLING_TITLE)
        self.members = await self._source(f"{prefix}/members", MEMBERS_TITLE)
        self.old = await self._source(f"{prefix}/old", "옛 문서", enabled=False)

        self.billing_v1 = await seed.version(self.billing, 1)
        self.billing_chunks = await seed.sections(
            self.billing_v1,
            BILLING_TITLE,
            self.chunking,
            [
                ("id-intro", "c-intro", "", "머리말"),
                ("id-cancel", "c-cancel", "구독 변경 또는 취소", "* 설정에서 구독을 **취소**할 수 있습니다."),
                ("id-cycle", "c-cycle", "결제 주기", "월별과 연간 결제를 고를 수 있습니다."),
            ],
        )
        # 같은 문서 판의 다른 청킹 설정 청크는 절 해석에서 빠져야 한다.
        await seed.sections(self.billing_v1, BILLING_TITLE, rechunking, [("id-cancel", "c-cancel-r", "구독 변경 또는 취소", "재청킹")])
        self.members_v1 = await seed.version(self.members, 1)
        self.members_chunks = await seed.sections(
            self.members_v1, MEMBERS_TITLE, self.chunking, [("m-invite", "c-invite", "멤버 초대", "멤버를 초대합니다.")]
        )
        old_v1 = await seed.version(self.old, 1)
        await seed.sections(old_v1, "옛 문서", self.chunking, [("o-1", "c-old", "옛 절", "옛 본문")])
        self.scope = await seed.index(self.group, self.chunking, self.embed, [self.billing_v1, self.members_v1, old_v1])
        await self.session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == self.scope.index_version_id)
            .values(status=IndexVersionStatus.ACTIVE)
        )
        await self.session.flush()
        self.embedder = FakeEmbedder()

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def _source(self, key: str, title: str, *, enabled: bool = True) -> DocumentSource:
        row = DocumentSource(
            document_group_id=self.group.id,
            document_key=key,
            source_type="GITBOOK",
            canonical_uri=f"https://docs.riido.io/{key}.md",
            title=title,
            enabled=enabled,
            created_at=_now(),
            updated_at=_now(),
        )
        await self.seed.add(row)
        return row

    # ------------------------------------------------------------------
    # 합성 입력
    # ------------------------------------------------------------------

    def _documents(self) -> List[Dict[str, Any]]:
        def citation(order: int, path: str, section: str) -> Dict[str, Any]:
            return {"order": order, "documentPath": path, "sectionPath": [section], "evidence": None}

        def subproblem(key: str, name: str, content: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "key": key,
                "legacyId": None,
                "name": name,
                "inclusionCriteria": [f"{name}을 묻는 질문"],
                "exclusionCriteria": ["다른 의도는 제외"],
                "canonical": {
                    "contentMarkdown": content,
                    "applicabilityRules": ["가격은 다루지 않는다"],
                    "citations": citations,
                },
                "reviewStatus": "approved",
            }

        billing = {
            "schemaVersion": 1,
            "document": {"path": self.billing_path, "title": BILLING_TITLE, "parentPath": "ws", "contentSha256": None},
            "subproblems": [
                subproblem(
                    "billing.cancel",
                    "구독 취소",
                    "설정에서 취소합니다 [1]. 멤버 초대는 따로 봅니다 [2].",
                    [
                        {**citation(1, self.billing_path, "구독 변경 또는 취소"), "evidence": "설정에서 구독을 취소할 수 있습니다."},
                        citation(2, self.members_path, "멤버 초대"),
                    ],
                ),
                subproblem("billing.cycle", "결제 주기", "월별과 연간 중 고릅니다 [1].", [citation(1, self.billing_path, "결제 주기")]),
                {**subproblem("billing.draft", "초안", "초안 [1].", [citation(1, self.billing_path, "결제 주기")]), "reviewStatus": "draft"},
            ],
            "notCandidates": [],
        }
        members = {
            "schemaVersion": 1,
            "document": {"path": self.members_path, "title": MEMBERS_TITLE, "parentPath": "ws", "contentSha256": None},
            "subproblems": [
                subproblem("members.invite", "멤버 초대", "멤버를 초대합니다 [1].", [citation(1, self.members_path, "멤버 초대")])
            ],
            "notCandidates": [],
        }
        return [billing, members]

    async def _run(
        self,
        documents: List[Dict[str, Any]],
        *,
        apply: bool = True,
        serving_state: Optional[QuestionSubproblemServingState] = None,
        include_draft: bool = False,
        embedder: Optional[FakeEmbedder] = None,
    ) -> SeedReport:
        parsed = parse_input([(f"f{i}.json", doc) for i, doc in enumerate(documents)], include_draft=include_draft)
        fake = embedder or self.embedder
        report = await run_seed(
            self.session,
            group_key=self.group.group_key,
            parsed=parsed,
            actor="seed-tester",
            apply=apply,
            serving_state=serving_state,
            include_draft=include_draft,
            embedder_factory=(lambda: fake) if apply else _refuse_embedder,
        )
        render_report(report)  # 보고 생성이 깨지지 않는지 함께 본다.
        return report

    # ------------------------------------------------------------------
    # 조회 도우미
    # ------------------------------------------------------------------

    async def _counts(self) -> Dict[str, int]:
        source_ids = [self.billing.id, self.members.id, self.old.id]
        group_filter = (QuestionProblemGroup.document_source_id.in_(source_ids)) | (
            QuestionProblemGroup.document_group_id == self.group.id
        )
        subproblem_ids = select(QuestionSubproblem.id).join(
            QuestionProblemGroup, QuestionProblemGroup.id == QuestionSubproblem.problem_group_id
        ).where(group_filter)
        canonical_ids = select(CanonicalAnswer.id).where(CanonicalAnswer.subproblem_id.in_(subproblem_ids))
        return {
            "problem_groups": await self.session.scalar(select(func.count()).select_from(QuestionProblemGroup).where(group_filter)),
            "subproblems": await self.session.scalar(select(func.count()).select_from(subproblem_ids.subquery())),
            "revisions": await self.session.scalar(
                select(func.count()).select_from(QuestionSubproblemRevision).where(QuestionSubproblemRevision.subproblem_id.in_(subproblem_ids))
            ),
            "canonicals": await self.session.scalar(select(func.count()).select_from(canonical_ids.subquery())),
            "citations": await self.session.scalar(
                select(func.count()).select_from(CanonicalAnswerCitation).where(CanonicalAnswerCitation.canonical_answer_id.in_(canonical_ids))
            ),
        }

    async def _subproblem(self, key: str):
        return (
            await self.session.execute(
                select(
                    QuestionSubproblem.id,
                    QuestionSubproblem.problem_group_id,
                    QuestionSubproblem.name,
                    QuestionSubproblem.inclusion_criteria,
                    QuestionSubproblem.exclusion_criteria,
                    QuestionSubproblem.current_version,
                    QuestionSubproblem.status,
                    QuestionSubproblem.serving_state,
                    QuestionSubproblem.created_by,
                    QuestionSubproblem.updated_at,
                )
                .join(QuestionProblemGroup, QuestionProblemGroup.id == QuestionSubproblem.problem_group_id)
                .where(
                    QuestionSubproblem.key == key,
                    QuestionProblemGroup.document_source_id.in_([self.billing.id, self.members.id]),
                )
            )
        ).one()

    async def _canonicals(self, subproblem_id) -> List[Any]:
        return (
            await self.session.execute(
                select(
                    CanonicalAnswer.id,
                    CanonicalAnswer.approval,
                    CanonicalAnswer.subproblem_version,
                    CanonicalAnswer.content_markdown,
                    CanonicalAnswer.applicability_rules,
                    CanonicalAnswer.origin,
                    CanonicalAnswer.approved_by,
                    CanonicalAnswer.valid_from,
                    CanonicalAnswer.valid_to,
                )
                .where(CanonicalAnswer.subproblem_id == subproblem_id)
                .order_by(CanonicalAnswer.created_at, CanonicalAnswer.id)
            )
        ).all()

    def _actions(self, report: SeedReport) -> Dict[str, tuple]:
        return {
            plan.seed.key: (plan.action, plan.embedding_action, plan.canonical_action)
            for plan in report.plan.subproblem_plans
        }

    # ------------------------------------------------------------------
    # 테스트
    # ------------------------------------------------------------------

    async def test_dry_run_plans_without_writing_or_embedding(self) -> None:
        report = await self._run(self._documents(), apply=False)

        self.assertEqual([], [issue.render() for issue in report.plan.errors])
        self.assertIsNone(report.applied)
        self.assertEqual(
            {key: (SubproblemAction.CREATE, EmbeddingAction.NEW_REVISION, CanonicalAction.CREATE) for key in ("billing.cancel", "billing.cycle", "members.invite")},
            self._actions(report),
        )
        self.assertIsNone(report.plan.no_document_group_id)
        self.assertEqual({self.billing.id, self.members.id}, set(report.plan.missing_document_problem_groups))
        self.assertEqual(
            {"problem_groups": 0, "subproblems": 0, "revisions": 0, "canonicals": 0, "citations": 0},
            await self._counts(),
        )
        self.assertEqual([], self.embedder.calls)

    async def test_first_apply_creates_rows_then_second_apply_writes_nothing(self) -> None:
        report = await self._run(self._documents(), serving_state=QuestionSubproblemServingState.SERVING)

        self.assertTrue(report.ok, [issue.render() for issue in report.plan.errors])
        # NO_DOCUMENT 1 + 켜진 문서 2(꺼진 문서는 만들지 않는다)
        self.assertEqual(
            {"problem_groups": 3, "subproblems": 3, "revisions": 3, "canonicals": 3, "citations": 4},
            await self._counts(),
        )
        self.assertEqual(3, report.applied.problem_groups_created)
        self.assertEqual(21, report.applied.embedding_input_tokens)
        self.assertEqual(1, len(self.embedder.calls))
        self.assertIn("구독 취소\n구독 취소을 묻는 질문", self.embedder.texts)
        no_document = await self.session.scalar(
            select(QuestionProblemGroup.kind).where(QuestionProblemGroup.document_group_id == self.group.id)
        )
        self.assertEqual(QuestionProblemGroupKind.NO_DOCUMENT, no_document)

        cancel = await self._subproblem("billing.cancel")
        billing_group = await self.session.scalar(
            select(QuestionProblemGroup.id).where(QuestionProblemGroup.document_source_id == self.billing.id)
        )
        self.assertEqual(billing_group, cancel.problem_group_id)
        self.assertEqual(("구독 취소", "구독 취소을 묻는 질문", "다른 의도는 제외"), (cancel.name, cancel.inclusion_criteria, cancel.exclusion_criteria))
        self.assertEqual((1, QuestionSubproblemStatus.APPROVED, QuestionSubproblemServingState.SERVING, "seed-tester"),
                         (cancel.current_version, cancel.status, cancel.serving_state, cancel.created_by))

        revision = (
            await self.session.execute(
                select(
                    QuestionSubproblemRevision.version,
                    QuestionSubproblemRevision.name_snapshot,
                    QuestionSubproblemRevision.inclusion_snapshot,
                    QuestionSubproblemRevision.exclusion_snapshot,
                    QuestionSubproblemRevision.inclusion_embedding,
                    QuestionSubproblemRevision.embedding_config_id,
                    QuestionSubproblemRevision.embedding_text_version,
                    QuestionSubproblemRevision.approved_by,
                ).where(QuestionSubproblemRevision.subproblem_id == cancel.id)
            )
        ).one()
        self.assertEqual(
            (1, "구독 취소", "구독 취소을 묻는 질문", "다른 의도는 제외", self.embed.id, SUBPROBLEM_EMBEDDING_TEXT_VERSION, "seed-tester"),
            (revision.version, revision.name_snapshot, revision.inclusion_snapshot, revision.exclusion_snapshot,
             revision.embedding_config_id, revision.embedding_text_version, revision.approved_by),
        )
        self.assertEqual(EMBEDDING_DIMENSIONS, len(revision.inclusion_embedding))

        (canonical,) = await self._canonicals(cancel.id)
        self.assertEqual(
            (CanonicalAnswerApproval.APPROVED, 1, CanonicalAnswerOrigin.AUTHORED, "seed-tester", {"rules": ["가격은 다루지 않는다"]}),
            (canonical.approval, canonical.subproblem_version, canonical.origin, canonical.approved_by, canonical.applicability_rules),
        )
        self.assertIsNotNone(canonical.valid_from)
        citations = (
            await self.session.execute(
                select(
                    CanonicalAnswerCitation.citation_order,
                    CanonicalAnswerCitation.chunk_id,
                    CanonicalAnswerCitation.document_version_id,
                    CanonicalAnswerCitation.document_title_snapshot,
                    CanonicalAnswerCitation.node_path_snapshot,
                    CanonicalAnswerCitation.source_uri_snapshot,
                )
                .where(CanonicalAnswerCitation.canonical_answer_id == canonical.id)
                .order_by(CanonicalAnswerCitation.citation_order)
            )
        ).all()
        self.assertEqual(
            [
                (1, self.billing_chunks[1], self.billing_v1.id, BILLING_TITLE, f"{BILLING_TITLE} > 구독 변경 또는 취소", self.billing.canonical_uri),
                (2, self.members_chunks[0], self.members_v1.id, MEMBERS_TITLE, f"{MEMBERS_TITLE} > 멤버 초대", self.members.canonical_uri),
            ],
            [tuple(row) for row in citations],
        )

        before = await self._counts()
        second_embedder = FakeEmbedder()
        second = await self._run(self._documents(), embedder=second_embedder)
        self.assertTrue(second.ok)
        self.assertEqual(
            {key: (SubproblemAction.UNCHANGED, EmbeddingAction.NONE, CanonicalAction.UNCHANGED) for key in ("billing.cancel", "billing.cycle", "members.invite")},
            self._actions(second),
        )
        self.assertFalse(any(plan.writes for plan in second.plan.subproblem_plans))
        self.assertEqual([], second_embedder.calls)
        self.assertEqual(before, await self._counts())
        applied = second.applied
        self.assertEqual(
            (0, 0, 0, 0, 0, 0, 0),
            (applied.problem_groups_created, applied.subproblems_created, applied.subproblems_updated,
             applied.revisions_created, applied.revisions_embedding_refreshed, applied.canonicals_created, applied.canonicals_replaced),
        )
        self.assertEqual(cancel.updated_at, (await self._subproblem("billing.cancel")).updated_at)
        self.assertEqual([canonical.id], [row.id for row in await self._canonicals(cancel.id)])

    async def test_criteria_change_bumps_version_and_replaces_canonical(self) -> None:
        await self._run(self._documents())
        cancel = await self._subproblem("billing.cancel")
        (old_canonical,) = await self._canonicals(cancel.id)

        documents = self._documents()
        documents[0]["subproblems"][0]["inclusionCriteria"] = ["구독을 해지하려는 질문"]
        embedder = FakeEmbedder()
        report = await self._run(documents, embedder=embedder)

        self.assertTrue(report.ok, [issue.render() for issue in report.plan.errors])
        actions = self._actions(report)
        self.assertEqual((SubproblemAction.UPDATE, EmbeddingAction.NEW_REVISION, CanonicalAction.REPLACE), actions["billing.cancel"])
        self.assertEqual((SubproblemAction.UNCHANGED, EmbeddingAction.NONE, CanonicalAction.UNCHANGED), actions["billing.cycle"])
        self.assertEqual([["구독 취소\n구독을 해지하려는 질문"]], embedder.calls)

        updated = await self._subproblem("billing.cancel")
        self.assertEqual((2, "구독을 해지하려는 질문"), (updated.current_version, updated.inclusion_criteria))
        versions = (
            await self.session.scalars(
                select(QuestionSubproblemRevision.version)
                .where(QuestionSubproblemRevision.subproblem_id == cancel.id)
                .order_by(QuestionSubproblemRevision.version)
            )
        ).all()
        self.assertEqual([1, 2], list(versions))

        old, new = await self._canonicals(cancel.id)
        self.assertEqual(old_canonical.id, old.id)
        self.assertEqual(CanonicalAnswerApproval.REVOKED, old.approval)
        self.assertIsNotNone(old.valid_to)
        self.assertEqual((CanonicalAnswerApproval.APPROVED, 2), (new.approval, new.subproblem_version))
        self.assertEqual(2, await self.session.scalar(
            select(func.count()).select_from(CanonicalAnswerCitation).where(CanonicalAnswerCitation.canonical_answer_id == new.id)
        ))

    async def test_canonical_text_change_replaces_canonical_only(self) -> None:
        await self._run(self._documents())
        cycle = await self._subproblem("billing.cycle")
        counts = await self._counts()

        documents = self._documents()
        documents[0]["subproblems"][1]["canonical"]["contentMarkdown"] = "결제 주기는 월별 또는 연간입니다 [1]."
        embedder = FakeEmbedder()
        report = await self._run(documents, embedder=embedder)

        self.assertEqual((SubproblemAction.UNCHANGED, EmbeddingAction.NONE, CanonicalAction.REPLACE), self._actions(report)["billing.cycle"])
        self.assertEqual([], embedder.calls)
        after = await self._counts()
        self.assertEqual(counts["revisions"], after["revisions"])
        self.assertEqual(counts["canonicals"] + 1, after["canonicals"])
        self.assertEqual(1, (await self._subproblem("billing.cycle")).current_version)
        old, new = await self._canonicals(cycle.id)
        self.assertEqual((CanonicalAnswerApproval.REVOKED, CanonicalAnswerApproval.APPROVED), (old.approval, new.approval))
        self.assertEqual((1, "결제 주기는 월별 또는 연간입니다 [1]."), (new.subproblem_version, new.content_markdown))

    async def _reindex_billing_with_same_sections(self) -> List[int]:
        """구독 및 결제 문서의 새 판을 같은 절 내용으로 만들고 새 ACTIVE 색인으로 바꾼다. 새 청크 id 를 돌려준다."""

        billing_v2 = await self.seed.version(self.billing, 2)
        new_chunks = await self.seed.sections(
            billing_v2,
            BILLING_TITLE,
            self.chunking,
            [
                ("id-intro", "c-intro", "", "머리말"),
                ("id-cancel", "c-cancel", "구독 변경 또는 취소", "* 설정에서 구독을 **취소**할 수 있습니다."),
                ("id-cycle", "c-cycle", "결제 주기", "월별과 연간 결제를 고를 수 있습니다."),
            ],
        )
        members_v1 = self.members_v1
        await self.session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == self.scope.index_version_id)
            .values(status=IndexVersionStatus.INACTIVE)
        )
        await self.session.flush()
        self.scope = await self.seed.index(self.group, self.chunking, self.embed, [billing_v2, members_v1])
        await self.session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == self.scope.index_version_id)
            .values(status=IndexVersionStatus.ACTIVE)
        )
        await self.session.flush()
        self.billing_v2 = billing_v2
        return new_chunks

    async def test_reindex_with_same_sections_keeps_canonical(self) -> None:
        await self._run(self._documents(), serving_state=QuestionSubproblemServingState.SERVING)
        cancel = await self._subproblem("billing.cancel")
        (canonical,) = await self._canonicals(cancel.id)
        new_chunks = await self._reindex_billing_with_same_sections()
        self.assertNotIn(self.billing_chunks[1], new_chunks)
        counts = await self._counts()

        embedder = FakeEmbedder()
        report = await self._run(self._documents(), embedder=embedder)
        self.assertTrue(report.ok, [issue.render() for issue in report.plan.errors])
        self.assertEqual(
            {key: (SubproblemAction.UNCHANGED, EmbeddingAction.NONE, CanonicalAction.UNCHANGED) for key in ("billing.cancel", "billing.cycle", "members.invite")},
            self._actions(report),
        )
        self.assertFalse(any(plan.writes for plan in report.plan.subproblem_plans))
        self.assertEqual([], embedder.calls)
        self.assertEqual(counts, await self._counts())
        self.assertEqual([(canonical.id, CanonicalAnswerApproval.APPROVED)], [(row.id, row.approval) for row in await self._canonicals(cancel.id)])
        # 계획은 새 색인의 청크를 해석하지만 쓰지 않았다. 저장된 인용은 옛 청크 그대로다.
        cancel_plan = next(plan for plan in report.plan.subproblem_plans if plan.seed.key == "billing.cancel")
        self.assertEqual((new_chunks[1], self.billing_v2.id), (cancel_plan.citations[0].chunk_id, cancel_plan.citations[0].document_version_id))
        stored = (
            await self.session.scalars(
                select(CanonicalAnswerCitation.chunk_id)
                .where(CanonicalAnswerCitation.canonical_answer_id == canonical.id)
                .order_by(CanonicalAnswerCitation.citation_order)
            )
        ).all()
        self.assertEqual([self.billing_chunks[1], self.members_chunks[0]], list(stored))

        # 서빙 게이트는 R17 로 새 색인의 같은 절(신원·내용 해시 일치, 2단계)을 찾는다.
        reader = QuestionCatalogReader(self.session)
        scope = await reader.load_index_scope(self.scope.index_version_id)
        inputs = await reader.load_gate_inputs(cancel.id, scope)
        resolutions = resolve_citations(inputs.citations, inputs.contexts_by_source_id)
        self.assertEqual([2, 1], [resolution.step for resolution in resolutions])
        self.assertEqual(new_chunks[1], resolutions[0].section.chunk_id)

        # 재색인 뒤 본문을 바꾸면 교체하고, 새 정본 인용에는 새 청크 id 를 쓴다.
        documents = self._documents()
        documents[0]["subproblems"][0]["canonical"]["contentMarkdown"] = "설정 화면에서 취소합니다 [1]. 멤버 초대는 따로 봅니다 [2]."
        replaced = await self._run(documents)
        self.assertEqual(CanonicalAction.REPLACE, self._actions(replaced)["billing.cancel"][2])
        self.assertEqual(CanonicalAction.UNCHANGED, self._actions(replaced)["billing.cycle"][2])
        old, new = await self._canonicals(cancel.id)
        self.assertEqual((CanonicalAnswerApproval.REVOKED, CanonicalAnswerApproval.APPROVED), (old.approval, new.approval))
        rows = (
            await self.session.execute(
                select(CanonicalAnswerCitation.chunk_id, CanonicalAnswerCitation.document_version_id)
                .where(CanonicalAnswerCitation.canonical_answer_id == new.id)
                .order_by(CanonicalAnswerCitation.citation_order)
            )
        ).all()
        self.assertEqual([(new_chunks[1], self.billing_v2.id), (self.members_chunks[0], self.members_v1.id)], [tuple(row) for row in rows])

    async def test_unresolved_section_and_disabled_document_block_all_writes(self) -> None:
        documents = self._documents()
        documents[0]["subproblems"][1]["canonical"]["citations"][0]["sectionPath"] = ["없는 절"]
        documents[1]["subproblems"][0]["canonical"]["contentMarkdown"] = "멤버를 초대합니다 [1]. 옛 문서 [2]."
        documents[1]["subproblems"][0]["canonical"]["citations"].append(
            {"order": 2, "documentPath": self.old_path, "sectionPath": ["옛 절"], "evidence": None}
        )
        report = await self._run(documents)

        codes = sorted(issue.code for issue in report.plan.errors)
        self.assertEqual([ERROR_DOCUMENT_DISABLED, ERROR_SECTION_NOT_FOUND], codes)
        section_error = next(issue for issue in report.plan.errors if issue.code == ERROR_SECTION_NOT_FOUND)
        self.assertEqual("billing.cycle", section_error.key)
        self.assertIn("없는 절", section_error.message)
        self.assertIn("결제 주기", section_error.message)  # 있는 절 목록을 보여 준다.
        self.assertIsNone(report.applied)
        self.assertEqual(
            {"problem_groups": 0, "subproblems": 0, "revisions": 0, "canonicals": 0, "citations": 0},
            await self._counts(),
        )
        self.assertEqual([], self.embedder.calls)
        self.assertIn("검증 실패", render_report(report))

    async def test_duplicate_key_across_documents_and_moving_key_are_rejected(self) -> None:
        documents = self._documents()
        documents[1]["subproblems"][0]["key"] = "billing.cancel"
        report = await self._run(documents)
        self.assertIn(ERROR_DUPLICATE_KEY, [issue.code for issue in report.plan.errors])
        self.assertEqual(0, (await self._counts())["subproblems"])

        await self._run(self._documents())
        moved = self._documents()
        invite = moved[1]["subproblems"].pop()
        invite["canonical"]["citations"][0] = {"order": 1, "documentPath": self.billing_path, "sectionPath": ["결제 주기"], "evidence": None}
        moved[0]["subproblems"].append(invite)
        counts = await self._counts()
        report = await self._run(moved)
        self.assertEqual([ERROR_KEY_IN_OTHER_PROBLEM_GROUP], [issue.code for issue in report.plan.errors])
        self.assertEqual("members.invite", report.plan.errors[0].key)
        self.assertEqual(counts, await self._counts())

    async def test_serving_state_option_and_untouched_report(self) -> None:
        await self._run(self._documents(), serving_state=QuestionSubproblemServingState.SHADOW)

        kept = await self._run(self._documents()[:1])
        self.assertEqual(QuestionSubproblemServingState.SHADOW, (await self._subproblem("billing.cancel")).serving_state)
        self.assertEqual({"ws" + self.seed.suffix + "/members": ["members.invite"]}, kept.plan.untouched)
        self.assertIn("members.invite", render_report(kept))

        stopped = await self._run(self._documents()[:1], serving_state=QuestionSubproblemServingState.STOPPED)
        self.assertEqual(2, stopped.applied.serving_state_changed)
        self.assertEqual(QuestionSubproblemServingState.STOPPED, (await self._subproblem("billing.cycle")).serving_state)
        self.assertEqual(QuestionSubproblemServingState.SHADOW, (await self._subproblem("members.invite")).serving_state)

    async def test_include_draft_loads_drafts(self) -> None:
        report = await self._run(self._documents(), include_draft=True)
        self.assertTrue(report.ok)
        self.assertEqual(QuestionSubproblemStatus.APPROVED, (await self._subproblem("billing.draft")).status)
        self.assertEqual(4, (await self._counts())["subproblems"])

    async def test_seeded_rows_feed_catalog_and_served_gate(self) -> None:
        report = await self._run(self._documents(), serving_state=QuestionSubproblemServingState.SERVING)
        self.assertTrue(report.ok)
        reader = QuestionCatalogReader(self.session)
        scope = await reader.load_index_scope(self.scope.index_version_id)
        self.assertEqual(
            IndexScope(self.scope.index_version_id, self.group.id, self.chunking.id, self.embed.id),
            scope,
        )

        catalog = await reader.load_subproblem_catalog(scope)
        self.assertEqual(["billing.cancel", "billing.cycle", "members.invite"], [item.key for item in catalog.items])
        self.assertEqual(0, catalog.skipped_count)
        cancel_item = catalog.items[0]
        self.assertEqual(EMBEDDING_DIMENSIONS, len(cancel_item.inclusion_embedding))
        self.assertEqual(("구독 취소을 묻는 질문",), cancel_item.inclusion_criteria)
        self.assertEqual(("가격은 다루지 않는다",), cancel_item.canonical_answer.applicability_rules)

        inputs = await reader.load_gate_inputs(cancel_item.subproblem_id, scope)
        resolutions = resolve_citations(inputs.citations, inputs.contexts_by_source_id)
        self.assertEqual([1, 1], [resolution.step for resolution in resolutions])
        presented = PresentedSubproblem(
            key=cancel_item.key,
            subproblem_id=cancel_item.subproblem_id,
            subproblem_version=cancel_item.current_version,
            problem_group_id=cancel_item.problem_group_id,
            document_source_id=cancel_item.document_source_id,
            document_key=cancel_item.document_key,
            canonical_answer_id=cancel_item.canonical_answer.canonical_answer_id,
            similarity=0.9,
            retrieval_rank=1,
            presented_order=1,
            inclusion_count=1,
            exclusion_count=1,
        )
        judgment = TurnJudgment(
            decision=ClassificationDecision.CONNECT,
            attribution=initial_attribution(ClassificationDecision.CONNECT, subproblem=presented),
            subproblem=presented,
        )
        gate = evaluate_cache_gate(
            judgment,
            subproblem=inputs.subproblem,
            canonical_answer=inputs.canonical_answer,
            citation_resolutions=resolutions,
            semantic_cache_enabled=True,
        )
        self.assertEqual(CacheAttemptOutcome.SERVED, gate.outcome)
        self.assertEqual(cancel_item.canonical_answer.canonical_answer_id, gate.canonical_answer_id)
        logs = served_citation_logs(gate)
        self.assertEqual(
            [
                (1, self.billing_chunks[1], self.billing_v1.id, BILLING_TITLE, f"{BILLING_TITLE} > 구독 변경 또는 취소"),
                (2, self.members_chunks[0], self.members_v1.id, MEMBERS_TITLE, f"{MEMBERS_TITLE} > 멤버 초대"),
            ],
            [
                (log.citation_order, log.chunk_id, log.document_version_id, log.document_title_snapshot, log.node_path_snapshot)
                for log in logs
            ],
        )


if __name__ == "__main__":
    unittest.main()
