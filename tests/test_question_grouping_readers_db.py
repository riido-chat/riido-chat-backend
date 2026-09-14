"""질문 판별 DB 읽기(catalog_reader, outline_reader) 로컬 DB 통합 테스트.

- 로컬 PostgreSQL 과 head 마이그레이션이 필요하다. 연결할 수 없으면 skip 한다.
- 외부 트랜잭션 하나에서 시드하고 마지막에 rollback 해 데이터를 남기지 않는다.

실행: DATABASE_URL=postgresql+asyncpg://riido:riido@localhost:5433/riido \
    .venv/bin/python -m unittest tests.test_question_grouping_readers_db -v
"""

import asyncio
import unittest
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    CacheAttemptOutcome,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    CanonicalAnswerOrigin,
    ChunkingConfig,
    ClassificationDecision,
    ContentNode,
    DocumentChunk,
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    IndexDocument,
    IndexVersion,
    IndexVersionStatus,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    QuestionSubproblemRevision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.question_grouping.attribution import initial_attribution
from app.question_grouping.catalog_reader import (
    CatalogDataError,
    IndexScopeNotFoundError,
    QuestionCatalogReader,
)
from app.question_grouping.constants import (
    REJECT_CITED_DOCUMENT_NOT_INDEXED,
    REJECT_CITED_SECTION_CHANGED,
    SUBPROBLEM_EMBEDDING_TEXT_VERSION,
)
from app.question_grouping.gate import evaluate_cache_gate, resolve_citations
from app.question_grouping.models import (
    IndexScope,
    PresentedSubproblem,
    TurnJudgment,
)
from app.question_grouping.outline_reader import (
    DocumentOutlineCache,
    DocumentOutlineReader,
)
from app.question_grouping.subproblem_search import rank_subproblems


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _vector(*weights: Tuple[int, float]) -> List[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for index, value in weights:
        vector[index] = value
    return vector


async def _available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


# 절 한 건: (신원 해시, 내용 해시, H2 제목, 본문)
Section = Tuple[str, str, str, str]


class _Seed:
    """문서 그룹 하나의 문서·색인·세부 문제 체인을 만든다."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.suffix = uuid.uuid4().hex[:10]
        self._counter = 0

    def _name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self.suffix}-{self._counter}"

    async def add(self, *rows: object) -> None:
        self.session.add_all(rows)
        await self.session.flush()

    async def chunking(self) -> ChunkingConfig:
        row = ChunkingConfig(
            version=self._name("chunk"), strategy="SECTION", max_tokens=512, created_at=_now()
        )
        await self.add(row)
        return row

    async def embedding(self) -> EmbeddingConfig:
        row = EmbeddingConfig(
            version=self._name("embed"),
            provider="openai",
            model_name="text-embedding-test",
            dimensions=EMBEDDING_DIMENSIONS,
            input_template_version="v1",
            created_at=_now(),
        )
        await self.add(row)
        return row

    async def group(self) -> DocumentGroup:
        row = DocumentGroup(group_key=self._name("G")[:50], name="판별 읽기 테스트", consumer_key="TEST")
        await self.add(row)
        return row

    async def source(self, group: DocumentGroup, key: str, title: str) -> DocumentSource:
        row = DocumentSource(
            document_group_id=group.id,
            document_key=f"{key}-{self.suffix}",
            source_type="LLMS_TXT",
            canonical_uri=f"https://docs.riido.io/{key}-{self.suffix}",
            title=title,
            created_at=_now(),
            updated_at=_now(),
        )
        await self.add(row)
        return row

    async def version(self, source: DocumentSource, version_no: int) -> DocumentVersion:
        row = DocumentVersion(
            document_source_id=source.id,
            version_no=version_no,
            raw_content_uri=f"raw/{source.id}/{version_no}.md",
            mime_type="text/markdown",
            raw_content_hash=self._name("raw"),
            normalized_content_hash=self._name("norm"),
            parser_name="markdown",
            parser_version="1",
            status=DocumentVersionStatus.READY,
            collected_at=_now(),
            created_at=_now(),
        )
        await self.add(row)
        return row

    async def sections(
        self,
        version: DocumentVersion,
        title: str,
        chunking: ChunkingConfig,
        sections: Sequence[Section],
    ) -> List[int]:
        nodes = []
        for order, (identity, content, heading, body) in enumerate(sections):
            path = [title, heading] if heading else [title]
            nodes.append(
                ContentNode(
                    document_version_id=version.id,
                    node_type="SECTION",
                    node_path=" > ".join(path),
                    node_order=order,
                    title=path[-1],
                    normalized_content=body,
                    content_hash=content,
                    node_identity_hash=identity,
                    node_identity_kind="path",
                    metadata_={"document_id": "doc", "section_id": identity, "section_path": path},
                    created_at=_now(),
                )
            )
        await self.add(*nodes)
        await self.add(
            *[
                DocumentChunk(id=node.id, chunking_config_id=chunking.id, chunk_index=index, created_at=_now())
                for index, node in enumerate(nodes)
            ]
        )
        return [node.id for node in nodes]

    async def index(
        self,
        group: DocumentGroup,
        chunking: ChunkingConfig,
        embedding: EmbeddingConfig,
        versions: Sequence[DocumentVersion],
    ) -> IndexScope:
        row = IndexVersion(
            document_group_id=group.id,
            version=self._name("index"),
            status=IndexVersionStatus.READY,
            chunking_config_id=chunking.id,
            embedding_config_id=embedding.id,
            created_at=_now(),
        )
        await self.add(row)
        await self.add(
            *[IndexDocument(index_version_id=row.id, document_version_id=version.id) for version in versions]
        )
        return IndexScope(
            index_version_id=row.id,
            document_group_id=group.id,
            chunking_config_id=chunking.id,
            embedding_config_id=embedding.id,
        )

    async def document_group(self, group: DocumentGroup, source: DocumentSource) -> QuestionProblemGroup:
        row = QuestionProblemGroup(kind=QuestionProblemGroupKind.DOCUMENT, document_source_id=source.id)
        await self.add(row)
        return row

    async def no_document_group(self, group: DocumentGroup) -> QuestionProblemGroup:
        row = QuestionProblemGroup(kind=QuestionProblemGroupKind.NO_DOCUMENT, document_group_id=group.id)
        await self.add(row)
        return row

    async def subproblem(
        self,
        problem_group: QuestionProblemGroup,
        key: str,
        *,
        status: QuestionSubproblemStatus = QuestionSubproblemStatus.APPROVED,
        serving_state: QuestionSubproblemServingState = QuestionSubproblemServingState.SERVING,
        current_version: int = 1,
        embedding: Optional[EmbeddingConfig] = None,
        vector: Optional[List[float]] = None,
        revisions: Sequence[Tuple[int, Optional[EmbeddingConfig], Optional[List[float]]]] = (),
    ) -> QuestionSubproblem:
        row = QuestionSubproblem(
            problem_group_id=problem_group.id,
            key=key,
            name=f"{key} 이름",
            inclusion_criteria=f"{key} 기준 하나\n- {key} 기준 둘\n",
            exclusion_criteria=f"{key} 제외",
            current_version=current_version,
            status=status,
            serving_state=serving_state,
            created_by="test",
        )
        await self.add(row)
        all_revisions = list(revisions) or [(current_version, embedding, vector)]
        await self.add(
            *[
                QuestionSubproblemRevision(
                    subproblem_id=row.id,
                    version=number,
                    name_snapshot=row.name,
                    inclusion_snapshot=row.inclusion_criteria,
                    inclusion_embedding=revision_vector if config is not None else None,
                    embedding_config_id=None if config is None else config.id,
                    embedding_text_version=None if config is None else SUBPROBLEM_EMBEDDING_TEXT_VERSION,
                    approved_by="test",
                )
                for number, config, revision_vector in all_revisions
            ]
        )
        return row

    async def canonical(
        self,
        subproblem: QuestionSubproblem,
        citations: Sequence[Tuple[int, int]],
        *,
        approval: CanonicalAnswerApproval = CanonicalAnswerApproval.APPROVED,
        rules: Optional[dict] = None,
        subproblem_version: int = 1,
    ) -> CanonicalAnswer:
        row = CanonicalAnswer(
            subproblem_id=subproblem.id,
            origin=CanonicalAnswerOrigin.AUTHORED,
            content_markdown=f"{subproblem.key} 정본 [1]",
            applicability_rules=rules,
            subproblem_version=subproblem_version,
            approval=approval,
            approved_by="test",
        )
        await self.add(row)
        await self.add(
            *[
                CanonicalAnswerCitation(
                    canonical_answer_id=row.id,
                    citation_order=order,
                    chunk_id=chunk_id,
                    document_version_id=version_id,
                    document_title_snapshot="옛 제목",
                    node_path_snapshot="옛 경로",
                    source_uri_snapshot="https://old",
                )
                for order, (chunk_id, version_id) in enumerate(citations, 1)
            ]
        )
        return row


class _DbTestCase(unittest.IsolatedAsyncioTestCase):
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
        self.reader = QuestionCatalogReader(self.session)

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()


class SubproblemCatalogDbTest(_DbTestCase):
    async def test_loads_approved_subproblems_of_enabled_documents_in_group(self) -> None:
        seed = self.seed
        chunking = await seed.chunking()
        embed, other_embed = await seed.embedding(), await seed.embedding()
        group, other_group = await seed.group(), await seed.group()
        billing = await seed.source(group, "workspaces/plans-and-billing", "구독 및 결제")
        version = await seed.version(billing, 1)
        chunks = await seed.sections(version, "구독 및 결제", chunking, [("id-1", "c-1", "구독 취소", "본문")])
        scope = await seed.index(group, chunking, embed, [version])
        other_source = await seed.source(other_group, "other/doc", "다른 그룹 문서")

        document_group = await seed.document_group(group, billing)
        no_document = await seed.no_document_group(group)
        other_problem_group = await seed.document_group(other_group, other_source)

        cancel = await seed.subproblem(
            document_group,
            "billing.cancel",
            current_version=2,
            revisions=[(1, other_embed, _vector((5, 1.0))), (2, embed, _vector((0, 1.0)))],
        )
        await seed.canonical(cancel, [(chunks[0], version.id)], approval=CanonicalAnswerApproval.REVOKED)
        approved = await seed.canonical(
            cancel,
            [(chunks[0], version.id)],
            rules={"rules": ["관리자만", "유료 플랜"]},
            subproblem_version=2,
        )
        # 가이드 밖 문제 그룹의 세부 문제는 임베딩이 있어도 카탈로그에 넣지 않는다(결정 C).
        await seed.subproblem(no_document, "outside.pricing", embedding=embed, vector=_vector((1, 1.0)))
        await seed.subproblem(document_group, "billing.draft", status=QuestionSubproblemStatus.DRAFT, embedding=embed, vector=_vector((0, 1.0)))
        await seed.subproblem(document_group, "billing.no-embedding")
        await seed.subproblem(document_group, "billing.old-config", embedding=other_embed, vector=_vector((0, 1.0)))
        await seed.subproblem(other_problem_group, "other.approved", embedding=embed, vector=_vector((0, 1.0)))
        await seed.subproblem(
            document_group,
            "billing.archived",
            status=QuestionSubproblemStatus.ARCHIVED,
            embedding=embed,
            vector=_vector((0, 1.0)),
        )

        self.assertEqual(scope, await self.reader.load_index_scope(scope.index_version_id))
        catalog = await self.reader.load_subproblem_catalog(scope)

        self.assertEqual(["billing.cancel"], [item.key for item in catalog.items])
        self.assertEqual(1, catalog.skipped_missing_embedding)
        self.assertEqual(1, catalog.skipped_embedding_config_mismatch)
        self.assertEqual(2, catalog.skipped_count)
        self.assertEqual((SUBPROBLEM_EMBEDDING_TEXT_VERSION,), catalog.embedding_text_versions)

        (cancel_item,) = catalog.items
        self.assertEqual(cancel.id, cancel_item.subproblem_id)
        self.assertEqual("billing.cancel 이름", cancel_item.name)
        self.assertEqual(("billing.cancel 기준 하나", "billing.cancel 기준 둘"), cancel_item.inclusion_criteria)
        self.assertEqual(("billing.cancel 제외",), cancel_item.exclusion_criteria)
        self.assertEqual(2, cancel_item.current_version)
        self.assertEqual(document_group.id, cancel_item.problem_group_id)
        self.assertEqual(billing.id, cancel_item.document_source_id)
        self.assertEqual(billing.document_key, cancel_item.document_key)
        self.assertEqual("구독 및 결제", cancel_item.document_title)
        self.assertEqual(QuestionSubproblemServingState.SERVING, cancel_item.serving_state)
        self.assertEqual(approved.id, cancel_item.canonical_answer.canonical_answer_id)
        self.assertEqual(("관리자만", "유료 플랜"), cancel_item.canonical_answer.applicability_rules)
        self.assertEqual(EMBEDDING_DIMENSIONS, len(cancel_item.inclusion_embedding))
        self.assertEqual(1.0, cancel_item.inclusion_embedding[0])
        self.assertEqual(0.0, cancel_item.inclusion_embedding[5])

        # 질문 벡터는 호출자가 넘긴다.
        ranked = rank_subproblems(_vector((1, 1.0), (0, 0.5)), catalog.items)
        self.assertEqual(["billing.cancel"], [c.item.key for c in ranked])

    async def test_excludes_subproblems_of_disabled_documents(self) -> None:
        seed = self.seed
        chunking = await seed.chunking()
        embed = await seed.embedding()
        group = await seed.group()
        enabled = await seed.source(group, "workspaces/plans-and-billing", "구독 및 결제")
        disabled = await seed.source(group, "workspaces/old-billing", "옛 결제")
        disabled.enabled = False
        await self.session.flush()
        version = await seed.version(enabled, 1)
        await seed.sections(version, "구독 및 결제", chunking, [("id-1", "c-1", "구독 취소", "본문")])
        scope = await seed.index(group, chunking, embed, [version])
        await seed.subproblem(
            await seed.document_group(group, enabled), "billing.cancel", embedding=embed, vector=_vector((0, 1.0))
        )
        disabled_group = await seed.document_group(group, disabled)
        await seed.subproblem(disabled_group, "billing.old", embedding=embed, vector=_vector((0, 1.0)))
        # 꺼진 문서의 key 는 켜진 문서의 key 와 겹쳐도 카탈로그 밖이라 오류가 아니다.
        await seed.subproblem(disabled_group, "billing.cancel", embedding=embed, vector=_vector((0, 1.0)))

        catalog = await self.reader.load_subproblem_catalog(scope)

        self.assertEqual(["billing.cancel"], [item.key for item in catalog.items])
        self.assertEqual(enabled.id, catalog.items[0].document_source_id)
        self.assertEqual(0, catalog.skipped_count)

    async def test_duplicate_key_across_documents_of_group_raises(self) -> None:
        seed = self.seed
        chunking = await seed.chunking()
        embed = await seed.embedding()
        group, other_group = await seed.group(), await seed.group()
        billing = await seed.source(group, "workspaces/plans-and-billing", "구독 및 결제")
        members = await seed.source(group, "workspaces/members", "멤버")
        other = await seed.source(other_group, "workspaces/plans-and-billing", "다른 그룹 결제")
        version = await seed.version(billing, 1)
        await seed.sections(version, "구독 및 결제", chunking, [("id-1", "c-1", "구독 취소", "본문")])
        scope = await seed.index(group, chunking, embed, [version])
        await seed.subproblem(
            await seed.document_group(group, billing), "shared.key", embedding=embed, vector=_vector((0, 1.0))
        )
        # 다른 문서 그룹의 같은 key 는 겹침이 아니다.
        await seed.subproblem(
            await seed.document_group(other_group, other), "shared.key", embedding=embed, vector=_vector((0, 1.0))
        )
        self.assertEqual(["shared.key"], [item.key for item in (await self.reader.load_subproblem_catalog(scope)).items])

        # 임베딩이 없어 빠질 세부 문제여도 key 가 겹치면 데이터 오류다.
        await seed.subproblem(await seed.document_group(group, members), "shared.key")

        with self.assertRaises(CatalogDataError) as raised:
            await self.reader.load_subproblem_catalog(scope)
        self.assertIn("shared.key", str(raised.exception))

    async def test_missing_index_scope_raises(self) -> None:
        with self.assertRaises(IndexScopeNotFoundError):
            await self.reader.load_index_scope(-1)


class GateInputsDbTest(_DbTestCase):
    """R17 시나리오마다 턴 색인 판을 따로 만들고 같은 정본 인용을 해석한다."""

    TITLE = "구독 및 결제"
    V1_SECTIONS: Sequence[Section] = (
        ("id-intro", "c-intro", "", "머리말"),
        ("id-cancel", "c-cancel", "구독 취소", "취소 본문"),
        ("id-refund", "c-refund", "환불", "환불 본문"),
    )

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        seed = self.seed
        self.chunking = await seed.chunking()
        self.embed = await seed.embedding()
        self.group = await seed.group()
        self.billing = await seed.source(self.group, "workspaces/plans-and-billing", self.TITLE)
        self.other = await seed.source(self.group, "notifications/settings", "알림 설정")
        self.v1 = await seed.version(self.billing, 1)
        self.v1_chunks = await seed.sections(self.v1, self.TITLE, self.chunking, self.V1_SECTIONS)
        self.other_v1 = await seed.version(self.other, 1)
        await seed.sections(self.other_v1, "알림 설정", self.chunking, [("id-n", "c-n", "알림 끄기", "본문")])
        problem_group = await seed.document_group(self.group, self.billing)
        self.subproblem = await seed.subproblem(problem_group, "billing.cancel", embedding=self.embed, vector=_vector((0, 1.0)))
        self.canonical = await seed.canonical(
            self.subproblem, [(self.v1_chunks[1], self.v1.id)], rules={"rules": ["관리자만"]}
        )

    async def _resolve(self, scope: IndexScope):
        inputs = await self.reader.load_gate_inputs(self.subproblem.id, scope)
        return inputs, resolve_citations(inputs.citations, inputs.contexts_by_source_id)

    async def _new_version(self, sections: Sequence[Section], version_no: int = 2):
        version = await self.seed.version(self.billing, version_no)
        chunks = await self.seed.sections(version, self.TITLE, self.chunking, sections)
        return version, chunks

    async def test_reads_state_canonical_and_citation_snapshot(self) -> None:
        scope = await self.seed.index(self.group, self.chunking, self.embed, [self.v1, self.other_v1])

        inputs, _ = await self._resolve(scope)

        self.assertEqual(self.subproblem.id, inputs.subproblem.subproblem_id)
        self.assertEqual(QuestionSubproblemStatus.APPROVED, inputs.subproblem.status)
        self.assertEqual(QuestionSubproblemServingState.SERVING, inputs.subproblem.serving_state)
        self.assertEqual(1, inputs.subproblem.current_version)
        self.assertEqual(self.canonical.id, inputs.canonical_answer.canonical_answer_id)
        self.assertEqual(1, inputs.canonical_answer.subproblem_version)
        self.assertEqual("billing.cancel 정본 [1]", inputs.canonical_answer.content_markdown)
        self.assertEqual(("관리자만",), inputs.canonical_answer.applicability_rules)
        (citation,) = inputs.citations
        self.assertEqual(
            (1, self.v1_chunks[1], self.v1.id, self.billing.id, "c-cancel", 1, "id-cancel"),
            (
                citation.citation_order,
                citation.chunk_id,
                citation.document_version_id,
                citation.document_source_id,
                citation.content_hash,
                citation.node_order,
                citation.node_identity_hash,
            ),
        )
        # 인용 문서만 맥락을 읽는다.
        self.assertEqual([self.billing.id], list(inputs.contexts_by_source_id))

    async def test_a_cited_version_still_indexed_uses_same_chunk(self) -> None:
        scope = await self.seed.index(self.group, self.chunking, self.embed, [self.v1, self.other_v1])

        inputs, (resolution,) = await self._resolve(scope)

        context = inputs.contexts_by_source_id[self.billing.id]
        self.assertEqual(self.v1.id, context.indexed_document_version_id)
        self.assertEqual([self.v1_chunks[1]], [section.chunk_id for section in context.sections])
        self.assertEqual(1, resolution.step)
        section = resolution.section
        self.assertEqual(
            (self.v1_chunks[1], self.v1.id, self.TITLE, f"{self.TITLE} > 구독 취소", self.billing.canonical_uri),
            (section.chunk_id, section.document_version_id, section.document_title, section.node_path, section.source_uri),
        )

        judgment = self._connect_judgment()
        gate = evaluate_cache_gate(
            judgment,
            subproblem=inputs.subproblem,
            canonical_answer=inputs.canonical_answer,
            citation_resolutions=(resolution,),
            semantic_cache_enabled=True,
        )
        self.assertEqual(CacheAttemptOutcome.SERVED, gate.outcome)
        self.assertEqual(self.canonical.id, gate.canonical_answer_id)
        self.assertEqual(self.v1_chunks[1], gate.served_citations[0].section.chunk_id)

    async def test_b_new_version_with_unchanged_section(self) -> None:
        v2, v2_chunks = await self._new_version(
            [
                ("id-intro", "c-intro-2", "", "바뀐 머리말"),
                ("id-new", "c-new", "새 절", "새 본문"),
                ("id-cancel", "c-cancel", "구독 취소", "취소 본문"),
            ]
        )
        scope = await self.seed.index(self.group, self.chunking, self.embed, [v2, self.other_v1])

        inputs, (resolution,) = await self._resolve(scope)

        self.assertEqual(v2.id, inputs.contexts_by_source_id[self.billing.id].indexed_document_version_id)
        self.assertEqual(2, resolution.step)
        self.assertEqual((v2_chunks[2], v2.id), (resolution.section.chunk_id, resolution.section.document_version_id))

    async def test_c_renamed_section_with_same_content(self) -> None:
        v2, v2_chunks = await self._new_version(
            [
                ("id-intro", "c-intro", "", "머리말"),
                ("id-refund", "c-refund", "환불", "환불 본문"),
                ("id-cancel-renamed", "c-cancel", "구독 해지", "취소 본문"),
                ("id-cancel-copy", "c-cancel", "부록", "취소 본문"),
            ]
        )
        scope = await self.seed.index(self.group, self.chunking, self.embed, [v2])

        _, (resolution,) = await self._resolve(scope)

        self.assertEqual(3, resolution.step)
        # 옛 node_order 1 과 가장 가까운 같은 내용 절(2)을 고른다.
        self.assertEqual(v2_chunks[2], resolution.section.chunk_id)
        self.assertEqual(f"{self.TITLE} > 구독 해지", resolution.section.node_path)

    async def test_d_changed_section_content_fails(self) -> None:
        v2, _ = await self._new_version(
            [
                ("id-intro", "c-intro", "", "머리말"),
                ("id-cancel", "c-cancel-2", "구독 취소", "바뀐 취소 본문"),
            ]
        )
        scope = await self.seed.index(self.group, self.chunking, self.embed, [v2])

        inputs, (resolution,) = await self._resolve(scope)

        context = inputs.contexts_by_source_id[self.billing.id]
        self.assertEqual(v2.id, context.indexed_document_version_id)
        self.assertEqual((), context.sections)
        self.assertFalse(resolution.passed)
        self.assertEqual(REJECT_CITED_SECTION_CHANGED, resolution.rejection_reason)

        gate = evaluate_cache_gate(
            self._connect_judgment(),
            subproblem=inputs.subproblem,
            canonical_answer=inputs.canonical_answer,
            citation_resolutions=(resolution,),
            semantic_cache_enabled=True,
        )
        self.assertEqual(CacheAttemptOutcome.REJECTED, gate.outcome)
        self.assertEqual((REJECT_CITED_SECTION_CHANGED,), gate.rejection_reasons)

    async def test_e_document_not_in_index_fails(self) -> None:
        # 옛 판 v1 은 그대로 있지만 턴 색인 판에 들지 않았다.
        scope = await self.seed.index(self.group, self.chunking, self.embed, [self.other_v1])

        inputs, (resolution,) = await self._resolve(scope)

        context = inputs.contexts_by_source_id[self.billing.id]
        self.assertIsNone(context.indexed_document_version_id)
        self.assertEqual((), context.sections)
        self.assertEqual(REJECT_CITED_DOCUMENT_NOT_INDEXED, resolution.rejection_reason)

    async def test_f_changed_chunking_config_of_same_version(self) -> None:
        rechunking = await self.seed.chunking()
        new_chunks = await self.seed.sections(self.v1, self.TITLE, rechunking, self.V1_SECTIONS)
        scope = await self.seed.index(self.group, rechunking, self.embed, [self.v1])

        inputs, (resolution,) = await self._resolve(scope)

        context = inputs.contexts_by_source_id[self.billing.id]
        self.assertEqual(self.v1.id, context.indexed_document_version_id)
        # 옛 청킹 설정의 인용 청크는 후보 절에 없다.
        self.assertEqual([new_chunks[1]], [section.chunk_id for section in context.sections])
        self.assertEqual(2, resolution.step)
        self.assertEqual((new_chunks[1], self.v1.id), (resolution.section.chunk_id, resolution.section.document_version_id))

    async def test_multiple_indexed_versions_pick_latest(self) -> None:
        v2, v2_chunks = await self._new_version(self.V1_SECTIONS)
        scope = await self.seed.index(self.group, self.chunking, self.embed, [self.v1, v2])

        inputs, (resolution,) = await self._resolve(scope)

        self.assertEqual(v2.id, inputs.contexts_by_source_id[self.billing.id].indexed_document_version_id)
        self.assertEqual((2, v2_chunks[1]), (resolution.step, resolution.section.chunk_id))

    async def test_missing_subproblem_and_canonical(self) -> None:
        scope = await self.seed.index(self.group, self.chunking, self.embed, [self.v1])

        empty = await self.reader.load_gate_inputs(uuid.uuid4(), scope)
        self.assertIsNone(empty.subproblem)
        self.assertIsNone(empty.canonical_answer)
        self.assertEqual((), empty.citations)

        problem_group = await self.seed.no_document_group(self.group)
        lonely = await self.seed.subproblem(
            problem_group, "outside.lonely", serving_state=QuestionSubproblemServingState.STOPPED
        )
        inputs = await self.reader.load_gate_inputs(lonely.id, scope)
        self.assertEqual(QuestionSubproblemServingState.STOPPED, inputs.subproblem.serving_state)
        self.assertIsNone(inputs.canonical_answer)
        self.assertEqual({}, dict(inputs.contexts_by_source_id))

    def _connect_judgment(self) -> TurnJudgment:
        presented = PresentedSubproblem(
            key="billing.cancel",
            subproblem_id=self.subproblem.id,
            subproblem_version=1,
            problem_group_id=self.subproblem.problem_group_id,
            document_source_id=self.billing.id,
            document_key=self.billing.document_key,
            canonical_answer_id=self.canonical.id,
            similarity=0.9,
            retrieval_rank=1,
            presented_order=1,
            inclusion_count=2,
            exclusion_count=1,
        )
        return TurnJudgment(
            decision=ClassificationDecision.CONNECT,
            attribution=initial_attribution(ClassificationDecision.CONNECT, subproblem=presented),
            subproblem=presented,
        )


class DocumentOutlineReaderDbTest(_DbTestCase):
    async def test_builds_outline_from_content_nodes_of_chunking_config(self) -> None:
        seed = self.seed
        chunking, rechunking = await seed.chunking(), await seed.chunking()
        group = await seed.group()
        source = await seed.source(group, "workspaces/plans-and-billing", "구독 및 결제")
        version = await seed.version(source, 1)
        sections: Sequence[Section] = (
            ("id-intro", "c-intro", "", "머리말"),
            ("id-cancel", "c-cancel", "구독 취소", "### 모바일에서\n본문\n### 웹에서\n본문"),
            ("id-refund", "c-refund", "환불", "환불 본문"),
        )
        await seed.sections(version, "구독 및 결제", chunking, sections)
        await seed.sections(version, "구독 및 결제", rechunking, sections[:2])
        cache = DocumentOutlineCache()
        reader = DocumentOutlineReader(self.session, cache=cache)

        outlines = await reader.load_outlines([version.id, -1], chunking_config_id=chunking.id)

        self.assertEqual([version.id], list(outlines))
        outline = outlines[version.id]
        self.assertEqual(source.id, outline.document_source_id)
        self.assertEqual(source.document_key, outline.document_key)
        self.assertEqual("구독 및 결제", outline.title)
        self.assertEqual("workspaces", outline.parent_path)
        self.assertEqual(
            ("구독 취소", "구독 취소 > 모바일에서", "구독 취소 > 웹에서", "환불"),
            outline.headings,
        )
        self.assertIn((version.id, chunking.id), cache)

        rechunked = await reader.load_outlines([version.id], chunking_config_id=rechunking.id)
        self.assertEqual(
            ("구독 취소", "구독 취소 > 모바일에서", "구독 취소 > 웹에서"),
            rechunked[version.id].headings,
        )
        # 절이 없는 청킹 설정은 결과에서 빠지고 캐시하지 않는다.
        unknown = await seed.chunking()
        self.assertEqual({}, await reader.load_outlines([version.id], chunking_config_id=unknown.id))
        self.assertNotIn((version.id, unknown.id), cache)
        self.assertEqual(2, len(cache))


if __name__ == "__main__":
    unittest.main()
