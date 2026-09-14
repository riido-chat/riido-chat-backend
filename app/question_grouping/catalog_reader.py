"""질문 판별과 캐시 게이트의 DB 읽기.

- 세부 문제 카탈로그: 문서 그룹의 켜진 문서에 속한 APPROVED 세부 문제, 현재 개정의 포함 기준
  임베딩, 승인 정본. 가이드 밖(NO_DOCUMENT) 문제 그룹의 세부 문제는 읽지 않는다.
- 게이트 입력: 게이트 직전에 다시 읽는 세부 문제 상태, 승인 정본, 인용과 R17 조회 결과.

이 계층은 쓰거나 commit 하지 않는다. ORM 엔티티 대신 칼럼만 골라 읽는다. 같은 세션의
identity map 에 엔티티가 남으면 게이트의 재조회가 앞서 읽은 값을 돌려받을 수 있기 때문이다.
"""

import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    IndexDocument,
    IndexVersion,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    QuestionSubproblemRevision,
    QuestionSubproblemStatus,
)
from app.question_grouping.models import (
    CanonicalCitationSnapshot,
    CatalogCanonicalAnswer,
    CitationIndexContext,
    GateCanonicalAnswer,
    GateInputs,
    GateSubproblemState,
    IndexedSection,
    IndexScope,
    SubproblemCatalog,
    SubproblemCatalogItem,
)
from app.question_grouping.payload import split_criteria

APPLICABILITY_RULES_FIELD = "rules"


class IndexScopeNotFoundError(LookupError):
    """턴의 색인 판을 찾지 못했다."""


class CatalogDataError(RuntimeError):
    """저장된 세부 문제·정본 데이터가 시드 규칙과 맞지 않는다."""


def applicability_rules_from_json(value: Any) -> Tuple[str, ...]:
    """canonical_answers.applicability_rules({"rules": [...]})를 규칙 목록으로 바꾼다.

    널과 빈 객체는 규칙 없음이다. 문자열 외 항목이나 다른 모양은 시드 규칙 위반이다.
    """

    if value is None:
        return ()
    if not isinstance(value, dict):
        raise CatalogDataError("applicability_rules 는 {\"rules\": [...]} 객체여야 합니다.")
    unknown = set(value) - {APPLICABILITY_RULES_FIELD}
    if unknown:
        raise CatalogDataError(f"applicability_rules 에 알 수 없는 키가 있습니다: {sorted(unknown)}")
    rules = value.get(APPLICABILITY_RULES_FIELD)
    if rules is None:
        return ()
    if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
        raise CatalogDataError("applicability_rules.rules 는 문자열 목록이어야 합니다.")
    return tuple(rule.strip() for rule in rules if rule.strip())


def _raise_on_duplicate_keys(rows: Sequence[Any], scope: IndexScope) -> None:
    """같은 문서 그룹 카탈로그에 key 가 두 번 나오면 데이터 오류다.

    DB 유니크는 문제 그룹(문서) 안에서만 key 를 막는다. payload 는 key 로 세부 문제를
    되돌리므로 문서 그룹 안에서 유일해야 한다. 호출 서비스는 판별 실패로 다룬다(fail-open).
    """

    documents_by_key: Dict[str, List[str]] = {}
    for row in rows:
        documents_by_key.setdefault(row.key, []).append(row.document_key)
    duplicates = {
        key: documents for key, documents in documents_by_key.items() if len(documents) > 1
    }
    if duplicates:
        raise CatalogDataError(
            "문서 그룹 안에서 세부 문제 key 가 겹칩니다: "
            f"document_group_id={scope.document_group_id}, "
            f"keys={sorted(duplicates)}"
        )


def _vector(value: Any) -> Tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(component) for component in value)


class QuestionCatalogReader:
    """AsyncSession 으로 세부 문제 카탈로그와 게이트 입력을 읽는다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # 색인 판 범위
    # ------------------------------------------------------------------

    async def load_index_scope(self, index_version_id: int) -> IndexScope:
        """턴 색인 판의 문서 그룹, 청킹 설정, 임베딩 설정. 상태는 보지 않는다(요청 스냅샷)."""

        row = (
            await self._session.execute(
                select(
                    IndexVersion.id,
                    IndexVersion.document_group_id,
                    IndexVersion.chunking_config_id,
                    IndexVersion.embedding_config_id,
                ).where(IndexVersion.id == index_version_id)
            )
        ).one_or_none()
        if row is None:
            raise IndexScopeNotFoundError(
                f"index_version_id={index_version_id} 색인 판이 없습니다."
            )
        return IndexScope(
            index_version_id=row.id,
            document_group_id=row.document_group_id,
            chunking_config_id=row.chunking_config_id,
            embedding_config_id=row.embedding_config_id,
        )

    # ------------------------------------------------------------------
    # 세부 문제 카탈로그
    # ------------------------------------------------------------------

    async def load_subproblem_catalog(self, scope: IndexScope) -> SubproblemCatalog:
        """문서 그룹의 APPROVED 세부 문제를 현재 개정 임베딩, 승인 정본과 함께 읽는다.

        포함 범위:
        - DOCUMENT 문제 그룹의 세부 문제만 읽는다. 문서의 document_group_id 로 문서 그룹에 속한다.
        - NO_DOCUMENT(가이드 밖) 문제 그룹 아래 세부 문제는 읽지 않는다. 정본이 인용할 문서가
          없어 캐시 단위가 될 수 없고, 시드도 만들지 않는다(결정 C).
        - 문서가 꺼져 있으면(document_sources.enabled = false) 그 문서의 세부 문제를 뺀다(결정 E).

        같은 문서 그룹의 서로 다른 문서에 같은 key 가 있으면 판별 payload 의 식별자가 겹치므로
        CatalogDataError 를 올린다(결정 D). 임베딩 누락·설정 불일치로 빼기 전에 검사한다.
        현재 개정에 임베딩이 없거나 턴 색인 판과 다른 임베딩 설정이면 빼고 건수만 센다.
        순서는 (key, id) 다.
        """

        subproblem = QuestionSubproblem
        revision = QuestionSubproblemRevision
        group = QuestionProblemGroup
        statement = (
            select(
                subproblem.id,
                subproblem.key,
                subproblem.name,
                subproblem.inclusion_criteria,
                subproblem.exclusion_criteria,
                subproblem.current_version,
                subproblem.serving_state,
                group.id.label("problem_group_id"),
                DocumentSource.id.label("document_source_id"),
                DocumentSource.document_key,
                DocumentSource.title.label("document_title"),
                revision.id.label("revision_id"),
                revision.inclusion_embedding,
                revision.embedding_config_id,
                revision.embedding_text_version,
                CanonicalAnswer.id.label("canonical_answer_id"),
                CanonicalAnswer.content_markdown,
                CanonicalAnswer.applicability_rules,
            )
            .join(group, group.id == subproblem.problem_group_id)
            .join(DocumentSource, DocumentSource.id == group.document_source_id)
            .outerjoin(
                revision,
                and_(
                    revision.subproblem_id == subproblem.id,
                    revision.version == subproblem.current_version,
                ),
            )
            .outerjoin(
                CanonicalAnswer,
                and_(
                    CanonicalAnswer.subproblem_id == subproblem.id,
                    CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
                ),
            )
            .where(
                subproblem.status == QuestionSubproblemStatus.APPROVED,
                group.kind == QuestionProblemGroupKind.DOCUMENT,
                DocumentSource.document_group_id == scope.document_group_id,
                DocumentSource.enabled.is_(True),
            )
            .order_by(subproblem.key, subproblem.id)
        )
        rows = (await self._session.execute(statement)).all()
        _raise_on_duplicate_keys(rows, scope)

        items: List[SubproblemCatalogItem] = []
        missing = 0
        mismatch = 0
        text_versions: Set[str] = set()
        for row in rows:
            if row.revision_id is None or row.inclusion_embedding is None:
                missing += 1
                continue
            if row.embedding_config_id != scope.embedding_config_id:
                mismatch += 1
                continue
            text_versions.add(row.embedding_text_version)
            items.append(self._catalog_item(row))
        return SubproblemCatalog(
            items=tuple(items),
            skipped_missing_embedding=missing,
            skipped_embedding_config_mismatch=mismatch,
            embedding_text_versions=tuple(sorted(text_versions)),
        )

    @staticmethod
    def _catalog_item(row: Any) -> SubproblemCatalogItem:
        canonical = None
        if row.canonical_answer_id is not None:
            canonical = CatalogCanonicalAnswer(
                canonical_answer_id=row.canonical_answer_id,
                content_markdown=row.content_markdown,
                applicability_rules=applicability_rules_from_json(
                    row.applicability_rules
                ),
            )
        return SubproblemCatalogItem(
            subproblem_id=row.id,
            key=row.key,
            name=row.name,
            inclusion_criteria=split_criteria(row.inclusion_criteria),
            exclusion_criteria=split_criteria(row.exclusion_criteria),
            current_version=row.current_version,
            problem_group_id=row.problem_group_id,
            document_source_id=row.document_source_id,
            document_key=row.document_key,
            serving_state=row.serving_state,
            document_title=row.document_title,
            canonical_answer=canonical,
            inclusion_embedding=_vector(row.inclusion_embedding),
        )

    # ------------------------------------------------------------------
    # 게이트 입력
    # ------------------------------------------------------------------

    async def load_gate_inputs(
        self,
        subproblem_id: uuid.UUID,
        scope: IndexScope,
    ) -> GateInputs:
        """게이트 직전의 세부 문제 상태, 승인 정본, 인용, 인용 문서의 턴 색인 맥락.

        세부 문제는 상태와 무관하게 읽어 게이트가 SUBPROBLEM_NOT_APPROVED 를 판단하게 한다.
        """

        state = await self._load_subproblem_state(subproblem_id)
        if state is None:
            return GateInputs()
        canonical = await self._load_approved_canonical(subproblem_id)
        if canonical is None:
            return GateInputs(subproblem=state)
        citations = await self.load_canonical_citations(canonical.canonical_answer_id)
        contexts = await self.load_citation_contexts(citations, scope)
        return GateInputs(
            subproblem=state,
            canonical_answer=canonical,
            citations=citations,
            contexts_by_source_id=contexts,
        )

    async def _load_subproblem_state(
        self, subproblem_id: uuid.UUID
    ) -> Optional[GateSubproblemState]:
        row = (
            await self._session.execute(
                select(
                    QuestionSubproblem.id,
                    QuestionSubproblem.status,
                    QuestionSubproblem.serving_state,
                    QuestionSubproblem.current_version,
                ).where(QuestionSubproblem.id == subproblem_id)
            )
        ).one_or_none()
        if row is None:
            return None
        return GateSubproblemState(
            subproblem_id=row.id,
            status=row.status,
            serving_state=row.serving_state,
            current_version=row.current_version,
        )

    async def _load_approved_canonical(
        self, subproblem_id: uuid.UUID
    ) -> Optional[GateCanonicalAnswer]:
        # 부분 유니크(uq_canonical_answers_subproblem_id_approved)라 많아야 한 행이다.
        row = (
            await self._session.execute(
                select(
                    CanonicalAnswer.id,
                    CanonicalAnswer.subproblem_version,
                    CanonicalAnswer.content_markdown,
                    CanonicalAnswer.applicability_rules,
                ).where(
                    CanonicalAnswer.subproblem_id == subproblem_id,
                    CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
                )
            )
        ).one_or_none()
        if row is None:
            return None
        return GateCanonicalAnswer(
            canonical_answer_id=row.id,
            subproblem_version=row.subproblem_version,
            content_markdown=row.content_markdown,
            applicability_rules=applicability_rules_from_json(row.applicability_rules),
        )

    async def load_canonical_citations(
        self, canonical_answer_id: uuid.UUID
    ) -> Tuple[CanonicalCitationSnapshot, ...]:
        """정본 인용과 인용 청크의 옛 절 해시. citation_order 순이다.

        canonical_answer_citations.chunk_id → document_chunks.id = content_nodes.id(공유 PK).
        인용의 document_version_id 와 청크 절의 문서 판이 다르면 데이터 오류다.
        """

        rows = (
            await self._session.execute(
                select(
                    CanonicalAnswerCitation.citation_order,
                    CanonicalAnswerCitation.chunk_id,
                    CanonicalAnswerCitation.document_version_id,
                    ContentNode.document_version_id.label("node_document_version_id"),
                    DocumentVersion.document_source_id,
                    ContentNode.content_hash,
                    ContentNode.node_order,
                    ContentNode.node_identity_hash,
                )
                .join(DocumentChunk, DocumentChunk.id == CanonicalAnswerCitation.chunk_id)
                .join(ContentNode, ContentNode.id == DocumentChunk.id)
                .join(
                    DocumentVersion,
                    DocumentVersion.id == CanonicalAnswerCitation.document_version_id,
                )
                .where(CanonicalAnswerCitation.canonical_answer_id == canonical_answer_id)
                .order_by(CanonicalAnswerCitation.citation_order)
            )
        ).all()
        citations = []
        for row in rows:
            if row.node_document_version_id != row.document_version_id:
                raise CatalogDataError(
                    "정본 인용의 문서 판과 인용 청크의 문서 판이 다릅니다: "
                    f"canonical_answer_id={canonical_answer_id}, "
                    f"citation_order={row.citation_order}"
                )
            citations.append(
                CanonicalCitationSnapshot(
                    citation_order=row.citation_order,
                    chunk_id=row.chunk_id,
                    document_version_id=row.document_version_id,
                    document_source_id=row.document_source_id,
                    content_hash=row.content_hash,
                    node_order=row.node_order,
                    node_identity_hash=row.node_identity_hash,
                )
            )
        return tuple(citations)

    async def load_citation_contexts(
        self,
        citations: Sequence[CanonicalCitationSnapshot],
        scope: IndexScope,
    ) -> Dict[int, CitationIndexContext]:
        """인용 문서마다 턴 색인 판에 든 문서 판과 그 판의 후보 절(R17).

        - 문서 판: 같은 document_source 중 턴 색인 판의 index_documents 에 든 판. 색인 판은
          문서마다 판 하나를 담는다. 둘 이상이면 version_no 가 가장 큰 판을 쓴다.
        - 절: 그 판의 content_nodes 중 document_chunks.chunking_config_id 가 턴 색인 판의
          청킹 설정인 것. R17 1~3단계는 모두 content_hash 가 같은 절만 통과시키므로
          (1단계의 같은 청크도 같은 절이다) 인용 절 해시와 같은 절만 읽는다.

        색인에 판이 없는 문서는 indexed_document_version_id=None 인 맥락을 넣는다.
        """

        source_ids = sorted({citation.document_source_id for citation in citations})
        if not source_ids:
            return {}
        indexed = await self._indexed_versions_by_source(source_ids, scope)
        sections = await self._candidate_sections(
            indexed.values(),
            {citation.content_hash for citation in citations},
            scope,
        )
        contexts: Dict[int, CitationIndexContext] = {}
        for source_id in source_ids:
            version_id = indexed.get(source_id)
            contexts[source_id] = CitationIndexContext(
                indexed_document_version_id=version_id,
                sections=tuple(
                    section
                    for section in sections
                    if version_id is not None
                    and section.document_version_id == version_id
                ),
            )
        return contexts

    async def _indexed_versions_by_source(
        self,
        source_ids: Sequence[int],
        scope: IndexScope,
    ) -> Dict[int, int]:
        rows = (
            await self._session.execute(
                select(
                    DocumentVersion.id,
                    DocumentVersion.document_source_id,
                    DocumentVersion.version_no,
                )
                .join(
                    IndexDocument,
                    and_(
                        IndexDocument.document_version_id == DocumentVersion.id,
                        IndexDocument.index_version_id == scope.index_version_id,
                    ),
                )
                .where(DocumentVersion.document_source_id.in_(source_ids))
                .order_by(
                    DocumentVersion.document_source_id,
                    DocumentVersion.version_no.desc(),
                )
            )
        ).all()
        chosen: Dict[int, int] = {}
        for row in rows:
            chosen.setdefault(row.document_source_id, row.id)
        return chosen

    async def _candidate_sections(
        self,
        document_version_ids: Iterable[int],
        content_hashes: Set[str],
        scope: IndexScope,
    ) -> Tuple[IndexedSection, ...]:
        version_ids = sorted(set(document_version_ids))
        if not version_ids or not content_hashes:
            return ()
        rows = (
            await self._session.execute(
                select(
                    ContentNode.id,
                    ContentNode.document_version_id,
                    ContentNode.content_hash,
                    ContentNode.node_order,
                    ContentNode.node_identity_hash,
                    ContentNode.node_path,
                    DocumentSource.title,
                    DocumentSource.canonical_uri,
                )
                .join(DocumentChunk, DocumentChunk.id == ContentNode.id)
                .join(DocumentVersion, DocumentVersion.id == ContentNode.document_version_id)
                .join(DocumentSource, DocumentSource.id == DocumentVersion.document_source_id)
                .where(
                    ContentNode.document_version_id.in_(version_ids),
                    ContentNode.content_hash.in_(sorted(content_hashes)),
                    DocumentChunk.chunking_config_id == scope.chunking_config_id,
                )
                .order_by(
                    ContentNode.document_version_id,
                    ContentNode.node_order,
                    ContentNode.id,
                )
            )
        ).all()
        return tuple(
            IndexedSection(
                chunk_id=row.id,
                document_version_id=row.document_version_id,
                content_hash=row.content_hash,
                node_order=row.node_order,
                node_identity_hash=row.node_identity_hash,
                document_title=row.title,
                node_path=row.node_path,
                source_uri=row.canonical_uri,
            )
            for row in rows
        )
