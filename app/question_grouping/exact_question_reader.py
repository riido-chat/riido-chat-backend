"""Read-only lookup for previously classified exact first-turn questions."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ClassificationDecision,
    DocumentSource,
    IndexVersion,
    QuestionClassification,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    RagRun,
)
from app.question_grouping.models import ExactQuestionLogMatch
from app.question_grouping.exact_question import (
    JUDGMENT_INPUT_SUBPROBLEM_CANDIDATES_FIELD,
    exact_question_hash,
    normalize_exact_question,
    presented_canonical_answer,
)


class ExactQuestionReader:
    """Read the production exact-match source query through an AsyncSession."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_exact_question_log_match(
        self,
        document_group_id: int,
        question: str,
        *,
        exclude_rag_run_id: Optional[uuid.UUID] = None,
    ) -> Optional[ExactQuestionLogMatch]:
        """같은 문서 그룹의 과거 첫 턴 질문 중 정규화 결과가 같은 로그의 최신 CONNECT 분류.

        조회는 첫 턴과 후속 턴 모두에서 사용자 원문으로 부르지만, 매핑 원천은 첫 턴 로그
        (turn_no = 1)뿐이다. 후속 턴 로그는 이전 문맥에 기대어 판별됐을 수 있어 원천으로 쓰지 않는다.
        ``rag_runs.query_hash`` 로 좁히고 현재(effective_to IS NULL) CONNECT 분류 중
        effective_from 이 가장 최근인 행(같으면 분류 id 가 큰 행)의 세부 문제를 쓴다.
        원천 분류가 기록한 세부 문제 개정과 제시 정본도 함께 돌려준다. 지금도 재사용할 수
        있는지는 서비스가 게이트 입력과 비교해 정한다(gate.exact_source_fallthrough_reasons).
        고른 행이 낡았어도 그 아래 옛 행으로 내려가지 않는다. 같은 질문의 최신 판단이 낡았으면
        더 옛 판단도 믿을 근거가 없기 때문이다.
        세부 문제가 서로 달라도 충돌로 보지 않는다. 운영자가 로그 하나를 다시 연결하면
        그 행이 가장 최근의 현재 분류가 되므로, 빠른 경로가 스스로 쓴 행을 포함한 옛 로그를
        모두 다시 연결하지 않아도 바로 반영된다.

        정렬은 SQL 에서 하지만 LIMIT 은 걸지 않는다. 해시만 같고 원문 정규화가 다른 행
        (해시 충돌이나 다른 규칙으로 잘못 채워진 query_hash 방어)이나 소속 재확인에 실패한 행이 맨 위에 있어도 그 아래의 유효한
        옛 행을 고를 수 있어야 하기 때문이다. 같은 해시의 첫 턴 로그만 읽으므로 행 수는 작다.
        """

        normalized = normalize_exact_question(question)
        if not normalized:
            return None
        classification = QuestionClassification
        subproblem = QuestionSubproblem
        problem_group = QuestionProblemGroup
        source = DocumentSource
        statement = (
            select(
                classification.id.label("classification_id"),
                classification.effective_from,
                classification.subproblem_version.label("source_subproblem_version"),
                classification.exact_cache_approved,
                classification.presented_canonical_answer_id.label(
                    "presented_canonical_column"
                ),
                classification.judgment_input[
                    JUDGMENT_INPUT_SUBPROBLEM_CANDIDATES_FIELD
                ]["items"].label("presented_items"),
                RagRun.id.label("rag_run_id"),
                RagRun.user_query,
                subproblem.id.label("subproblem_id"),
                subproblem.key,
                subproblem.problem_group_id,
                subproblem.current_version,
                problem_group.kind,
                problem_group.document_group_id.label("no_document_owner_id"),
                source.id.label("document_source_id"),
                source.document_key,
                source.document_group_id.label("document_owner_id"),
            )
            .select_from(RagRun)
            .join(IndexVersion, IndexVersion.id == RagRun.index_version_id)
            .join(
                classification,
                and_(
                    classification.rag_run_id == RagRun.id,
                    classification.effective_to.is_(None),
                    classification.decision == ClassificationDecision.CONNECT,
                    classification.exact_cache_approved.is_(True),
                ),
            )
            .join(subproblem, subproblem.id == classification.subproblem_id)
            .join(problem_group, problem_group.id == subproblem.problem_group_id)
            .outerjoin(source, source.id == problem_group.document_source_id)
            .where(
                IndexVersion.document_group_id == document_group_id,
                RagRun.turn_no == 1,
                RagRun.query_hash == exact_question_hash(question),
                or_(
                    and_(
                        problem_group.kind == QuestionProblemGroupKind.DOCUMENT,
                        source.document_group_id == document_group_id,
                    ),
                    and_(
                        problem_group.kind == QuestionProblemGroupKind.NO_DOCUMENT,
                        problem_group.document_group_id == document_group_id,
                    ),
                ),
            )
            # 가장 최근에 확정된 현재 분류가 먼저 온다. 같은 시각이면 id 가 큰 행이 이긴다.
            .order_by(classification.effective_from.desc(), classification.id.desc())
        )
        if exclude_rag_run_id is not None:
            statement = statement.where(RagRun.id != exclude_rag_run_id)
        rows = (await self._session.execute(statement)).all()

        chosen: Optional[Any] = None
        matched_count = 0
        for row in rows:
            if normalize_exact_question(row.user_query) != normalized:
                continue
            owner_id = (
                row.document_owner_id
                if row.kind == QuestionProblemGroupKind.DOCUMENT
                else row.no_document_owner_id
            )
            # 위 조회가 이미 거르지만, 조회가 바뀌어도 다른 그룹 세부 문제를 쓰지 않게 한 번 더 막는다.
            if owner_id != document_group_id:
                continue
            matched_count += 1
            if chosen is None:
                chosen = row
        if chosen is None:
            return None
        canonical_recorded, canonical_answer_id = presented_canonical_answer(
            chosen.presented_items, chosen.subproblem_id
        )
        # 제시 목록이 없어도(subproblemCandidates 가 없거나 JSON null) 판별 행 칸에 제시 정본이
        # 있으면 그 값을 쓴다. 칸이 널이면 정본 없이 판별했는지 알 수 없어 기록 없음으로 둔다.
        if not canonical_recorded and chosen.presented_canonical_column is not None:
            canonical_recorded = True
            canonical_answer_id = chosen.presented_canonical_column
        return ExactQuestionLogMatch(
            subproblem_id=chosen.subproblem_id,
            key=chosen.key,
            problem_group_id=chosen.problem_group_id,
            current_version=chosen.current_version,
            document_source_id=chosen.document_source_id,
            document_key=chosen.document_key,
            normalized_question=normalized,
            source_rag_run_id=chosen.rag_run_id,
            classification_id=chosen.classification_id,
            matched_count=matched_count,
            source_subproblem_version=chosen.source_subproblem_version,
            source_canonical_recorded=canonical_recorded,
            source_canonical_answer_id=canonical_answer_id,
            source_exact_cache_approved=chosen.exact_cache_approved,
            source_effective_from=chosen.effective_from,
        )
