"""질문 판별과 캐시 시도의 DB 쓰기.

- 열린 ONLINE 분류 실행 조회/생성
- 문서 그룹별 과거 첫 턴 질문 로그 정확 일치 조회(가장 최근의 현재 CONNECT 분류)
- 질문 임베딩, 판별 행(question_classifications), 캐시 시도(question_cache_attempts)
- 문제 그룹 ensure(INSERT … ON CONFLICT DO NOTHING 후 SELECT)
- 턴 끝 인용 귀속 갱신(problem_group_id, attribution_source 두 칸만)
- SERVED 턴의 answer_citations 입력(CitationLog) 구성

트랜잭션 규칙:
- 이 계층은 flush 까지만 하고 commit 하지 않는다. 트랜잭션 경계는 호출자가 정한다.
- 턴에 속한 쓰기는 RagLogStore 와 같은 순서(conversation → rag_run → 나머지)로 잠그고,
  PROCESSING 이 아닌 턴에는 쓰지 않는다(ValueError).
- 판별 행은 한 번 쓰고 judgment_input 은 고치지 않는다. 게이트 결과도 insert 전에
  judgment_input.gate 에 담는다(결정 A). 턴 끝 갱신은 두 칸만 바꾼다.
- 분류 실행의 row_count 는 0 으로 두고 갱신하지 않는다(결정 F). 개수는 조회 시 센다.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.log_store import MIN_CITATIONS, CitationLog, RagLogStore
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    OPEN_ONLINE_CLASSIFICATION_RUN_CONSTRAINT,
    AttributionSource,
    CacheAttemptOutcome,
    ClassificationDecision,
    ClassificationRun,
    ClassificationRunKind,
    DocumentSource,
    DocumentVersion,
    IndexVersion,
    QuestionCacheAttempt,
    QuestionClassification,
    QuestionEmbedding,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    RagRun,
)
from app.question_grouping.attribution import citation_attribution
from app.question_grouping.models import (
    AttributionTarget,
    CitedDocument,
    ExactQuestionLogMatch,
    GateResult,
    TurnJudgment,
)
from app.question_grouping.exact_question import (
    exact_question_hash,
    normalize_exact_question,
)

# question_cache_attempts.rejection_reasons 원소 길이(VARCHAR(50)).
MAX_REJECTION_REASON_LENGTH = 50
# 판별 행 judgment_input 에 반드시 있어야 하는 게이트 칸(결정 A).
JUDGMENT_INPUT_GATE_FIELD = "gate"

CANONICAL_OUTCOMES = frozenset(
    {
        CacheAttemptOutcome.SERVED,
        CacheAttemptOutcome.SHADOW,
        CacheAttemptOutcome.GROUP_DISABLED,
    }
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _violates_constraint(exc: IntegrityError, constraint_name: str) -> bool:
    """asyncpg 원 예외의 constraint_name 으로 어떤 유니크를 어겼는지 확인한다."""

    orig = exc.orig
    for candidate in (orig, getattr(orig, "__cause__", None)):
        if getattr(candidate, "constraint_name", None) == constraint_name:
            return True
    return constraint_name in str(orig)


# ---------------------------------------------------------------------------
# 순수 검증과 변환
# ---------------------------------------------------------------------------


def validate_turn_judgment(judgment: TurnJudgment) -> None:
    """판별 행 CHECK(connect_subproblem, connect_attribution, subproblem_version)와
    초기 귀속 규칙을 insert 전에 확인한다. 위반이면 ValueError.
    """

    attribution = judgment.attribution
    source = attribution.attribution_source
    is_connect = judgment.decision == ClassificationDecision.CONNECT
    if judgment.failed and judgment.decision != ClassificationDecision.UNCLASSIFIED:
        raise ValueError("판별 실패 행의 decision 은 UNCLASSIFIED 여야 합니다.")
    if judgment.failed and source != AttributionSource.NONE:
        raise ValueError("판별 실패 행의 초기 귀속은 NONE 이어야 합니다.")
    if is_connect != (judgment.subproblem is not None):
        raise ValueError("CONNECT 와 세부 문제는 함께 있거나 함께 없어야 합니다.")
    if is_connect != (source == AttributionSource.SUBPROBLEM):
        raise ValueError("CONNECT 와 SUBPROBLEM 귀속은 함께 있거나 함께 없어야 합니다.")
    if source == AttributionSource.CITATION:
        raise ValueError("CITATION 귀속은 턴 끝 갱신으로만 씁니다.")
    if source == AttributionSource.SUBPROBLEM:
        subproblem = judgment.subproblem
        if (
            attribution.problem_group_id is not None
            and subproblem is not None
            and attribution.problem_group_id != subproblem.problem_group_id
        ):
            raise ValueError("SUBPROBLEM 귀속의 문제 그룹이 세부 문제의 문제 그룹과 다릅니다.")
    if source == AttributionSource.DOCUMENT and (
        attribution.document_source_id is None
        or attribution.group_kind != QuestionProblemGroupKind.DOCUMENT
    ):
        raise ValueError("DOCUMENT 귀속에는 문서 id 가 필요합니다.")
    if (
        source == AttributionSource.NONE
        and attribution.group_kind != QuestionProblemGroupKind.NO_DOCUMENT
    ):
        raise ValueError("NONE 귀속은 NO_DOCUMENT 문제 그룹을 가리켜야 합니다.")


def validate_gate_result(gate: GateResult) -> None:
    """question_cache_attempts CHECK(outcome_canonical_answer, outcome_rejection_reasons)
    를 insert 전에 확인한다. 2-252 는 SKIPPED 를 쓰지 않는다. 위반이면 ValueError.
    """

    if gate.outcome == CacheAttemptOutcome.SKIPPED:
        raise ValueError("SKIPPED 캐시 시도는 기록하지 않습니다.")
    if (gate.outcome in CANONICAL_OUTCOMES) != (gate.canonical_answer_id is not None):
        raise ValueError(
            f"{gate.outcome.value} 캐시 시도의 정본 id 가 CHECK 와 맞지 않습니다."
        )
    if (gate.outcome == CacheAttemptOutcome.REJECTED) != bool(gate.rejection_reasons):
        raise ValueError(
            f"{gate.outcome.value} 캐시 시도의 거부 사유가 CHECK 와 맞지 않습니다."
        )
    for reason in gate.rejection_reasons:
        if not reason or len(reason) > MAX_REJECTION_REASON_LENGTH:
            raise ValueError(f"거부 사유는 1~{MAX_REJECTION_REASON_LENGTH}자여야 합니다.")


def served_citation_logs(gate: GateResult) -> Tuple[CitationLog, ...]:
    """SERVED 게이트의 인용 해석을 complete_rag_run 입력(CitationLog)으로 바꾼다.

    R17 로 찾은 현재 판의 청크 id, 문서 판 id, 현재 제목·절 경로·URI 스냅샷을 쓰고
    citation_order 는 정본 인용 번호를 그대로 둔다(본문 [n] 과 맞춘다).
    """

    if gate.outcome != CacheAttemptOutcome.SERVED:
        raise ValueError(f"SERVED 게이트만 인용을 복사합니다: {gate.outcome.value}")
    resolutions = sorted(gate.served_citations, key=lambda item: item.citation_order)
    if len(resolutions) < MIN_CITATIONS:
        raise ValueError(f"SERVED 인용은 최소 {MIN_CITATIONS}개여야 합니다.")
    orders = [resolution.citation_order for resolution in resolutions]
    if len(set(orders)) != len(orders) or orders[0] < 1:
        raise ValueError(f"인용 번호가 1 이상이고 서로 달라야 합니다: {orders}")
    logs = []
    for resolution in resolutions:
        section = resolution.section
        if section is None:
            raise ValueError(
                f"해석하지 못한 인용은 복사할 수 없습니다: {resolution.citation_order}"
            )
        logs.append(
            CitationLog(
                chunk_id=section.chunk_id,
                document_version_id=section.document_version_id,
                citation_order=resolution.citation_order,
                document_title_snapshot=section.document_title,
                node_path_snapshot=section.node_path,
                source_uri_snapshot=section.source_uri,
            )
        )
    return tuple(logs)


# ---------------------------------------------------------------------------
# 저장 계층
# ---------------------------------------------------------------------------


class QuestionGroupingStore:
    """AsyncSession 으로 질문 판별 결과를 기록한다. commit 하지 않는다."""

    def __init__(
        self,
        session: AsyncSession,
        log_store: Optional[RagLogStore] = None,
    ) -> None:
        self._session = session
        self._log_store = log_store or RagLogStore(session)

    async def find_exact_question_log_match(
        self,
        document_group_id: int,
        question: str,
        *,
        exclude_rag_run_id: uuid.UUID,
    ) -> Optional[ExactQuestionLogMatch]:
        """같은 문서 그룹의 과거 첫 턴 질문 중 정규화 결과가 같은 로그의 최신 CONNECT 분류.

        ``rag_runs.query_hash`` 로 좁히고 현재(effective_to IS NULL) CONNECT 분류 중
        effective_from 이 가장 최근인 행(같으면 분류 id 가 큰 행)의 세부 문제를 쓴다.
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
        rows = (
            await self._session.execute(
                select(
                    classification.id.label("classification_id"),
                    classification.effective_from,
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
                    ),
                )
                .join(subproblem, subproblem.id == classification.subproblem_id)
                .join(problem_group, problem_group.id == subproblem.problem_group_id)
                .outerjoin(source, source.id == problem_group.document_source_id)
                .where(
                    IndexVersion.document_group_id == document_group_id,
                    RagRun.turn_no == 1,
                    RagRun.query_hash == exact_question_hash(question),
                    RagRun.id != exclude_rag_run_id,
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
        ).all()

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
        )

    # ------------------------------------------------------------------
    # 분류 실행
    # ------------------------------------------------------------------

    async def get_or_open_online_run(
        self,
        *,
        document_group_id: int,
        index_version_id: int,
        model: str,
        prompt_version: str,
        actor: Optional[str] = None,
    ) -> int:
        """같은 설정의 열린 ONLINE 실행 id. 없으면 새로 연다.

        - 같은 문서 그룹의 다른 설정(색인 판, 모델, 프롬프트 판)으로 열린 ONLINE 실행은
          닫지 않는다. 색인 전환 중에는 옛 색인 판 턴과 새 색인 판 턴이 섞여 들어오므로,
          닫으면 두 설정의 실행이 번갈아 열리고 닫힌다. 설정마다 열린 실행이 하나씩 공존한다.
        - 동시 턴이 먼저 열어 부분 유니크(uq_classification_runs_open_online)를 어기면
          savepoint 만 되돌리고 다시 조회한다. 호출자 트랜잭션은 계속 쓸 수 있다.
        - 턴에 속하지 않은 공유 행이라 턴 잠금을 잡지 않는다. 턴 잠금과 섞을 때는 별도
          트랜잭션에서 부르고 commit 하거나(계획 1절 5단계), 턴 잠금 뒤에 부른다.
        - row_count 는 0 으로 두고 갱신하지 않는다.
        """

        found = await self._find_open_online_run(
            document_group_id, index_version_id, model, prompt_version
        )
        if found is not None:
            return found

        await self._session.flush()
        try:
            async with self._session.begin_nested():
                run = ClassificationRun(
                    document_group_id=document_group_id,
                    index_version_id=index_version_id,
                    kind=ClassificationRunKind.ONLINE,
                    model=model,
                    prompt_version=prompt_version,
                    row_count=0,
                    started_at=_utcnow(),
                    finished_at=None,
                    actor=actor,
                )
                self._session.add(run)
                await self._session.flush()
            return run.id
        except IntegrityError as exc:
            if not _violates_constraint(exc, OPEN_ONLINE_CLASSIFICATION_RUN_CONSTRAINT):
                raise

        found = await self._find_open_online_run(
            document_group_id, index_version_id, model, prompt_version
        )
        if found is None:
            raise RuntimeError(
                "열린 ONLINE 분류 실행 유니크를 어겼지만 다시 조회되지 않습니다: "
                f"document_group_id={document_group_id}, index_version_id={index_version_id}"
            )
        return found

    async def _find_open_online_run(
        self,
        document_group_id: int,
        index_version_id: int,
        model: str,
        prompt_version: str,
    ) -> Optional[int]:
        return await self._session.scalar(
            select(ClassificationRun.id).where(
                ClassificationRun.document_group_id == document_group_id,
                ClassificationRun.index_version_id == index_version_id,
                ClassificationRun.model == model,
                ClassificationRun.prompt_version == prompt_version,
                ClassificationRun.kind == ClassificationRunKind.ONLINE,
                ClassificationRun.finished_at.is_(None),
            )
        )

    # ------------------------------------------------------------------
    # 질문 임베딩
    # ------------------------------------------------------------------

    async def insert_question_embedding(
        self,
        rag_run_id: uuid.UUID,
        *,
        embedding: Sequence[float],
        embedding_config_id: int,
    ) -> None:
        """판별과 같은 resolved_query 의 질문 벡터. 턴당 한 행(PK rag_run_id)."""

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"질문 임베딩은 {EMBEDDING_DIMENSIONS}차원이어야 합니다: {len(embedding)}"
            )
        await self._log_store.lock_processing_run(rag_run_id)
        self._session.add(
            QuestionEmbedding(
                rag_run_id=rag_run_id,
                embedding=[float(value) for value in embedding],
                embedding_config_id=embedding_config_id,
                created_at=_utcnow(),
            )
        )
        await self._session.flush()

    # ------------------------------------------------------------------
    # 문제 그룹
    # ------------------------------------------------------------------

    async def ensure_document_problem_group(self, document_source_id: int) -> uuid.UUID:
        """문서의 DOCUMENT 문제 그룹 id. 없으면 만든다(동시 호출에도 한 행)."""

        table = QuestionProblemGroup.__table__
        await self._session.execute(
            pg_insert(table)
            .values(
                id=uuid.uuid4(),
                kind=QuestionProblemGroupKind.DOCUMENT,
                document_source_id=document_source_id,
                document_group_id=None,
            )
            .on_conflict_do_nothing(index_elements=[table.c.document_source_id])
        )
        group_id = await self._session.scalar(
            select(QuestionProblemGroup.id).where(
                QuestionProblemGroup.document_source_id == document_source_id
            )
        )
        if group_id is None:
            raise RuntimeError(
                f"DOCUMENT 문제 그룹을 만들지 못했습니다: document_source_id={document_source_id}"
            )
        return group_id

    async def ensure_no_document_problem_group(self, document_group_id: int) -> uuid.UUID:
        """문서 그룹의 NO_DOCUMENT(가이드 밖) 문제 그룹 id. 없으면 만든다."""

        table = QuestionProblemGroup.__table__
        await self._session.execute(
            pg_insert(table)
            .values(
                id=uuid.uuid4(),
                kind=QuestionProblemGroupKind.NO_DOCUMENT,
                document_source_id=None,
                document_group_id=document_group_id,
            )
            .on_conflict_do_nothing(index_elements=[table.c.document_group_id])
        )
        group_id = await self._session.scalar(
            select(QuestionProblemGroup.id).where(
                QuestionProblemGroup.document_group_id == document_group_id
            )
        )
        if group_id is None:
            raise RuntimeError(
                f"NO_DOCUMENT 문제 그룹을 만들지 못했습니다: document_group_id={document_group_id}"
            )
        return group_id

    async def _initial_problem_group(
        self,
        judgment: TurnJudgment,
        document_group_id: int,
    ) -> uuid.UUID:
        attribution: AttributionTarget = judgment.attribution
        source = attribution.attribution_source
        if source == AttributionSource.SUBPROBLEM:
            assert judgment.subproblem is not None  # validate_turn_judgment 가 보장
            return attribution.problem_group_id or judgment.subproblem.problem_group_id
        if source == AttributionSource.DOCUMENT:
            assert attribution.document_source_id is not None
            return await self.ensure_document_problem_group(attribution.document_source_id)
        return await self.ensure_no_document_problem_group(document_group_id)

    # ------------------------------------------------------------------
    # 판별 행
    # ------------------------------------------------------------------

    async def insert_classification(
        self,
        rag_run_id: uuid.UUID,
        *,
        run_id: int,
        judgment: TurnJudgment,
        judgment_input: Mapping[str, Any],
    ) -> int:
        """턴의 현재 판별 행을 한 번 쓴다.

        - decision, subproblem_id·subproblem_version(CONNECT 만), 초기 귀속(SUBPROBLEM,
          DOCUMENT, NONE)은 judgment 에서 온다. DOCUMENT/NONE 문제 그룹은 ensure 한다.
          NONE 의 문서 그룹은 분류 실행의 document_group_id 다.
        - effective_from 은 지금, effective_to 는 널, is_composite 는 false, confidence 는
          정규화 결과에 있을 때만 채운다.
        - judgment_input 은 게이트 결과(gate 칸)까지 담아 한 번에 쓴다(결정 A).
        - 분류 실행의 색인 판은 턴의 색인 판과 같아야 한다.
        - 같은 턴에 현재 행이 이미 있으면 부분 유니크
          (uq_question_classifications_rag_run_id_current)가 IntegrityError 를 낸다.
        """

        validate_turn_judgment(judgment)
        if JUDGMENT_INPUT_GATE_FIELD not in judgment_input:
            raise ValueError("judgment_input 에 게이트 결과(gate)가 필요합니다.")

        rag_run = await self._log_store.lock_processing_run(rag_run_id)
        run = (
            await self._session.execute(
                select(
                    ClassificationRun.document_group_id,
                    ClassificationRun.index_version_id,
                ).where(ClassificationRun.id == run_id)
            )
        ).one_or_none()
        if run is None:
            raise ValueError(f"존재하지 않는 분류 실행입니다: {run_id}")
        if run.index_version_id != rag_run.index_version_id:
            raise ValueError(
                "분류 실행의 색인 판이 턴의 색인 판과 다릅니다: "
                f"run={run.index_version_id}, turn={rag_run.index_version_id}"
            )

        problem_group_id = await self._initial_problem_group(
            judgment, run.document_group_id
        )
        normalized = judgment.normalized
        subproblem = judgment.subproblem
        row = QuestionClassification(
            rag_run_id=rag_run_id,
            subproblem_id=None if subproblem is None else subproblem.subproblem_id,
            problem_group_id=problem_group_id,
            run_id=run_id,
            decision=judgment.decision,
            subproblem_version=judgment.subproblem_version,
            attribution_source=judgment.attribution.attribution_source,
            is_composite=False,
            confidence=None if normalized is None else normalized.confidence,
            judgment_input=dict(judgment_input),
            effective_from=_utcnow(),
            effective_to=None,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # 캐시 시도
    # ------------------------------------------------------------------

    async def insert_cache_attempt(
        self,
        rag_run_id: uuid.UUID,
        *,
        classification_id: int,
        gate: GateResult,
        latency_ms: Optional[int] = None,
    ) -> int:
        """턴의 캐시 시도 한 행. 정본 id 는 SERVED/SHADOW/GROUP_DISABLED 만, 거부 사유는
        REJECTED 만, FAILED 는 둘 다 없다. 판별 행은 같은 턴의 현재 행이어야 하고
        정본 id 를 담는 결과는 CONNECT 행에만 붙는다.
        """

        validate_gate_result(gate)
        if latency_ms is not None and latency_ms < 0:
            raise ValueError("latency_ms 는 0 이상이어야 합니다.")

        await self._log_store.lock_processing_run(rag_run_id)
        classification = (
            await self._session.execute(
                select(
                    QuestionClassification.rag_run_id,
                    QuestionClassification.decision,
                    QuestionClassification.effective_to,
                ).where(QuestionClassification.id == classification_id)
            )
        ).one_or_none()
        if classification is None or classification.rag_run_id != rag_run_id:
            raise ValueError(
                f"턴의 판별 행이 아닙니다: classification_id={classification_id}"
            )
        if classification.effective_to is not None:
            raise ValueError("현재 판별 행에만 캐시 시도를 붙입니다.")
        if (
            gate.outcome in CANONICAL_OUTCOMES
            and classification.decision != ClassificationDecision.CONNECT
        ):
            raise ValueError(f"{gate.outcome.value} 캐시 시도는 CONNECT 판별 행이 필요합니다.")

        attempt = QuestionCacheAttempt(
            rag_run_id=rag_run_id,
            classification_id=classification_id,
            outcome=gate.outcome,
            canonical_answer_id=gate.canonical_answer_id,
            rejection_reasons=list(gate.rejection_reasons) or None,
            latency_ms=latency_ms,
            created_at=_utcnow(),
        )
        self._session.add(attempt)
        await self._session.flush()
        return attempt.id

    # ------------------------------------------------------------------
    # 턴 끝 인용 귀속
    # ------------------------------------------------------------------

    async def finalize_citation_attribution(
        self,
        classification_id: int,
        citations: Sequence[CitationLog],
    ) -> bool:
        """COMPLETED 로 끝날 턴의 인용으로 판별 행의 문제 그룹을 CITATION 으로 덮어쓴다.

        - complete_rag_run 과 같은 트랜잭션에서, complete_rag_run 보다 먼저 부른다(턴이 아직
          PROCESSING 이어야 쓴다).
        - 인용 행 수가 가장 많은 문서, 동수면 가장 앞 번호로 인용된 문서의 DOCUMENT 문제
          그룹을 ensure 하고 problem_group_id, attribution_source 두 칸만 바꾼다.
          judgment_input 은 건드리지 않는다.
        - CONNECT 행, 현재 행이 아닌 행, 인용이 없는 턴은 바꾸지 않고 False 를 돌려준다.
        """

        rag_run_id = await self._session.scalar(
            select(QuestionClassification.rag_run_id).where(
                QuestionClassification.id == classification_id
            )
        )
        if rag_run_id is None:
            raise ValueError(f"존재하지 않는 판별 행입니다: {classification_id}")
        await self._log_store.lock_processing_run(rag_run_id)

        row = (
            await self._session.execute(
                select(
                    QuestionClassification.decision,
                    QuestionClassification.effective_to,
                )
                .where(QuestionClassification.id == classification_id)
                .with_for_update()
            )
        ).one()
        if (
            not citations
            or row.decision == ClassificationDecision.CONNECT
            or row.effective_to is not None
        ):
            return False

        source_by_version = await self._document_sources_by_version(
            citation.document_version_id for citation in citations
        )
        target = citation_attribution(
            [
                CitedDocument(
                    citation_order=citation.citation_order,
                    document_source_id=source_by_version[citation.document_version_id],
                )
                for citation in citations
            ]
        )
        assert target is not None and target.document_source_id is not None
        problem_group_id = await self.ensure_document_problem_group(
            target.document_source_id
        )
        result = await self._session.execute(
            update(QuestionClassification)
            .where(
                QuestionClassification.id == classification_id,
                QuestionClassification.effective_to.is_(None),
                QuestionClassification.decision != ClassificationDecision.CONNECT,
            )
            .values(
                problem_group_id=problem_group_id,
                attribution_source=AttributionSource.CITATION,
            )
            .execution_options(synchronize_session="fetch")
        )
        await self._session.flush()
        return result.rowcount > 0

    async def _document_sources_by_version(
        self, document_version_ids: Iterable[int]
    ) -> Dict[int, int]:
        version_ids = sorted(set(document_version_ids))
        rows = (
            await self._session.execute(
                select(DocumentVersion.id, DocumentVersion.document_source_id).where(
                    DocumentVersion.id.in_(version_ids)
                )
            )
        ).all()
        found = {row.id: row.document_source_id for row in rows}
        missing = [version_id for version_id in version_ids if version_id not in found]
        if missing:
            raise ValueError(f"존재하지 않는 문서 판을 인용했습니다: {missing}")
        return found
