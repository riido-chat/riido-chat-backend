"""질문 로그 콘솔 조회(QuestionInsightService) 로컬 DB 통합 테스트.

- 로컬 PostgreSQL 과 head 마이그레이션이 필요하다. 연결할 수 없으면 skip 한다.
- 외부 트랜잭션 하나에서 시드하고 마지막에 rollback 해 데이터를 남기지 않는다.
- 새 문서 그룹 두 개에만 시드하므로 공유 DB 의 다른 데이터가 결과에 섞이지 않는다.

실행: DATABASE_URL=postgresql+asyncpg://riido:riido@localhost:5433/riido \
    .venv/bin/python -m unittest tests.test_admin_question_insights_db -v

픽스처 (문서 그룹 G, 턴 created_at = BASE + 분)

| 턴 | 분 | 상태 | 현재 분류 행 | 캐시 시도 | 문서 | 세부 문제 | 답변 상태 |
| T01 | 1 | COMPLETED | CONNECT SA1 | SERVED | A | SA1 | CACHED_ANSWER |
| T02 | 2 | COMPLETED | CONNECT SA1 | SHADOW | A | SA1 | ANSWERED |
| T03 | 3 | COMPLETED | CONNECT SB1 | GROUP_DISABLED | B | SB1 | ANSWERED |
| T04 | 4 | COMPLETED | CONNECT SB1 | SERVED | B | SB1 | CACHED_ANSWER |
| T05 | 5 | COMPLETED | SEPARATE CITATION A | REJECTED | A | - | ANSWERED |
| T06 | 5 | COMPLETED | UNCLASSIFIED CITATION C | FAILED | C | - | ANSWERED |
| T07 | 7 | COMPLETED | 없음(판별 꺼짐) | 없음 | - | - | ANSWERED |
| T08 | 8 | WITHHELD 근거 부족 | CONNECT SB2 | REJECTED | B | SB2 | WITHHELD |
| T09 | 9 | WITHHELD 근거 부족 | SEPARATE DOCUMENT A | REJECTED | A | - | WITHHELD |
| T10 | 10 | WITHHELD 근거 부족 | SEPARATE DOCUMENT C | 없음 | C | - | WITHHELD |
| T11 | 11 | WITHHELD 질문 모호 | 없음(resolved 널) | 없음 | - | - | WITHHELD |
| T12 | 12 | WITHHELD 범위 밖 | SEPARATE NONE 가이드 밖 | REJECTED | - | - | WITHHELD |
| T13 | 13 | WITHHELD 검증 실패 | CONNECT SA2 | REJECTED | A | SA2 | WITHHELD |
| T14 | 14 | ERROR | CONNECT SA3(보관) | REJECTED | A | SA3 | ERROR |
| T15 | 15 | ERROR | 없음 | 없음 | - | - | ERROR |
| T16 | 16 | COMPLETED (색인 판 2) | 옛 행 CONNECT SB1, 현재 SEPARATE CITATION C | 없음 | C | - | ANSWERED |
| T17 | 17 | PROCESSING | CONNECT SA1 | - | 제외 | | |
| T18 | 18 | CANCELLED | CONNECT SB1 | - | 제외 | | |
| T22 | 22 | COMPLETED | CONNECT SB3 | REJECTED | B | SB3 | ANSWERED |
| T23 | 23 | WITHHELD 근거 부족 | CONNECT SB4 | REJECTED | B | SB4 | WITHHELD |
| T24 | 24 | ERROR | UNCLASSIFIED NONE 가이드 밖 | FAILED | - | - | ERROR |
| T25~T29 | 25~29 | COMPLETED | 없음 | 없음 | - | - | ANSWERED (검색용 원문) |
| T30 | 30 | WITHHELD 질문 모호 | SEPARATE DOCUMENT B | 없음 | B | - | WITHHELD |

문서 그룹 H: T19 COMPLETED CONNECT SH1, T20 COMPLETED CITATION 으로 G 의 문서 B(데이터 이상). G 에 섞이면 안 된다.
"""

import asyncio
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.admin.question_insights.schema import (
    AnswerStatus,
    ApplyStatus,
    SubproblemPresence,
    WithheldReason,
)
from app.admin.question_insights.service import (
    CanonicalAnswerView,
    DocumentDetail,
    DocumentList,
    DocumentRef,
    DocumentRow,
    DocumentSummary,
    FrequentSubproblem,
    InvalidQuestionLogRequestError,
    QuestionDashboard,
    QuestionInsightService,
    QuestionListFilters,
    QuestionRow,
    SubproblemDetail,
    SubproblemNotFoundError,
    SubproblemRow,
    WithheldDocument,
    WithheldReasonCounts,
)
from app.chat.log_store import RagLogStore
from app.core.config import get_settings
from app.database.models import AnswerStatus as RunStatus
from app.database.models import (
    AttributionSource,
    CacheAttemptOutcome,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    CanonicalAnswerOrigin,
    ClassificationDecision,
    ClassificationRun,
    ClassificationRunKind,
    ContextStrategy,
    DocumentGroup,
    DocumentSource,
    QuestionCacheAttempt,
    QuestionClassification,
    QuestionProblemGroup,
    QuestionSubproblem,
    QuestionSubproblemStatus,
    RagRun,
)
from app.document.ingestion_service import (
    DocumentGroupNotFoundError,
    DocumentNotFoundError,
)
from tests.test_question_grouping_readers_db import _available, _Seed

BASE = datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)
UNKNOWN_ID = 9_000_000_000_000

IE = WithheldReason.INSUFFICIENT_EVIDENCE
AQ = WithheldReason.AMBIGUOUS_QUESTION
OOS = WithheldReason.OUT_OF_SCOPE
UA = WithheldReason.UNVERIFIABLE_ANSWER

# 질문 목록 기본 정렬(created_at DESC, id DESC). T05 와 T06 은 같은 시각이고 T06 의 id 가 크다.
ORDER = [
    "T30", "T29", "T28", "T27", "T26", "T25", "T24", "T23", "T22", "T16",
    "T15", "T14", "T13", "T12", "T11", "T10", "T09", "T08", "T07", "T06",
    "T05", "T04", "T03", "T02", "T01",
]  # fmt: skip

# 턴마다 (표시 질문, 문서 키, 세부 문제 키, 답변 상태, 보류 사유, 분)
EXPECTED_ROWS: Dict[str, Tuple[str, Optional[str], Optional[str], AnswerStatus, Optional[WithheldReason], int]] = {
    "T01": ("결제 실패 시 재시도 방법", "A", "SA1", AnswerStatus.CACHED_ANSWER, None, 1),
    "T02": ("결제 알림이 두 번 와요", "A", "SA1", AnswerStatus.ANSWERED, None, 2),
    "T03": ("멤버 추가하면 요금이 늘어요?", "B", "SB1", AnswerStatus.ANSWERED, None, 3),
    "T04": ("좌석 추가 요금", "B", "SB1", AnswerStatus.CACHED_ANSWER, None, 4),
    "T05": ("T05 원문", "A", None, AnswerStatus.ANSWERED, None, 5),
    "T06": ("T06 원문", "C", None, AnswerStatus.ANSWERED, None, 5),
    "T07": ("T07 원문", None, None, AnswerStatus.ANSWERED, None, 7),
    "T08": ("T08 원문", "B", "SB2", AnswerStatus.WITHHELD, IE, 8),
    "T09": ("T09 원문", "A", None, AnswerStatus.WITHHELD, IE, 9),
    "T10": ("T10 원문", "C", None, AnswerStatus.WITHHELD, IE, 10),
    "T11": ("그거 어떻게 해요?", None, None, AnswerStatus.WITHHELD, AQ, 11),
    "T12": ("계약서 검토 해줄 수 있나요?", None, None, AnswerStatus.WITHHELD, OOS, 12),
    "T13": ("T13 원문", "A", "SA2", AnswerStatus.WITHHELD, UA, 13),
    "T14": ("T14 원문", "A", "SA3", AnswerStatus.ERROR, None, 14),
    "T15": ("T15 원문", None, None, AnswerStatus.ERROR, None, 15),
    "T16": ("T16 재작성", "C", None, AnswerStatus.ANSWERED, None, 16),
    "T22": ("T22 원문", "B", "SB3", AnswerStatus.ANSWERED, None, 22),
    "T23": ("T23 원문", "B", "SB4", AnswerStatus.WITHHELD, IE, 23),
    "T24": ("T24 원문", None, None, AnswerStatus.ERROR, None, 24),
    "T25": ("100% 환불 되나요", None, None, AnswerStatus.ANSWERED, None, 25),
    "T26": ("1000명 좌석 요금", None, None, AnswerStatus.ANSWERED, None, 26),
    "T27": ("A_B 설정", None, None, AnswerStatus.ANSWERED, None, 27),
    "T28": ("AxB 설정", None, None, AnswerStatus.ANSWERED, None, 28),
    "T29": ("C:\\temp 경로", None, None, AnswerStatus.ANSWERED, None, 29),
    "T30": ("T30 원문", "B", None, AnswerStatus.WITHHELD, AQ, 30),
}


class QuestionInsightServiceDbTest(unittest.IsolatedAsyncioTestCase):
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
        self.service = QuestionInsightService(self.session)
        await self._build_fixture()

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    # ------------------------------------------------------------------
    # 픽스처
    # ------------------------------------------------------------------

    async def _build_fixture(self) -> None:
        seed = self.seed
        chunking = await seed.chunking()
        embedding = await seed.embedding()
        self.group = await seed.group()
        self.other_group = await seed.group()

        self.docs: Dict[str, DocumentSource] = {
            "A": await seed.source(self.group, "billing", "구독 및 결제"),
            "B": await seed.source(self.group, "members", "멤버"),
            "C": await seed.source(self.group, "guide", "가이드"),
            "D": await seed.source(self.group, "team", "팀"),
            # 제목이 비어 document_key 로 보인다.
            "E": await seed.source(self.group, "slack", "  "),
            "F": await seed.source(self.group, "archived", "보관 문서"),
            "I": await seed.source(self.group, "ganada", "가나"),
            "X": await seed.source(self.group, "no-problem-group", "문제 그룹 없음"),
            "H": await seed.source(self.other_group, "members", "멤버"),
        }
        a_version = await seed.version(self.docs["A"], 1)
        a_chunks = await seed.sections(
            a_version,
            "구독 및 결제",
            chunking,
            [("id-fail", "c-fail", "결제 실패", "본문"), ("id-refund", "c-refund", "환불", "본문")],
        )
        b_version = await seed.version(self.docs["B"], 1)
        b_chunks = await seed.sections(
            b_version,
            "멤버",
            chunking,
            [("id-fee", "c-fee", "요금 계산 방식", "본문"), ("id-role", "c-role", "권한 등급", "본문")],
        )
        self.index_1 = (await seed.index(self.group, chunking, embedding, [a_version, b_version])).index_version_id
        self.index_2 = (await seed.index(self.group, chunking, embedding, [a_version])).index_version_id
        other_index = (await seed.index(self.other_group, chunking, embedding, [])).index_version_id

        pg = {key: await seed.document_group(self.group, self.docs[key]) for key in "ABCDEFI"}
        pg["H"] = await seed.document_group(self.other_group, self.docs["H"])
        pg["N"] = await seed.no_document_group(self.group)
        self.problem_groups = pg

        approved = QuestionSubproblemStatus.APPROVED
        self.subproblems: Dict[str, QuestionSubproblem] = {
            "SA1": await self._subproblem(pg["A"], "결제 실패 처리", approved),
            "SA2": await self._subproblem(pg["A"], "구독 취소", QuestionSubproblemStatus.DRAFT),
            "SA3": await self._subproblem(pg["A"], "가격 옛 안내", QuestionSubproblemStatus.ARCHIVED),
            "SB1": await self._subproblem(pg["B"], "멤버 추가 시 과금", approved),
            "SB2": await self._subproblem(pg["B"], "멤버 권한", approved),
            "SB3": await self._subproblem(pg["B"], "멤버 초대 메일", approved),
            "SB4": await self._subproblem(pg["B"], "멤버 삭제", approved),
            "SE1": await self._subproblem(pg["E"], "슬랙 연동", QuestionSubproblemStatus.DRAFT),
            "SF1": await self._subproblem(pg["F"], "보관된 것", QuestionSubproblemStatus.ARCHIVED),
            "SI1": await self._subproblem(pg["I"], "가나 세부", approved),
            "SH1": await self._subproblem(pg["H"], "다른 그룹 세부", approved),
        }
        sp = self.subproblems

        a_fail = (a_chunks[0], a_version.id)
        a_refund = (a_chunks[1], a_version.id)
        b_fee = (b_chunks[0], b_version.id)
        b_role = (b_chunks[1], b_version.id)
        self.canonical_a1 = await self._canonical(
            sp["SA1"],
            "결제 실패 정본",
            {"rules": ["환불 절차는 다루지 않습니다", "연간 계약의 중도 좌석 변경은 다루지 않습니다"]},
            # 인용 번호 2 를 먼저 넣어도 번호 1 을 쓴다.
            [
                (2, a_refund, "구독 및 결제", "구독 및 결제 > 환불"),
                (1, a_fail, "구독 및 결제", "구독 및 결제 > 결제 실패"),
            ],
        )
        await self._canonical(
            sp["SA2"],
            "내린 정본",
            None,
            [(1, a_refund, "구독 및 결제", "구독 및 결제 > 환불")],
            approval=CanonicalAnswerApproval.REVOKED,
        )
        await self._canonical(sp["SA3"], "보관 세부 문제 정본", None, [(1, a_fail, "구독 및 결제", "구독 및 결제 > 결제 실패")])
        self.canonical_b1 = await self._canonical(
            sp["SB1"],
            "멤버 과금 정본",
            None,
            # 절 경로에 문서 제목이 없으면 앞에 붙인다.
            [(1, b_fee, "멤버", "요금 계산 방식"), (2, b_role, "멤버", "멤버 > 권한 등급")],
        )
        # 적용 범위 규칙 모양이 틀린 정본. 인용 행이 없다.
        await self._canonical(sp["SI1"], "가나 정본", {"bad": ["x"]}, [])

        run = await self._run(self.group, self.index_1)
        other_run = await self._run(self.other_group, other_index)
        conversation = await RagLogStore(self.session).create_conversation()
        self.conversation_id = conversation.id
        self._turn_no = 0
        self.turns: Dict[str, RagRun] = {}

        low_id, high_id = sorted([uuid.uuid4(), uuid.uuid4()])

        C, W, E_ = RunStatus.COMPLETED, RunStatus.WITHHELD, RunStatus.ERROR
        SEPARATE = ClassificationDecision.SEPARATE
        UNCLASSIFIED = ClassificationDecision.UNCLASSIFIED
        CIT, DOC, NONE = AttributionSource.CITATION, AttributionSource.DOCUMENT, AttributionSource.NONE
        O = CacheAttemptOutcome

        t = await self._turn("T01", C, 1, user_query="결제가 안 돼요", resolved_query="결제 실패 시 재시도 방법")
        cls = await self._connect(t, "SA1", run)
        await self._attempt(t, O.SERVED, cls, self.canonical_a1)

        t = await self._turn("T02", C, 2, user_query="결제 알림이 두 번 와요", resolved_query="   ")
        cls = await self._connect(t, "SA1", run)
        await self._attempt(t, O.SHADOW, cls, self.canonical_a1)

        t = await self._turn("T03", C, 3, user_query="멤버 추가하면 요금이 늘어요?")
        cls = await self._connect(t, "SB1", run)
        await self._attempt(t, O.GROUP_DISABLED, cls, self.canonical_b1)

        t = await self._turn("T04", C, 4, user_query="좌석 추가 요금")
        cls = await self._connect(t, "SB1", run)
        await self._attempt(t, O.SERVED, cls, self.canonical_b1)

        t = await self._turn("T05", C, 5, turn_id=low_id)
        cls = await self._classify(t, SEPARATE, pg["A"], CIT, run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T06", C, 5, turn_id=high_id)
        cls = await self._classify(t, UNCLASSIFIED, pg["C"], CIT, run)
        await self._attempt(t, O.FAILED, cls)

        await self._turn("T07", C, 7)

        t = await self._turn("T08", W, 8, reason=IE)
        cls = await self._connect(t, "SB2", run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T09", W, 9, reason=IE)
        cls = await self._classify(t, SEPARATE, pg["A"], DOC, run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T10", W, 10, reason=IE)
        await self._classify(t, SEPARATE, pg["C"], DOC, run)

        await self._turn("T11", W, 11, reason=AQ, user_query="그거 어떻게 해요?")

        t = await self._turn("T12", W, 12, reason=OOS, user_query="계약서 검토 해줄 수 있나요?")
        cls = await self._classify(t, SEPARATE, pg["N"], NONE, run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T13", W, 13, reason=UA)
        cls = await self._connect(t, "SA2", run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T14", E_, 14)
        cls = await self._connect(t, "SA3", run)
        await self._attempt(t, O.REJECTED, cls)

        await self._turn("T15", E_, 15)

        # 재분류: 옛 행은 B 의 세부 문제, 현재 행은 C 인용 귀속이다. 다른 색인 판의 턴이다.
        t = await self._turn("T16", C, 16, iv=self.index_2, resolved_query="T16 재작성")
        await self._connect(t, "SB1", run, effective_from=BASE, effective_to=BASE + timedelta(minutes=1))
        await self._classify(t, SEPARATE, pg["C"], CIT, run, effective_from=BASE + timedelta(minutes=1))

        t = await self._turn("T17", RunStatus.PROCESSING, 17)
        await self._connect(t, "SA1", run)
        t = await self._turn("T18", RunStatus.CANCELLED, 18)
        await self._connect(t, "SB1", run)

        t = await self._turn("T19", C, 19, iv=other_index)
        await self._connect(t, "SH1", other_run)
        t = await self._turn("T20", C, 20, iv=other_index)
        await self._classify(t, SEPARATE, pg["B"], CIT, other_run)

        t = await self._turn("T22", C, 22)
        cls = await self._connect(t, "SB3", run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T23", W, 23, reason=IE)
        cls = await self._connect(t, "SB4", run)
        await self._attempt(t, O.REJECTED, cls)

        t = await self._turn("T24", E_, 24)
        cls = await self._classify(t, UNCLASSIFIED, pg["N"], NONE, run)
        await self._attempt(t, O.FAILED, cls)

        await self._turn("T25", C, 25, user_query="100% 환불 되나요")
        await self._turn("T26", C, 26, user_query="1000명 좌석 요금")
        await self._turn("T27", C, 27, user_query="A_B 설정")
        await self._turn("T28", C, 28, user_query="AxB 설정")
        await self._turn("T29", C, 29, user_query="C:\\temp 경로")

        t = await self._turn("T30", W, 30, reason=AQ)
        await self._classify(t, SEPARATE, pg["B"], DOC, run)

    async def _subproblem(
        self, problem_group: QuestionProblemGroup, name: str, status: QuestionSubproblemStatus
    ) -> QuestionSubproblem:
        row = QuestionSubproblem(
            problem_group_id=problem_group.id,
            key=f"key-{uuid.uuid4().hex[:12]}",
            name=name,
            inclusion_criteria=f"{name} 기준",
            current_version=1,
            status=status,
            created_by="test",
        )
        await self.seed.add(row)
        return row

    async def _canonical(
        self,
        subproblem: QuestionSubproblem,
        content: str,
        rules: Optional[dict],
        citations: Sequence[Tuple[int, Tuple[int, int], str, str]],
        *,
        approval: CanonicalAnswerApproval = CanonicalAnswerApproval.APPROVED,
    ) -> CanonicalAnswer:
        row = CanonicalAnswer(
            subproblem_id=subproblem.id,
            origin=CanonicalAnswerOrigin.AUTHORED,
            content_markdown=content,
            applicability_rules=rules,
            subproblem_version=1,
            approval=approval,
            approved_by="test",
        )
        await self.seed.add(row)
        for order, (chunk_id, version_id), title, path in citations:
            await self.seed.add(
                CanonicalAnswerCitation(
                    canonical_answer_id=row.id,
                    citation_order=order,
                    chunk_id=chunk_id,
                    document_version_id=version_id,
                    document_title_snapshot=title,
                    node_path_snapshot=path,
                    source_uri_snapshot="https://docs.riido.io/x",
                )
            )
        return row

    async def _run(self, group: DocumentGroup, index_version_id: int) -> ClassificationRun:
        row = ClassificationRun(
            document_group_id=group.id,
            index_version_id=index_version_id,
            kind=ClassificationRunKind.ONLINE,
            model="test-model",
            prompt_version=f"test-{uuid.uuid4().hex[:8]}",
            started_at=BASE,
            finished_at=BASE,
        )
        await self.seed.add(row)
        return row

    async def _turn(
        self,
        key: str,
        status: RunStatus,
        minute: int,
        *,
        iv: Optional[int] = None,
        reason: Optional[WithheldReason] = None,
        user_query: Optional[str] = None,
        resolved_query: Optional[str] = None,
        turn_id: Optional[uuid.UUID] = None,
    ) -> RagRun:
        self._turn_no += 1
        row = RagRun(
            id=turn_id or uuid.uuid4(),
            conversation_id=self.conversation_id,
            turn_no=self._turn_no,
            index_version_id=iv or self.index_1,
            user_query=user_query or f"{key} 원문",
            resolved_query=resolved_query,
            context_strategy=ContextStrategy.NEW_TOPIC,
            status=status,
            withheld_reason_code=None if reason is None else reason.value,
            created_at=BASE + timedelta(minutes=minute),
        )
        await self.seed.add(row)
        self.turns[key] = row
        return row

    async def _classify(
        self,
        turn: RagRun,
        decision: ClassificationDecision,
        problem_group: QuestionProblemGroup,
        attribution: AttributionSource,
        run: ClassificationRun,
        *,
        subproblem: Optional[QuestionSubproblem] = None,
        effective_from: Optional[datetime] = None,
        effective_to: Optional[datetime] = None,
    ) -> QuestionClassification:
        row = QuestionClassification(
            rag_run_id=turn.id,
            subproblem_id=None if subproblem is None else subproblem.id,
            subproblem_version=None if subproblem is None else 1,
            problem_group_id=problem_group.id,
            run_id=run.id,
            decision=decision,
            attribution_source=attribution,
            effective_from=effective_from or BASE,
            effective_to=effective_to,
        )
        await self.seed.add(row)
        return row

    async def _connect(self, turn: RagRun, subproblem_key: str, run: ClassificationRun, **kwargs: Any) -> QuestionClassification:
        subproblem = self.subproblems[subproblem_key]
        problem_group = await self.session.get(QuestionProblemGroup, subproblem.problem_group_id)
        return await self._classify(
            turn,
            ClassificationDecision.CONNECT,
            problem_group,
            AttributionSource.SUBPROBLEM,
            run,
            subproblem=subproblem,
            **kwargs,
        )

    async def _attempt(
        self,
        turn: RagRun,
        outcome: CacheAttemptOutcome,
        classification: QuestionClassification,
        canonical: Optional[CanonicalAnswer] = None,
    ) -> None:
        await self.seed.add(
            QuestionCacheAttempt(
                rag_run_id=turn.id,
                classification_id=classification.id,
                outcome=outcome,
                canonical_answer_id=None if canonical is None else canonical.id,
                rejection_reasons=["TEST_REJECTED"] if outcome is CacheAttemptOutcome.REJECTED else None,
            )
        )

    # ------------------------------------------------------------------
    # 기대값 도우미
    # ------------------------------------------------------------------

    def _doc(self, key: str) -> int:
        return self.docs[key].id

    def _sp(self, key: str) -> uuid.UUID:
        return self.subproblems[key].id

    def _titles(self) -> Dict[str, str]:
        return {
            "A": "구독 및 결제",
            "B": "멤버",
            "C": "가이드",
            "E": self.docs["E"].document_key,
            "I": "가나",
            "H": "멤버",
        }

    def _row(self, key: str) -> QuestionRow:
        question, doc, subproblem, answer_status, reason, minute = EXPECTED_ROWS[key]
        return QuestionRow(
            rag_run_id=self.turns[key].id,
            question=question,
            document_id=None if doc is None else self._doc(doc),
            document_title=None if doc is None else self._titles()[doc],
            subproblem_id=None if subproblem is None else self._sp(subproblem),
            subproblem_name=None if subproblem is None else self.subproblems[subproblem].name,
            asked_at=BASE + timedelta(minutes=minute),
            answer_status=answer_status,
            withheld_reason=reason,
        )

    async def _keys(self, group: Optional[DocumentGroup] = None, **filters: Any) -> Tuple[List[str], int]:
        page = await self.service.list_questions(
            (group or self.group).id, QuestionListFilters(size=100, **filters)
        )
        by_id = {turn.id: key for key, turn in self.turns.items()}
        return [by_id[item.rag_run_id] for item in page.items], page.total_count

    # ------------------------------------------------------------------
    # 대시보드
    # ------------------------------------------------------------------

    async def test_dashboard(self) -> None:
        titles = self._titles()
        expected = QuestionDashboard(
            question_count=25,
            unanswerable_count=8,
            withheld_reason_counts=WithheldReasonCounts(
                insufficient_evidence=4,
                ambiguous_question=2,
                out_of_scope=1,
                unverifiable_answer=1,
            ),
            # 질문 수 내림차순 → 이름 오름차순. 보관 SA3(1건)과 0건 세부 문제는 빠지고 6위 SB3 이 잘린다.
            frequent_subproblems=[
                FrequentSubproblem(self._sp("SA1"), "결제 실패 처리", self._doc("A"), titles["A"], 2),
                FrequentSubproblem(self._sp("SB1"), "멤버 추가 시 과금", self._doc("B"), titles["B"], 2),
                FrequentSubproblem(self._sp("SA2"), "구독 취소", self._doc("A"), titles["A"], 1),
                FrequentSubproblem(self._sp("SB2"), "멤버 권한", self._doc("B"), titles["B"], 1),
                FrequentSubproblem(self._sp("SB4"), "멤버 삭제", self._doc("B"), titles["B"], 1),
            ],
            # 근거 부족 내림차순 → 질문 수 내림차순(A 6 > C 3, 제목은 C 가 앞). 0건 문서는 빠진다.
            withheld_documents=[
                WithheldDocument(self._doc("B"), titles["B"], 2),
                WithheldDocument(self._doc("A"), titles["A"], 1),
                WithheldDocument(self._doc("C"), titles["C"], 1),
            ],
        )
        self.assertEqual(expected, await self.service.get_dashboard(self.group.id))

    async def test_dashboard_of_other_group_does_not_see_group_turns(self) -> None:
        expected = QuestionDashboard(
            question_count=2,
            unanswerable_count=0,
            withheld_reason_counts=WithheldReasonCounts(0, 0, 0, 0),
            frequent_subproblems=[
                FrequentSubproblem(self._sp("SH1"), "다른 그룹 세부", self._doc("H"), "멤버", 1)
            ],
            withheld_documents=[],
        )
        self.assertEqual(expected, await self.service.get_dashboard(self.other_group.id))

    # ------------------------------------------------------------------
    # 문서 목록
    # ------------------------------------------------------------------

    async def test_list_documents(self) -> None:
        titles = self._titles()
        expected = DocumentList(
            # 질문 수 내림차순 → 보류 수 내림차순(B 3 > A 2) → 제목 오름차순(document_key "slack-" < "가나").
            # D(질문·세부 문제 없음), F(보관 세부 문제만), X(문제 그룹 없음)는 행이 없다.
            items=[
                DocumentRow(self._doc("B"), titles["B"], 6, 3, 4),
                DocumentRow(self._doc("A"), titles["A"], 6, 2, 2),
                DocumentRow(self._doc("C"), titles["C"], 3, 1, 0),
                DocumentRow(self._doc("E"), titles["E"], 0, 0, 1),
                DocumentRow(self._doc("I"), titles["I"], 0, 0, 1),
            ],
            no_document_question_count=2,
            unclassified_question_count=8,
        )
        self.assertEqual(expected, await self.service.list_documents(self.group.id))
        dashboard = await self.service.get_dashboard(self.group.id)
        self.assertEqual(
            dashboard.question_count,
            sum(row.question_count for row in expected.items)
            + expected.no_document_question_count
            + expected.unclassified_question_count,
        )

    async def test_list_documents_of_other_group(self) -> None:
        # T20 은 G 의 문서 문제 그룹을 가리키지만 H 문서가 아니므로 문서 없음으로 센다.
        expected = DocumentList(
            items=[DocumentRow(self._doc("H"), "멤버", 1, 0, 1)],
            no_document_question_count=1,
            unclassified_question_count=0,
        )
        self.assertEqual(expected, await self.service.list_documents(self.other_group.id))

    # ------------------------------------------------------------------
    # 문서 상세
    # ------------------------------------------------------------------

    async def test_document_detail_with_canonical_and_archived(self) -> None:
        expected = DocumentDetail(
            document=DocumentRef(self._doc("A"), "구독 및 결제"),
            summary=DocumentSummary(
                question_count=6,
                insufficient_evidence_count=1,
                cached_answer_count=1,
                subproblem_count=2,
            ),
            subproblems=[
                SubproblemRow(self._sp("SA1"), "결제 실패 처리", 2, "구독 및 결제 > 결제 실패", ApplyStatus.APPLIED),
                # 내린(REVOKED) 정본만 있으면 적용 필요이고 근거 절이 없다.
                SubproblemRow(self._sp("SA2"), "구독 취소", 1, None, ApplyStatus.NEEDS_CANONICAL),
            ],
        )
        self.assertEqual(expected, await self.service.get_document_detail(self.group.id, self._doc("A")))

    async def test_document_detail_orders_subproblems(self) -> None:
        expected = DocumentDetail(
            document=DocumentRef(self._doc("B"), "멤버"),
            summary=DocumentSummary(6, 2, 1, 4),
            subproblems=[
                SubproblemRow(self._sp("SB1"), "멤버 추가 시 과금", 2, "멤버 > 요금 계산 방식", ApplyStatus.APPLIED),
                SubproblemRow(self._sp("SB2"), "멤버 권한", 1, None, ApplyStatus.NEEDS_CANONICAL),
                SubproblemRow(self._sp("SB4"), "멤버 삭제", 1, None, ApplyStatus.NEEDS_CANONICAL),
                SubproblemRow(self._sp("SB3"), "멤버 초대 메일", 1, None, ApplyStatus.NEEDS_CANONICAL),
            ],
        )
        self.assertEqual(expected, await self.service.get_document_detail(self.group.id, self._doc("B")))

    async def test_document_full_detail_includes_canonical_and_excludes_archived(self) -> None:
        detail = await self.service.get_document_full_detail(self.group.id, self._doc("A"))

        self.assertEqual(
            [self._sp("SA1"), self._sp("SA2")],
            [item.subproblem_id for item in detail.subproblems],
        )
        applied, needs_canonical = detail.subproblems
        self.assertEqual(
            ("결제 실패 처리 기준",), applied.inclusion_criteria
        )
        self.assertEqual((), applied.exclusion_criteria)
        self.assertEqual(ApplyStatus.APPLIED, applied.apply_status)
        self.assertEqual(
            CanonicalAnswerView(
                "결제 실패 정본",
                ["환불 절차는 다루지 않습니다", "연간 계약의 중도 좌석 변경은 다루지 않습니다"],
            ),
            applied.canonical_answer,
        )
        self.assertEqual(("구독 취소 기준",), needs_canonical.inclusion_criteria)
        self.assertEqual((), needs_canonical.exclusion_criteria)
        self.assertEqual(ApplyStatus.NEEDS_CANONICAL, needs_canonical.apply_status)
        self.assertIsNone(needs_canonical.canonical_answer)

        # SA3 는 같은 문서의 ARCHIVED 세부 문제라 통합 응답에 포함하지 않는다.
        self.assertNotIn(self._sp("SA3"), [item.subproblem_id for item in detail.subproblems])

    async def test_document_full_detail_checks_group_and_document_ownership(self) -> None:
        with self.assertRaises(DocumentNotFoundError):
            await self.service.get_document_full_detail(self.group.id, self._doc("H"))
        with self.assertRaises(DocumentGroupNotFoundError):
            await self.service.get_document_full_detail(UNKNOWN_ID, self._doc("A"))

    async def test_document_detail_edge_documents(self) -> None:
        cases = {
            "C": DocumentDetail(DocumentRef(self._doc("C"), "가이드"), DocumentSummary(3, 1, 0, 0), []),
            "D": DocumentDetail(DocumentRef(self._doc("D"), "팀"), DocumentSummary(0, 0, 0, 0), []),
            "E": DocumentDetail(
                DocumentRef(self._doc("E"), self.docs["E"].document_key),
                DocumentSummary(0, 0, 0, 1),
                [SubproblemRow(self._sp("SE1"), "슬랙 연동", 0, None, ApplyStatus.NEEDS_CANONICAL)],
            ),
            "F": DocumentDetail(DocumentRef(self._doc("F"), "보관 문서"), DocumentSummary(0, 0, 0, 0), []),
            # 인용 행이 없는 APPROVED 정본은 적용중이고 근거 절이 없다.
            "I": DocumentDetail(
                DocumentRef(self._doc("I"), "가나"),
                DocumentSummary(0, 0, 0, 1),
                [SubproblemRow(self._sp("SI1"), "가나 세부", 0, None, ApplyStatus.APPLIED)],
            ),
            # 문제 그룹 행이 없는 그룹 소속 문서는 0 과 빈 목록이다.
            "X": DocumentDetail(DocumentRef(self._doc("X"), "문제 그룹 없음"), DocumentSummary(0, 0, 0, 0), []),
        }
        for key, expected in cases.items():
            with self.subTest(document=key):
                self.assertEqual(expected, await self.service.get_document_detail(self.group.id, self._doc(key)))

    # ------------------------------------------------------------------
    # 세부 문제 펼침
    # ------------------------------------------------------------------

    async def test_subproblem_detail(self) -> None:
        self.assertEqual(
            SubproblemDetail(
                self._sp("SA1"),
                "결제 실패 처리",
                ApplyStatus.APPLIED,
                CanonicalAnswerView(
                    "결제 실패 정본",
                    ["환불 절차는 다루지 않습니다", "연간 계약의 중도 좌석 변경은 다루지 않습니다"],
                ),
            ),
            await self.service.get_subproblem_detail(self.group.id, self._doc("A"), self._sp("SA1")),
        )
        self.assertEqual(
            SubproblemDetail(self._sp("SB1"), "멤버 추가 시 과금", ApplyStatus.APPLIED, CanonicalAnswerView("멤버 과금 정본", [])),
            await self.service.get_subproblem_detail(self.group.id, self._doc("B"), self._sp("SB1")),
        )
        self.assertEqual(
            SubproblemDetail(self._sp("SA2"), "구독 취소", ApplyStatus.NEEDS_CANONICAL, None),
            await self.service.get_subproblem_detail(self.group.id, self._doc("A"), self._sp("SA2")),
        )
        self.assertEqual(
            SubproblemDetail(self._sp("SE1"), "슬랙 연동", ApplyStatus.NEEDS_CANONICAL, None),
            await self.service.get_subproblem_detail(self.group.id, self._doc("E"), self._sp("SE1")),
        )
        with self.assertLogs("app.admin.question_insights.service", "WARNING"):
            self.assertEqual(
                SubproblemDetail(self._sp("SI1"), "가나 세부", ApplyStatus.APPLIED, CanonicalAnswerView("가나 정본", [])),
                await self.service.get_subproblem_detail(self.group.id, self._doc("I"), self._sp("SI1")),
            )

    # ------------------------------------------------------------------
    # 질문 목록
    # ------------------------------------------------------------------

    async def test_list_questions_rows_and_default_page(self) -> None:
        full = await self.service.list_questions(self.group.id, QuestionListFilters(size=100))
        self.assertEqual(25, full.total_count)
        self.assertEqual([self._row(key) for key in ORDER], full.items)

        default = await self.service.list_questions(self.group.id)
        self.assertEqual((1, 20, 25), (default.page, default.size, default.total_count))
        self.assertEqual([self._row(key) for key in ORDER[:20]], default.items)
        self.assertEqual(timezone.utc.utcoffset(None), default.items[0].asked_at.utcoffset())

    async def test_list_questions_pagination(self) -> None:
        second = await self.service.list_questions(self.group.id, QuestionListFilters(page=2))
        self.assertEqual((2, 20, 25), (second.page, second.size, second.total_count))
        self.assertEqual([self._row(key) for key in ORDER[20:]], second.items)

        fourth = await self.service.list_questions(self.group.id, QuestionListFilters(page=4, size=7))
        self.assertEqual([self._row(key) for key in ORDER[21:]], fourth.items)
        self.assertEqual(25, fourth.total_count)

        beyond = await self.service.list_questions(self.group.id, QuestionListFilters(page=3))
        self.assertEqual((3, 20, 25, []), (beyond.page, beyond.size, beyond.total_count, beyond.items))

        empty = await self.service.list_questions(
            self.group.id, QuestionListFilters(q="없는 검색어", page=2)
        )
        self.assertEqual((2, 20, 0, []), (empty.page, empty.size, empty.total_count, empty.items))

    async def test_list_questions_answer_status_filter(self) -> None:
        cases = {
            AnswerStatus.ANSWERED: ["T29", "T28", "T27", "T26", "T25", "T22", "T16", "T07", "T06", "T05", "T03", "T02"],
            AnswerStatus.CACHED_ANSWER: ["T04", "T01"],
            AnswerStatus.WITHHELD: ["T30", "T23", "T13", "T12", "T11", "T10", "T09", "T08"],
            AnswerStatus.ERROR: ["T24", "T15", "T14"],
        }
        for answer_status, keys in cases.items():
            with self.subTest(answer_status=answer_status):
                self.assertEqual((keys, len(keys)), await self._keys(answer_status=answer_status))

    async def test_list_questions_document_and_subproblem_filters(self) -> None:
        self.assertEqual(
            (["T30", "T23", "T22", "T08", "T04", "T03"], 6), await self._keys(document_id=self._doc("B"))
        )
        # 그룹 소속이 아닌 문서, 질문 없는 문서, 없는 문서는 빈 목록이다.
        self.assertEqual(([], 0), await self._keys(document_id=self._doc("H")))
        self.assertEqual(([], 0), await self._keys(document_id=self._doc("D")))
        self.assertEqual(([], 0), await self._keys(document_id=UNKNOWN_ID))

        present = ["T23", "T22", "T14", "T13", "T08", "T04", "T03", "T02", "T01"]
        self.assertEqual((present, 9), await self._keys(subproblem_presence=SubproblemPresence.PRESENT))
        absent = [key for key in ORDER if key not in present]
        self.assertEqual((absent, 16), await self._keys(subproblem_presence=SubproblemPresence.ABSENT))

        self.assertEqual((["T02", "T01"], 2), await self._keys(subproblem_id=self._sp("SA1")))
        # 보관된 세부 문제도 목록 필터로는 조회된다. 재분류 전 옛 행과 제외 상태는 들어오지 않는다.
        self.assertEqual((["T14"], 1), await self._keys(subproblem_id=self._sp("SA3")))
        self.assertEqual((["T04", "T03"], 2), await self._keys(subproblem_id=self._sp("SB1")))
        self.assertEqual(([], 0), await self._keys(subproblem_id=self._sp("SH1")))
        self.assertEqual(
            (["T04", "T03"], 2),
            await self._keys(subproblem_presence=SubproblemPresence.PRESENT, subproblem_id=self._sp("SB1")),
        )

        self.assertEqual(
            (["T30", "T23", "T08"], 3),
            await self._keys(answer_status=AnswerStatus.WITHHELD, document_id=self._doc("B")),
        )
        self.assertEqual(
            (["T30"], 1),
            await self._keys(
                answer_status=AnswerStatus.WITHHELD,
                document_id=self._doc("B"),
                subproblem_presence=SubproblemPresence.ABSENT,
            ),
        )

    async def test_list_questions_search(self) -> None:
        cases = {
            "재시도": ["T01"],  # 표시 질문(재작성)
            "안 돼요": ["T01"],  # 사용자 원문
            "  재시도  ": ["T01"],
            "결제": ["T02", "T01"],  # T02 는 재작성이 공백이라 원문이 표시 질문
            "100%": ["T25"],  # % 는 글자 그대로. "1000명" 은 걸리지 않는다
            "a_b": ["T27"],  # _ 는 글자 그대로, 대소문자 무시. "AxB" 는 걸리지 않는다
            "C:\\": ["T29"],
            "t16 재작성": ["T16"],
        }
        for q, keys in cases.items():
            with self.subTest(q=q):
                self.assertEqual((keys, len(keys)), await self._keys(q=q))
        self.assertEqual(
            (["T01"], 1), await self._keys(q="결제", answer_status=AnswerStatus.CACHED_ANSWER)
        )

    async def test_list_questions_of_other_group(self) -> None:
        page = await self.service.list_questions(self.other_group.id)
        self.assertEqual(2, page.total_count)
        self.assertEqual([self.turns["T20"].id, self.turns["T19"].id], [item.rag_run_id for item in page.items])
        # T20 은 G 문서의 문제 그룹을 가리키지만 H 에서는 문서 없음이다.
        self.assertEqual((None, None), (page.items[0].document_id, page.items[0].document_title))
        self.assertEqual((self._doc("H"), "멤버", self._sp("SH1")), (
            page.items[1].document_id, page.items[1].document_title, page.items[1].subproblem_id
        ))

    # ------------------------------------------------------------------
    # 오류와 읽기 전용
    # ------------------------------------------------------------------

    async def test_unknown_group_is_not_found(self) -> None:
        calls = [
            self.service.get_dashboard(UNKNOWN_ID),
            self.service.list_documents(UNKNOWN_ID),
            self.service.get_document_detail(UNKNOWN_ID, self._doc("A")),
            self.service.get_subproblem_detail(UNKNOWN_ID, self._doc("A"), self._sp("SA1")),
            self.service.list_questions(UNKNOWN_ID),
        ]
        for call in calls:
            with self.subTest(call=call.__qualname__), self.assertRaises(DocumentGroupNotFoundError) as caught:
                await call
            self.assertEqual((404, "NOT_FOUND"), (caught.exception.status_code, caught.exception.code))

    async def test_document_outside_group_is_not_found(self) -> None:
        for document_id in (self._doc("H"), UNKNOWN_ID):
            with self.subTest(document_id=document_id):
                with self.assertRaises(DocumentNotFoundError):
                    await self.service.get_document_detail(self.group.id, document_id)
                with self.assertRaises(DocumentNotFoundError):
                    await self.service.get_subproblem_detail(self.group.id, document_id, self._sp("SH1"))

    async def test_subproblem_not_in_document_or_archived_is_not_found(self) -> None:
        cases = [
            ("A", self._sp("SB1")),  # 다른 문서의 세부 문제
            ("A", self._sp("SA3")),  # 보관
            ("F", self._sp("SF1")),  # 보관
            ("A", uuid.uuid4()),  # 없음
        ]
        for document_key, subproblem_id in cases:
            with self.subTest(document=document_key, subproblem_id=subproblem_id):
                with self.assertRaises(SubproblemNotFoundError) as caught:
                    await self.service.get_subproblem_detail(self.group.id, self._doc(document_key), subproblem_id)
                self.assertEqual((404, "NOT_FOUND"), (caught.exception.status_code, caught.exception.code))
        # 다른 그룹 문서의 세부 문제는 그 그룹 경로로는 조회된다.
        detail = await self.service.get_subproblem_detail(self.other_group.id, self._doc("H"), self._sp("SH1"))
        self.assertEqual(ApplyStatus.NEEDS_CANONICAL, detail.apply_status)

    async def test_invalid_filters_are_rejected(self) -> None:
        invalid = [
            {"page": 0},
            {"size": 0},
            {"size": 101},
            {"q": "가" * 101},
            {"answer_status": "UNKNOWN"},
            {"subproblem_presence": SubproblemPresence.ABSENT, "subproblem_id": self._sp("SA1")},
        ]
        for kwargs in invalid:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                with self.assertRaises(InvalidQuestionLogRequestError) as caught:
                    QuestionListFilters(**kwargs)
                self.assertEqual((422, "INVALID_REQUEST"), (caught.exception.status_code, caught.exception.code))

    async def test_reads_do_not_write(self) -> None:
        await self.service.get_dashboard(self.group.id)
        await self.service.list_documents(self.group.id)
        await self.service.get_document_detail(self.group.id, self._doc("A"))
        await self.service.get_subproblem_detail(self.group.id, self._doc("A"), self._sp("SA1"))
        await self.service.list_questions(self.group.id, QuestionListFilters(q="결제"))
        self.assertFalse(self.session.new or self.session.dirty or self.session.deleted)
        self.assertTrue(self.transaction.is_active)


if __name__ == "__main__":
    unittest.main()
