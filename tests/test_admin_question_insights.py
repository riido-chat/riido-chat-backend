"""질문 로그 콘솔 조회의 순수 규칙(답변 상태 매핑, 입력 검증, 근거 문서 절) 단위 테스트."""

import unittest
import uuid
from http import HTTPStatus

from sqlalchemy.dialects import postgresql

from app.admin.question_insights.schema import (
    AnswerStatus,
    SubproblemPresence,
    WithheldReason,
)
from app.admin.question_insights.service import (
    InvalidQuestionLogRequestError,
    QuestionListFilters,
    ResolvedAnswerStatus,
    answer_status_condition,
    compose_source_section,
    escape_like,
    resolve_answer_status,
)
from app.database.models import AnswerStatus as RunStatus
from app.database.models import CacheAttemptOutcome, QuestionCacheAttempt, RagRun


class ResolveAnswerStatusTest(unittest.TestCase):
    def test_completed_is_cached_only_when_served(self) -> None:
        self.assertEqual(
            ResolvedAnswerStatus(AnswerStatus.CACHED_ANSWER, None),
            resolve_answer_status(RunStatus.COMPLETED, None, CacheAttemptOutcome.SERVED),
        )
        self.assertEqual(
            (AnswerStatus.ANSWERED, None),
            resolve_answer_status(RunStatus.COMPLETED, None, None),
        )
        for outcome in (
            CacheAttemptOutcome.SHADOW,
            CacheAttemptOutcome.GROUP_DISABLED,
            CacheAttemptOutcome.REJECTED,
            CacheAttemptOutcome.SKIPPED,
            CacheAttemptOutcome.FAILED,
        ):
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    (AnswerStatus.ANSWERED, None),
                    resolve_answer_status(RunStatus.COMPLETED, None, outcome),
                )

    def test_accepts_plain_strings(self) -> None:
        self.assertEqual(
            (AnswerStatus.CACHED_ANSWER, None),
            resolve_answer_status("COMPLETED", None, "SERVED"),
        )
        self.assertEqual(
            (AnswerStatus.WITHHELD, WithheldReason.OUT_OF_SCOPE),
            resolve_answer_status("WITHHELD", "OUT_OF_SCOPE", None),
        )

    def test_withheld_carries_reason(self) -> None:
        for reason in WithheldReason:
            with self.subTest(reason=reason):
                self.assertEqual(
                    (AnswerStatus.WITHHELD, reason),
                    resolve_answer_status(RunStatus.WITHHELD, reason.value, None),
                )

    def test_withheld_with_unknown_or_missing_reason_has_no_reason(self) -> None:
        with self.assertLogs("app.admin.question_insights.service", "WARNING"):
            self.assertEqual(
                (AnswerStatus.WITHHELD, None),
                resolve_answer_status(RunStatus.WITHHELD, "SOMETHING_NEW", None),
            )
        self.assertEqual(
            (AnswerStatus.WITHHELD, None),
            resolve_answer_status(RunStatus.WITHHELD, None, None),
        )

    def test_error_ignores_cache_outcome(self) -> None:
        self.assertEqual(
            (AnswerStatus.ERROR, None),
            resolve_answer_status(RunStatus.ERROR, None, CacheAttemptOutcome.SERVED),
        )

    def test_uncounted_statuses_raise(self) -> None:
        for status in (RunStatus.PROCESSING, RunStatus.CANCELLED):
            with self.subTest(status=status), self.assertRaises(ValueError):
                resolve_answer_status(status, None, None)


class AnswerStatusConditionTest(unittest.TestCase):
    def _sql(self, answer_status: AnswerStatus) -> str:
        condition = answer_status_condition(
            answer_status, RagRun.status, QuestionCacheAttempt.outcome
        )
        return str(
            condition.compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )

    def test_answered_includes_missing_attempt(self) -> None:
        sql = self._sql(AnswerStatus.ANSWERED)
        self.assertIn("rag_runs.status = 'COMPLETED'", sql)
        self.assertIn("question_cache_attempts.outcome IS DISTINCT FROM 'SERVED'", sql)

    def test_cached_requires_served(self) -> None:
        sql = self._sql(AnswerStatus.CACHED_ANSWER)
        self.assertIn("rag_runs.status = 'COMPLETED'", sql)
        self.assertIn("question_cache_attempts.outcome = 'SERVED'", sql)

    def test_withheld_and_error_use_run_status_only(self) -> None:
        self.assertEqual("rag_runs.status = 'WITHHELD'", self._sql(AnswerStatus.WITHHELD))
        self.assertEqual("rag_runs.status = 'ERROR'", self._sql(AnswerStatus.ERROR))


class QuestionListFiltersTest(unittest.TestCase):
    def assertInvalid(self, **kwargs: object) -> InvalidQuestionLogRequestError:
        with self.assertRaises(InvalidQuestionLogRequestError) as caught:
            QuestionListFilters(**kwargs)  # type: ignore[arg-type]
        self.assertEqual("INVALID_REQUEST", caught.exception.code)
        self.assertEqual(HTTPStatus.UNPROCESSABLE_ENTITY, caught.exception.status_code)
        return caught.exception

    def test_defaults(self) -> None:
        filters = QuestionListFilters()
        self.assertIsNone(filters.answer_status)
        self.assertIsNone(filters.document_id)
        self.assertIsNone(filters.subproblem_presence)
        self.assertIsNone(filters.subproblem_id)
        self.assertIsNone(filters.q)
        self.assertEqual(1, filters.page)
        self.assertEqual(20, filters.size)
        self.assertEqual(0, filters.offset)

    def test_query_is_trimmed_and_blank_means_no_search(self) -> None:
        self.assertEqual("멤버 추가", QuestionListFilters(q="  멤버 추가 \t").q)
        self.assertIsNone(QuestionListFilters(q="   ").q)
        self.assertIsNone(QuestionListFilters(q="").q)
        self.assertEqual("가" * 100, QuestionListFilters(q=" " + "가" * 100 + " ").q)
        self.assertInvalid(q="가" * 101)
        self.assertInvalid(q=123)

    def test_page_and_size_bounds(self) -> None:
        self.assertEqual(40, QuestionListFilters(page=3, size=20).offset)
        self.assertEqual(1, QuestionListFilters(size=1).size)
        self.assertEqual(100, QuestionListFilters(size=100).size)
        self.assertInvalid(page=0)
        self.assertInvalid(page=-1)
        self.assertInvalid(size=0)
        self.assertInvalid(size=101)
        self.assertInvalid(page="1")
        self.assertInvalid(size=True)

    def test_enums_and_ids_are_coerced(self) -> None:
        subproblem_id = uuid.uuid4()
        filters = QuestionListFilters(
            answer_status="WITHHELD",
            subproblem_presence="PRESENT",
            subproblem_id=str(subproblem_id),
            document_id=7,
        )
        self.assertIs(AnswerStatus.WITHHELD, filters.answer_status)
        self.assertIs(SubproblemPresence.PRESENT, filters.subproblem_presence)
        self.assertEqual(subproblem_id, filters.subproblem_id)
        self.assertEqual(7, filters.document_id)
        self.assertInvalid(answer_status="HELD")
        self.assertInvalid(subproblem_presence="MAYBE")
        self.assertInvalid(subproblem_id="not-a-uuid")
        self.assertInvalid(document_id="7")

    def test_absent_with_subproblem_id_is_invalid(self) -> None:
        self.assertInvalid(
            subproblem_presence=SubproblemPresence.ABSENT, subproblem_id=uuid.uuid4()
        )
        filters = QuestionListFilters(
            subproblem_presence=SubproblemPresence.PRESENT, subproblem_id=uuid.uuid4()
        )
        self.assertIs(SubproblemPresence.PRESENT, filters.subproblem_presence)


class EscapeLikeTest(unittest.TestCase):
    def test_escapes_wildcards_and_escape_char(self) -> None:
        self.assertEqual("100\\%", escape_like("100%"))
        self.assertEqual("a\\_b", escape_like("a_b"))
        self.assertEqual("C:\\\\temp", escape_like("C:\\temp"))
        self.assertEqual("평범한 검색", escape_like("평범한 검색"))


class ComposeSourceSectionTest(unittest.TestCase):
    def test_title_prefix_is_not_duplicated(self) -> None:
        self.assertEqual(
            "멤버 > 요금 계산 방식",
            compose_source_section("멤버", "멤버 > 요금 계산 방식"),
        )

    def test_title_is_prepended_when_path_lacks_it(self) -> None:
        self.assertEqual(
            "멤버 > 요금 계산 방식",
            compose_source_section("멤버", "요금 계산 방식"),
        )

    def test_partial_snapshots(self) -> None:
        self.assertEqual("멤버", compose_source_section("멤버", None))
        self.assertEqual("멤버", compose_source_section("멤버", "멤버"))
        self.assertEqual("요금 > 좌석", compose_source_section(None, "요금 > 좌석"))
        self.assertEqual("요금", compose_source_section("  ", "요금"))
        self.assertIsNone(compose_source_section(None, None))
        self.assertIsNone(compose_source_section("", "  "))


if __name__ == "__main__":
    unittest.main()
