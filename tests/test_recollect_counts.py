"""재탐색 배치 결과 집계를 검증한다.

집계는 DB 없이 실행 목록만으로 결정되므로 단위 테스트로 덮는다.
"""

import unittest

from app.database.models import (
    ExecutionStatus,
    IngestionResultCode,
    IngestionRun,
)
from app.document.recollect_service import REMOVED_ACTION, _count_results


def _run(
    status: ExecutionStatus,
    result_code=None,
    *,
    removed: bool = False,
) -> IngestionRun:
    run = IngestionRun()
    run.status = status
    run.result_code = result_code
    run.summary = {"recollect_action": REMOVED_ACTION} if removed else {}
    return run


class RecollectCountsTest(unittest.TestCase):
    def test_duplicate_is_counted_as_no_change(self) -> None:
        counts = _count_results([
            _run(ExecutionStatus.SUCCESS, IngestionResultCode.DUPLICATE_CONTENT),
        ])

        self.assertEqual(1, counts["total"])
        self.assertEqual(1, counts["no_change"])
        self.assertEqual(0, counts["created"])

    def test_every_run_lands_in_exactly_one_bucket(self) -> None:
        counts = _count_results([
            _run(ExecutionStatus.SUCCESS, IngestionResultCode.CREATED),
            _run(ExecutionStatus.SUCCESS, IngestionResultCode.UPDATED),
            _run(ExecutionStatus.SUCCESS, IngestionResultCode.NO_CHANGE),
            _run(ExecutionStatus.SUCCESS, IngestionResultCode.DUPLICATE_CONTENT),
            _run(ExecutionStatus.FAILED),
        ])

        self.assertEqual(
            counts["total"],
            counts["created"]
            + counts["updated"]
            + counts["no_change"]
            + counts["failed"],
        )

    def test_removed_is_outside_total(self) -> None:
        counts = _count_results([
            _run(ExecutionStatus.SUCCESS, IngestionResultCode.NO_CHANGE),
            _run(ExecutionStatus.SUCCESS, removed=True),
        ])

        self.assertEqual(1, counts["total"])
        self.assertEqual(1, counts["removed"])
