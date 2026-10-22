import unittest
import uuid

from app.database.models import (
    AnswerStatus,
    AttributionSource,
    ClassificationDecision,
    QuestionProblemGroupKind,
)
from app.question_grouping.attribution import (
    citation_attribution,
    initial_attribution,
    most_cited_document_source_id,
    should_attribute_by_citation,
)
from app.question_grouping.models import (
    CitedDocument,
    PresentedDocument,
    PresentedSubproblem,
)


SUBPROBLEM = PresentedSubproblem(
    key="notifications.mute",
    subproblem_id=uuid.UUID(int=1),
    subproblem_version=1,
    problem_group_id=uuid.UUID(int=10),
    document_source_id=5,
    document_key="notifications/settings",
    canonical_answer_id=None,
    similarity=0.6,
    retrieval_rank=1,
    presented_order=2,
    inclusion_count=1,
    exclusion_count=0,
)
DOCUMENT = PresentedDocument(
    id="D2",
    document_source_id=8,
    document_version_id=80,
    document_key="views/custom-views",
    title="사용자 지정 보기",
    parent_path="views",
    score=0.03,
    retrieval_rank=1,
    presented_order=2,
)


def _cited(*pairs):
    return [CitedDocument(citation_order=order, document_source_id=source) for order, source in pairs]


class InitialAttributionTest(unittest.TestCase):
    def test_connect_uses_subproblem_group_even_with_document(self) -> None:
        target = initial_attribution(
            ClassificationDecision.CONNECT, subproblem=SUBPROBLEM, document=DOCUMENT
        )

        self.assertEqual(AttributionSource.SUBPROBLEM, target.attribution_source)
        self.assertEqual(QuestionProblemGroupKind.DOCUMENT, target.group_kind)
        self.assertEqual(SUBPROBLEM.problem_group_id, target.problem_group_id)
        self.assertIsNone(target.document_source_id)

    def test_connect_requires_subproblem(self) -> None:
        with self.assertRaisesRegex(ValueError, "세부 문제"):
            initial_attribution(ClassificationDecision.CONNECT)

    def test_matched_document_is_document_attribution(self) -> None:
        for decision in (ClassificationDecision.SEPARATE, ClassificationDecision.UNCLASSIFIED):
            with self.subTest(decision=decision):
                target = initial_attribution(decision, document=DOCUMENT)

                self.assertEqual(AttributionSource.DOCUMENT, target.attribution_source)
                self.assertEqual(QuestionProblemGroupKind.DOCUMENT, target.group_kind)
                self.assertEqual(8, target.document_source_id)
                self.assertIsNone(target.problem_group_id)

    def test_no_document_or_failure_is_none_attribution(self) -> None:
        target = initial_attribution(ClassificationDecision.UNCLASSIFIED)

        self.assertEqual(AttributionSource.NONE, target.attribution_source)
        self.assertEqual(QuestionProblemGroupKind.NO_DOCUMENT, target.group_kind)
        self.assertIsNone(target.problem_group_id)
        self.assertIsNone(target.document_source_id)


class CitationAttributionTest(unittest.TestCase):
    def test_only_completed_non_connect_turns_with_citations(self) -> None:
        self.assertTrue(
            should_attribute_by_citation(
                ClassificationDecision.SEPARATE, AnswerStatus.COMPLETED, 1
            )
        )
        # 판별 실패 턴도 인용과 함께 완료되면 CITATION 이다.
        self.assertTrue(
            should_attribute_by_citation(
                ClassificationDecision.UNCLASSIFIED, AnswerStatus.COMPLETED, 2
            )
        )
        # 서빙된 턴과 서빙되지 않은 CONNECT 턴은 SUBPROBLEM 을 유지한다.
        self.assertFalse(
            should_attribute_by_citation(
                ClassificationDecision.CONNECT, AnswerStatus.COMPLETED, 1
            )
        )
        for status in (AnswerStatus.WITHHELD, AnswerStatus.ERROR, AnswerStatus.CANCELLED):
            with self.subTest(status=status):
                self.assertFalse(
                    should_attribute_by_citation(
                        ClassificationDecision.SEPARATE, status, 1
                    )
                )
        self.assertFalse(
            should_attribute_by_citation(
                ClassificationDecision.SEPARATE, AnswerStatus.COMPLETED, 0
            )
        )

    def test_most_cited_document_wins(self) -> None:
        self.assertEqual(9, most_cited_document_source_id(_cited((1, 4), (2, 9), (3, 9))))

    def test_tie_goes_to_document_of_citation_one(self) -> None:
        self.assertEqual(4, most_cited_document_source_id(_cited((2, 9), (1, 4))))
        self.assertEqual(
            4, most_cited_document_source_id(_cited((3, 9), (1, 4), (2, 4), (4, 9)))
        )

    def test_tie_outside_citation_one_uses_earliest_tied_citation(self) -> None:
        citations = _cited((1, 1), (2, 9), (3, 4), (4, 9), (5, 4))

        self.assertEqual(9, most_cited_document_source_id(citations))

    def test_single_citation_and_empty(self) -> None:
        self.assertEqual(3, most_cited_document_source_id(_cited((1, 3))))
        self.assertIsNone(most_cited_document_source_id([]))
        self.assertIsNone(citation_attribution([]))

    def test_citation_attribution_targets_document_group(self) -> None:
        target = citation_attribution(_cited((1, 4), (2, 9), (3, 9)))

        self.assertEqual(AttributionSource.CITATION, target.attribution_source)
        self.assertEqual(QuestionProblemGroupKind.DOCUMENT, target.group_kind)
        self.assertEqual(9, target.document_source_id)


if __name__ == "__main__":
    unittest.main()
