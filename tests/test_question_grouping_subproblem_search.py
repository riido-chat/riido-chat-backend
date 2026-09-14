import math
import unittest
import uuid
from dataclasses import replace
from typing import Sequence

from app.database.models import QuestionSubproblemServingState
from app.question_grouping.constants import (
    SUBPROBLEM_CANDIDATE_TOP_K,
    SUBPROBLEM_EMBEDDING_TEXT_VERSION,
)
from app.question_grouping.models import SubproblemCatalogItem
from app.question_grouping.subproblem_search import (
    build_subproblem_embedding_text,
    cosine_similarity,
    rank_subproblems,
    subproblem_embedding_text,
)


def _item(key: str, embedding: Sequence[float], *, id_int: int = 0) -> SubproblemCatalogItem:
    return SubproblemCatalogItem(
        subproblem_id=uuid.UUID(int=id_int) if id_int else uuid.uuid5(uuid.NAMESPACE_URL, key),
        key=key,
        name=f"{key} 이름",
        inclusion_criteria=("기준 하나", "기준 둘"),
        exclusion_criteria=(),
        current_version=1,
        problem_group_id=uuid.UUID(int=99),
        document_source_id=1,
        document_key="docs/doc",
        serving_state=QuestionSubproblemServingState.UNUSED,
        inclusion_embedding=tuple(embedding),
    )


class EmbeddingTextTest(unittest.TestCase):
    def test_name_then_one_inclusion_criterion_per_line(self) -> None:
        text = build_subproblem_embedding_text(
            " 구독 취소 ", [" 구독을 해지하는 방법 ", "", "결제 중단"]
        )

        self.assertEqual("구독 취소\n구독을 해지하는 방법\n결제 중단", text)
        self.assertEqual("name-inclusion-v1", SUBPROBLEM_EMBEDDING_TEXT_VERSION)

    def test_item_text_ignores_exclusion_criteria(self) -> None:
        item = _item("billing.cancel", (1.0, 0.0))
        item = replace(item, exclusion_criteria=("환불 문의",))

        self.assertEqual("billing.cancel 이름\n기준 하나\n기준 둘", subproblem_embedding_text(item))

    def test_rejects_empty_name_criteria_and_unknown_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "이름"):
            build_subproblem_embedding_text(" ", ["기준"])
        with self.assertRaisesRegex(ValueError, "포함 기준"):
            build_subproblem_embedding_text("이름", [" "])
        with self.assertRaisesRegex(ValueError, "판"):
            build_subproblem_embedding_text("이름", ["기준"], text_version="name-v0")


class CosineTest(unittest.TestCase):
    def test_cosine_and_zero_vector(self) -> None:
        self.assertAlmostEqual(1.0, cosine_similarity([1.0, 1.0], [2.0, 2.0]))
        self.assertAlmostEqual(0.0, cosine_similarity([1.0, 0.0], [0.0, 3.0]))
        self.assertAlmostEqual(1 / math.sqrt(2), cosine_similarity([1.0, 0.0], [1.0, 1.0]))
        self.assertEqual(0.0, cosine_similarity([0.0, 0.0], [1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "차원"):
            cosine_similarity([1.0], [1.0, 0.0])


class RankSubproblemsTest(unittest.TestCase):
    def test_orders_by_similarity_and_cuts_top_k(self) -> None:
        items = [
            _item("far", (0.0, 1.0)),
            _item("near", (1.0, 0.1)),
            _item("mid", (1.0, 1.0)),
        ]

        ranked = rank_subproblems((1.0, 0.0), items, top_k=2)

        self.assertEqual(["near", "mid"], [candidate.item.key for candidate in ranked])
        self.assertEqual([1, 2], [candidate.retrieval_rank for candidate in ranked])
        self.assertAlmostEqual(1.0 / math.sqrt(1.01), ranked[0].similarity)
        self.assertAlmostEqual(1.0 / math.sqrt(2), ranked[1].similarity)

    def test_default_top_k_and_fewer_items(self) -> None:
        items = [_item(f"k{index}", (1.0, float(index))) for index in range(7)]

        self.assertEqual(SUBPROBLEM_CANDIDATE_TOP_K, len(rank_subproblems((1.0, 0.0), items)))
        self.assertEqual(2, len(rank_subproblems((1.0, 0.0), items[:2])))
        self.assertEqual((), rank_subproblems((1.0, 0.0), []))

    def test_ties_follow_key_then_id_even_for_duplicate_keys(self) -> None:
        items = [
            _item("b", (1.0, 0.0), id_int=1),
            _item("a", (2.0, 0.0), id_int=3),
            _item("a", (3.0, 0.0), id_int=2),
        ]

        ranked = rank_subproblems((1.0, 0.0), items)

        self.assertEqual(
            [("a", 2), ("a", 3), ("b", 1)],
            [(c.item.key, c.item.subproblem_id.int) for c in ranked],
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            rank_subproblems((1.0,), [], top_k=0)
        with self.assertRaisesRegex(ValueError, "질문 벡터"):
            rank_subproblems((), [_item("a", (1.0,))])
        with self.assertRaisesRegex(ValueError, "임베딩이 없습니다"):
            rank_subproblems((1.0,), [_item("a", ())])
        with self.assertRaisesRegex(ValueError, "차원"):
            rank_subproblems((1.0, 0.0), [_item("a", (1.0,))])


if __name__ == "__main__":
    unittest.main()
