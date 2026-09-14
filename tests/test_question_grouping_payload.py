import hashlib
import json
import unittest
import uuid
from pathlib import Path
from typing import Optional, Sequence

from app.database.models import QuestionSubproblemServingState
from app.question_grouping.constants import (
    DOCUMENT_SHUFFLE_STREAM,
    JUDGE_PROMPT_VERSION,
    SUBPROBLEM_SHUFFLE_STREAM,
)
from app.question_grouping.models import (
    CatalogCanonicalAnswer,
    DocumentCandidate,
    DocumentOutline,
    SubproblemCandidate,
    SubproblemCatalogItem,
)
from app.question_grouping.payload import (
    build_judge_presentation,
    numbered_rules,
    presentation_judgment_input,
    presentation_seed,
    split_criteria,
)
from app.question_grouping.prompt_v7_2 import (
    JUDGE_INSTRUCTIONS,
    JUDGE_INSTRUCTIONS_SHA256,
    JUDGE_OUTPUT_SCHEMA,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POC_PROMPT = REPO_ROOT / "evaluation/question_grouping_poc/prompts/classifier_v7_2.md"
SEED = "6f1c1d52-3f55-4d47-9e2c-6c1b8f4f0a11"


def subproblem_candidate(
    index: int,
    *,
    inclusion: Sequence[str] = ("포함 기준 하나", "포함 기준 둘"),
    exclusion: Sequence[str] = ("제외 기준",),
    canonical: bool = True,
    rules: Sequence[str] = ("환불은 다루지 않는다",),
    document_title: Optional[str] = "구독 및 결제",
) -> SubproblemCandidate:
    return SubproblemCandidate(
        item=SubproblemCatalogItem(
            subproblem_id=uuid.UUID(int=index),
            key=f"billing.subproblem-{index}",
            name=f"세부 문제 {index}",
            inclusion_criteria=tuple(inclusion),
            exclusion_criteria=tuple(exclusion),
            current_version=index + 1,
            problem_group_id=uuid.UUID(int=1000 + index),
            document_source_id=100 + index,
            document_key=f"workspaces/doc-{index}",
            serving_state=QuestionSubproblemServingState.SERVING,
            document_title=document_title,
            canonical_answer=(
                CatalogCanonicalAnswer(
                    canonical_answer_id=uuid.UUID(int=2000 + index),
                    content_markdown=f"정본 {index} [1]",
                    applicability_rules=tuple(rules),
                )
                if canonical
                else None
            ),
        ),
        similarity=round(0.9 - index / 100, 2),
        retrieval_rank=index,
    )


def document_candidate(index: int) -> DocumentCandidate:
    return DocumentCandidate(
        outline=DocumentOutline(
            document_source_id=500 + index,
            document_version_id=900 + index,
            document_key=f"folder-{index}/doc-{index}",
            title=f"문서 {index}",
            parent_path=f"folder-{index}",
            headings=(f"절 {index}", f"절 {index} > 하위"),
        ),
        score=round(0.03 - index / 1000, 3),
        retrieval_rank=index,
    )


def presentation(subproblem_count: int = 5, document_count: int = 5, seed: str = SEED):
    return build_judge_presentation(
        "  구독을 취소하려면 어떻게 하나요?  ",
        [subproblem_candidate(index) for index in range(1, subproblem_count + 1)],
        [document_candidate(index) for index in range(1, document_count + 1)],
        seed=seed,
    )


class PromptTest(unittest.TestCase):
    def test_instructions_match_checked_in_hash(self) -> None:
        digest = hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest()

        self.assertEqual(JUDGE_INSTRUCTIONS_SHA256, digest)
        self.assertTrue(JUDGE_INSTRUCTIONS.startswith("# Question grouping classifier v7.2"))
        self.assertEqual(JUDGE_INSTRUCTIONS, JUDGE_INSTRUCTIONS.strip())

    @unittest.skipUnless(POC_PROMPT.is_file(), "PoC 프롬프트 파일이 없는 환경")
    def test_instructions_are_byte_identical_to_poc_prompt(self) -> None:
        self.assertEqual(
            POC_PROMPT.read_text(encoding="utf-8").strip(),
            JUDGE_INSTRUCTIONS,
        )

    def test_schema_is_strict_and_orders_document_fields_last(self) -> None:
        self.assertFalse(JUDGE_OUTPUT_SCHEMA["additionalProperties"])
        properties = list(JUDGE_OUTPUT_SCHEMA["properties"])
        self.assertEqual(properties, JUDGE_OUTPUT_SCHEMA["required"])
        self.assertEqual(
            ["documentDecision", "documentCandidateId", "documentRationaleCode"],
            properties[-3:],
        )
        self.assertEqual("decision", properties[0])

    @unittest.skipUnless(POC_PROMPT.is_file(), "PoC 코드가 없는 환경")
    def test_schema_matches_poc_request_schema(self) -> None:
        from evaluation.question_grouping_poc.openai_client import CLASSIFIER_SCHEMA_V7_2

        self.assertEqual(
            json.dumps(CLASSIFIER_SCHEMA_V7_2),
            json.dumps(JUDGE_OUTPUT_SCHEMA),
        )


class CriteriaTextTest(unittest.TestCase):
    def test_splits_one_criterion_per_line(self) -> None:
        self.assertEqual(
            ("첫 기준", "둘째 기준"),
            split_criteria("- 첫 기준\r\n\n * 둘째 기준 \n"),
        )
        self.assertEqual((), split_criteria(None))
        self.assertEqual((), split_criteria(""))

    def test_numbers_rules_from_one(self) -> None:
        self.assertEqual({"I1": "a", "I2": "b"}, numbered_rules("I", ["a", "b"]))
        self.assertEqual({}, numbered_rules("E", []))


class JudgePresentationTest(unittest.TestCase):
    def test_payload_uses_keys_document_keys_and_numbered_rules(self) -> None:
        result = presentation(subproblem_count=1, document_count=1)
        payload = result.payload

        self.assertEqual(
            [
                "policyVersion",
                "question",
                "contextSummary",
                "candidateCount",
                "candidates",
                "documentCandidateCount",
                "documentCandidates",
            ],
            list(payload),
        )
        self.assertEqual(JUDGE_PROMPT_VERSION, payload["policyVersion"])
        self.assertEqual("구독을 취소하려면 어떻게 하나요?", payload["question"])
        self.assertIsNone(payload["contextSummary"])
        self.assertEqual(
            {
                "rank": 1,
                "group": {"id": "workspaces/doc-1", "name": "구독 및 결제"},
                "subproblem": {
                    "id": "billing.subproblem-1",
                    "name": "세부 문제 1",
                    "inclusionCriteria": {"I1": "포함 기준 하나", "I2": "포함 기준 둘"},
                    "exclusionCriteria": {"E1": "제외 기준"},
                    "canonicalAnswer": {
                        "contentMarkdown": "정본 1 [1]",
                        "applicabilityRules": ["환불은 다루지 않는다"],
                    },
                },
            },
            payload["candidates"][0],
        )
        self.assertEqual(
            [
                {
                    "id": "D1",
                    "title": "문서 1",
                    "parentPath": "folder-1",
                    "headings": ["절 1", "절 1 > 하위"],
                }
            ],
            payload["documentCandidates"],
        )

    def test_payload_carries_no_scores_or_database_ids(self) -> None:
        result = presentation()
        serialized = json.dumps(result.payload, ensure_ascii=False)

        for item in result.subproblems:
            self.assertNotIn(str(item.subproblem_id), serialized)
            self.assertNotIn(str(item.problem_group_id), serialized)
        for forbidden in ("similarity", "score", "retrievalRank", "documentVersionId"):
            self.assertNotIn(forbidden, serialized)
        for document in result.payload["documentCandidates"]:
            self.assertEqual({"id", "title", "parentPath", "headings"}, set(document))

    def test_omits_optional_group_name_and_canonical_fields(self) -> None:
        same_name = subproblem_candidate(1, document_title="세부 문제 1", rules=())
        no_title = subproblem_candidate(2, document_title=None, canonical=False)

        result = build_judge_presentation("질문", [same_name, no_title], [], seed=SEED)
        by_key = {
            item["subproblem"]["id"]: item for item in result.payload["candidates"]
        }

        first = by_key["billing.subproblem-1"]
        self.assertEqual({"id": "workspaces/doc-1"}, first["group"])
        self.assertEqual(
            {"contentMarkdown": "정본 1 [1]"},
            first["subproblem"]["canonicalAnswer"],
        )
        second = by_key["billing.subproblem-2"]
        self.assertNotIn("canonicalAnswer", second["subproblem"])
        self.assertEqual({"id": "workspaces/doc-2"}, second["group"])
        self.assertEqual(0, result.payload["documentCandidateCount"])
        self.assertEqual([], result.payload["documentCandidates"])

    def test_same_seed_reproduces_order_regardless_of_input_order(self) -> None:
        subproblems = [subproblem_candidate(index) for index in range(1, 6)]
        documents = [document_candidate(index) for index in range(1, 6)]

        first = build_judge_presentation("질문", subproblems, documents, seed=SEED)
        second = build_judge_presentation(
            "질문", list(reversed(subproblems)), list(reversed(documents)), seed=SEED
        )

        self.assertEqual(first.payload, second.payload)
        self.assertEqual(first.subproblems, second.subproblems)
        self.assertEqual(first.documents, second.documents)

    def test_streams_are_independent(self) -> None:
        full = presentation(subproblem_count=5, document_count=5)
        fewer_documents = presentation(subproblem_count=5, document_count=2)

        self.assertEqual(
            [item.key for item in full.subproblems],
            [item.key for item in fewer_documents.subproblems],
        )
        self.assertEqual(f"{SEED}:{SUBPROBLEM_SHUFFLE_STREAM}", full.subproblem_seed)
        self.assertEqual(f"{SEED}:{DOCUMENT_SHUFFLE_STREAM}", full.document_seed)

    def test_different_seeds_shuffle_presentation(self) -> None:
        orders = {
            tuple(item.key for item in presentation(seed=f"seed-{index}").subproblems)
            for index in range(10)
        }

        self.assertGreater(len(orders), 1)

    def test_ids_and_ranks_follow_presented_order(self) -> None:
        result = presentation()

        self.assertEqual(
            ["D1", "D2", "D3", "D4", "D5"],
            [item.id for item in result.documents],
        )
        self.assertEqual(
            [1, 2, 3, 4, 5],
            [item["rank"] for item in result.payload["candidates"]],
        )
        self.assertEqual(
            [item.key for item in result.subproblems],
            [item["subproblem"]["id"] for item in result.payload["candidates"]],
        )
        self.assertEqual(
            [item.title for item in result.documents],
            [item["title"] for item in result.payload["documentCandidates"]],
        )
        self.assertEqual([1, 2, 3, 4, 5], [item.presented_order for item in result.documents])

    def test_presented_lists_map_keys_back_to_database_ids(self) -> None:
        result = presentation()
        item = result.subproblem_by_key()["billing.subproblem-3"]

        self.assertEqual(uuid.UUID(int=3), item.subproblem_id)
        self.assertEqual(4, item.subproblem_version)
        self.assertEqual(uuid.UUID(int=1003), item.problem_group_id)
        self.assertEqual(uuid.UUID(int=2003), item.canonical_answer_id)
        self.assertEqual(3, item.retrieval_rank)
        self.assertEqual((2, 1), (item.inclusion_count, item.exclusion_count))
        document = next(doc for doc in result.documents if doc.document_version_id == 902)
        self.assertEqual(result.document_by_id()[document.id], document)
        self.assertEqual(502, document.document_source_id)

    def test_judgment_input_records_mapping_and_seeds(self) -> None:
        result = presentation(subproblem_count=1, document_count=1)

        block = presentation_judgment_input(
            result, document_retrieval={"dense": 10, "bm25": 10, "rrfK": 60}
        )

        self.assertEqual(
            {
                "seed": f"{SEED}:{SUBPROBLEM_SHUFFLE_STREAM}",
                "items": [
                    {
                        "key": "billing.subproblem-1",
                        "subproblemId": str(uuid.UUID(int=1)),
                        "subproblemVersion": 2,
                        "problemGroupId": str(uuid.UUID(int=1001)),
                        "documentKey": "workspaces/doc-1",
                        "canonicalAnswerId": str(uuid.UUID(int=2001)),
                        "similarity": 0.89,
                        "retrievalRank": 1,
                        "presentedOrder": 1,
                    }
                ],
            },
            block["subproblemCandidates"],
        )
        self.assertEqual(
            {
                "seed": f"{SEED}:{DOCUMENT_SHUFFLE_STREAM}",
                "retrieval": {"dense": 10, "bm25": 10, "rrfK": 60},
                "items": [
                    {
                        "id": "D1",
                        "documentSourceId": 501,
                        "documentVersionId": 901,
                        "documentKey": "folder-1/doc-1",
                        "title": "문서 1",
                        "parentPath": "folder-1",
                        "score": 0.029,
                        "retrievalRank": 1,
                        "presentedOrder": 1,
                    }
                ],
            },
            block["documentCandidates"],
        )
        json.dumps(block)

    def test_seed_comes_from_rag_run_id(self) -> None:
        rag_run_id = uuid.UUID(SEED)

        self.assertEqual(SEED, presentation_seed(rag_run_id))

    def test_rejects_invalid_candidate_lists(self) -> None:
        duplicate_key = subproblem_candidate(1)
        with self.assertRaisesRegex(ValueError, "key"):
            build_judge_presentation("질문", [duplicate_key, duplicate_key], [], seed=SEED)
        with self.assertRaisesRegex(ValueError, "세부 문제 후보"):
            build_judge_presentation(
                "질문", [subproblem_candidate(index) for index in range(1, 7)], [], seed=SEED
            )
        with self.assertRaisesRegex(ValueError, "문서 후보"):
            build_judge_presentation(
                "질문", [], [document_candidate(index) for index in range(1, 7)], seed=SEED
            )
        with self.assertRaisesRegex(ValueError, "문서"):
            build_judge_presentation(
                "질문", [], [document_candidate(1), document_candidate(1)], seed=SEED
            )
        with self.assertRaisesRegex(ValueError, "질문"):
            build_judge_presentation("  ", [], [], seed=SEED)


if __name__ == "__main__":
    unittest.main()
