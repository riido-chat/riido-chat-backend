"""세부 문제 시드 스크립트의 DB·API 없는 단위 테스트. 입력은 모두 인라인 합성 fixture 다."""

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from app.database.models import (
    QuestionProblemGroupKind,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.ops.seed_question_grouping import (
    ERROR_ANSWER_CONTENT,
    ERROR_CANONICAL_CITATIONS,
    ERROR_CITATION_ORDER,
    ERROR_CRITERIA,
    ERROR_DOCUMENT_PATH,
    ERROR_DUPLICATE_DOCUMENT,
    ERROR_DUPLICATE_KEY,
    ERROR_INPUT_READ,
    ERROR_KEY,
    ERROR_PLACEMENT,
    ERROR_REVIEW_STATUS,
    ERROR_SCHEMA,
    CanonicalAction,
    EmbeddingAction,
    ExistingCanonical,
    ExistingRevision,
    ExistingSubproblem,
    ParsedInput,
    ResolvedCitation,
    SeedPlan,
    SeedReport,
    SubproblemAction,
    build_parser,
    canonical_content_hash,
    document_key_from_path,
    evidence_found,
    existing_canonical_hash,
    join_criteria,
    key_error,
    load_input,
    local_section_path,
    parse_input,
    plan_subproblem,
    render_report,
)
from app.question_grouping.constants import SUBPROBLEM_EMBEDDING_TEXT_VERSION

DOC_PATH = "workspaces/plans-and-billing.md"
OTHER_PATH = "workspaces/members.md"


def _subproblem(key: str = "plans-and-billing.cancel", **overrides: Any) -> Dict[str, Any]:
    item = {
        "key": key,
        "legacyId": None,
        "name": "구독 취소 방법",
        "inclusionCriteria": ["유료 구독을 취소하려는 질문"],
        "exclusionCriteria": ["환불을 묻는 질문은 제외"],
        "canonical": {
            "contentMarkdown": "설정에서 구독을 취소할 수 있습니다 [1].",
            "applicabilityRules": ["환불은 다루지 않는다"],
            "citations": [
                {
                    "order": 1,
                    "documentPath": DOC_PATH,
                    "sectionPath": ["구독 변경 또는 취소"],
                    "evidence": "구독을 취소할 수 있습니다.",
                }
            ],
        },
        "reviewStatus": "approved",
    }
    item.update(overrides)
    return item


def _document(path: str = DOC_PATH, subproblems: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "document": {"path": path, "title": "구독 및 결제", "parentPath": "workspaces", "contentSha256": "abc"},
        "subproblems": [_subproblem()] if subproblems is None else subproblems,
        "notCandidates": [],
    }


def _parse(*documents: Dict[str, Any], include_draft: bool = False) -> ParsedInput:
    return parse_input([(f"file-{index}.json", doc) for index, doc in enumerate(documents)], include_draft=include_draft)


def _codes(parsed: ParsedInput) -> List[str]:
    return [issue.code for issue in parsed.errors]


class ParseInputTest(unittest.TestCase):
    def test_parses_valid_document(self) -> None:
        parsed = _parse(_document())
        self.assertEqual([], _codes(parsed))
        (document,) = parsed.documents
        self.assertEqual(DOC_PATH, document.path)
        (item,) = document.subproblems
        self.assertEqual("plans-and-billing.cancel", item.key)
        self.assertEqual(("유료 구독을 취소하려는 질문",), item.inclusion_criteria)
        self.assertEqual(("환불은 다루지 않는다",), item.applicability_rules)
        self.assertEqual(("구독 변경 또는 취소",), item.citations[0].section_path)

    def test_filters_by_review_status(self) -> None:
        document = _document(
            subproblems=[
                _subproblem("a.approved"),
                _subproblem("a.draft", reviewStatus="draft"),
                _subproblem("a.rejected", reviewStatus="rejected"),
            ]
        )
        default = _parse(document)
        self.assertEqual(["a.approved"], [s.key for s in default.documents[0].subproblems])
        self.assertEqual(("a.draft",), default.documents[0].skipped_draft_keys)
        self.assertEqual(("a.rejected",), default.documents[0].skipped_rejected_keys)

        with_draft = _parse(document, include_draft=True)
        self.assertEqual(["a.approved", "a.draft"], [s.key for s in with_draft.documents[0].subproblems])
        self.assertEqual(("a.rejected",), with_draft.documents[0].skipped_rejected_keys)

    def test_skipped_items_are_not_validated(self) -> None:
        broken_draft = _subproblem("BAD KEY", reviewStatus="draft")
        del broken_draft["canonical"]
        parsed = _parse(_document(subproblems=[_subproblem(), broken_draft]))
        self.assertEqual([], _codes(parsed))

    def test_unknown_review_status_is_error(self) -> None:
        parsed = _parse(_document(subproblems=[_subproblem(reviewStatus="APPROVED")]))
        self.assertEqual([ERROR_REVIEW_STATUS], _codes(parsed))

    def test_schema_errors(self) -> None:
        wrong_version = _document()
        wrong_version["schemaVersion"] = 2
        missing_canonical = _subproblem()
        del missing_canonical["canonical"]
        bad_order = _subproblem()
        bad_order["canonical"]["citations"][0]["order"] = "1"
        parsed = _parse(wrong_version, _document(OTHER_PATH, [missing_canonical, bad_order]))
        self.assertEqual([ERROR_SCHEMA, ERROR_SCHEMA, ERROR_SCHEMA], _codes(parsed))

    def test_load_input_from_directory_reads_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invite = _subproblem("members.invite")
            invite["canonical"]["citations"][0]["documentPath"] = OTHER_PATH
            for name, doc in (
                ("workspaces__plans-and-billing", _document()),
                ("workspaces__members", _document(OTHER_PATH, [invite])),
            ):
                folder = root / "documents" / name
                folder.mkdir(parents=True)
                (folder / "subproblems.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            (root / "documents" / "broken").mkdir()
            (root / "documents" / "broken" / "subproblems.json").write_text("{", encoding="utf-8")

            parsed = load_input(root, include_draft=False)
            self.assertEqual({DOC_PATH, OTHER_PATH}, {document.path for document in parsed.documents})
            self.assertEqual([ERROR_INPUT_READ], _codes(parsed))

            single = load_input(root / "documents" / "workspaces__members" / "subproblems.json", include_draft=False)
            self.assertEqual([OTHER_PATH], [document.path for document in single.documents])

        with self.assertRaises(FileNotFoundError):
            load_input(Path(directory) / "missing", include_draft=False)


class ValidationTest(unittest.TestCase):
    def test_key_rules(self) -> None:
        self.assertIsNone(key_error("plans-and-billing.cancel-subscription"))
        self.assertIsNone(key_error("0.a"))
        for bad in ("Plans.cancel", ".a", "-a", "a_b", "a b", "가.b", ""):
            self.assertIsNotNone(key_error(bad), bad)
        self.assertIsNone(key_error("a" * 200))
        self.assertIsNotNone(key_error("a" * 201))
        self.assertEqual([ERROR_KEY], _codes(_parse(_document(subproblems=[_subproblem("Bad.Key")]))))

    def test_placement_rule_first_citation_must_be_document(self) -> None:
        item = _subproblem()
        item["canonical"]["citations"][0]["documentPath"] = OTHER_PATH
        self.assertEqual([ERROR_PLACEMENT], _codes(_parse(_document(subproblems=[item]))))

        # 추가 인용은 다른 문서여도 된다.
        item = _subproblem()
        item["canonical"]["contentMarkdown"] = "취소합니다 [1]. 멤버도 봅니다 [2]."
        item["canonical"]["citations"].append(
            {"order": 2, "documentPath": OTHER_PATH, "sectionPath": ["멤버"], "evidence": None}
        )
        self.assertEqual([], _codes(_parse(_document(subproblems=[item]))))

    def test_citation_marker_and_order_rules(self) -> None:
        missing_marker = _subproblem()
        missing_marker["canonical"]["contentMarkdown"] = "번호 없는 본문"
        extra_marker = _subproblem("a.extra")
        extra_marker["canonical"]["contentMarkdown"] = "A [1]. B [2]."
        gap = _subproblem("a.gap")
        gap["canonical"]["contentMarkdown"] = "A [2]."
        gap["canonical"]["citations"][0]["order"] = 2
        no_citations = _subproblem("a.none")
        no_citations["canonical"]["citations"] = []
        parsed = _parse(_document(subproblems=[missing_marker, extra_marker, gap, no_citations]))
        by_key = {}
        for issue in parsed.errors:
            by_key.setdefault(issue.key, []).append(issue.code)
        self.assertEqual([ERROR_CANONICAL_CITATIONS], by_key["plans-and-billing.cancel"])
        self.assertEqual([ERROR_CANONICAL_CITATIONS], by_key["a.extra"])
        # [2] 하나만 있으면 번호 집합은 맞지만 1..N 이 아니고 인용 [1] 도 없다.
        self.assertEqual([ERROR_CITATION_ORDER], by_key["a.gap"])
        self.assertEqual([ERROR_CANONICAL_CITATIONS], by_key["a.none"])

    def test_answer_content_rules(self) -> None:
        link = _subproblem()
        link["canonical"]["contentMarkdown"] = "[설정](https://app.riido.io) 에서 취소합니다 [1]."
        adjacent = _subproblem("a.adjacent")
        adjacent["canonical"]["contentMarkdown"] = "취소합니다 [1][2]."
        adjacent["canonical"]["citations"].append(
            {"order": 2, "documentPath": DOC_PATH, "sectionPath": ["다른 절"], "evidence": None}
        )
        source = _subproblem("a.source")
        source["canonical"]["contentMarkdown"] = "SOURCE_1 을 보세요 [1]."
        parsed = _parse(_document(subproblems=[link, adjacent, source]))
        self.assertEqual(
            [("plans-and-billing.cancel", ERROR_ANSWER_CONTENT), ("a.source", ERROR_ANSWER_CONTENT)],
            [(issue.key, issue.code) for issue in parsed.errors],
        )

    def test_criteria_must_round_trip_one_per_line(self) -> None:
        newline = _subproblem(inclusionCriteria=["한 줄\n두 줄"])
        bullet = _subproblem("a.bullet", exclusionCriteria=["- 기호로 시작"])
        empty = _subproblem("a.empty", inclusionCriteria=[])
        parsed = _parse(_document(subproblems=[newline, bullet, empty]))
        self.assertEqual([ERROR_CRITERIA] * 3, _codes(parsed))

    def test_duplicate_keys_and_documents_across_files(self) -> None:
        parsed = _parse(
            _document(),
            _document(OTHER_PATH, [_subproblem()]),
        )
        self.assertIn(ERROR_DUPLICATE_KEY, _codes(parsed))
        self.assertIn(ERROR_PLACEMENT, _codes(parsed))
        parsed = _parse(_document(), _document(subproblems=[_subproblem("a.other")]))
        self.assertEqual([ERROR_DUPLICATE_DOCUMENT], _codes(parsed))

    def test_document_path_rules(self) -> None:
        self.assertEqual("workspaces/plans-and-billing", document_key_from_path(DOC_PATH))
        item = _subproblem()
        item["canonical"]["citations"][0]["documentPath"] = "/abs/doc.md"
        parsed = _parse(_document("../doc.txt", [item]))
        self.assertIn(ERROR_DOCUMENT_PATH, _codes(parsed))


DOC_KEY = "workspaces/plans-and-billing"
SECTION = ("구독 변경 또는 취소",)
STORED_CITATION = (1, DOC_KEY, SECTION)


def _citation(
    order: int = 1,
    chunk_id: int = 10,
    version_id: int = 5,
    *,
    document_key: str = DOC_KEY,
    section_path: tuple = SECTION,
) -> ResolvedCitation:
    return ResolvedCitation(
        order=order,
        chunk_id=chunk_id,
        document_version_id=version_id,
        document_source_id=1,
        document_title="구독 및 결제",
        node_path="구독 및 결제 > " + " > ".join(section_path),
        source_uri="https://docs.riido.io/workspaces/plans-and-billing.md",
        document_key=document_key,
        section_path=section_path,
    )


def _seed(**overrides: Any):
    return _parse(_document(subproblems=[_subproblem(**overrides)])).documents[0].subproblems[0]


def _existing(
    seed: Any,
    *,
    version: int = 1,
    embedding_config_id: int = 7,
    text_version: str = SUBPROBLEM_EMBEDDING_TEXT_VERSION,
    has_embedding: bool = True,
    canonical_hash_inputs: Optional[Dict[str, Any]] = None,
    serving_state: QuestionSubproblemServingState = QuestionSubproblemServingState.SHADOW,
) -> ExistingSubproblem:
    canonical_inputs = {
        "content_markdown": seed.content_markdown,
        "applicability_rules": seed.applicability_rules,
        "subproblem_version": version,
        "citations": (STORED_CITATION,),
    }
    canonical_inputs.update(canonical_hash_inputs or {})
    return ExistingSubproblem(
        subproblem_id=uuid.uuid4(),
        problem_group_id=uuid.uuid4(),
        problem_group_kind=QuestionProblemGroupKind.DOCUMENT,
        document_source_id=1,
        document_key="workspaces/plans-and-billing",
        key=seed.key,
        name=seed.name,
        inclusion_criteria=join_criteria(seed.inclusion_criteria),
        exclusion_criteria=join_criteria(seed.exclusion_criteria),
        current_version=version,
        status=QuestionSubproblemStatus.APPROVED,
        serving_state=serving_state,
        current_revision=ExistingRevision(
            revision_id=3,
            version=version,
            has_embedding=has_embedding,
            embedding_config_id=embedding_config_id,
            embedding_text_version=text_version,
        ),
        approved_canonical=ExistingCanonical(canonical_answer_id=uuid.uuid4(), **canonical_inputs),
    )


class PlanSubproblemTest(unittest.TestCase):
    def _plan(self, seed: Any, existing: Optional[ExistingSubproblem], citations=None, serving_state=None):
        return plan_subproblem(
            DOC_PATH,
            seed,
            existing,
            citations or [_citation()],
            embedding_config_id=7,
            serving_state=serving_state,
        )

    def test_new_subproblem(self) -> None:
        plan = self._plan(_seed(), None)
        self.assertEqual(SubproblemAction.CREATE, plan.action)
        self.assertEqual(1, plan.target_version)
        self.assertEqual(EmbeddingAction.NEW_REVISION, plan.embedding_action)
        self.assertEqual("구독 취소 방법\n유료 구독을 취소하려는 질문", plan.embedding_text)
        self.assertEqual(CanonicalAction.CREATE, plan.canonical_action)
        self.assertEqual(QuestionSubproblemServingState.UNUSED, plan.serving_state)
        self.assertEqual(
            QuestionSubproblemServingState.SERVING,
            self._plan(_seed(), None, serving_state=QuestionSubproblemServingState.SERVING).serving_state,
        )

    def test_unchanged_writes_nothing(self) -> None:
        seed = _seed()
        plan = self._plan(seed, _existing(seed))
        self.assertEqual(SubproblemAction.UNCHANGED, plan.action)
        self.assertEqual(EmbeddingAction.NONE, plan.embedding_action)
        self.assertEqual(CanonicalAction.UNCHANGED, plan.canonical_action)
        self.assertEqual(QuestionSubproblemServingState.SHADOW, plan.serving_state)
        self.assertFalse(plan.writes)

    def test_existing_serving_state_changes_only_when_explicit(self) -> None:
        seed = _seed()
        plan = self._plan(seed, _existing(seed), serving_state=QuestionSubproblemServingState.STOPPED)
        self.assertTrue(plan.serving_state_changed)
        self.assertTrue(plan.writes)
        self.assertEqual(SubproblemAction.UNCHANGED, plan.action)

    def test_stored_criteria_formatting_does_not_count_as_change(self) -> None:
        seed = _seed()
        existing = _existing(seed)
        existing = ExistingSubproblem(**{**existing.__dict__, "inclusion_criteria": "- 유료 구독을 취소하려는 질문\n"})
        self.assertEqual(SubproblemAction.UNCHANGED, self._plan(seed, existing).action)

    def test_definition_change_bumps_version_and_replaces_canonical(self) -> None:
        old = _seed()
        seed = _seed(inclusionCriteria=["유료 구독을 해지하려는 질문"], name="구독 해지")
        plan = self._plan(seed, _existing(old, version=2))
        self.assertEqual(SubproblemAction.UPDATE, plan.action)
        self.assertEqual(("name", "inclusion"), plan.definition_changes)
        self.assertEqual(3, plan.target_version)
        self.assertEqual(EmbeddingAction.NEW_REVISION, plan.embedding_action)
        # 본문이 같아도 세부 문제 판이 올라 정본을 교체한다.
        self.assertEqual(CanonicalAction.REPLACE, plan.canonical_action)

    def test_exclusion_change_alone_is_definition_change(self) -> None:
        old = _seed()
        seed = _seed(exclusionCriteria=[])
        plan = self._plan(seed, _existing(old))
        self.assertEqual(("exclusion",), plan.definition_changes)

    def test_canonical_changes_replace_without_new_revision(self) -> None:
        old = _seed()
        text = copy.deepcopy(_subproblem())
        text["canonical"]["contentMarkdown"] = "설정 화면에서 구독을 취소합니다 [1]."
        seed = _parse(_document(subproblems=[text])).documents[0].subproblems[0]
        plan = self._plan(seed, _existing(old))
        self.assertEqual(SubproblemAction.UNCHANGED, plan.action)
        self.assertEqual(EmbeddingAction.NONE, plan.embedding_action)
        self.assertEqual(CanonicalAction.REPLACE, plan.canonical_action)

        rules = _seed()
        plan = self._plan(rules, _existing(rules, canonical_hash_inputs={"applicability_rules": ("다른 규칙",)}))
        self.assertEqual(CanonicalAction.REPLACE, plan.canonical_action)

        # 인용 절 경로나 문서가 바뀌면 교체한다.
        plan = self._plan(rules, _existing(rules), citations=[_citation(section_path=("결제 주기",))])
        self.assertEqual(CanonicalAction.REPLACE, plan.canonical_action)
        plan = self._plan(rules, _existing(rules), citations=[_citation(document_key="workspaces/members")])
        self.assertEqual(CanonicalAction.REPLACE, plan.canonical_action)

        # 저장된 적용 범위를 읽지 못했으면(None) 교체한다.
        plan = self._plan(rules, _existing(rules, canonical_hash_inputs={"applicability_rules": None}))
        self.assertEqual(CanonicalAction.REPLACE, plan.canonical_action)

    def test_reindexed_chunk_ids_keep_canonical(self) -> None:
        # 재색인으로 청크 id·문서 판 id 만 바뀌고 절이 같으면 정본을 두고 게이트(R17)에 맡긴다.
        seed = _seed()
        plan = self._plan(seed, _existing(seed), citations=[_citation(chunk_id=11, version_id=6)])
        self.assertEqual(CanonicalAction.UNCHANGED, plan.canonical_action)
        self.assertFalse(plan.writes)
        # 새로 넣는 정본에 쓸 청크 id 는 이번에 해석한 것이다.
        self.assertEqual((11, 6), (plan.citations[0].chunk_id, plan.citations[0].document_version_id))

    def test_stale_embedding_refreshes_current_revision(self) -> None:
        seed = _seed()
        for existing in (
            _existing(seed, embedding_config_id=8),
            _existing(seed, has_embedding=False),
            _existing(seed, text_version="old-text-v0"),
        ):
            plan = self._plan(seed, existing)
            self.assertEqual(SubproblemAction.UNCHANGED, plan.action)
            self.assertEqual(EmbeddingAction.REFRESH_CURRENT_REVISION, plan.embedding_action)
            self.assertEqual(CanonicalAction.UNCHANGED, plan.canonical_action)


class CanonicalHashTest(unittest.TestCase):
    def test_hash_is_stable_and_order_insensitive_for_citations(self) -> None:
        first = canonical_content_hash("본문 [1][2]", ("규칙 A", "규칙 B"), [(1, "a/doc", ("절",)), (2, "b/doc", ("절", "하위"))], 1)
        second = canonical_content_hash("본문 [1][2]", ["규칙 A", "규칙 B"], [(2, "b/doc", ["절", "하위"]), (1, "a/doc", ["절"])], 1)
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))
        # 입력 문서 경로(.md)와 DB 문서 키, 본문 앞뒤 공백은 같은 해시다.
        self.assertEqual(
            first,
            canonical_content_hash(" 본문 [1][2]\n", ("규칙 A", "규칙 B"), [(1, "a/doc.md", ("절",)), (2, "b/doc", ("절", "하위"))], 1),
        )

    def test_each_component_changes_hash(self) -> None:
        citation = (1, "a/doc", ("절",))
        base = ("본문 [1]", ("규칙",), [citation], 1)
        reference = canonical_content_hash(*base)
        variants = [
            ("본문 [1].", ("규칙",), [citation], 1),
            ("본문 [1]", ("규칙", "추가"), [citation], 1),
            ("본문 [1]", (), [citation], 1),
            ("본문 [1]", None, [citation], 1),
            ("본문 [1]", ("규칙",), [(1, "b/doc", ("절",))], 1),
            ("본문 [1]", ("규칙",), [(1, "a/doc", ("다른 절",))], 1),
            ("본문 [1]", ("규칙",), [(1, "a/doc", ("절", "하위"))], 1),
            ("본문 [1]", ("규칙",), [(2, "a/doc", ("절",))], 1),
            ("본문 [1]", ("규칙",), [citation], 2),
        ]
        hashes = {canonical_content_hash(*variant) for variant in variants}
        self.assertEqual(len(variants), len(hashes))
        self.assertNotIn(reference, hashes)

    def test_existing_hash_matches_desired_hash(self) -> None:
        existing = ExistingCanonical(
            canonical_answer_id=uuid.uuid4(),
            content_markdown="본문 [1]",
            applicability_rules=("규칙",),
            subproblem_version=1,
            citations=(STORED_CITATION,),
        )
        self.assertEqual(
            canonical_content_hash("본문 [1]", ["규칙"], [_citation().hash_key], 1),
            existing_canonical_hash(existing),
        )


class HelperTest(unittest.TestCase):
    def test_local_section_path_prefers_metadata(self) -> None:
        self.assertEqual(("절",), local_section_path("문서 > 다른", {"section_path": ["문서", "절"]}))
        self.assertEqual(("절 A",), local_section_path("문서 > 절 A", None))
        self.assertEqual((), local_section_path("문서", {"section_path": ["문서"]}))
        self.assertEqual((), local_section_path(None, None))

    def test_evidence_ignores_markup_and_quote_shapes(self) -> None:
        content = "#### 복구\n\n* 삭제한 항목은 ‘복구’ 버튼으로 되돌립니다.\n* **복구된** 항목은 원래 위치로 갑니다."
        self.assertTrue(evidence_found("삭제한 항목은 '복구' 버튼으로 되돌립니다. 복구된 항목은 원래 위치로 갑니다.", content))
        self.assertFalse(evidence_found("삭제한 항목은 영구히 사라집니다.", content))

    def test_cli_arguments(self) -> None:
        args = build_parser().parse_args(["--group-key", "G", "--input", "in", "--actor", " kim "])
        self.assertFalse(args.apply)
        self.assertFalse(args.include_draft)
        self.assertIsNone(args.serving_state)
        self.assertEqual("kim", args.actor)
        args = build_parser().parse_args(
            ["--group-key", "G", "--input", "in", "--actor", "kim", "--apply", "--include-draft", "--serving-state", "SHADOW"]
        )
        self.assertTrue(args.apply and args.include_draft)
        self.assertEqual("SHADOW", args.serving_state)
        with self.assertRaises(SystemExit):
            with patch("sys.stderr"):
                build_parser().parse_args(["--group-key", "G", "--input", "in", "--actor", "kim", "--serving-state", "ON"])

    def test_render_report_for_validation_failure(self) -> None:
        parsed = _parse(_document(subproblems=[_subproblem("Bad.Key")]))
        report = SeedReport(plan=SeedPlan(group_key="G", errors=list(parsed.errors)), parsed=parsed, apply=False)
        text = render_report(report, database="postgresql+asyncpg://riido:***@localhost/riido")
        self.assertIn("DRY-RUN", text)
        self.assertIn("[KEY_INVALID]", text)
        self.assertIn("검증 실패", text)
        self.assertNotIn("riido:riido", text)


if __name__ == "__main__":
    unittest.main()
