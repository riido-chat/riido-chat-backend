import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
from openai import APITimeoutError
from pydantic import ValidationError

from app.answering.generator import (
    ANSWER_PROMPT_V17,
    ANSWER_PROMPT_VERSION,
    ANSWER_REPAIR_PROMPT_V17,
    ANSWER_REPAIR_PROMPT_VERSION,
    GENERATION_PROMPT_VERSION,
    MAX_CONTEXT_SOURCES,
    OPENAI_GENERATION_MODEL,
    PROCEDURE_EVIDENCE_RULES,
    SOURCE_PLANNING_PROMPT_VERSION,
    SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
    SOURCE_PLANNING_REPAIR_PROMPT_V11,
    SOURCE_PLANNING_PROMPT_V11,
    OpenAIGenerator,
    build_answer_input,
    build_answer_repair_input,
    build_generation_context,
    build_generation_input,
    build_source_planning_repair_input,
    count_distinct_citations,
    select_required_sources,
)
from app.answering.models import (
    GenerationAnswerType,
    GenerationAnswerScope,
    GenerationEvidenceRequirement,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStageTrace,
    GenerationStatus,
    GenerationWithheldReason,
)
from app.retrieval.models import HybridRetrievalResult, RetrievalChunk


class GenerationResultTest(unittest.TestCase):
    def test_answerable_requires_answer_and_null_reason(self) -> None:
        result = GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="핵심 답변입니다. [SOURCE_1]",
            withheld_reason=None,
        )

        self.assertEqual(GenerationStatus.ANSWERABLE, result.status)

        invalid_cases = (
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_markdown": None,
                "withheld_reason": None,
            },
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_markdown": "  ",
                "withheld_reason": None,
            },
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_markdown": "답변",
                "withheld_reason": GenerationWithheldReason.OUT_OF_SCOPE,
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    GenerationResult(**values)

    def test_withheld_requires_null_answer_and_reason(self) -> None:
        for reason in GenerationWithheldReason:
            with self.subTest(reason=reason):
                result = GenerationResult(
                    status=GenerationStatus.WITHHELD,
                    answer_markdown=None,
                    withheld_reason=reason,
                )
                self.assertEqual(reason, result.withheld_reason)

        invalid_cases = (
            {
                "status": GenerationStatus.WITHHELD,
                "answer_markdown": "답변",
                "withheld_reason": GenerationWithheldReason.OUT_OF_SCOPE,
            },
            {
                "status": GenerationStatus.WITHHELD,
                "answer_markdown": None,
                "withheld_reason": None,
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    GenerationResult(**values)

    def test_structured_output_requires_all_fields_and_forbids_extra_fields(self) -> None:
        schema = GenerationResult.model_json_schema()

        self.assertEqual(
            {"status", "answer_markdown", "withheld_reason"},
            set(schema["required"]),
        )
        self.assertFalse(schema["additionalProperties"])

        with self.assertRaises(ValidationError):
            GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
            )
        with self.assertRaises(ValidationError):
            GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
                withheld_reason=GenerationWithheldReason.OUT_OF_SCOPE,
                unexpected="field",
            )


class GenerationSourcePlanTest(unittest.TestCase):
    def test_answerable_requires_information_unit_evidence(self) -> None:
        plan = self._answerable_plan("SOURCE_1")

        self.assertEqual(GenerationStatus.ANSWERABLE, plan.status)
        self.assertEqual(
            ["SOURCE_1"],
            plan.evidence_requirements[0].source_ids,
        )

        invalid_cases = (
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_type": GenerationAnswerType.GENERAL,
                "answer_scope": GenerationAnswerScope.SUMMARY,
                "evidence_requirements": [],
                "withheld_reason": None,
            },
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_type": GenerationAnswerType.GENERAL,
                "answer_scope": GenerationAnswerScope.SUMMARY,
                "evidence_requirements": [
                    GenerationEvidenceRequirement(
                        information_unit="설정 방법",
                        source_ids=["SOURCE_1"],
                    )
                ],
                "withheld_reason": GenerationWithheldReason.OUT_OF_SCOPE,
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    GenerationSourcePlan(**values)

    def test_definition_and_feature_summary_require_one_unit_and_source(self) -> None:
        invalid_evidence = (
            [
                GenerationEvidenceRequirement(
                    information_unit="기능",
                    source_ids=["SOURCE_1", "SOURCE_2"],
                )
            ],
            [
                GenerationEvidenceRequirement(
                    information_unit="기능",
                    source_ids=["SOURCE_1"],
                ),
                GenerationEvidenceRequirement(
                    information_unit="설정",
                    source_ids=["SOURCE_2"],
                ),
            ],
        )

        for answer_type in (
            GenerationAnswerType.DEFINITION,
            GenerationAnswerType.FEATURE_SUMMARY,
        ):
            for evidence_requirements in invalid_evidence:
                with self.subTest(
                    answer_type=answer_type,
                    evidence_requirements=evidence_requirements,
                ):
                    with self.assertRaises(ValidationError):
                        GenerationSourcePlan(
                            status=GenerationStatus.ANSWERABLE,
                            answer_type=answer_type,
                            answer_scope=GenerationAnswerScope.SUMMARY,
                            evidence_requirements=evidence_requirements,
                            withheld_reason=None,
                        )

    def test_procedure_summary_allows_multiple_sources(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.PROCEDURE,
            answer_scope=GenerationAnswerScope.SUMMARY,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="설정 방법",
                    source_ids=["SOURCE_1", "SOURCE_2"],
                )
            ],
            withheld_reason=None,
        )

        self.assertEqual(
            ["SOURCE_1", "SOURCE_2"],
            plan.evidence_requirements[0].source_ids,
        )

    def test_detailed_definition_allows_multiple_units_and_sources(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.DEFINITION,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="정의",
                    source_ids=["SOURCE_1"],
                ),
                GenerationEvidenceRequirement(
                    information_unit="상세 역할",
                    source_ids=["SOURCE_2", "SOURCE_3"],
                ),
            ],
            withheld_reason=None,
        )

        self.assertEqual(2, len(plan.evidence_requirements))

    def test_multi_detail_allows_multiple_information_units_and_sources(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.PROCEDURE,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="기간과 시작 요일",
                    source_ids=["SOURCE_1", "SOURCE_2"],
                ),
                GenerationEvidenceRequirement(
                    information_unit="자동화",
                    source_ids=["SOURCE_3"],
                ),
            ],
            withheld_reason=None,
        )

        self.assertEqual(2, len(plan.evidence_requirements))
        self.assertEqual(
            ["SOURCE_1", "SOURCE_2"],
            plan.evidence_requirements[0].source_ids,
        )

    def test_withheld_requires_empty_evidence_and_reason(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.WITHHELD,
            answer_type=GenerationAnswerType.GENERAL,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[],
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )

        self.assertEqual(
            GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
            plan.withheld_reason,
        )

        with self.assertRaises(ValidationError):
            GenerationSourcePlan(
                status=GenerationStatus.WITHHELD,
                answer_type=GenerationAnswerType.GENERAL,
                answer_scope=GenerationAnswerScope.SUMMARY,
                evidence_requirements=[],
                withheld_reason=None,
            )

    def test_evidence_rejects_duplicate_source_ids(self) -> None:
        with self.assertRaises(ValidationError):
            GenerationEvidenceRequirement(
                information_unit="설정 방법",
                source_ids=["SOURCE_1", "SOURCE_1"],
            )

    def test_structured_output_requires_all_fields_and_forbids_extra(self) -> None:
        schema = GenerationSourcePlan.model_json_schema()

        self.assertEqual(
            {
                "status",
                "answer_type",
                "answer_scope",
                "evidence_requirements",
                "withheld_reason",
            },
            set(schema["required"]),
        )
        self.assertFalse(schema["additionalProperties"])

    @staticmethod
    def _answerable_plan(*source_ids: str) -> GenerationSourcePlan:
        return GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.PROCEDURE,
            answer_scope=(
                GenerationAnswerScope.SUMMARY
                if len(source_ids) == 1
                else GenerationAnswerScope.MULTI_DETAIL
            ),
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="설정 방법",
                    source_ids=list(source_ids),
                )
            ],
            withheld_reason=None,
        )


class GenerationContextTest(unittest.TestCase):
    def test_assigns_source_ids_in_hybrid_order(self) -> None:
        results = [self._result(index) for index in range(1, 6)]

        sources = build_generation_context(results)

        self.assertEqual(
            [f"SOURCE_{index}" for index in range(1, 6)],
            [source.source_id for source in sources],
        )
        self.assertEqual(
            [result.chunk for result in results],
            [source.chunk for source in sources],
        )

    def test_rejects_more_than_five_sources(self) -> None:
        results = [self._result(index) for index in range(MAX_CONTEXT_SOURCES + 1)]

        with self.assertRaisesRegex(ValueError, "최대 5개"):
            build_generation_context(results)

    def test_generation_input_exposes_only_allowed_context_fields(self) -> None:
        source = build_generation_context([self._result(1)])[0]

        generation_input = build_generation_input("사용자 질문", [source])

        self.assertIn("SOURCE_1", generation_input)
        self.assertIn(source.chunk.document_title, generation_input)
        self.assertIn(" > ".join(source.chunk.section_path), generation_input)
        self.assertIn(source.chunk.content, generation_input)
        self.assertIn("사용자 질문", generation_input)
        self.assertNotIn("Chunk ID:", generation_input)
        self.assertNotIn(source.chunk.source_url, generation_input)
        self.assertNotIn("rrf", generation_input.lower())

    def test_generation_input_normalizes_escaped_opening_bracket(self) -> None:
        result = self._result(1)
        result = replace(
            result,
            chunk=replace(
                result.chunk,
                content=r"1. \[설정 > 워크스페이스 > 알림]에서 관리",
            ),
        )
        source = build_generation_context([result])[0]

        generation_input = build_generation_input("질문", [source])

        self.assertIn("[설정 > 워크스페이스 > 알림]", generation_input)
        self.assertNotIn(r"\[설정", generation_input)

    def test_source_planning_prompt_preserves_scope_and_all_evidence(self) -> None:
        self.assertIn("질문을 answer_type으로 분류", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("DEFINITION", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("FEATURE_SUMMARY", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("PROCEDURE", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("GENERAL", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("가능 여부, 조건·제한, 공개 범위", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("사용자가 지금 해결하려는 핵심", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("대상의 의미 자체를 묻는 경우에만", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("여러 기능을 넓게 요청할 때만", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("GENERAL을 우선", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("질문이 직접 요구한 정보 단위만", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("질문 전체를 하나의 정보 단위", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("가장 직접적인 SOURCE 하나만", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("설정 위치·절차·설정 항목·값 범위", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("관련 없는", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("4~5개 SOURCE가 필요해도", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("정보 단위 하나라도", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("INSUFFICIENT_EVIDENCE", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("제품의 기능·설정·사용 가능 여부", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("넓은 일반 권한보다", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("구체적인 제한·예외가 질문에 대한 결론", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("문서 제목과 Section Path", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("Top-5에서 앞선 SOURCE 하나만", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("하위 대상을 별도로 허용한다고 추측하지", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("주제 자체와", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("자동화", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("EvidenceRequirement와 source_ids를 각각", SOURCE_PLANNING_PROMPT_V11)
        self.assertIn("정확히 하나만 작성", SOURCE_PLANNING_PROMPT_V11)

    def test_source_planning_repair_prompt_only_corrects_structure(self) -> None:
        self.assertIn(SOURCE_PLANNING_PROMPT_V11, SOURCE_PLANNING_REPAIR_PROMPT_V11)
        self.assertIn("Backend 구조 검증에 실패", SOURCE_PLANNING_REPAIR_PROMPT_V11)
        self.assertIn("질문의 의미와 정보 범위는 바꾸지 마세요", SOURCE_PLANNING_REPAIR_PROMPT_V11)
        self.assertIn("각각 정확히 하나만", SOURCE_PLANNING_REPAIR_PROMPT_V11)

    def test_procedure_evidence_rules_are_shared_by_planning_and_answer(self) -> None:
        for prompt in (SOURCE_PLANNING_PROMPT_V11, ANSWER_PROMPT_V17):
            with self.subTest(prompt=prompt[:30]):
                self.assertIn(PROCEDURE_EVIDENCE_RULES, prompt)
                self.assertIn("요청한 행동을 수행할 수 있을 때만", prompt)
                self.assertIn("행동의 대상과 목적이 같아야", prompt)
                self.assertIn("메뉴 이름, 사용 가능 여부", prompt)
                self.assertIn("수행할 핵심 동작이 필요", prompt)
                self.assertIn("한 단계로 충분", prompt)
                self.assertIn("관련 안내로 질문 범위를 바꾸거나", prompt)

    def test_answer_prompt_avoids_partial_or_duplicate_answers(self) -> None:
        self.assertIn("기본 답변은 간결하지만", ANSWER_PROMPT_V17)
        self.assertIn('"자세히", "전부",', ANSWER_PROMPT_V17)
        self.assertIn("근거 있는 내용을 빠짐없이 확장", ANSWER_PROMPT_V17)
        self.assertIn("모든 정보 단위에 답하세요", ANSWER_PROMPT_V17)
        self.assertIn("정보 단위를 생략하거나", ANSWER_PROMPT_V17)
        self.assertIn("반드시 가장 직접적인 SOURCE 하나만 선택", ANSWER_PROMPT_V17)
        self.assertIn("중복 SOURCE는 사용하거나 인용하지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("SOURCE별로 답변 문단을 만들거나", ANSWER_PROMPT_V17)
        self.assertIn("필요한 최소한의 SOURCE만 인용", ANSWER_PROMPT_V17)

    def test_answer_prompt_formats_only_broad_feature_summaries(self) -> None:
        self.assertIn("Answer Type이 FEATURE_SUMMARY", ANSWER_PROMPT_V17)
        self.assertIn("짧은 소개 문장 하나", ANSWER_PROMPT_V17)
        self.assertIn("기능이 두 개 이상", ANSWER_PROMPT_V17)
        self.assertIn("모든 핵심 기능", ANSWER_PROMPT_V17)
        self.assertIn("`기능명: 설명`", ANSWER_PROMPT_V17)
        self.assertIn("실제로 하는 일과 직접적인 결과", ANSWER_PROMPT_V17)
        self.assertIn("각 목록 항목 끝에 해당 SOURCE marker", ANSWER_PROMPT_V17)
        self.assertIn("기존 방식의 불편함", ANSWER_PROMPT_V17)
        self.assertIn("사용자가 직접 물었을 때만", ANSWER_PROMPT_V17)
        self.assertIn("다른 종류의 SUMMARY에 적용하지 마세요", ANSWER_PROMPT_V17)

    def test_answer_prompt_formats_procedures_for_compact_execution(self) -> None:
        self.assertIn("Answer Type이 PROCEDURE", ANSWER_PROMPT_V17)
        self.assertIn(
            "`설정 위치 → 실제 절차\n  → 설정 항목 → 필수 주의사항`",
            ANSWER_PROMPT_V17,
        )
        self.assertIn("빈 소제목을 출력하지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("첫 문장에서 바로 안내", ANSWER_PROMPT_V17)
        self.assertIn("번호 목록을 사용", ANSWER_PROMPT_V17)
        self.assertIn("`항목명: 설명`", ANSWER_PROMPT_V17)
        self.assertIn("값 범위·기본값·변경 결과", ANSWER_PROMPT_V17)
        self.assertIn("각 단계와 설정 항목은 한두 문장", ANSWER_PROMPT_V17)
        self.assertIn("관련 없는 기능 소개, FAQ, 배경 설명", ANSWER_PROMPT_V17)

    def test_answer_prompt_formats_general_summaries_as_helpful_short_answers(
        self,
    ) -> None:
        self.assertIn("Answer Type이 GENERAL", ANSWER_PROMPT_V17)
        self.assertIn("Answer Scope가 SUMMARY", ANSWER_PROMPT_V17)
        self.assertIn("기본적으로 2~4문장", ANSWER_PROMPT_V17)
        self.assertIn("결론을 자연스럽고 직접적으로", ANSWER_PROMPT_V17)
        self.assertIn("가능 여부만 한 문장으로", ANSWER_PROMPT_V17)
        self.assertIn("판단 이유나 꼭 알아야 할 조건", ANSWER_PROMPT_V17)
        self.assertIn("간단한 방법이 한두 단계", ANSWER_PROMPT_V17)
        self.assertIn("긴 절차와 관련 없는 배경", ANSWER_PROMPT_V17)
        self.assertIn("각 사실 문장 끝에는 해당 SOURCE marker", ANSWER_PROMPT_V17)

    def test_answer_prompt_uses_backend_citation_identity_limit(self) -> None:
        self.assertNotIn("최대 3개", ANSWER_PROMPT_V17)
        self.assertIn("원문 URL과 Section Path", ANSWER_PROMPT_V17)
        self.assertIn("필요한 근거를 생략하지 마세요", ANSWER_PROMPT_V17)

    def test_prompt_forbids_multiline_content_inside_table_cells(self) -> None:
        self.assertIn("표를 사용하지 말고", ANSWER_PROMPT_V17)
        self.assertIn("항목별 소제목이나 목록으로 설명하세요", ANSWER_PROMPT_V17)
        self.assertIn("모든 셀은 반드시 한 줄로 끝내고", ANSWER_PROMPT_V17)
        self.assertIn("코드 블록이나", ANSWER_PROMPT_V17)
        self.assertIn("줄바꿈을 절대 넣지 마세요", ANSWER_PROMPT_V17)

    def test_prompt_defines_terms_before_riido_role_and_value(self) -> None:
        self.assertIn("용어의 의미를 직접 물으면", ANSWER_PROMPT_V17)
        self.assertIn("첫 문장에 그 용어 자체의 쉬운 의미", ANSWER_PROMPT_V17)
        self.assertIn("그 용어가 어떤 종류인지", ANSWER_PROMPT_V17)
        self.assertIn("핵심 사용 주체나 목적", ANSWER_PROMPT_V17)
        self.assertIn("첫 문장만으로 이해할 수 있게 끝내고", ANSWER_PROMPT_V17)
        self.assertIn("같은 문장에", ANSWER_PROMPT_V17)
        self.assertIn("이어 붙이지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("둘째 문장으로 미루지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("용어 의미를 먼저 설명한 뒤", ANSWER_PROMPT_V17)
        self.assertIn("뤼이도에서의 역할과 사용 가치", ANSWER_PROMPT_V17)
        self.assertIn("최대 두 개의 짧은 문단", ANSWER_PROMPT_V17)
        self.assertIn("X의 종류와 핵심 사용 주체 또는 목적", ANSWER_PROMPT_V17)
        self.assertIn("X와 뤼이도의 관계를 직접 설명하면", ANSWER_PROMPT_V17)
        self.assertIn("질문 이해에 필요한", ANSWER_PROMPT_V17)
        self.assertIn("한두 문장으로 안내", ANSWER_PROMPT_V17)
        self.assertIn("둘째 문단을 억지로 만들지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("문단 끝에 해당 SOURCE marker", ANSWER_PROMPT_V17)
        self.assertIn("두 문단 사이에 빈 줄 하나", ANSWER_PROMPT_V17)
        self.assertIn("문장 중간에 강제 줄바꿈하지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("전체 기능 목록, 배경 문제", ANSWER_PROMPT_V17)

    def test_prompt_does_not_invent_unsupported_term_definitions(self) -> None:
        self.assertIn("일반 지식으로 정의를 보완하지 말고", ANSWER_PROMPT_V17)
        self.assertIn("WITHHELD 여부를 판단하세요", ANSWER_PROMPT_V17)

    def test_prompt_forbids_links_urls_and_html(self) -> None:
        self.assertEqual("v24", GENERATION_PROMPT_VERSION)
        self.assertEqual("v15", SOURCE_PLANNING_PROMPT_VERSION)
        self.assertEqual("v15-repair-1", SOURCE_PLANNING_REPAIR_PROMPT_VERSION)
        self.assertEqual("v20", ANSWER_PROMPT_VERSION)
        self.assertEqual("v20-repair-1", ANSWER_REPAIR_PROMPT_VERSION)
        self.assertIn("넓은 허용 규칙과 구체적인 제한", ANSWER_PROMPT_V17)
        self.assertIn("상위 공간과 그 내부 대상", ANSWER_PROMPT_V17)
        self.assertIn("내부 식별자를 답변 문장에 직접 노출", ANSWER_PROMPT_V17)
        self.assertIn(
            "Markdown 링크 문법과 HTML을 사용하지 마세요",
            ANSWER_PROMPT_V17,
        )
        self.assertIn(
            "코드 블록이나 백틱 인라인 코드 안에 넣고",
            ANSWER_PROMPT_V17,
        )
        self.assertIn("그 밖의 URL은 본문에 쓰지 마세요", ANSWER_PROMPT_V17)
        self.assertIn("별도 citations 영역", ANSWER_PROMPT_V17)

    def test_selects_required_sources_in_first_evidence_order(self) -> None:
        sources = build_generation_context(
            [self._result(index) for index in range(1, 4)]
        )
        plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.PROCEDURE,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="두 번째",
                    source_ids=["SOURCE_2", "SOURCE_1"],
                ),
                GenerationEvidenceRequirement(
                    information_unit="세 번째",
                    source_ids=["SOURCE_1", "SOURCE_3"],
                ),
            ],
            withheld_reason=None,
        )

        selected = select_required_sources(plan, sources)

        self.assertEqual(
            ["SOURCE_2", "SOURCE_1", "SOURCE_3"],
            [source.source_id for source in selected],
        )

    def test_rejects_source_plan_id_missing_from_context(self) -> None:
        plan = GenerationSourcePlanTest._answerable_plan("SOURCE_9")

        with self.assertRaisesRegex(RuntimeError, "SOURCE_9"):
            select_required_sources(plan, [])

    def test_counts_citations_after_same_section_merge(self) -> None:
        results = [self._result(index) for index in range(1, 5)]
        results[1] = replace(
            results[1],
            chunk=replace(
                results[1].chunk,
                source_url=results[0].chunk.source_url,
                section_path=results[0].chunk.section_path,
            ),
        )
        sources = build_generation_context(results)

        self.assertEqual(3, count_distinct_citations(sources))

    def test_answer_input_contains_required_coverage(self) -> None:
        sources = build_generation_context([self._result(1)])
        plan = GenerationSourcePlanTest._answerable_plan("SOURCE_1")

        answer_input = build_answer_input("질문", sources, plan)

        self.assertIn("## Answer Contract", answer_input)
        self.assertIn("Answer Type: PROCEDURE", answer_input)
        self.assertIn("Answer Scope: SUMMARY", answer_input)
        self.assertIn("## Required Answer Coverage", answer_input)
        self.assertIn("설정 방법: SOURCE_1", answer_input)

    @staticmethod
    def _result(index: int) -> HybridRetrievalResult:
        chunk = RetrievalChunk(
            document_id=f"document-{index}",
            section_id=f"section-{index}",
            document_title=f"문서 {index}",
            section_path=(f"문서 {index}", f"섹션 {index}"),
            source_url=f"https://docs.riido.io/{index}",
            category="guide",
            content=f"본문 {index}",
            chunk_id=index,
            document_version_id=100 + index,
            index_version_id=1,
        )
        return HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.1,
            final_rank=index,
            bm25_rank=index,
            vector_rank=index,
        )


class OpenAIGeneratorTest(unittest.IsolatedAsyncioTestCase):
    async def test_selects_sources_before_requesting_answer_with_prompt_v7(
        self,
    ) -> None:
        plan = self._answerable_plan("SOURCE_1")
        expected = self._answerable_result()
        client = self._client_with_responses(plan, expected)
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        result = await generator.generate("질문", sources)

        self.assertEqual(expected, result)
        self.assertEqual(
            [
                call(
                    model=OPENAI_GENERATION_MODEL,
                    instructions=SOURCE_PLANNING_PROMPT_V11,
                    input=build_generation_input("질문", sources),
                    text_format=GenerationSourcePlan,
                ),
                call(
                    model=OPENAI_GENERATION_MODEL,
                    instructions=ANSWER_PROMPT_V17,
                    input=build_answer_input("질문", sources, plan),
                    text_format=GenerationResult,
                ),
            ],
            client.responses.parse.await_args_list,
        )

    async def test_answer_receives_only_sources_required_by_plan(self) -> None:
        sources = build_generation_context(
            [GenerationContextTest._result(index) for index in range(1, 4)]
        )
        plan = self._answerable_plan("SOURCE_2")
        client = self._client_with_responses(plan, self._answerable_result())
        generator = OpenAIGenerator(client=client)

        await generator.generate("질문", sources)

        answer_input = client.responses.parse.await_args_list[1].kwargs["input"]
        self.assertIn("### SOURCE_2", answer_input)
        self.assertNotIn("### SOURCE_1", answer_input)
        self.assertNotIn("### SOURCE_3", answer_input)

    async def test_source_plan_withheld_skips_answer_generation(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.WITHHELD,
            answer_type=GenerationAnswerType.GENERAL,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[],
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )
        client = self._client_with_responses(plan)
        generator = OpenAIGenerator(client=client)

        result = await generator.generate("질문", [])

        self.assertEqual(GenerationStatus.WITHHELD, result.status)
        self.assertEqual(
            GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
            result.withheld_reason,
        )
        client.responses.parse.assert_awaited_once()

    async def test_allows_four_and_five_citations_without_replanning(self) -> None:
        for count in (4, 5):
            with self.subTest(count=count):
                sources = build_generation_context(
                    [GenerationContextTest._result(index) for index in range(1, count + 1)]
                )
                plan = self._answerable_plan(
                    *(source.source_id for source in sources)
                )
                answer = GenerationResult(
                    status=GenerationStatus.ANSWERABLE,
                    answer_markdown=" ".join(
                        f"근거 [{source.source_id}]" for source in sources
                    ),
                    withheld_reason=None,
                )
                client = self._client_with_responses(plan, answer)
                result = await OpenAIGenerator(client=client).generate_with_trace(
                    "복합 질문", sources
                )
                self.assertEqual(answer, result.result)
                self.assertEqual(tuple(sources), result.stage_trace.selected_sources)
                self.assertEqual(1, result.stage_trace.planning_attempt_count)
                self.assertEqual(0, result.stage_trace.planning_regeneration_count)
                self.assertEqual(2, client.responses.parse.await_count)
                self.assertEqual(
                    build_answer_input("복합 질문", sources, plan),
                    client.responses.parse.await_args_list[1].kwargs["input"],
                )

    async def test_invalid_planned_source_returns_error_without_answer_call(
        self,
    ) -> None:
        plan = self._answerable_plan("SOURCE_9")
        client = self._client_with_responses(plan)
        generator = OpenAIGenerator(client=client)

        call_result = await generator.generate_with_trace("질문", [])

        self.assertIsInstance(call_result.error, RuntimeError)
        self.assertIn("SOURCE_9", str(call_result.error))
        self.assertFalse(call_result.trace.succeeded)
        self.assertEqual(plan, call_result.stage_trace.source_plan)
        self.assertEqual((), call_result.stage_trace.selected_sources)
        self.assertIsNone(call_result.stage_trace.pre_validation_result)
        client.responses.parse.assert_awaited_once()

    async def test_retries_one_transient_source_planning_error(self) -> None:
        expected = self._answerable_result()
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
                self._response(self._answerable_plan("SOURCE_1")),
                self._response(expected),
            ]
        )
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        result = await generator.generate("질문", sources)

        self.assertEqual(expected, result)
        self.assertEqual(3, client.responses.parse.await_count)

    async def test_repairs_one_source_plan_structure_validation_error(self) -> None:
        validation_error = self._single_source_validation_error()
        repaired_plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.FEATURE_SUMMARY,
            answer_scope=GenerationAnswerScope.SUMMARY,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="핵심 기능",
                    source_ids=["SOURCE_1"],
                )
            ],
            withheld_reason=None,
        )
        answer = self._answerable_result()
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                validation_error,
                self._response(repaired_plan),
                self._response(answer),
            ]
        )
        sources = build_generation_context(
            [GenerationContextTest._result(index) for index in range(1, 3)]
        )

        call_result = await OpenAIGenerator(client=client).generate_with_trace(
            "어떤 기능을 제공해?",
            sources,
        )

        repair_request = client.responses.parse.await_args_list[1].kwargs
        self.assertEqual(
            SOURCE_PLANNING_REPAIR_PROMPT_V11,
            repair_request["instructions"],
        )
        self.assertEqual(
            build_source_planning_repair_input(
                "어떤 기능을 제공해?",
                sources,
                str(validation_error),
            ),
            repair_request["input"],
        )
        self.assertEqual(answer, call_result.result)
        self.assertIsNone(call_result.error)
        self.assertEqual(2, call_result.stage_trace.planning_attempt_count)
        self.assertEqual(1, call_result.stage_trace.planning_regeneration_count)
        self.assertEqual(
            repaired_plan,
            call_result.stage_trace.planning_regeneration_result,
        )
        self.assertEqual(
            SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
            call_result.stage_trace.planning_regeneration_model_call.prompt_version,
        )
        self.assertEqual(3, client.responses.parse.await_count)

    async def test_returns_error_when_source_plan_structure_repair_fails(
        self,
    ) -> None:
        first_error = self._single_source_validation_error()
        second_error = self._single_source_validation_error()
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[first_error, second_error]
        )

        call_result = await OpenAIGenerator(client=client).generate_with_trace(
            "어떤 기능을 제공해?",
            [],
        )

        self.assertIs(second_error, call_result.error)
        self.assertEqual(2, call_result.stage_trace.planning_attempt_count)
        self.assertEqual(1, call_result.stage_trace.planning_regeneration_count)
        self.assertFalse(
            call_result.stage_trace.planning_regeneration_model_call.succeeded
        )
        self.assertIsNone(call_result.stage_trace.planning_regeneration_result)
        self.assertEqual(2, client.responses.parse.await_count)

    async def test_stops_after_one_retry_when_source_planning_error_continues(
        self,
    ) -> None:
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        with self.assertRaises(APITimeoutError):
            await generator.generate("질문", [])

        self.assertEqual(2, client.responses.parse.await_count)

    async def test_does_not_retry_non_transient_source_planning_error(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=ValueError("invalid"))
        generator = OpenAIGenerator(client=client)

        with self.assertRaisesRegex(ValueError, "invalid"):
            await generator.generate("질문", [])

        client.responses.parse.assert_awaited_once()

    async def test_rejects_missing_source_plan_output(self) -> None:
        client = self._client_with_responses(None)
        generator = OpenAIGenerator(client=client)

        with self.assertRaisesRegex(RuntimeError, "Structured Output"):
            await generator.generate("질문", [])

    def test_requires_api_key_and_configures_client_retry_and_timeout(self) -> None:
        with patch(
            "app.answering.generator.get_settings",
            return_value=SimpleNamespace(openai_api_key=None),
        ):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIGenerator()

        with patch(
            "app.answering.generator.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key"),
        ), patch("app.answering.generator.AsyncOpenAI") as client_class:
            OpenAIGenerator()

        client_class.assert_called_once_with(
            api_key="test-key",
            max_retries=0,
            timeout=30.0,
        )

    async def test_trace_reports_tokens_without_retry_on_first_success(self) -> None:
        plan = self._answerable_plan("SOURCE_1")
        answer = self._answerable_result()
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                self._response(
                    plan,
                    input_tokens=400,
                    output_tokens=50,
                ),
                self._response(
                    answer,
                    input_tokens=1200,
                    output_tokens=300,
                ),
            ]
        )
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        generation_call = await generator.generate_with_trace("질문", sources)

        self.assertIsNone(generation_call.error)
        self.assertTrue(generation_call.trace.succeeded)
        self.assertEqual(0, generation_call.trace.retry_count)
        self.assertEqual(1600, generation_call.trace.input_tokens)
        self.assertEqual(350, generation_call.trace.output_tokens)
        self.assertEqual(
            GENERATION_PROMPT_VERSION,
            generation_call.trace.prompt_version,
        )
        self.assertEqual(
            OPENAI_GENERATION_MODEL,
            generation_call.trace.model_name,
        )
        self.assertEqual(plan, generation_call.stage_trace.source_plan)
        self.assertEqual(
            (sources[0],),
            generation_call.stage_trace.selected_sources,
        )
        self.assertEqual(
            answer,
            generation_call.stage_trace.pre_validation_result,
        )
        self.assertIsNone(generation_call.stage_trace.validation_error)

    async def test_trace_counts_answer_retry_as_a_single_logical_call(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                self._response(self._answerable_plan("SOURCE_1")),
                APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
                self._response(self._answerable_result()),
            ]
        )
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        generation_call = await generator.generate_with_trace("질문", sources)

        self.assertIsNone(generation_call.error)
        self.assertEqual(1, generation_call.trace.retry_count)

    async def test_trace_keeps_last_source_planning_error_when_attempts_fail(
        self,
    ) -> None:
        error = APITimeoutError(httpx.Request("POST", "https://api.openai.com"))
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        generation_call = await generator.generate_with_trace("질문", [])

        self.assertIs(error, generation_call.error)
        self.assertIsNone(generation_call.result)
        self.assertFalse(generation_call.trace.succeeded)
        self.assertEqual(1, generation_call.trace.retry_count)
        self.assertIsNotNone(generation_call.trace.error_message)

    async def test_validation_regeneration_reuses_question_sources_and_coverage(
        self,
    ) -> None:
        sources = tuple(
            build_generation_context([GenerationContextTest._result(1)])
        )
        plan = self._answerable_plan("SOURCE_1")
        invalid_answer = GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="잘못된 인용 [SOURCE_9]",
            withheld_reason=None,
        )
        repaired_answer = self._answerable_result()
        stage_trace = GenerationStageTrace(
            source_plan=plan,
            selected_sources=sources,
            pre_validation_result=invalid_answer,
            planning_attempt_count=1,
            answer_attempt_count=1,
        )
        client = self._client_with_responses(repaired_answer)
        generator = OpenAIGenerator(client=client)

        call_result = await generator.regenerate_answer_with_trace(
            "원 질문",
            stage_trace,
            "SOURCE_9는 전달되지 않았습니다.",
        )

        request = client.responses.parse.await_args.kwargs
        self.assertEqual(ANSWER_REPAIR_PROMPT_V17, request["instructions"])
        self.assertEqual(GenerationResult, request["text_format"])
        self.assertEqual(
            build_answer_repair_input(
                "원 질문",
                sources,
                plan,
                invalid_answer,
                "SOURCE_9는 전달되지 않았습니다.",
            ),
            request["input"],
        )
        self.assertEqual(repaired_answer, call_result.result)
        self.assertEqual(
            invalid_answer,
            call_result.stage_trace.pre_validation_result,
        )
        self.assertEqual(
            repaired_answer,
            call_result.stage_trace.validation_regeneration_result,
        )
        self.assertEqual(1, call_result.stage_trace.validation_regeneration_count)
        self.assertEqual(2, call_result.stage_trace.answer_attempt_count)
        self.assertEqual(ANSWER_REPAIR_PROMPT_VERSION, call_result.trace.prompt_version)
        client.responses.parse.assert_awaited_once()

    async def test_validation_regeneration_keeps_api_retry_separate_and_bounded(
        self,
    ) -> None:
        sources = tuple(
            build_generation_context([GenerationContextTest._result(1)])
        )
        stage_trace = GenerationStageTrace(
            source_plan=self._answerable_plan("SOURCE_1"),
            selected_sources=sources,
            pre_validation_result=self._answerable_result(),
            planning_attempt_count=1,
            answer_attempt_count=1,
        )
        error = APITimeoutError(httpx.Request("POST", "https://api.openai.com"))
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        call_result = await generator.regenerate_answer_with_trace(
            "질문",
            stage_trace,
            "marker 없음",
        )

        self.assertIs(error, call_result.error)
        self.assertEqual(2, client.responses.parse.await_count)
        self.assertEqual(1, call_result.trace.retry_count)
        self.assertEqual(1, call_result.stage_trace.validation_regeneration_count)
        self.assertEqual(3, call_result.stage_trace.answer_attempt_count)
        self.assertIsNone(
            call_result.stage_trace.validation_regeneration_result
        )

    @staticmethod
    def _client_with_responses(*results: object) -> Mock:
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[OpenAIGeneratorTest._response(result) for result in results]
        )
        return client

    @staticmethod
    def _response(
        result: object,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> SimpleNamespace:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return SimpleNamespace(output_parsed=result, usage=usage)

    @staticmethod
    def _answerable_plan(*source_ids: str) -> GenerationSourcePlan:
        return GenerationSourcePlanTest._answerable_plan(*source_ids)

    @staticmethod
    def _answerable_result() -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="답변입니다. [SOURCE_1]",
            withheld_reason=None,
        )

    @staticmethod
    def _single_source_validation_error() -> ValidationError:
        with unittest.TestCase().assertRaises(ValidationError) as caught:
            GenerationSourcePlan(
                status=GenerationStatus.ANSWERABLE,
                answer_type=GenerationAnswerType.FEATURE_SUMMARY,
                answer_scope=GenerationAnswerScope.SUMMARY,
                evidence_requirements=[
                    GenerationEvidenceRequirement(
                        information_unit="핵심 기능",
                        source_ids=["SOURCE_1", "SOURCE_2"],
                    )
                ],
                withheld_reason=None,
            )
        return caught.exception


if __name__ == "__main__":
    unittest.main()
