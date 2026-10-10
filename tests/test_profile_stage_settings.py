"""프로필 버전 구성 행으로 단계별 모델, 프롬프트 버전, 요청 설정을 고르는 규칙(#211 2-a).

- 구성 행 → 단계별 설정: 행의 모델, 프롬프트 버전(등록부 본문), params 를 쓰고 params 가 비면 코드 기본값
- 클라이언트: 재작성, 판별, 생성(계획, 계획 재시도, 답변, 답변 재시도)이 호출마다 받은 설정으로 요청하고
  model_calls 에 남길 관측값도 그 설정이다
- fail closed: 행 없음, 등록부에 없는 버전, 지문 불일치, 묶음 버전 불일치, 출력 스키마 불일치 등
- 두 서비스가 서로 다른 프롬프트 버전을 골라 같은 프로세스 클라이언트로 동시에 답한다(테스트 전용 등록부)

실제 프롬프트 파일은 추가하지 않는다. 테스트 전용 버전은 등록부 조회와 지문 표를 patch 해서 만든다.
"""

import asyncio
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import prompts
from app.answering.generator import (
    ANSWER_PROMPT_V17,
    ANSWER_REPAIR_PROMPT_V17,
    GENERATION_PROMPT_VERSION,
    OPENAI_GENERATION_MODEL,
    SOURCE_PLANNING_PROMPT_V11,
    GenerationStageSettings,
    OpenAIGenerator,
    with_repair_instructions,
)
from app.answering.models import GenerationResult, GenerationSourcePlan
from app.answering.service import GenerationService
from app.chat.log_store import RagLogStore
from app.chat.query_rewrite import (
    OPENAI_QUERY_REWRITE_MODEL,
    QUERY_REWRITE_MAX_OUTPUT_TOKENS,
    QUERY_REWRITE_PROMPT_V4,
    QUERY_REWRITE_PROMPT_VERSION,
    QueryRewriteDecision,
    QueryRewriteOutput,
    QueryRewriteService,
)
from app.chat.service import ChatService
from app.core.stage_settings import StageSettings
from app.profiles import stage_settings as stage_settings_module
from app.profiles.services import HELP_CHATBOT, HELP_CHATBOT_TEST
from app.profiles.stage_settings import (
    ProfileStageSettingsError,
    component_problems,
    stage_settings_from_components,
)
from app.prompts.fingerprints import composite_fingerprint, load_fingerprint_table, sha256_text
from app.question_grouping.constants import JUDGE_MODEL, JUDGE_PROMPT_VERSION
from app.question_grouping.judge_client import QuestionJudgeClient
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS
from app.retrieval.models import HybridRetrievalResult, HybridSearchCall, RetrievalResult
from app.core.model_trace import ModelCallTrace
from tests.profile_components import component_rows
from tests.test_help_chatbot_identity import ALPHA_CHUNK, ANSWER, SOURCE_PLAN


def _revision(rows: List[SimpleNamespace], revision_id: int = 1, profile_id: int = 7) -> SimpleNamespace:
    """구성 행과 옛 칸이 같은 프로필 버전."""

    by_stage = {row.stage: row for row in rows}
    return SimpleNamespace(
        id=revision_id,
        profile_id=profile_id,
        document_group_id=23,
        semantic_cache_enabled=False,
        exact_cache_enabled=False,
        generation_model_name=by_stage["generation"].model_name,
        generation_prompt_version=by_stage["generation"].prompt_version,
        query_rewrite_model_name=by_stage["rewrite"].model_name,
        query_rewrite_prompt_version=by_stage["rewrite"].prompt_version,
    )


def _row(rows: List[SimpleNamespace], stage: str) -> SimpleNamespace:
    return next(row for row in rows if row.stage == stage)


# ---------------------------------------------------------------------------
# 테스트 전용 프롬프트 버전(실제 파일 없음)
# ---------------------------------------------------------------------------

FAKE_ANSWER_VERSION = "v23-test"
FAKE_ANSWER_REPAIR_VERSION = "v23-test-repair-1"
FAKE_GENERATION_VERSION = "v40-test"
FAKE_ANSWER_TEXT = "[테스트 전용] 답변 지시문. 근거 밖 내용은 쓰지 않는다."
FAKE_ANSWER_REPAIR_SUFFIX = "[테스트 전용] 직전 답변의 검증 실패를 고친다."
FAKE_ANSWER_REPAIR_TEXT = with_repair_instructions(FAKE_ANSWER_TEXT, FAKE_ANSWER_REPAIR_SUFFIX)
FAKE_PROMPTS = {
    ("generation.answer", FAKE_ANSWER_VERSION): FAKE_ANSWER_TEXT,
    ("generation.answer_repair", FAKE_ANSWER_REPAIR_VERSION): FAKE_ANSWER_REPAIR_SUFFIX,
}


def _fake_table() -> Dict[str, str]:
    table = dict(load_fingerprint_table())
    table[f"generation.answer@{FAKE_ANSWER_VERSION}"] = sha256_text(FAKE_ANSWER_TEXT)
    table[f"generation.answer_repair@{FAKE_ANSWER_REPAIR_VERSION}"] = sha256_text(
        FAKE_ANSWER_REPAIR_TEXT
    )
    # 출력 스키마는 바꾸지 않은 버전이다.
    table[f"generation.answer.output_schema@{FAKE_ANSWER_VERSION}"] = table[
        "generation.answer.output_schema@v23"
    ]
    parts = {
        key: table[key]
        for key in (
            "generation.source_planning@v34",
            "generation.source_planning_repair@v34-repair-1",
            "generation.source_planning.output_schema@v34",
            f"generation.answer@{FAKE_ANSWER_VERSION}",
            f"generation.answer_repair@{FAKE_ANSWER_REPAIR_VERSION}",
            f"generation.answer.output_schema@{FAKE_ANSWER_VERSION}",
        )
    }
    table[f"generation@{FAKE_GENERATION_VERSION}"] = composite_fingerprint(parts)
    return table


@contextmanager
def fake_registry() -> Iterator[None]:
    """등록부와 지문 표에 테스트 전용 답변 프롬프트 버전을 더한다."""

    real_get = prompts.get

    def get(component: str, version: str) -> str:
        return FAKE_PROMPTS.get((component, version)) or real_get(component, version)

    table = _fake_table()
    with patch("app.prompts.get", side_effect=get), patch.object(
        stage_settings_module, "_fingerprint_table", return_value=table
    ):
        yield


def fake_answer_rows(generation_model: str) -> List[SimpleNamespace]:
    """답변과 답변 재시도, 묶음 버전만 테스트 전용 버전으로 고른 구성 행."""

    table = _fake_table()
    rows = component_rows(generation_model=generation_model)
    for stage, version, key in (
        ("answer", FAKE_ANSWER_VERSION, "generation.answer"),
        ("answer_repair", FAKE_ANSWER_REPAIR_VERSION, "generation.answer_repair"),
        ("generation", FAKE_GENERATION_VERSION, "generation"),
    ):
        row = _row(rows, stage)
        row.prompt_version = version
        row.prompt_key = f"{key}@{version}"
        row.prompt_sha256 = table[row.prompt_key]
    return rows


# ---------------------------------------------------------------------------
# 구성 행 → 단계별 설정
# ---------------------------------------------------------------------------


class StageSettingsResolutionTest(TestCase):
    def test_code_components_resolve_to_code_defaults(self) -> None:
        rows = component_rows()
        settings = stage_settings_from_components(_revision(rows), rows)

        self.assertEqual(1, settings.revision_id)
        self.assertEqual(
            (OPENAI_QUERY_REWRITE_MODEL, QUERY_REWRITE_PROMPT_VERSION, QUERY_REWRITE_PROMPT_V4),
            (settings.rewrite.model_name, settings.rewrite.prompt_version, settings.rewrite.instructions),
        )
        self.assertEqual({"max_output_tokens": QUERY_REWRITE_MAX_OUTPUT_TOKENS}, settings.rewrite.request_params())
        self.assertEqual(
            (JUDGE_MODEL, JUDGE_PROMPT_VERSION, JUDGE_INSTRUCTIONS),
            (settings.judge.model_name, settings.judge.prompt_version, settings.judge.instructions),
        )
        generation = settings.generation
        self.assertEqual((OPENAI_GENERATION_MODEL, GENERATION_PROMPT_VERSION), (generation.model_name, generation.prompt_version))
        self.assertEqual(SOURCE_PLANNING_PROMPT_V11, generation.source_planning.instructions)
        self.assertEqual(ANSWER_PROMPT_V17, generation.answer.instructions)
        self.assertEqual(ANSWER_REPAIR_PROMPT_V17, generation.answer_repair.instructions)
        self.assertEqual({}, generation.answer.request_params())

    def test_null_params_use_stage_defaults_like_dev_backfill_rows(self) -> None:
        # DEV 백필 행은 재작성 params 가 NULL 이다. 코드가 보내던 max_output_tokens 를 그대로 보낸다.
        rows = component_rows()
        _row(rows, "rewrite").params = None
        settings = stage_settings_from_components(_revision(rows), rows)
        self.assertEqual({"max_output_tokens": QUERY_REWRITE_MAX_OUTPUT_TOKENS}, settings.rewrite.request_params())

    def test_each_stage_takes_the_rows_model_and_params(self) -> None:
        rows = component_rows(generation_model="gen-x", rewrite_model="rewrite-x")
        _row(rows, "rewrite").params = {"max_output_tokens": 300}
        judge = _row(rows, "judge")
        judge.model_name = "judge-x"
        judge.params = {"reasoning": {"effort": "medium"}, "max_output_tokens": 2048}
        _row(rows, "answer").params = {"max_output_tokens": 4000}

        settings = stage_settings_from_components(_revision(rows), rows)

        self.assertEqual(("rewrite-x", {"max_output_tokens": 300}), (settings.rewrite.model_name, settings.rewrite.request_params()))
        self.assertEqual(
            ("judge-x", {"reasoning": {"effort": "medium"}, "max_output_tokens": 2048}),
            (settings.judge.model_name, settings.judge.request_params()),
        )
        generation = settings.generation
        self.assertEqual(
            {"gen-x"},
            {
                generation.model_name,
                generation.source_planning.model_name,
                generation.source_planning_repair.model_name,
                generation.answer.model_name,
                generation.answer_repair.model_name,
            },
        )
        self.assertEqual({"max_output_tokens": 4000}, generation.answer.request_params())
        self.assertEqual({}, generation.source_planning.request_params())

    def test_registered_test_only_version_is_read_from_registry(self) -> None:
        with fake_registry():
            rows = fake_answer_rows(OPENAI_GENERATION_MODEL)
            settings = stage_settings_from_components(_revision(rows), rows)

        self.assertEqual(FAKE_GENERATION_VERSION, settings.generation.prompt_version)
        self.assertEqual(
            (FAKE_ANSWER_VERSION, FAKE_ANSWER_TEXT),
            (settings.generation.answer.prompt_version, settings.generation.answer.instructions),
        )
        self.assertEqual(FAKE_ANSWER_REPAIR_TEXT, settings.generation.answer_repair.instructions)
        self.assertEqual(SOURCE_PLANNING_PROMPT_V11, settings.generation.source_planning.instructions)


class FailClosedTest(TestCase):
    def _problems(self, mutate) -> tuple:
        rows = component_rows()
        revision = _revision(rows)
        mutate(rows, revision)
        return component_problems(revision, rows)

    def assertFailsWith(self, mutate, expected: str) -> None:
        problems = self._problems(mutate)
        self.assertTrue(any(expected in problem for problem in problems), problems)
        rows = component_rows()
        revision = _revision(rows)
        mutate(rows, revision)
        with self.assertRaises(ProfileStageSettingsError):
            stage_settings_from_components(revision, rows)

    def test_code_components_have_no_problems(self) -> None:
        self.assertEqual((), self._problems(lambda rows, revision: None))

    def test_missing_stage_row(self) -> None:
        for stage in ("rewrite", "judge", "generation", "source_planning", "answer_repair"):
            with self.subTest(stage=stage):
                self.assertFailsWith(
                    lambda rows, _r, stage=stage: rows.remove(_row(rows, stage)),
                    f"{stage}: 구성 행 없음",
                )

    def test_prompt_version_missing_from_registry(self) -> None:
        def mutate(rows, revision):
            row = _row(rows, "judge")
            row.prompt_version = "question-grouping-v99"
            row.prompt_key = "judge@question-grouping-v99"

        self.assertFailsWith(mutate, "실행 중인 코드에 없는 프롬프트 버전 judge@question-grouping-v99")

    def test_sha256_mismatch(self) -> None:
        def mutate(rows, revision):
            _row(rows, "answer").prompt_sha256 = "0" * 64

        self.assertFailsWith(mutate, "generation.answer@v23 지문 행 000000000000")

    def test_missing_sha256(self) -> None:
        self.assertFailsWith(
            lambda rows, _r: setattr(_row(rows, "rewrite"), "prompt_sha256", None),
            "rewrite@v10 지문 없음",
        )

    def test_repair_sha_is_checked_against_base_plus_suffix(self) -> None:
        # 재시도 행의 지문이 덧붙임만의 지문이면 실제로 보내는 지시문과 다르다.
        def mutate(rows, revision):
            _row(rows, "answer_repair").prompt_sha256 = sha256_text(
                prompts.get("generation.answer_repair", "v23-repair-1")
            )

        self.assertFailsWith(mutate, "answer_repair: generation.answer_repair@v23-repair-1 지문 행")

    def test_composite_missing_from_table(self) -> None:
        def mutate(rows, revision):
            row = _row(rows, "generation")
            row.prompt_version = "v41"
            row.prompt_key = "generation@v41"
            revision.generation_prompt_version = "v41"

        self.assertFailsWith(mutate, "실행 중인 코드에 없는 묶음 버전 generation@v41")

    def test_composite_must_match_sub_stage_rows(self) -> None:
        # 하위 단계가 다른 버전을 가리키면 묶음 지문을 다시 계산한 값이 표와 다르다.
        with fake_registry():
            rows = fake_answer_rows(OPENAI_GENERATION_MODEL)
            for row in rows:
                if row.stage == "generation":
                    row.prompt_version = GENERATION_PROMPT_VERSION
                    row.prompt_key = f"generation@{GENERATION_PROMPT_VERSION}"
                    row.prompt_sha256 = load_fingerprint_table()[row.prompt_key]
            problems = component_problems(_revision(rows), rows)
        self.assertIn("generation: generation@v40 가 하위 단계 구성 행과 맞지 않음", problems)

    def test_sub_stage_model_must_match_composite_model(self) -> None:
        self.assertFailsWith(
            lambda rows, _r: setattr(_row(rows, "answer"), "model_name", "other-model"),
            "answer: 모델 other-model 가 묶음 행 모델",
        )

    def test_unsupported_params(self) -> None:
        cases = {
            "지원하지 않는 요청 설정 temperature": {"temperature": 0.2},
            "max_output_tokens 가 양의 정수가 아님": {"max_output_tokens": 0},
            "reasoning 이 객체가 아님": {"reasoning": "low"},
            "요청 설정이 객체가 아님": ["max_output_tokens"],
        }
        for expected, params in cases.items():
            with self.subTest(expected=expected):
                self.assertFailsWith(
                    lambda rows, _r, params=params: setattr(_row(rows, "judge"), "params", params),
                    expected,
                )
        self.assertFailsWith(
            lambda rows, _r: setattr(_row(rows, "generation"), "params", {"max_output_tokens": 10}),
            "묶음 행에는 요청 설정을 두지 않음",
        )

    def test_output_schema_mismatch(self) -> None:
        # 실행 중인 코드가 보내는 스키마가 그 버전의 기록과 다르면 그 버전을 서빙하지 않는다.
        schemas = dict(stage_settings_module._code_output_schemas())
        schemas[stage_settings_module.Stage.REWRITE] = "f" * 64
        with patch.object(stage_settings_module, "_code_output_schemas", return_value=schemas):
            problems = self._problems(lambda rows, revision: None)
        self.assertEqual(
            ("rewrite: 출력 스키마 rewrite.output_schema@v10 표 c8c7956a086f, 코드 ffffffffffff",),
            problems,
        )

    def test_legacy_columns_must_match_rows(self) -> None:
        self.assertFailsWith(
            lambda _rows, revision: setattr(revision, "generation_model_name", "other"),
            "프로필 버전 칸 generation_model_name=other 가 구성 행과 다름",
        )

    def test_prompt_key_must_match_version(self) -> None:
        self.assertFailsWith(
            lambda rows, _r: setattr(_row(rows, "rewrite"), "prompt_key", "rewrite@v9"),
            "프롬프트 키 rewrite@v9 가 rewrite@v10 와 다름",
        )


# ---------------------------------------------------------------------------
# 클라이언트가 호출마다 받은 설정으로 요청한다
# ---------------------------------------------------------------------------


class _RecordingResponses:
    def __init__(self, outputs: Dict[Any, Any]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._outputs = outputs

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        output = self._outputs[kwargs["text_format"]]
        if isinstance(output, Exception):
            self._outputs[kwargs["text_format"]] = SOURCE_PLAN
            raise output
        return SimpleNamespace(output_parsed=output, status="completed", incomplete_details=None, usage=None)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(status="completed", incomplete_details=None, output_text="{}", usage=None)


class _RecordingClient:
    def __init__(self, outputs: Optional[Dict[Any, Any]] = None) -> None:
        self.responses = _RecordingResponses(
            outputs
            or {
                GenerationSourcePlan: SOURCE_PLAN,
                GenerationResult: ANSWER,
                QueryRewriteOutput: QueryRewriteOutput(
                    decision=QueryRewriteDecision.NEW_TOPIC,
                    selected_turn_no=None,
                    context_phrase=None,
                    resolved_query=None,
                ),
            }
        )


def _stage(name: str, **params: Any) -> StageSettings:
    return StageSettings(
        model_name=f"{name}-model",
        prompt_version=f"{name}-version",
        instructions=f"{name} 지시문",
        params=params,
    )


def _generation_stages() -> GenerationStageSettings:
    return GenerationStageSettings(
        model_name="generation-model",
        prompt_version="generation-composite",
        source_planning=_stage("planning"),
        source_planning_repair=_stage("planning-repair"),
        answer=_stage("answer", max_output_tokens=4000),
        answer_repair=_stage("answer-repair", reasoning={"effort": "low"}),
    )


class ClientStageSettingsTest(IsolatedAsyncioTestCase):
    async def test_query_rewrite_uses_stage_model_prompt_and_params(self) -> None:
        client = _RecordingClient()
        checkpoints = []

        async def before(provider, model, version):
            checkpoints.append((provider, model, version))

        call = await QueryRewriteService(client=client).rewrite(
            "질문", [], before_model_call=before, stage=_stage("rewrite", max_output_tokens=300)
        )

        request = client.responses.calls[0]
        self.assertEqual(
            ("rewrite-model", "rewrite 지시문", 300),
            (request["model"], request["instructions"], request["max_output_tokens"]),
        )
        self.assertEqual([("openai", "rewrite-model", "rewrite-version")], checkpoints)
        self.assertEqual(("rewrite-model", "rewrite-version"), (call.trace.model_name, call.trace.prompt_version))

    async def test_query_rewrite_without_stage_keeps_code_request(self) -> None:
        client = _RecordingClient()
        await QueryRewriteService(client=client).rewrite("질문", [])
        request = client.responses.calls[0]
        self.assertEqual(
            (OPENAI_QUERY_REWRITE_MODEL, QUERY_REWRITE_PROMPT_V4, QUERY_REWRITE_MAX_OUTPUT_TOKENS),
            (request["model"], request["instructions"], request["max_output_tokens"]),
        )

    async def test_judge_uses_stage_model_prompt_and_params(self) -> None:
        client = _RecordingClient()
        checkpoints = []

        async def before(provider, model, version):
            checkpoints.append((provider, model, version))

        call = await QuestionJudgeClient(client=client).judge(
            {"q": 1},
            before_model_call=before,
            stage=_stage("judge", reasoning={"effort": "medium"}, max_output_tokens=2048),
        )

        request = client.responses.calls[0]
        self.assertEqual("judge-model", request["model"])
        self.assertEqual("judge 지시문", request["instructions"])
        self.assertEqual({"effort": "medium"}, request["reasoning"])
        self.assertEqual(2048, request["max_output_tokens"])
        self.assertEqual("json_schema", request["text"]["format"]["type"])
        self.assertEqual([("openai", "judge-model", "judge-version")], checkpoints)
        self.assertEqual(("judge-model", "judge-version"), (call.trace.model_name, call.trace.prompt_version))

    async def test_generation_uses_each_sub_stage_and_records_composite(self) -> None:
        planning_error = None
        try:
            GenerationSourcePlan.model_validate({})
        except ValidationError as error:
            planning_error = error
        client = _RecordingClient(
            {GenerationSourcePlan: planning_error, GenerationResult: ANSWER}
        )
        generator = OpenAIGenerator(client=client)
        stages = _generation_stages()
        source = SimpleNamespace(source_id="SOURCE_1", chunk=ALPHA_CHUNK)

        call = await generator.generate_with_trace("질문", [source], stages)
        repair = await generator.regenerate_answer_with_trace(
            "질문", call.stage_trace, "검증 실패", stages
        )

        self.assertEqual(
            [
                ("planning-model", "planning 지시문", set()),
                ("planning-repair-model", "planning-repair 지시문", set()),
                ("answer-model", "answer 지시문", {"max_output_tokens"}),
                ("answer-repair-model", "answer-repair 지시문", {"reasoning"}),
            ],
            [
                (
                    request["model"],
                    request["instructions"],
                    set(request) - {"model", "instructions", "input", "text_format"},
                )
                for request in client.responses.calls
            ],
        )
        self.assertEqual(
            ("generation-model", "generation-composite"),
            (call.trace.model_name, call.trace.prompt_version),
        )
        planning_trace = call.stage_trace.planning_regeneration_model_call
        self.assertEqual(
            ("planning-repair-model", "planning-repair-version"),
            (planning_trace.model_name, planning_trace.prompt_version),
        )
        self.assertEqual(
            ("answer-repair-model", "answer-repair-version"),
            (repair.trace.model_name, repair.trace.prompt_version),
        )

    async def test_generation_service_checkpoints_the_composite(self) -> None:
        client = _RecordingClient()
        service = GenerationService(OpenAIGenerator(client=client))
        checkpoints = []

        async def before(provider, model, version):
            checkpoints.append((provider, model, version))

        result = await service.generate_answer(
            "질문",
            [HybridRetrievalResult(chunk=ALPHA_CHUNK, rrf_score=0.03, final_rank=1, bm25_rank=1, vector_rank=1)],
            before_model_call=before,
            stages=_generation_stages(),
        )

        self.assertEqual([("openai", "generation-model", "generation-composite")], checkpoints)
        self.assertEqual(
            ("generation-model", "generation-composite"),
            (result.model_call.model_name, result.model_call.prompt_version),
        )


# ---------------------------------------------------------------------------
# 두 서비스가 서로 다른 프롬프트 버전으로 동시에 답한다
# ---------------------------------------------------------------------------


class _FixedRetriever:
    async def search_with_trace(self, _query, *, before_model_call) -> HybridSearchCall:
        await before_model_call("openai", "text-embedding-test", None)
        await asyncio.sleep(0)
        return HybridSearchCall(
            bm25_results=(RetrievalResult(chunk=ALPHA_CHUNK, score=3.0, rank=1),),
            vector_results=(RetrievalResult(chunk=ALPHA_CHUNK, score=0.9, rank=1),),
            fused_results=(
                HybridRetrievalResult(chunk=ALPHA_CHUNK, rrf_score=0.03, final_rank=1, bm25_rank=1, vector_rank=1),
            ),
            bm25_latency_ms=1,
            vector_latency_ms=1,
            embedding_call=ModelCallTrace(provider="openai", model_name="text-embedding-test", succeeded=True, latency_ms=1),
        )


def _log_store(model_calls: List[Dict[str, Any]]) -> AsyncMock:
    store = AsyncMock(spec=RagLogStore)
    ids = iter(range(1, 100))

    async def start_model_call(**kwargs: Any) -> Any:
        model_calls.append(kwargs)
        return SimpleNamespace(id=next(ids))

    store.create_conversation.return_value = SimpleNamespace(id=uuid.uuid4())
    store.start_rag_run.return_value = SimpleNamespace(id=uuid.uuid4(), turn_no=1)
    store.start_model_call.side_effect = start_model_call
    return store


class TwoServicesDifferentPromptVersionsTest(IsolatedAsyncioTestCase):
    async def test_services_answer_concurrently_with_their_own_versions(self) -> None:
        # 프로세스가 공유하는 클라이언트 하나를 두 서비스가 함께 쓴다. 설정은 호출마다 넘긴다.
        client = _RecordingClient()
        generation_service = GenerationService(OpenAIGenerator(client=client))
        rows_by_revision = {
            1: component_rows(),
            2: fake_answer_rows("gpt-test-b"),
        }
        revision_by_service = {
            HELP_CHATBOT.profile_key: _revision(rows_by_revision[1], revision_id=1, profile_id=7),
            HELP_CHATBOT_TEST.profile_key: _revision(rows_by_revision[2], revision_id=2, profile_id=8),
        }

        async def resolve(_session, *, service, conversation_id=None):
            return revision_by_service[service.profile_key]

        async def read(_session, revision_id):
            return rows_by_revision[revision_id]

        model_calls: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
        services = {
            label: ChatService(
                retriever=None,
                generation_service=generation_service,
                query_rewrite_service=QueryRewriteService(client=client),
                log_store=_log_store(model_calls[label]),
                session=AsyncMock(spec=AsyncSession),
                index_version_id=None,
                service=route,
                retriever_factory=lambda _group_id: (_FixedRetriever(), 41),
            )
            for label, route in (("A", HELP_CHATBOT), ("B", HELP_CHATBOT_TEST))
        }

        with fake_registry(), patch(
            "app.chat.service.resolve_chat_profile_revision", side_effect=resolve
        ), patch("app.profiles.stage_settings.read_profile_components", side_effect=read):
            responses = await asyncio.gather(
                services["A"].answer_question("질문"),
                services["B"].answer_question("질문"),
            )

        self.assertEqual(["COMPLETED", "COMPLETED"], [response.status.value for response in responses])
        answer_requests = {
            request["model"]: request["instructions"]
            for request in client.responses.calls
            if request["text_format"] is GenerationResult
        }
        self.assertEqual(
            {OPENAI_GENERATION_MODEL: ANSWER_PROMPT_V17, "gpt-test-b": FAKE_ANSWER_TEXT},
            answer_requests,
        )
        # 계획 단계는 두 프로필 버전이 같은 버전을 골랐다.
        self.assertEqual(
            {SOURCE_PLANNING_PROMPT_V11},
            {
                request["instructions"]
                for request in client.responses.calls
                if request["text_format"] is GenerationSourcePlan
            },
        )
        generation_logs = {
            label: [
                (call["model_name"], call["prompt_version"])
                for call in calls
                if call["purpose"] == "ANSWER_GENERATION"
            ]
            for label, calls in model_calls.items()
        }
        self.assertEqual(
            {
                "A": [(OPENAI_GENERATION_MODEL, GENERATION_PROMPT_VERSION)],
                "B": [("gpt-test-b", FAKE_GENERATION_VERSION)],
            },
            generation_logs,
        )
        # 턴 행에는 각자 고른 프로필 버전이 남는다.
        for label, revision_id in (("A", 1), ("B", 2)):
            store = services[label]._log_store
            self.assertEqual(revision_id, store.start_rag_run.await_args.kwargs["profile_revision_id"])


if __name__ == "__main__":
    import unittest

    unittest.main()
