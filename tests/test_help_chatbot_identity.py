"""HELP_CHATBOT 턴 흐름이 모델에 보내는 요청과 단계 순서를 스냅샷으로 고정한다.

ChatService 부터 실제 OpenAIGenerator, QueryRewriteService, QuestionJudgeClient, OpenAIEmbedder,
HybridRetriever, VectorRetriever, QuestionGroupingService 를 그대로 쓰고 바깥 경계만 가짜로
바꾼다. 가짜는 OpenAI 클라이언트(요청을 기록하고 정해진 응답을 돌려줌), BM25 검색기와 pgvector
조회(고정 검색 결과), 로그 저장과 세션(DB 쓰기 없음), 판별 저장소와 카탈로그와 outline 읽기
(고정 카탈로그, 정확 일치 조회 결과)다.

모델 호출마다 단계, 모델, 추론 설정, 출력 한도, 요청 인자 이름, 지시문과 입력과 출력 스키마의
sha256 을 남기고, 시나리오마다 단계 순서와 model_calls 기록(목적, 모델, 프롬프트 판)과
진행 단계와 응답 상태를 남긴다. 결과를 tests/fixtures/help_chatbot_identity.json 과
바이트 단위로 비교한다.

동작을 일부러 바꾼 PR 에서만 `UPDATE_IDENTITY_SNAPSHOT=1` 환경변수를 켜고 이 테스트를 돌려
fixture 를 다시 만든다. 다시 만든 diff 는 리뷰에서 그대로 확인한다.
"""

import json
import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.answering import generator as generation_prompts
from app.answering.generator import OPENAI_GENERATION_MODEL, OpenAIGenerator
from app.answering.models import (
    GenerationAnswerScope,
    GenerationAnswerType,
    GenerationEvidenceRequirement,
    GenerationPlanningStatus,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStatus,
)
from app.answering.service import GenerationService
from app.chat.log_store import RagLogStore
from app.chat.progress import ProgressStage
from app.chat.query_rewrite import (
    OPENAI_QUERY_REWRITE_MODEL,
    QUERY_REWRITE_PROMPT_V4,
    QUERY_REWRITE_PROMPT_VERSION,
    QueryRewriteCandidateTurn,
    QueryRewriteDecision,
    QueryRewriteOutput,
    QueryRewriteService,
    QueryRewriteTurnStatus,
)
from app.chat.service import ChatService
from app.core.prompt_fingerprints import (
    load_fingerprint_table,
    sha256_json,
    sha256_text,
    text_format_of,
)
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    ChatProfileRevisionStatus,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.question_grouping.judge_client import QuestionJudgeClient
from app.question_grouping.models import (
    CanonicalCitationSnapshot,
    CatalogCanonicalAnswer,
    CitationIndexContext,
    DocumentOutline,
    ExactQuestionLogMatch,
    GateCanonicalAnswer,
    GateInputs,
    GateSubproblemState,
    IndexedSection,
    IndexScope,
    SubproblemCatalog,
    SubproblemCatalogItem,
)
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS
from app.question_grouping.service import QuestionGroupingService
from app.question_grouping.store import validate_gate_result, validate_turn_judgment
from app.retrieval.embedding import OpenAIEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalChunk, RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "help_chatbot_identity.json"
UPDATE_ENV = "UPDATE_IDENTITY_SNAPSHOT"

# 고정 식별자. 판별 payload 의 섞기 seed 가 rag_run_id 에서 나오므로 바꾸면 입력 지문이 바뀐다.
CONVERSATION_ID = uuid.UUID("00000000-0000-4000-8000-000000000211")
RAG_RUN_ID = uuid.UUID("00000000-0000-4000-8000-000000002111")
PREVIOUS_RAG_RUN_ID = uuid.UUID("00000000-0000-4000-8000-000000002110")
SOURCE_RAG_RUN_ID = uuid.UUID("00000000-0000-4000-8000-000000002100")
PROFILE_REVISION_ID = 21
DOCUMENT_GROUP_ID = 31
INDEX_VERSION_ID = 41
CLASSIFICATION_RUN_ID = 51
CHUNKING_CONFIG_ID = 61
EMBEDDING_CONFIG_ID = 71

# 실제 서비스 문구와 섞이지 않도록 모든 질문과 문서는 합성 표시를 붙인다.
FIRST_QUESTION = "[합성 테스트] 가짜 기능 알파는 어떻게 켜나요?"
FOLLOW_UP_QUESTION = "[합성 테스트] 그 기능을 끄려면요?"
FOLLOW_UP_CONTEXT_PHRASE = "가짜 기능 알파"
FOLLOW_UP_RESOLVED_QUERY = "[합성 테스트] 가짜 기능 알파를 끄려면요?"
PREVIOUS_ANSWER = "가짜 기능 알파는 합성 설정 화면에서 켭니다. [1]"
CANONICAL_ANSWER = "합성 정본: 가짜 기능 알파는 합성 설정 화면의 스위치로 켭니다. [1]"
GENERATED_ANSWER = "가짜 기능 알파는 합성 설정 화면에서 스위치를 눌러 켭니다. [SOURCE_1]"

ALPHA_SOURCE_ID, BETA_SOURCE_ID = 101, 102
ALPHA_VERSION_ID, BETA_VERSION_ID = 201, 202
ALPHA_CHUNK_ID, BETA_CHUNK_ID = 301, 302
ALPHA_KEY, BETA_KEY = "synthetic/alpha", "synthetic/beta"
ALPHA_SUBPROBLEM = uuid.UUID("00000000-0000-4000-8000-00000000a001")
BETA_SUBPROBLEM = uuid.UUID("00000000-0000-4000-8000-00000000b001")
ALPHA_GROUP = uuid.UUID("00000000-0000-4000-8000-00000000a002")
BETA_GROUP = uuid.UUID("00000000-0000-4000-8000-00000000b002")
ALPHA_CANONICAL = uuid.UUID("00000000-0000-4000-8000-00000000a003")

SCOPE = IndexScope(
    index_version_id=INDEX_VERSION_ID,
    document_group_id=DOCUMENT_GROUP_ID,
    chunking_config_id=CHUNKING_CONFIG_ID,
    embedding_config_id=EMBEDDING_CONFIG_ID,
)


def _unit_vector(index: int) -> List[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


def _chunk(chunk_id: int, version_id: int, key: str, title: str, section: str, content: str) -> RetrievalChunk:
    return RetrievalChunk(
        document_id=key,
        section_id=f"{key}:{chunk_id}",
        document_title=title,
        section_path=(title, section),
        source_url=f"https://synthetic.invalid/{key}",
        category="synthetic",
        content=content,
        chunk_id=chunk_id,
        document_version_id=version_id,
        index_version_id=INDEX_VERSION_ID,
    )


ALPHA_CHUNK = _chunk(
    ALPHA_CHUNK_ID,
    ALPHA_VERSION_ID,
    ALPHA_KEY,
    "합성 문서 알파",
    "켜고 끄기",
    "가짜 기능 알파는 합성 설정 화면의 스위치를 눌러 켜거나 끕니다.",
)
BETA_CHUNK = _chunk(
    BETA_CHUNK_ID,
    BETA_VERSION_ID,
    BETA_KEY,
    "합성 문서 베타",
    "초기화",
    "가짜 기능 베타는 합성 관리 화면에서 초기화합니다.",
)


def _catalog_item(subproblem_id, key, group_id, source_id, document_key, vector, canonical) -> SubproblemCatalogItem:
    return SubproblemCatalogItem(
        subproblem_id=subproblem_id,
        key=key,
        name=f"[합성] {key}",
        inclusion_criteria=(f"[합성] {key} 방법을 묻는다",),
        exclusion_criteria=("[합성] 다른 기능을 묻는다",),
        current_version=1,
        problem_group_id=group_id,
        document_source_id=source_id,
        document_key=document_key,
        serving_state=QuestionSubproblemServingState.SERVING,
        document_title=f"[합성] {document_key}",
        canonical_answer=(
            CatalogCanonicalAnswer(canonical_answer_id=ALPHA_CANONICAL, content_markdown=CANONICAL_ANSWER)
            if canonical
            else None
        ),
        inclusion_embedding=tuple(vector),
    )


CATALOG = SubproblemCatalog(
    items=(
        _catalog_item(ALPHA_SUBPROBLEM, "alpha.enable", ALPHA_GROUP, ALPHA_SOURCE_ID, ALPHA_KEY, _unit_vector(0), True),
        _catalog_item(BETA_SUBPROBLEM, "beta.reset", BETA_GROUP, BETA_SOURCE_ID, BETA_KEY, _unit_vector(1), False),
    ),
    skipped_missing_embedding=0,
    embedding_text_versions=("name-inclusion-v1",),
)
OUTLINES = {
    ALPHA_VERSION_ID: DocumentOutline(ALPHA_SOURCE_ID, ALPHA_VERSION_ID, ALPHA_KEY, "합성 문서 알파", "synthetic", ("켜고 끄기",)),
    BETA_VERSION_ID: DocumentOutline(BETA_SOURCE_ID, BETA_VERSION_ID, BETA_KEY, "합성 문서 베타", "synthetic", ("초기화",)),
}
GATE_INPUTS = GateInputs(
    subproblem=GateSubproblemState(
        ALPHA_SUBPROBLEM,
        QuestionSubproblemStatus.APPROVED,
        QuestionSubproblemServingState.SERVING,
        1,
    ),
    canonical_answer=GateCanonicalAnswer(ALPHA_CANONICAL, 1, CANONICAL_ANSWER),
    citations=(
        CanonicalCitationSnapshot(
            citation_order=1,
            chunk_id=ALPHA_CHUNK_ID,
            document_version_id=ALPHA_VERSION_ID,
            document_source_id=ALPHA_SOURCE_ID,
            content_hash="synthetic-alpha-hash",
            node_order=1,
            node_identity_hash="synthetic-alpha-node",
        ),
    ),
    contexts_by_source_id={
        ALPHA_SOURCE_ID: CitationIndexContext(
            ALPHA_VERSION_ID,
            (
                IndexedSection(
                    chunk_id=ALPHA_CHUNK_ID,
                    document_version_id=ALPHA_VERSION_ID,
                    content_hash="synthetic-alpha-hash",
                    node_order=1,
                    node_identity_hash="synthetic-alpha-node",
                    document_title="합성 문서 알파",
                    node_path="합성 문서 알파 > 켜고 끄기",
                    source_uri=f"https://synthetic.invalid/{ALPHA_KEY}",
                ),
            ),
        )
    },
)
EXACT_MATCH = ExactQuestionLogMatch(
    subproblem_id=ALPHA_SUBPROBLEM,
    key="alpha.enable",
    problem_group_id=ALPHA_GROUP,
    current_version=1,
    document_source_id=ALPHA_SOURCE_ID,
    document_key=ALPHA_KEY,
    normalized_question=FIRST_QUESTION,
    source_rag_run_id=SOURCE_RAG_RUN_ID,
    classification_id=901,
    matched_count=1,
)
PREVIOUS_TURN = QueryRewriteCandidateTurn(
    rag_run_id=PREVIOUS_RAG_RUN_ID,
    turn_no=1,
    status=QueryRewriteTurnStatus.COMPLETED,
    user_query=FIRST_QUESTION,
    resolved_query=FIRST_QUESTION,
    answer_content=PREVIOUS_ANSWER,
    withheld_reason_code=None,
)


def _judge_output(decision: str) -> str:
    connect = decision == "CONNECT"
    return json.dumps(
        {
            "decision": decision,
            "groupId": ALPHA_KEY if connect else None,
            "subproblemId": "alpha.enable" if connect else None,
            "confidence": 0.9 if connect else 0.6,
            "rationaleCode": "SAME_ASK" if connect else "DIFFERENT_ASK",
            "matchedCriteria": ["I1"] if connect else [],
            "conflictingCriteria": [],
            "ambiguityReason": None,
            "documentDecision": "SUBPROBLEM_DOCUMENT" if connect else "NONE",
            "documentCandidateId": None,
            "documentRationaleCode": None if connect else "NO_CANDIDATE_COVERS",
        }
    )


SOURCE_PLAN = GenerationSourcePlan(
    status=GenerationPlanningStatus.ANSWERABLE,
    answer_type=GenerationAnswerType.PROCEDURE,
    answer_scope=GenerationAnswerScope.SUMMARY,
    evidence_requirements=[
        GenerationEvidenceRequirement(information_unit="합성 설정 방법", source_ids=["SOURCE_1"])
    ],
    optional_context=[],
    unanswered_information=[],
    related_guidance=[],
    withheld_reason=None,
)
ANSWER = GenerationResult(
    status=GenerationStatus.ANSWERABLE,
    limitation_markdown=None,
    answer_markdown=GENERATED_ANSWER,
    withheld_reason=None,
)

# 지시문으로 생성 단계 이름을 정한다. 목록에 없는 지시문은 unknown 으로 남아 스냅샷이 깨진다.
_PARSE_STAGES = {
    generation_prompts.SOURCE_PLANNING_PROMPT_V11: "generation.source_planning",
    generation_prompts.SOURCE_PLANNING_REPAIR_PROMPT_V11: "generation.source_planning_repair",
    generation_prompts.ANSWER_PROMPT_V17: "generation.answer",
    generation_prompts.ANSWER_REPAIR_PROMPT_V17: "generation.answer_repair",
    QUERY_REWRITE_PROMPT_V4: "query_rewrite",
}


# ---------------------------------------------------------------------------
# 가짜 경계
# ---------------------------------------------------------------------------


_CONTENT_KEYS = frozenset(
    {"model", "instructions", "input", "text", "text_format", "reasoning", "max_output_tokens"}
)


class _Recorder:
    """단계 순서와 모델 요청을 한곳에 쌓는다."""

    def __init__(self) -> None:
        self.stages: List[str] = []
        self.model_requests: List[Dict[str, Any]] = []
        self.model_call_logs: List[Dict[str, Any]] = []
        self.progress: List[str] = []

    def request(self, stage: str, kwargs: Dict[str, Any], *, schema: Any, payload: Any) -> None:
        self.stages.append(stage)
        instructions = kwargs.get("instructions")
        self.model_requests.append(
            {
                "stage": stage,
                "model": kwargs.get("model"),
                "reasoning": kwargs.get("reasoning"),
                "maxOutputTokens": kwargs.get("max_output_tokens"),
                "requestKeys": sorted(kwargs),
                # store, dimensions, encoding_format 같은 나머지 스칼라 인자.
                "options": {
                    key: value
                    for key, value in sorted(kwargs.items())
                    if key not in _CONTENT_KEYS and isinstance(value, (str, int, float, bool))
                },
                "instructionsSha256": None if instructions is None else sha256_text(instructions),
                "inputSha256": sha256_text(payload) if isinstance(payload, str) else sha256_json(payload),
                "outputSchemaSha256": None if schema is None else sha256_json(schema),
            }
        )


class _FakeResponses:
    def __init__(self, recorder: _Recorder, outputs: Dict[str, Any]) -> None:
        self._recorder = recorder
        self._outputs = outputs

    async def parse(self, **kwargs: Any) -> Any:
        stage = _PARSE_STAGES.get(kwargs.get("instructions"), "unknown")
        self._recorder.request(
            stage,
            kwargs,
            schema=text_format_of(kwargs["text_format"]),
            payload=kwargs["input"],
        )
        return SimpleNamespace(
            output_parsed=self._outputs[stage],
            status="completed",
            incomplete_details=None,
            usage=None,
        )

    async def create(self, **kwargs: Any) -> Any:
        stage = "judge" if kwargs.get("instructions") == JUDGE_INSTRUCTIONS else "unknown"
        self._recorder.request(
            stage,
            kwargs,
            schema=kwargs.get("text", {}).get("format"),
            payload=kwargs["input"],
        )
        return SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output_text=self._outputs[stage],
            usage=None,
        )


class _FakeAsyncOpenAI:
    def __init__(self, recorder: _Recorder, outputs: Dict[str, Any]) -> None:
        self.responses = _FakeResponses(recorder, outputs)


class _FakeEmbeddingsRaw:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def create(self, **kwargs: Any) -> Any:
        self._recorder.request("query_embedding", kwargs, schema=None, payload=list(kwargs["input"]))
        response = SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=_unit_vector(0))
                for index, _ in enumerate(kwargs["input"])
            ],
            usage=None,
        )
        return SimpleNamespace(parse=lambda: response, retries_taken=0)


class _FakeSyncOpenAI:
    def __init__(self, recorder: _Recorder) -> None:
        self.embeddings = SimpleNamespace(with_raw_response=_FakeEmbeddingsRaw(recorder))

    def with_options(self, **_options: Any) -> "_FakeSyncOpenAI":
        return self


class _FixedBm25:
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        return [
            RetrievalResult(chunk=ALPHA_CHUNK, score=3.5, rank=1),
            RetrievalResult(chunk=BETA_CHUNK, score=1.25, rank=2),
        ][:top_k]


class _FixedSearchReader:
    async def similarity_search(self, embedding: Sequence[float], top_k: int) -> List[Tuple[RetrievalChunk, float]]:
        return [(ALPHA_CHUNK, 0.875), (BETA_CHUNK, 0.5)][:top_k]


class _FakeGroupingStore:
    def __init__(self, recorder: _Recorder, exact_match: Optional[ExactQuestionLogMatch]) -> None:
        self._recorder = recorder
        self._exact_match = exact_match
        self._classifications = 0

    async def find_exact_question_log_match(self, document_group_id, question, *, exclude_rag_run_id):
        self._recorder.stages.append("exact_cache_lookup")
        return self._exact_match

    async def get_or_open_online_run(self, **_kwargs: Any) -> int:
        return CLASSIFICATION_RUN_ID

    async def insert_question_embedding(self, rag_run_id, *, embedding, embedding_config_id) -> None:
        return None

    async def insert_classification(self, rag_run_id, *, run_id, judgment, judgment_input) -> int:
        validate_turn_judgment(judgment)
        json.dumps(judgment_input, allow_nan=False)
        self._classifications += 1
        return 800 + self._classifications

    async def insert_cache_attempt(self, rag_run_id, *, classification_id, gate, latency_ms) -> int:
        validate_gate_result(gate)
        self._recorder.stages.append(f"cache_attempt:{gate.outcome.value}")
        return 700

    async def finalize_citation_attribution(self, classification_id, citations) -> bool:
        return True


class _FakeCatalogReader:
    async def load_index_scope(self, index_version_id: int) -> IndexScope:
        return SCOPE

    async def load_subproblem_catalog(self, scope: IndexScope) -> SubproblemCatalog:
        return CATALOG

    async def load_gate_inputs(self, subproblem_id, scope) -> GateInputs:
        return GATE_INPUTS


class _FakeOutlineReader:
    async def load_outlines(self, version_ids: Sequence[int], *, chunking_config_id: int):
        return {version: OUTLINES[version] for version in version_ids if version in OUTLINES}


def _log_store(recorder: _Recorder, turn_no: int) -> AsyncMock:
    store = AsyncMock(spec=RagLogStore)
    next_ids = iter(range(1, 100))

    async def start_model_call(**kwargs: Any) -> Any:
        recorder.model_call_logs.append(
            {
                "purpose": kwargs["purpose"],
                "provider": kwargs.get("provider"),
                "modelName": kwargs.get("model_name"),
                "promptVersion": kwargs.get("prompt_version"),
                "classificationRun": kwargs.get("classification_run_id") is not None,
            }
        )
        return SimpleNamespace(id=next(next_ids))

    def terminal(label: str):
        async def effect(*_args: Any, **_kwargs: Any) -> None:
            recorder.stages.append(label)

        return effect

    store.create_conversation.return_value = SimpleNamespace(id=CONVERSATION_ID)
    store.start_rag_run.return_value = SimpleNamespace(id=RAG_RUN_ID, turn_no=turn_no)
    store.start_model_call.side_effect = start_model_call
    store.get_query_rewrite_candidates.return_value = [PREVIOUS_TURN] if turn_no > 1 else []
    store.complete_rag_run.side_effect = terminal("rag_run:COMPLETED")
    store.withhold_rag_run.side_effect = terminal("rag_run:WITHHELD")
    store.fail_rag_run.side_effect = terminal("rag_run:FAILED")
    return store


def _profile_revision(*, caches_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        id=PROFILE_REVISION_ID,
        document_group_id=DOCUMENT_GROUP_ID,
        semantic_cache_enabled=caches_enabled,
        exact_cache_enabled=caches_enabled,
        generation_model_name=OPENAI_GENERATION_MODEL,
        generation_prompt_version=generation_prompts.GENERATION_PROMPT_VERSION,
        query_rewrite_model_name=OPENAI_QUERY_REWRITE_MODEL,
        query_rewrite_prompt_version=QUERY_REWRITE_PROMPT_VERSION,
    )


# ---------------------------------------------------------------------------
# 시나리오
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # 첫 턴: 정확 일치 없음, 판별은 SEPARATE 라 캐시 거절, 생성으로 답한다.
    "first_turn": dict(turn_no=1, question=FIRST_QUESTION, caches_enabled=True, exact_match=None, judge="SEPARATE"),
    # 후속 턴: Query Rewrite 가 앞 턴 문맥으로 질의를 확정한 뒤 같은 흐름을 탄다.
    "follow_up_turn": dict(turn_no=2, question=FOLLOW_UP_QUESTION, caches_enabled=True, exact_match=None, judge="SEPARATE"),
    # 정확 일치 캐시: 승인된 과거 첫 턴 질문과 같아 모델 호출 없이 정본으로 답한다.
    "exact_cache_serve": dict(turn_no=1, question=FIRST_QUESTION, caches_enabled=True, exact_match=EXACT_MATCH, judge=None),
    # 의미 캐시: 판별이 CONNECT 라 게이트가 정본을 서빙하고 생성은 부르지 않는다.
    "semantic_cache_serve": dict(turn_no=1, question=FIRST_QUESTION, caches_enabled=True, exact_match=None, judge="CONNECT"),
    # 캐시 꺼짐: 프로필의 두 캐시 스위치가 꺼져 판별 없이 검색과 생성만 한다.
    "caches_disabled": dict(turn_no=1, question=FIRST_QUESTION, caches_enabled=False, exact_match=None, judge=None),
}


async def run_scenario(
    *,
    turn_no: int,
    question: str,
    caches_enabled: bool,
    exact_match: Optional[ExactQuestionLogMatch],
    judge: Optional[str],
) -> Dict[str, Any]:
    recorder = _Recorder()
    outputs: Dict[str, Any] = {
        "generation.source_planning": SOURCE_PLAN,
        "generation.answer": ANSWER,
        "query_rewrite": QueryRewriteOutput(
            decision=QueryRewriteDecision.FOLLOW_UP_RESOLVED,
            selected_turn_no=1,
            context_phrase=FOLLOW_UP_CONTEXT_PHRASE,
            resolved_query=FOLLOW_UP_RESOLVED_QUERY,
        ),
    }
    if judge is not None:
        outputs["judge"] = _judge_output(judge)

    async_client = _FakeAsyncOpenAI(recorder, outputs)
    embedder = OpenAIEmbedder(client=_FakeSyncOpenAI(recorder))
    session = AsyncMock(spec=AsyncSession)
    log_store = _log_store(recorder, turn_no)
    retriever = HybridRetriever(
        bm25_retriever=_FixedBm25(),
        vector_retriever=VectorRetriever(embedder=embedder, store=_FixedSearchReader()),
    )
    grouping = QuestionGroupingService(
        session,
        log_store,
        QuestionJudgeClient(client=async_client),
        embedder,
        store=_FakeGroupingStore(recorder, exact_match),
        catalog_reader=_FakeCatalogReader(),
        outline_reader=_FakeOutlineReader(),
    )
    service = ChatService(
        retriever=None,
        generation_service=GenerationService(OpenAIGenerator(client=async_client)),
        query_rewrite_service=QueryRewriteService(client=async_client),
        log_store=log_store,
        session=session,
        index_version_id=None,
        profile_status=ChatProfileRevisionStatus.PUBLISHED,
        retriever_factory=lambda _group_id: (retriever, INDEX_VERSION_ID),
        question_grouping=grouping,
        question_grouping_enabled=True,
    )

    async def on_progress_stage(stage: ProgressStage) -> None:
        recorder.progress.append(stage.value)

    with patch(
        "app.chat.service.resolve_chat_profile_revision",
        new=AsyncMock(return_value=_profile_revision(caches_enabled=caches_enabled)),
    ):
        response = await service.answer_question(
            question,
            None if turn_no == 1 else CONVERSATION_ID,
            on_progress_stage=on_progress_stage,
        )

    answer = getattr(response, "answer", None)
    answer_markdown = getattr(answer, "answer_markdown", None)
    return {
        "stages": recorder.stages,
        "progressStages": recorder.progress,
        "modelCallLogs": recorder.model_call_logs,
        "modelRequests": recorder.model_requests,
        "response": {
            "status": response.status.value,
            "answerMarkdownSha256": None if answer_markdown is None else sha256_text(answer_markdown),
            "citationCount": len(getattr(response, "citations", None) or []),
        },
    }


async def build_snapshot() -> Dict[str, Any]:
    return {
        "description": (
            "HELP_CHATBOT 턴 흐름의 모델 요청과 단계 순서. "
            "tests/test_help_chatbot_identity.py 가 만든다. 손으로 고치지 않는다."
        ),
        "scenarios": {name: await run_scenario(**spec) for name, spec in SCENARIOS.items()},
    }


def serialize(snapshot: Dict[str, Any]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class HelpChatbotIdentityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.snapshot = await build_snapshot()
        self.serialized = serialize(self.snapshot)
        if os.environ.get(UPDATE_ENV) == "1":
            FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE_PATH.write_text(self.serialized, "utf-8")

    async def test_snapshot_matches_fixture_byte_for_byte(self) -> None:
        self.assertTrue(
            FIXTURE_PATH.exists(),
            f"{FIXTURE_PATH.name} 가 없습니다. {UPDATE_ENV}=1 로 이 테스트를 돌려 만드세요.",
        )
        expected_text = FIXTURE_PATH.read_text("utf-8")
        expected = json.loads(expected_text)
        message = (
            "HELP_CHATBOT 동작 스냅샷이 달라졌습니다. 의도한 동작 변경이면 "
            f"{UPDATE_ENV}=1 로 fixture 를 다시 만들고 판과 시스템 버전을 함께 올리세요."
        )
        for name in sorted(set(expected["scenarios"]) | set(self.snapshot["scenarios"])):
            with self.subTest(scenario=name):
                self.assertEqual(
                    expected["scenarios"].get(name),
                    self.snapshot["scenarios"].get(name),
                    message,
                )
        self.assertEqual(expected_text.encode("utf-8"), self.serialized.encode("utf-8"), message)

    async def test_every_request_hash_is_a_registered_fingerprint(self) -> None:
        registered = set(load_fingerprint_table().values())
        for name, scenario in self.snapshot["scenarios"].items():
            for request in scenario["modelRequests"]:
                for field in ("instructionsSha256", "outputSchemaSha256"):
                    digest = request[field]
                    if digest is None:
                        continue
                    with self.subTest(scenario=name, stage=request["stage"], field=field):
                        self.assertIn(digest, registered)

    async def test_scenarios_cover_expected_paths(self) -> None:
        scenarios = self.snapshot["scenarios"]
        stages = {name: [request["stage"] for request in value["modelRequests"]] for name, value in scenarios.items()}

        self.assertEqual([], stages["exact_cache_serve"])
        self.assertNotIn("generation.answer", stages["semantic_cache_serve"])
        self.assertIn("judge", stages["semantic_cache_serve"])
        self.assertIn("query_rewrite", stages["follow_up_turn"])
        self.assertNotIn("query_rewrite", stages["first_turn"])
        self.assertNotIn("judge", stages["caches_disabled"])
        for value in scenarios.values():
            self.assertNotIn("unknown", [request["stage"] for request in value["modelRequests"]])


if __name__ == "__main__":
    unittest.main()
