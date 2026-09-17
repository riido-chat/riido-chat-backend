"""QuestionGroupingService 분기 단위 테스트. DB 와 API 없이 가짜 협력자로 순서와 기록을 고정한다."""

import json
import logging
import unittest
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

from app.chat.log_store import CitationLog
from app.core.model_trace import ModelCallTrace
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    AttributionSource,
    CacheAttemptOutcome,
    ClassificationDecision,
    ExecutionStatus,
    ModelCallPurpose,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.question_grouping.catalog_reader import CatalogDataError
from app.question_grouping.constants import (
    INVALID_UNKNOWN_SUBPROBLEM_KEY,
    INVALID_UNPARSEABLE_OUTPUT,
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER,
    REJECT_CANONICAL_ANSWER_NOT_FOUND,
    REJECT_CANONICAL_DATA_INVALID,
    REJECT_CANONICAL_SERVE_FAILED,
    REJECT_CITED_SECTION_CHANGED,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
    REJECT_SUBPROBLEM_VERSION_MISMATCH,
)
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
    JudgeCall,
    JudgeFailure,
    JudgeFailureKind,
    SubproblemCatalog,
    SubproblemCatalogItem,
)
from app.question_grouping.service import (
    FAILURE_STAGE_CANDIDATES,
    FAILURE_STAGE_JUDGE,
    FAILURE_STAGE_QUESTION_EMBEDDING,
    GroupingTurn,
    QuestionGroupingService,
)
from app.question_grouping.store import validate_gate_result, validate_turn_judgment
from app.retrieval.embedding import EmbeddingResponse
from app.retrieval.models import HybridSearchCall, RetrievalChunk, RetrievalResult

RAG_RUN_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
RUN_ID = 7
GROUP_ID = 3
INDEX_ID = 5
SCOPE = IndexScope(index_version_id=INDEX_ID, document_group_id=GROUP_ID, chunking_config_id=2, embedding_config_id=9)
QUESTION = "구독을 취소하면 환불되나요?"
BILLING_SOURCE, MEMBERS_SOURCE = 11, 12
BILLING_VERSION, MEMBERS_VERSION = 21, 22
BILLING_KEY, MEMBERS_KEY = "workspaces/plans-and-billing", "workspaces/members"
CANCEL_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
INVITE_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
CANONICAL_ID = uuid.UUID("44444444-4444-4444-8444-444444444444")
CANCEL_GROUP = uuid.UUID("55555555-5555-4555-8555-555555555555")
INVITE_GROUP = uuid.UUID("66666666-6666-4666-8666-666666666666")
CITED_CHUNK = 501


def _vec(index: int) -> Tuple[float, ...]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return tuple(vector)


def _item(subproblem_id, key, group_id, source_id, document_key, vector, canonical=True) -> SubproblemCatalogItem:
    return SubproblemCatalogItem(
        subproblem_id=subproblem_id,
        key=key,
        name=f"{key} 이름",
        inclusion_criteria=("기준 하나", "기준 둘"),
        exclusion_criteria=("제외",),
        current_version=1,
        problem_group_id=group_id,
        document_source_id=source_id,
        document_key=document_key,
        serving_state=QuestionSubproblemServingState.SERVING,
        document_title="문서",
        canonical_answer=(
            CatalogCanonicalAnswer(canonical_answer_id=CANONICAL_ID, content_markdown="정본 [1]") if canonical else None
        ),
        inclusion_embedding=vector,
    )


CATALOG = SubproblemCatalog(
    items=(
        _item(CANCEL_ID, "billing.cancel", CANCEL_GROUP, BILLING_SOURCE, BILLING_KEY, _vec(0)),
        _item(INVITE_ID, "members.invite", INVITE_GROUP, MEMBERS_SOURCE, MEMBERS_KEY, _vec(1), canonical=False),
    ),
    skipped_missing_embedding=1,
    embedding_text_versions=("name-inclusion-v1",),
)

OUTLINES = {
    BILLING_VERSION: DocumentOutline(BILLING_SOURCE, BILLING_VERSION, BILLING_KEY, "구독 및 결제", "workspaces", ("구독 취소",)),
    MEMBERS_VERSION: DocumentOutline(MEMBERS_SOURCE, MEMBERS_VERSION, MEMBERS_KEY, "멤버", "workspaces", ("초대",)),
}

SECTION = IndexedSection(
    chunk_id=CITED_CHUNK,
    document_version_id=BILLING_VERSION,
    content_hash="c-cancel",
    node_order=1,
    node_identity_hash="id-cancel",
    document_title="구독 및 결제",
    node_path="구독 및 결제 > 구독 취소",
    source_uri="https://docs.riido.io/billing",
)
CITATION = CanonicalCitationSnapshot(
    citation_order=1,
    chunk_id=CITED_CHUNK,
    document_version_id=BILLING_VERSION,
    document_source_id=BILLING_SOURCE,
    content_hash="c-cancel",
    node_order=1,
    node_identity_hash="id-cancel",
)


def _gate_inputs(
    *,
    serving_state: QuestionSubproblemServingState = QuestionSubproblemServingState.SERVING,
    current_version: int = 1,
    canonical: bool = True,
    section: IndexedSection = SECTION,
) -> GateInputs:
    state = GateSubproblemState(CANCEL_ID, QuestionSubproblemStatus.APPROVED, serving_state, current_version)
    if not canonical:
        return GateInputs(subproblem=state)
    return GateInputs(
        subproblem=state,
        canonical_answer=GateCanonicalAnswer(CANONICAL_ID, 1, "현재 정본 [1]"),
        citations=(CITATION,),
        contexts_by_source_id={BILLING_SOURCE: CitationIndexContext(BILLING_VERSION, (section,))},
    )


def _result(chunk_id: int, version_id: int, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk=RetrievalChunk(
            document_id=f"doc-{version_id}",
            section_id=f"doc-{version_id}:{chunk_id}",
            document_title="문서",
            section_path=("문서",),
            source_url="https://docs.riido.io",
            category=None,
            content="본문",
            chunk_id=chunk_id,
            document_version_id=version_id,
            index_version_id=INDEX_ID,
        ),
        score=1.0,
        rank=rank,
    )


def _search(retrieval_query: str = QUESTION, embedding: Optional[Tuple[float, ...]] = None) -> HybridSearchCall:
    results = (_result(CITED_CHUNK, BILLING_VERSION, 1), _result(601, MEMBERS_VERSION, 2))
    return HybridSearchCall(
        bm25_results=results,
        vector_results=results,
        retrieval_query=retrieval_query,
        query_embedding=_vec(0) if embedding is None else embedding,
    )


def _trace(succeeded: bool = True, error: Optional[str] = None) -> ModelCallTrace:
    return ModelCallTrace(
        provider=JUDGE_PROVIDER,
        model_name=JUDGE_MODEL,
        succeeded=succeeded,
        latency_ms=1200,
        retry_count=0 if succeeded else 1,
        input_tokens=5000,
        output_tokens=300,
        cached_input_tokens=4000,
        reasoning_tokens=120,
        prompt_version=JUDGE_PROMPT_VERSION,
        error_message=error,
    )


def _connect_output(key: str = "billing.cancel", group: str = BILLING_KEY) -> str:
    return json.dumps(
        {
            "decision": "CONNECT",
            "groupId": group,
            "subproblemId": key,
            "confidence": 0.91,
            "rationaleCode": "SAME_ASK",
            "ambiguityReason": None,
            "matchedCriteria": ["I1"],
            "conflictingCriteria": [],
            "documentDecision": "SUBPROBLEM_DOCUMENT",
            "documentCandidateId": None,
            "documentRationaleCode": None,
        }
    )


def _separate_output(document_id: Optional[str]) -> str:
    return json.dumps(
        {
            "decision": "SEPARATE",
            "groupId": None,
            "subproblemId": None,
            "confidence": 0.7,
            "rationaleCode": "DIFFERENT_ASK",
            "ambiguityReason": None,
            "matchedCriteria": [],
            "conflictingCriteria": [],
            "documentDecision": "MATCHED" if document_id else "NONE",
            "documentCandidateId": document_id,
            "documentRationaleCode": "ASK_COVERED" if document_id else "NO_CANDIDATE_COVERS",
        }
    )


# ---------------------------------------------------------------------------
# 가짜 협력자
# ---------------------------------------------------------------------------


class FakeSession:
    def __init__(self, events: List[Any]) -> None:
        self.events = events

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")


class FakeLogStore:
    def __init__(self, events: List[Any], fail_start: bool = False) -> None:
        self.events = events
        self.fail_start = fail_start
        self.started: List[Dict[str, Any]] = []
        self.finished: Dict[int, Dict[str, Any]] = {}
        self._next_id = 900

    async def lock_processing_run(self, rag_run_id: uuid.UUID) -> None:
        self.events.append("lock")

    async def start_model_call(self, **kwargs: Any) -> Any:
        if self.fail_start:
            raise RuntimeError("DB down")
        self._next_id += 1
        self.started.append({"id": self._next_id, **kwargs})
        self.events.append(("start", kwargs["purpose"]))
        return SimpleNamespace(id=self._next_id)

    async def finish_model_call(self, model_call_id: int, **kwargs: Any) -> None:
        self.finished[model_call_id] = kwargs
        self.events.append(("finish", model_call_id, kwargs["status"]))


class FakeStore:
    def __init__(self, events: List[Any], fail_classification: bool = False) -> None:
        self.events = events
        self.fail_classification = fail_classification
        self.embeddings: List[Dict[str, Any]] = []
        self.classifications: List[Dict[str, Any]] = []
        self.attempts: List[Dict[str, Any]] = []
        self.finalized: List[Any] = []
        self.exact_match: Optional[ExactQuestionLogMatch] = None
        self.exact_lookups: List[Tuple[Any, ...]] = []

    async def find_exact_question_log_match(self, document_group_id, question, *, exclude_rag_run_id):
        self.exact_lookups.append((document_group_id, question, exclude_rag_run_id))
        self.events.append("exact_lookup")
        return self.exact_match

    async def get_or_open_online_run(self, **kwargs: Any) -> int:
        self.events.append(("open_run", kwargs["model"], kwargs["prompt_version"]))
        return RUN_ID

    async def insert_question_embedding(self, rag_run_id, *, embedding, embedding_config_id) -> None:
        self.embeddings.append({"embedding": tuple(embedding), "config": embedding_config_id})
        self.events.append("embedding")

    async def insert_classification(self, rag_run_id, *, run_id, judgment, judgment_input) -> int:
        if self.fail_classification:
            raise RuntimeError("insert failed")
        validate_turn_judgment(judgment)
        assert "gate" in judgment_input
        json.dumps(judgment_input, allow_nan=False)
        self.classifications.append({"run_id": run_id, "judgment": judgment, "input": judgment_input})
        self.events.append("classification")
        return 100 + len(self.classifications)

    async def insert_cache_attempt(self, rag_run_id, *, classification_id, gate, latency_ms) -> int:
        validate_gate_result(gate)
        self.attempts.append({"classification_id": classification_id, "gate": gate, "latency_ms": latency_ms})
        self.events.append("attempt")
        return 200 + len(self.attempts)

    async def finalize_citation_attribution(self, classification_id, citations) -> bool:
        self.finalized.append((classification_id, list(citations)))
        return True


class FakeCatalogReader:
    def __init__(self, events: List[Any], gate_inputs: Optional[GateInputs] = None) -> None:
        self.events = events
        self.gate_inputs = gate_inputs if gate_inputs is not None else _gate_inputs()
        self.catalog_error: Optional[Exception] = None
        self.gate_error: Optional[Exception] = None

    async def load_index_scope(self, index_version_id: int) -> IndexScope:
        return SCOPE

    async def load_subproblem_catalog(self, scope: IndexScope) -> SubproblemCatalog:
        if self.catalog_error is not None:
            raise self.catalog_error
        return CATALOG

    async def load_gate_inputs(self, subproblem_id, scope) -> GateInputs:
        self.events.append("gate_inputs")
        if self.gate_error is not None:
            raise self.gate_error
        return self.gate_inputs


class FakeOutlineReader:
    def __init__(self, outlines: Optional[Dict[int, DocumentOutline]] = None) -> None:
        self.outlines = OUTLINES if outlines is None else outlines

    async def load_outlines(self, version_ids: Sequence[int], *, chunking_config_id: int):
        return {version: self.outlines[version] for version in version_ids if version in self.outlines}


class FakeJudgeClient:
    provider = JUDGE_PROVIDER
    model_name = JUDGE_MODEL
    prompt_version = JUDGE_PROMPT_VERSION

    def __init__(self, events: List[Any], call: Any = None) -> None:
        self.events = events
        self.call = call if call is not None else JudgeCall(trace=_trace(), output_text=_connect_output())
        self.payloads: List[Any] = []

    async def judge(self, payload, *, before_model_call=None) -> JudgeCall:
        self.payloads.append(payload)
        await before_model_call(JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_PROMPT_VERSION)
        self.events.append("judge_api")
        if isinstance(self.call, BaseException):
            raise self.call
        return self.call


class FakeEmbedder:
    def __init__(self, events: List[Any], error: Optional[Exception] = None, vector: Tuple[float, ...] = _vec(1)) -> None:
        self.events = events
        self.error = error
        self.vector = vector
        self.calls: List[Tuple[List[str], Dict[str, Any]]] = []

    def embed_many_with_usage(self, texts, **kwargs) -> EmbeddingResponse:
        self.calls.append((list(texts), kwargs))
        self.events.append("embed_api")
        if self.error is not None:
            raise self.error
        return EmbeddingResponse(embeddings=[list(self.vector)], input_tokens=12)


class _ServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        service_logger = logging.getLogger("app.question_grouping.service")
        service_logger.disabled = True
        self.addCleanup(setattr, service_logger, "disabled", False)
        self.events: List[Any] = []
        self.session = FakeSession(self.events)
        self.log_store = FakeLogStore(self.events)
        self.store = FakeStore(self.events)
        self.catalog = FakeCatalogReader(self.events)
        self.outlines = FakeOutlineReader()
        self.judge_client = FakeJudgeClient(self.events)
        self.embedder = FakeEmbedder(self.events)
        self.turn = GroupingTurn(RAG_RUN_ID, GROUP_ID, INDEX_ID, RUN_ID)

    def service(self) -> QuestionGroupingService:
        return QuestionGroupingService(
            self.session,
            self.log_store,
            self.judge_client,
            self.embedder,
            store=self.store,
            catalog_reader=self.catalog,
            outline_reader=self.outlines,
        )

    async def run_turn(self, *, semantic_cache_enabled: bool = True, search: Optional[HybridSearchCall] = None):
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, search or _search())
        self.assertTrue(prepared.ready, prepared.failure)
        judged = await service.judge(prepared)
        recorded = await service.record_judgment_and_gate(
            prepared, judged, semantic_cache_enabled=semantic_cache_enabled
        )
        return service, prepared, judged, recorded

    def judgment_input(self, index: int = -1) -> Dict[str, Any]:
        return self.store.classifications[index]["input"]

    def judge_call_id(self) -> int:
        return next(
            call["id"] for call in self.log_store.started
            if call["purpose"] == ModelCallPurpose.QUESTION_CLASSIFICATION.value
        )


# ---------------------------------------------------------------------------


class OpenOnlineRunTest(_ServiceTestCase):
    async def test_opens_run_with_judge_model_and_commits(self) -> None:
        turn = await self.service().open_online_run(
            rag_run_id=RAG_RUN_ID, document_group_id=GROUP_ID, index_version_id=INDEX_ID
        )

        self.assertEqual(GroupingTurn(RAG_RUN_ID, GROUP_ID, INDEX_ID, RUN_ID), turn)
        self.assertEqual([("open_run", JUDGE_MODEL, JUDGE_PROMPT_VERSION), "commit"], self.events)


class ConnectServedTest(_ServiceTestCase):
    async def test_served_builds_citation_logs_and_does_not_commit(self) -> None:
        _, prepared, judged, recorded = await self.run_turn()

        self.assertTrue(prepared.vector.reused_retrieval_embedding)
        self.assertEqual([], self.embedder.calls)
        judge_call_id = self.judge_call_id()
        self.assertEqual(
            [
                ("start", ModelCallPurpose.QUESTION_CLASSIFICATION.value),
                "commit",
                "judge_api",
                "lock",
                ("finish", judge_call_id, ExecutionStatus.SUCCESS),
                "embedding",
                "gate_inputs",
                "classification",
                "attempt",
            ],
            self.events,
        )
        started = self.log_store.started[0]
        self.assertEqual((RAG_RUN_ID, RUN_ID, JUDGE_PROMPT_VERSION), (started["rag_run_id"], started["classification_run_id"], started["prompt_version"]))
        self.assertEqual(4000, self.log_store.finished[judge_call_id]["cached_input_tokens"])

        self.assertEqual(CacheAttemptOutcome.SERVED, recorded.gate.outcome)
        self.assertTrue(recorded.served)
        self.assertEqual(ClassificationDecision.CONNECT, recorded.judgment.decision)
        self.assertEqual(AttributionSource.SUBPROBLEM, recorded.judgment.attribution.attribution_source)
        self.assertEqual("현재 정본 [1]", recorded.served_answer_markdown)
        self.assertEqual(
            (CitationLog(CITED_CHUNK, BILLING_VERSION, 1, "구독 및 결제", "구독 및 결제 > 구독 취소", "https://docs.riido.io/billing"),),
            recorded.served_citations,
        )
        self.assertEqual((101, 201), (recorded.classification_id, recorded.cache_attempt_id))
        self.assertEqual(CANONICAL_ID, self.store.attempts[0]["gate"].canonical_answer_id)
        self.assertEqual({"embedding": _vec(0), "config": SCOPE.embedding_config_id}, self.store.embeddings[0])

        data = self.judgment_input()
        self.assertEqual("SERVED", data["gate"]["outcome"])
        self.assertEqual(str(CANONICAL_ID), data["gate"]["canonicalAnswerId"])
        self.assertEqual(1, data["gate"]["citationResolutions"][0]["step"])
        self.assertIsNone(data["failure"])
        self.assertEqual("CONNECT", data["output"]["decision"])
        self.assertEqual(str(RAG_RUN_ID), data["presentationSeed"])
        self.assertEqual(
            {"input": 5000, "output": 300, "cached": 4000, "reasoning": 120, "latencyMs": 1200, "retryCount": 0},
            data["usage"],
        )
        self.assertEqual(str(CANCEL_ID), next(item["subproblemId"] for item in data["subproblemCandidates"]["items"] if item["key"] == "billing.cancel"))
        self.assertEqual({"dense": 10, "bm25": 10, "rrfK": 60}, data["documentCandidates"]["retrieval"])
        self.assertEqual({"itemCount": 2, "skippedMissingEmbedding": 1, "skippedEmbeddingConfigMismatch": 0}, data["catalog"])
        self.assertEqual(
            {"embeddingConfigId": 9, "subproblemTextVersion": ["name-inclusion-v1"], "reusedRetrievalEmbedding": True, "retrievalQuery": QUESTION, "usage": None},
            data["embedding"],
        )

    async def test_judgment_input_round_trips_through_json(self) -> None:
        await self.run_turn()

        data = self.judgment_input()
        self.assertEqual(data, json.loads(json.dumps(data, allow_nan=False)))

        def walk(value: Any) -> None:
            self.assertNotIsInstance(value, uuid.UUID)
            if isinstance(value, dict):
                for inner in value.values():
                    walk(inner)
            elif isinstance(value, (list, tuple)):
                for inner in value:
                    walk(inner)

        walk(data)


class ConnectNotServedTest(_ServiceTestCase):
    async def test_shadow(self) -> None:
        self.catalog.gate_inputs = _gate_inputs(serving_state=QuestionSubproblemServingState.SHADOW)
        _, _, _, recorded = await self.run_turn()

        self.assertEqual(CacheAttemptOutcome.SHADOW, recorded.gate.outcome)
        self.assertEqual(CANONICAL_ID, recorded.gate.canonical_answer_id)
        self.assertEqual((), recorded.served_citations)
        self.assertIsNone(recorded.served_answer_markdown)
        self.assertEqual("SHADOW", self.judgment_input()["gate"]["outcome"])

    async def test_group_disabled(self) -> None:
        _, _, _, recorded = await self.run_turn(semantic_cache_enabled=False)

        self.assertEqual(CacheAttemptOutcome.GROUP_DISABLED, recorded.gate.outcome)
        self.assertFalse(recorded.served)
        self.assertEqual((), recorded.served_citations)

    async def test_rejected_when_canonical_missing(self) -> None:
        self.catalog.gate_inputs = _gate_inputs(canonical=False)
        _, _, _, recorded = await self.run_turn()

        self.assertEqual((REJECT_CANONICAL_ANSWER_NOT_FOUND,), recorded.gate.rejection_reasons)
        self.assertIsNone(self.judgment_input()["gate"]["canonicalAnswerId"])

    async def test_rejected_on_version_mismatch_keeps_canonical_id_in_judgment_input(self) -> None:
        self.catalog.gate_inputs = _gate_inputs(current_version=2)
        _, _, _, recorded = await self.run_turn()

        self.assertEqual(CacheAttemptOutcome.REJECTED, recorded.gate.outcome)
        self.assertEqual((REJECT_SUBPROBLEM_VERSION_MISMATCH,), recorded.gate.rejection_reasons)
        self.assertIsNone(self.store.attempts[0]["gate"].canonical_answer_id)
        self.assertEqual(str(CANONICAL_ID), self.judgment_input()["gate"]["canonicalAnswerId"])
        self.assertEqual(ClassificationDecision.CONNECT, self.store.classifications[0]["judgment"].decision)

    async def test_rejected_when_cited_section_changed(self) -> None:
        changed = IndexedSection(CITED_CHUNK + 1, BILLING_VERSION, "c-other", 1, "id-cancel")
        self.catalog.gate_inputs = _gate_inputs(section=changed)
        _, _, _, recorded = await self.run_turn()

        self.assertEqual((REJECT_CITED_SECTION_CHANGED,), recorded.gate.rejection_reasons)
        resolution = self.judgment_input()["gate"]["citationResolutions"][0]
        self.assertEqual(REJECT_CITED_SECTION_CHANGED, resolution["rejectionReason"])

    async def test_gate_data_error_rejects_without_failing_turn(self) -> None:
        self.catalog.gate_error = CatalogDataError("정본 인용의 문서 판과 인용 청크의 문서 판이 다릅니다")
        _, _, _, recorded = await self.run_turn()

        self.assertEqual((REJECT_CANONICAL_DATA_INVALID,), recorded.gate.rejection_reasons)
        self.assertIn("dataError", self.judgment_input()["gate"])
        self.assertEqual(ClassificationDecision.CONNECT, recorded.judgment.decision)


class NotConnectedTest(_ServiceTestCase):
    async def test_separate_with_matched_document_is_rejected_without_gate_reload(self) -> None:
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search())
        billing_id = next(item.id for item in prepared.presentation.documents if item.document_version_id == BILLING_VERSION)
        self.judge_client.call = JudgeCall(trace=_trace(), output_text=_separate_output(billing_id))
        judged = await service.judge(prepared)
        recorded = await service.record_judgment_and_gate(prepared, judged, semantic_cache_enabled=True)

        self.assertEqual((REJECT_CLASSIFICATION_NOT_CONNECTED,), recorded.gate.rejection_reasons)
        self.assertNotIn("gate_inputs", self.events)
        self.assertEqual(AttributionSource.DOCUMENT, recorded.judgment.attribution.attribution_source)
        self.assertEqual(BILLING_SOURCE, recorded.judgment.attribution.document_source_id)


class JudgeFailureTest(_ServiceTestCase):
    def assert_failed_row(self, recorded, kind: JudgeFailureKind, model_call_status: ExecutionStatus) -> Dict[str, Any]:
        self.assertEqual(CacheAttemptOutcome.FAILED, recorded.gate.outcome)
        self.assertEqual(ClassificationDecision.UNCLASSIFIED, recorded.judgment.decision)
        self.assertEqual(AttributionSource.NONE, recorded.judgment.attribution.attribution_source)
        self.assertNotIn("gate_inputs", self.events)
        self.assertEqual(model_call_status, self.log_store.finished[self.judge_call_id()]["status"])
        data = self.judgment_input()
        self.assertEqual("FAILED", data["gate"]["outcome"])
        self.assertEqual(FAILURE_STAGE_JUDGE, data["failure"]["stage"])
        self.assertEqual(kind.value, data["failure"]["kind"])
        self.assertNotIn(QUESTION, data["failure"]["safeMessage"])
        # 판별 실패 턴도 질문 벡터와 후보는 남긴다.
        self.assertEqual(1, len(self.store.embeddings))
        self.assertEqual(2, len(data["subproblemCandidates"]["items"]))
        return data

    async def test_api_error(self) -> None:
        failure = JudgeFailure(JudgeFailureKind.API_ERROR, "OpenAI 판별 호출 실패: HTTP 500")
        self.judge_client.call = JudgeCall(trace=_trace(False, failure.safe_message), failure=failure)
        _, _, _, recorded = await self.run_turn()

        data = self.assert_failed_row(recorded, JudgeFailureKind.API_ERROR, ExecutionStatus.FAILED)
        self.assertEqual(1, data["failure"]["retryCount"])
        self.assertIsNone(data["output"])
        self.assertEqual("OpenAI 판별 호출 실패: HTTP 500", self.log_store.finished[self.judge_call_id()]["error_message"])

    async def test_incomplete_response(self) -> None:
        failure = JudgeFailure(JudgeFailureKind.INCOMPLETE_RESPONSE, "판별 응답이 완료되지 않았습니다: status=incomplete, reason=max_output_tokens")
        self.judge_client.call = JudgeCall(trace=_trace(False, failure.safe_message), failure=failure)
        _, _, _, recorded = await self.run_turn()

        self.assert_failed_row(recorded, JudgeFailureKind.INCOMPLETE_RESPONSE, ExecutionStatus.FAILED)

    async def test_unparseable_output_finishes_model_call_success(self) -> None:
        self.judge_client.call = JudgeCall(trace=_trace(), output_text="{not json")
        _, _, _, recorded = await self.run_turn()

        data = self.assert_failed_row(recorded, JudgeFailureKind.INVALID_OUTPUT, ExecutionStatus.SUCCESS)
        self.assertEqual(INVALID_UNPARSEABLE_OUTPUT, data["normalization"]["invalidReason"])
        self.assertEqual("{not json", data["output"])
        self.assertIsNotNone(data["usage"])

    async def test_unknown_subproblem_key_is_invalid(self) -> None:
        self.judge_client.call = JudgeCall(trace=_trace(), output_text=_connect_output(key="billing.unknown"))
        _, _, _, recorded = await self.run_turn()

        data = self.assert_failed_row(recorded, JudgeFailureKind.INVALID_OUTPUT, ExecutionStatus.SUCCESS)
        self.assertEqual(INVALID_UNKNOWN_SUBPROBLEM_KEY, data["normalization"]["invalidReason"])
        self.assertEqual("billing.unknown", data["output"]["subproblemId"])

    async def test_unexpected_client_exception_is_fail_open(self) -> None:
        self.judge_client.call = RuntimeError("boom " + QUESTION)
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search())
        judged = await service.judge(prepared)
        recorded = await service.record_judgment_and_gate(prepared, judged, semantic_cache_enabled=True)

        self.assertEqual(CacheAttemptOutcome.FAILED, recorded.gate.outcome)
        self.assertEqual(ExecutionStatus.FAILED, self.log_store.finished[judged.model_call_id]["status"])
        data = self.judgment_input()
        self.assertEqual(JudgeFailureKind.INTERNAL_ERROR.value, data["failure"]["kind"])
        self.assertNotIn(QUESTION, json.dumps(data["failure"], ensure_ascii=False))

    async def test_checkpoint_write_failure_is_fail_closed(self) -> None:
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search())
        self.log_store.fail_start = True

        with self.assertRaisesRegex(RuntimeError, "DB down"):
            await service.judge(prepared)
        self.assertNotIn("judge_api", self.events)

    async def test_record_write_failure_propagates(self) -> None:
        self.store.fail_classification = True
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search())
        judged = await service.judge(prepared)

        with self.assertRaisesRegex(RuntimeError, "insert failed"):
            await service.record_judgment_and_gate(prepared, judged, semantic_cache_enabled=True)
        self.assertEqual([], self.store.attempts)


class PreparationFailureTest(_ServiceTestCase):
    async def test_catalog_data_error_records_failed_row_without_judge(self) -> None:
        self.catalog.catalog_error = CatalogDataError("문서 그룹 안에서 세부 문제 key 가 겹칩니다: keys=['billing.cancel']")
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search())

        self.assertFalse(prepared.ready)
        self.assertEqual(FAILURE_STAGE_CANDIDATES, prepared.failure_stage)
        with self.assertRaises(ValueError):
            await service.judge(prepared)
        with self.assertRaises(ValueError):
            await service.record_judgment_and_gate(prepared, None, semantic_cache_enabled=True)  # type: ignore[arg-type]

        recorded = await service.record_preparation_failure(prepared)

        self.assertEqual(["lock", "embedding", "classification", "attempt"], self.events)
        self.assertEqual(CacheAttemptOutcome.FAILED, recorded.gate.outcome)
        self.assertEqual(AttributionSource.NONE, recorded.judgment.attribution.attribution_source)
        data = self.judgment_input()
        self.assertEqual(
            {"stage": FAILURE_STAGE_CANDIDATES, "kind": "INTERNAL_ERROR", "retryCount": None},
            {key: data["failure"][key] for key in ("stage", "kind", "retryCount")},
        )
        self.assertIn("key 가 겹칩니다", data["failure"]["safeMessage"])
        self.assertIsNone(data["subproblemCandidates"])
        self.assertIsNone(data["usage"])
        self.assertEqual("FAILED", data["gate"]["outcome"])

    async def test_missing_outline_is_candidate_failure(self) -> None:
        self.outlines.outlines = {BILLING_VERSION: OUTLINES[BILLING_VERSION]}
        prepared = await self.service().prepare(self.turn, QUESTION, _search())

        self.assertEqual(FAILURE_STAGE_CANDIDATES, prepared.failure_stage)
        self.assertEqual("판별 후보를 만들지 못했습니다: ValueError", prepared.failure.safe_message)

    async def test_wrong_dimension_vector_is_not_stored(self) -> None:
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search(embedding=(1.0, 0.0)))

        self.assertEqual(FAILURE_STAGE_QUESTION_EMBEDDING, prepared.failure_stage)
        self.assertIsNone(prepared.vector.embedding)
        await service.record_preparation_failure(prepared)
        self.assertEqual([], self.store.embeddings)

    async def test_record_preparation_failure_requires_failure(self) -> None:
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search())
        with self.assertRaises(ValueError):
            await service.record_preparation_failure(prepared)


class QueryExpansionTest(_ServiceTestCase):
    async def test_reembeds_resolved_query_when_retrieval_query_differs(self) -> None:
        # 검색 벡터(_vec(0))는 billing.cancel, 재임베딩 벡터(_vec(1))는 members.invite 에 가깝다.
        search = _search(retrieval_query="뤼이도 서비스 소개", embedding=_vec(0))
        _, prepared, judged, recorded = await self.run_turn(search=search)

        self.assertEqual([([QUESTION], {"sdk_max_retries": 0, "timeout": 30.0})], self.embedder.calls)
        embedding_call = self.log_store.started[0]
        self.assertEqual(
            (ModelCallPurpose.QUERY_EMBEDDING.value, RAG_RUN_ID, RUN_ID),
            (embedding_call["purpose"], embedding_call["rag_run_id"], embedding_call["classification_run_id"]),
        )
        judge_call_id = self.judge_call_id()
        self.assertEqual(
            [
                ("start", ModelCallPurpose.QUERY_EMBEDDING.value),
                "commit",
                "embed_api",
                ("finish", embedding_call["id"], ExecutionStatus.SUCCESS),
                ("start", ModelCallPurpose.QUESTION_CLASSIFICATION.value),
                "commit",
                "judge_api",
                "lock",
                ("finish", judge_call_id, ExecutionStatus.SUCCESS),
                "embedding",
                "gate_inputs",
                "classification",
                "attempt",
            ],
            self.events,
        )
        self.assertTrue(judged.embedding_call_finished)
        self.assertEqual(12, self.log_store.finished[embedding_call["id"]]["input_tokens"])
        self.assertEqual(_vec(1), self.store.embeddings[0]["embedding"])
        items = sorted(prepared.presentation.subproblems, key=lambda item: item.retrieval_rank)
        self.assertEqual("members.invite", items[0].key)
        # 문서 후보는 검색 결과에서 고른다.
        self.assertEqual({BILLING_VERSION, MEMBERS_VERSION}, {item.document_version_id for item in prepared.presentation.documents})
        data = self.judgment_input()
        self.assertFalse(data["embedding"]["reusedRetrievalEmbedding"])
        self.assertEqual("뤼이도 서비스 소개", data["embedding"]["retrievalQuery"])
        self.assertEqual(12, data["embedding"]["usage"]["input"])
        self.assertEqual(CacheAttemptOutcome.SERVED, recorded.gate.outcome)

    async def test_reembedding_failure_is_fail_open(self) -> None:
        self.embedder.error = ValueError("bad input")
        service = self.service()
        prepared = await service.prepare(self.turn, QUESTION, _search(retrieval_query="확장 질의"))

        self.assertEqual(FAILURE_STAGE_QUESTION_EMBEDDING, prepared.failure_stage)
        recorded = await service.record_preparation_failure(prepared)

        embedding_call_id = self.log_store.started[0]["id"]
        self.assertEqual(ExecutionStatus.FAILED, self.log_store.finished[embedding_call_id]["status"])
        self.assertEqual([], self.store.embeddings)
        self.assertEqual(CacheAttemptOutcome.FAILED, recorded.gate.outcome)
        data = self.judgment_input()
        self.assertEqual(FAILURE_STAGE_QUESTION_EMBEDDING, data["failure"]["stage"])
        self.assertEqual(0, data["failure"]["retryCount"])
        self.assertIsNone(data["embedding"]["embeddingConfigId"])

    async def test_transient_embedding_error_is_retried(self) -> None:
        from openai import APIConnectionError
        import httpx

        errors = [APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))]
        original = self.embedder.embed_many_with_usage

        def flaky(texts, **kwargs):
            if errors:
                self.embedder.calls.append((list(texts), kwargs))
                raise errors.pop()
            return original(texts, **kwargs)

        self.embedder.embed_many_with_usage = flaky  # type: ignore[assignment]
        with patch("app.question_grouping.service.asyncio.sleep") as sleep:
            prepared = await self.service().prepare(self.turn, QUESTION, _search(retrieval_query="확장 질의"))

        self.assertTrue(prepared.ready)
        self.assertEqual(2, len(self.embedder.calls))
        self.assertEqual(1, prepared.vector.trace.retry_count)
        sleep.assert_awaited_once()


class SeedTest(_ServiceTestCase):
    async def test_presentation_is_reproducible_from_rag_run_id(self) -> None:
        service = self.service()
        first = await service.prepare(self.turn, QUESTION, _search())
        second = await service.prepare(self.turn, QUESTION, _search())
        other = await service.prepare(
            GroupingTurn(uuid.UUID("99999999-9999-4999-8999-999999999999"), GROUP_ID, INDEX_ID, RUN_ID), QUESTION, _search()
        )

        self.assertEqual(str(RAG_RUN_ID), first.seed)
        self.assertEqual(json.dumps(first.presentation.payload, ensure_ascii=False), json.dumps(second.presentation.payload, ensure_ascii=False))
        self.assertEqual(first.presentation.subproblems, second.presentation.subproblems)
        self.assertEqual(first.presentation.documents, second.presentation.documents)
        self.assertNotEqual(first.seed, other.seed)
        self.assertEqual(f"{RAG_RUN_ID}:subproblems", first.presentation.subproblem_seed)

        judged = await service.judge(first)
        await service.record_judgment_and_gate(first, judged, semantic_cache_enabled=True)
        data = self.judgment_input()
        self.assertEqual(str(RAG_RUN_ID), data["presentationSeed"])
        self.assertEqual(f"{RAG_RUN_ID}:documents", data["documentCandidates"]["seed"])


class ServeFailureTest(_ServiceTestCase):
    async def test_rolls_back_and_rewrites_as_rejected(self) -> None:
        service, prepared, judged, served = await self.run_turn()
        self.events.clear()

        retried = await service.record_serve_failure(prepared, judged, served)

        judge_call_id = self.judge_call_id()
        self.assertEqual(
            ["rollback", "lock", ("finish", judge_call_id, ExecutionStatus.SUCCESS), "embedding", "classification", "attempt"],
            self.events,
        )
        self.assertEqual(CacheAttemptOutcome.REJECTED, retried.gate.outcome)
        self.assertEqual((REJECT_CANONICAL_SERVE_FAILED,), retried.gate.rejection_reasons)
        self.assertFalse(retried.served)
        self.assertEqual((), retried.served_citations)
        self.assertEqual(ClassificationDecision.CONNECT, retried.judgment.decision)
        data = self.judgment_input()
        self.assertEqual(
            {"outcome": "REJECTED", "canonicalAnswerId": str(CANONICAL_ID), "rejectionReasons": [REJECT_CANONICAL_SERVE_FAILED]},
            {key: data["gate"][key] for key in ("outcome", "canonicalAnswerId", "rejectionReasons")},
        )
        self.assertEqual(CITED_CHUNK, data["gate"]["citationResolutions"][0]["chunkId"])
        self.assertEqual("SERVED", data["serveFailure"]["gateOutcome"])
        self.assertEqual(served.latency_ms, self.store.attempts[-1]["latency_ms"])

    async def test_reembedding_call_finished_at_checkpoint_is_not_finished_again(self) -> None:
        service, prepared, judged, served = await self.run_turn(search=_search(retrieval_query="확장 질의"))
        self.events.clear()

        await service.record_serve_failure(prepared, judged, served)

        finishes = [event for event in self.events if isinstance(event, tuple) and event[0] == "finish"]
        self.assertEqual([("finish", self.judge_call_id(), ExecutionStatus.SUCCESS)], finishes)

    async def test_only_served_records_can_be_rewritten(self) -> None:
        service, prepared, judged, recorded = await self.run_turn(semantic_cache_enabled=False)
        with self.assertRaises(ValueError):
            await service.record_serve_failure(prepared, judged, recorded)


class FinalizeAttributionTest(_ServiceTestCase):
    CITATIONS = [CitationLog(CITED_CHUNK, BILLING_VERSION, 1)]

    async def test_connect_and_empty_citations_skip_store(self) -> None:
        service, _, _, connect = await self.run_turn()

        self.assertFalse(await service.finalize_attribution(connect, self.CITATIONS))
        self.assertEqual([], self.store.finalized)

    async def test_non_connect_delegates_to_store(self) -> None:
        self.judge_client.call = JudgeCall(trace=_trace(), output_text=_separate_output(None))
        service, _, _, recorded = await self.run_turn()

        self.assertFalse(await service.finalize_attribution(recorded, []))
        self.assertTrue(await service.finalize_attribution(recorded, self.CITATIONS))
        self.assertEqual([(recorded.classification_id, self.CITATIONS)], self.store.finalized)
        self.assertNotIn("commit", self.events[-2:])


def _exact_match(subproblem_id: uuid.UUID = CANCEL_ID, key: str = "billing.cancel") -> ExactQuestionLogMatch:
    return ExactQuestionLogMatch(
        subproblem_id=subproblem_id,
        key=key,
        problem_group_id=CANCEL_GROUP,
        current_version=1,
        document_source_id=BILLING_SOURCE,
        document_key=BILLING_KEY,
        normalized_question=QUESTION,
        source_rag_run_id=uuid.UUID("77777777-7777-4777-8777-777777777777"),
        classification_id=9,
        matched_count=3,
    )


class ExactQuestionTest(_ServiceTestCase):
    async def test_exact_cache_can_serve_when_semantic_cache_is_disabled(self) -> None:
        self.store.exact_match = _exact_match()

        result = await self.service().record_exact_question(
            self.turn,
            QUESTION,
            semantic_cache_enabled=False,
            exact_cache_enabled=True,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.recorded.served)
        self.assertEqual(CacheAttemptOutcome.SERVED, result.recorded.gate.outcome)

    async def test_single_log_match_records_connect_and_serves_without_llm(self) -> None:
        self.store.exact_match = _exact_match()

        result = await self.service().record_exact_question(
            self.turn, QUESTION, semantic_cache_enabled=True
        )

        self.assertIsNotNone(result)
        self.assertEqual([(GROUP_ID, QUESTION, RAG_RUN_ID)], self.store.exact_lookups)
        self.assertEqual(["exact_lookup", "gate_inputs", "classification", "attempt"], self.events)
        self.assertEqual([], self.judge_client.payloads)
        self.assertEqual([], self.embedder.calls)
        self.assertEqual([], self.store.embeddings)
        recorded = result.recorded
        self.assertTrue(recorded.served)
        self.assertEqual(ClassificationDecision.CONNECT, recorded.judgment.decision)
        self.assertEqual(CANCEL_ID, recorded.judgment.subproblem.subproblem_id)
        self.assertEqual(1, recorded.judgment.subproblem.subproblem_version)
        self.assertEqual("현재 정본 [1]", recorded.served_answer_markdown)
        self.assertEqual(RUN_ID, self.store.classifications[0]["run_id"])
        data = self.judgment_input()
        self.assertEqual("QUESTION_LOG_EXACT", data["matchSource"])
        self.assertEqual(
            {
                "normalizedQuestion": QUESTION,
                "sourceRagRunId": "77777777-7777-4777-8777-777777777777",
                "sourceClassificationId": 9,
                "matchedCount": 3,
            },
            data["exactQuestionMatch"],
        )

    async def test_gate_rejection_is_recorded_and_returned(self) -> None:
        self.store.exact_match = _exact_match()
        self.catalog.gate_inputs = _gate_inputs(canonical=False)

        result = await self.service().record_exact_question(
            self.turn, QUESTION, semantic_cache_enabled=True
        )

        self.assertFalse(result.recorded.served)
        self.assertEqual(CacheAttemptOutcome.REJECTED, result.recorded.gate.outcome)
        self.assertEqual(1, len(self.store.classifications))
        self.assertEqual(1, len(self.store.attempts))

    async def test_no_match_writes_nothing(self) -> None:
        result = await self.service().record_exact_question(
            self.turn, QUESTION, semantic_cache_enabled=True
        )

        self.assertIsNone(result)
        self.assertEqual(["exact_lookup"], self.events)
        self.assertEqual([], self.store.classifications)
        self.assertEqual([], self.store.attempts)

    async def test_store_selected_latest_subproblem_is_connected_as_is(self) -> None:
        # 최신 분류 선택은 저장소가 한다. 서비스는 고른 세부 문제를 충돌 검사 없이 그대로 연결한다.
        self.store.exact_match = _exact_match(INVITE_ID, "members.invite")
        self.catalog.gate_inputs = _gate_inputs(canonical=False)

        result = await self.service().record_exact_question(
            self.turn, QUESTION, semantic_cache_enabled=True
        )

        self.assertEqual(INVITE_ID, result.recorded.judgment.subproblem.subproblem_id)
        self.assertEqual(INVITE_ID, self.store.classifications[0]["judgment"].subproblem.subproblem_id)

if __name__ == "__main__":
    unittest.main()
