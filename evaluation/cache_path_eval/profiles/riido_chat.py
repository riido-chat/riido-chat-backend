"""Concrete read-only Riido cache-path callbacks over production components."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from app.answering.models import FinalWithheldReason
from app.answering.trace import ReplayContext
from app.answering.use_case import AnswerQuestion
from app.chat.query_rewrite import QueryRewriteService
from app.core.model_trace import ModelCallTrace
from app.database.models import CacheAttemptOutcome, ClassificationDecision
from app.question_grouping.attribution import subproblem_attribution
from app.question_grouping.cache_decision import CacheDecisionResult, CacheDecisionService
from app.question_grouping.judge_client import QuestionJudgeClient
from app.question_grouping.models import (
    GateResult,
    PresentedSubproblem,
    TurnJudgment,
)
from app.question_grouping.constants import REJECT_CANONICAL_DATA_INVALID
from app.question_grouping.payload import presentation_seed
from app.question_grouping.question_vector import resolve_question_vector
from app.retrieval.analyzer import KiwiAnalyzer
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_PROVIDER
from app.retrieval.embedding_config import EMBEDDING_INPUT_TEMPLATE_VERSION
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import HybridSearchCall
from app.retrieval.vector_retriever import VectorRetriever
from evaluation.cache_path_eval.evidence import (
    CacheRuntimeEvidence,
    ModelCallEvidence,
    RetrievalCandidateEvidence,
    SubproblemCandidateEvidence,
)
from evaluation.cache_path_eval.execution import (
    CacheDecisionCallbacks,
    CacheDecisionMetrics,
)
from evaluation.cache_path_eval.golden_inputs import FrozenCacheCorpusLoader
from evaluation.cache_path_eval.readonly_service import ReadOnlyRiidoServiceData
from evaluation.cache_path_eval.run import CachePathEvaluationProfilePort
from evaluation.cache_path_eval.store import CachePathRunMetadata


class FrozenInputLoaderPort(Protocol):
    def load_turn(self, item_id: str, turn_no: int) -> Any: ...


@dataclass
class _TurnState:
    seed: str
    calls: list[tuple[str, ModelCallTrace]]
    search: Optional[HybridSearchCall] = None
    semantic: Optional[CacheDecisionResult] = None
    exact: Optional[CacheDecisionResult] = None
    exact_classification_id: Optional[int] = None
    exact_fallthrough_reasons: tuple[str, ...] = ()
    candidate_presentation: Any = None
    selected_subproblem_id: Optional[str] = None
    selected_canonical_answer_id: Optional[str] = None
    candidate_failure: Optional[BaseException] = None
    query_resolved: bool = False
    resolved_query: Optional[str] = None
    cache_selection_enabled: bool = False
    closed: bool = False


class RiidoChatCachePathEvaluationProfile(CachePathEvaluationProfilePort):
    """Builds one isolated read snapshot and production-backed callback bundle per turn."""

    def __init__(
        self,
        *,
        service_data: ReadOnlyRiidoServiceData,
        input_loader: FrozenInputLoaderPort,
        metadata: CachePathRunMetadata,
        embedder: Any,
        query_rewrite_service: QueryRewriteService,
        judge_client: QuestionJudgeClient,
        analyzer_factory: Callable[[], KiwiAnalyzer] = KiwiAnalyzer,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._service_data = service_data
        self._input_loader = input_loader
        self._metadata = metadata
        self._embedder = embedder
        self._rewrite = query_rewrite_service
        self._judge = judge_client
        self._analyzer_factory = analyzer_factory
        self._clock = clock

    async def callbacks_for_turn(
        self, selected_turn: Mapping[str, Any]
    ) -> CacheDecisionCallbacks:
        item_id, turn_no = _reference(selected_turn)
        golden = self._input_loader.load_turn(item_id, turn_no)
        manager = self._service_data.open_snapshot(self._metadata)
        snapshot = await manager.__aenter__()
        state = _TurnState(
            seed=presentation_seed(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{self._metadata.run_id}:{item_id}:{turn_no}")
            ),
            calls=[],
        )
        try:
            selection = snapshot.selection
            _validate_snapshot_selection(selection, self._metadata)
            _validate_embedding_identity(selection.embedding_config, self._metadata)
            retriever: Optional[HybridRetriever] = None
            decision_service = CacheDecisionService(
                snapshot.catalog_reader, snapshot.outline_reader
            )

            async def resolve_query() -> Optional[str]:
                if turn_no == 1:
                    state.query_resolved = True
                    state.resolved_query = golden.question
                    return golden.question
                if not golden.candidates:
                    raise ValueError(
                        "후속 질문에 명시적인 이전 대화 문맥이 없어 재작성할 수 없습니다."
                    )
                rewritten = await self._rewrite.rewrite(
                    golden.question,
                    golden.candidates,
                    stage=selection.stage_settings.rewrite,
                )
                state.calls.append(("QUERY_REWRITE", rewritten.trace))
                state.query_resolved = rewritten.resolution is not None and rewritten.resolution.should_retrieve
                state.resolved_query = (
                    None
                    if rewritten.resolution is None
                    else rewritten.resolution.resolved_query
                )
                return (
                    state.resolved_query
                )

            async def retrieve(resolved_query: str) -> HybridSearchCall:
                nonlocal retriever
                if retriever is None:
                    chunks = await snapshot.search_reader.load_active_chunks()
                    if not chunks:
                        raise ValueError("선택한 index snapshot에 검색 chunk가 없습니다.")
                    retriever = HybridRetriever(
                        BM25Retriever(chunks, analyzer=self._analyzer_factory()),
                        VectorRetriever(self._embedder, snapshot.search_reader),
                    )
                result = await retriever.search_with_trace(resolved_query)
                state.search = result
                if result.embedding_call is not None:
                    state.calls.append(("QUERY_EMBEDDING", result.embedding_call))
                return result

            async def inspect_exact() -> Any:
                if not selection.exact_cache_enabled:
                    return None
                match = await snapshot.exact_reader.find_exact_question_log_match(
                    selection.document_group_id, golden.question
                )
                if match is None:
                    return None
                state.exact_classification_id = match.classification_id
                inspection = await decision_service.inspect_exact_source(
                    match,
                    index_version_id=selection.index_version_id,
                    document_group_id=selection.document_group_id,
                )
                if not inspection.scope_matches_document_group:
                    return None
                if inspection.fallthrough_reasons:
                    state.exact_fallthrough_reasons = inspection.fallthrough_reasons
                    return None
                state.exact_classification_id = match.classification_id
                presented = PresentedSubproblem(
                    key=match.key,
                    subproblem_id=match.subproblem_id,
                    subproblem_version=match.source_subproblem_version,
                    problem_group_id=match.problem_group_id,
                    document_source_id=match.document_source_id,
                    document_key=match.document_key or "",
                    canonical_answer_id=(
                        None
                        if inspection.inputs is None
                        or inspection.inputs.canonical_answer is None
                        else inspection.inputs.canonical_answer.canonical_answer_id
                    ),
                    similarity=1.0,
                    retrieval_rank=1,
                    presented_order=1,
                    inclusion_count=0,
                    exclusion_count=0,
                )
                judgment = TurnJudgment(
                    decision=ClassificationDecision.CONNECT,
                    attribution=subproblem_attribution(match.problem_group_id),
                    subproblem=presented,
                )
                if inspection.data_error is not None:
                    state.exact = CacheDecisionResult(
                        gate=GateResult(
                            outcome=CacheAttemptOutcome.REJECTED,
                            rejection_reasons=(REJECT_CANONICAL_DATA_INVALID,),
                        ),
                        data_error=inspection.data_error,
                    )
                else:
                    state.exact = await decision_service.decide_gate(
                        judgment,
                        scope=inspection.scope,
                        semantic_cache_enabled=selection.exact_cache_enabled,
                        inputs=inspection.inputs,
                    )
                state.selected_subproblem_id = str(match.subproblem_id)
                canonical = (
                    None
                    if state.exact.inputs is None
                    else state.exact.inputs.canonical_answer
                )
                state.selected_canonical_answer_id = (
                    None
                    if canonical is None
                    else str(canonical.canonical_answer_id)
                )
                return state.exact

            async def handle_exact(decision: Any) -> bool:
                return (
                    isinstance(decision, CacheDecisionResult)
                    and decision.gate.outcome == CacheAttemptOutcome.SERVED
                )

            async def should_choose_source() -> bool:
                return selection.semantic_cache_enabled and state.exact is None

            async def choose_source() -> CacheDecisionResult:
                state.cache_selection_enabled = True
                # Production records a fresh exact rejection and continues through
                # retrieval, while suppressing a duplicate semantic judgment.
                if state.exact is not None:
                    return state.exact
                if state.search is None:
                    raise RuntimeError("검색 결과 없이 의미 캐시 판정을 시작했습니다.")
                assert state.search.retrieval_query is not None
                vector = await resolve_question_vector(
                    _last_resolved_query(state, golden.question),
                    state.search,
                    self._embedder,
                    clock=self._clock,
                )
                if vector.trace is not None:
                    state.calls.append(("QUERY_EMBEDDING", vector.trace))
                if vector.error is not None or vector.embedding is None:
                    state.candidate_failure = vector.error
                    state.semantic = CacheDecisionResult(
                        gate=GateResult(outcome=CacheAttemptOutcome.FAILED)
                    )
                    return state.semantic

                resolved_query = _last_resolved_query(state, golden.question)
                prepared = await decision_service.prepare_candidates(
                    resolved_query=resolved_query,
                    embedding=vector.embedding,
                    index_version_id=selection.index_version_id,
                    document_group_id=selection.document_group_id,
                    retrieval=state.search,
                    seed=state.seed,
                )
                if not prepared.ready:
                    state.candidate_failure = prepared.error
                    state.semantic = CacheDecisionResult(
                        gate=GateResult(outcome=CacheAttemptOutcome.FAILED)
                    )
                    return state.semantic
                state.candidate_presentation = prepared.presentation
                decision = await decision_service.judge(
                    client=self._judge,
                    presentation=prepared.presentation,
                    model=selection.stage_settings.judge.model_name,
                    prompt_version=selection.stage_settings.judge.prompt_version,
                    stage=selection.stage_settings.judge,
                    clock=self._clock,
                )
                state.calls.append(("QUESTION_CLASSIFICATION", decision.call.trace))
                state.selected_subproblem_id = (
                    None
                    if decision.judgment.subproblem is None
                    else str(decision.judgment.subproblem.subproblem_id)
                )
                gate_result = await decision_service.decide_gate(
                    decision.judgment,
                    scope=prepared.scope,
                    semantic_cache_enabled=selection.semantic_cache_enabled,
                )
                state.semantic = gate_result
                state.selected_canonical_answer_id = (
                    None
                    if gate_result.inputs.canonical_answer is None
                    else str(gate_result.inputs.canonical_answer.canonical_answer_id)
                )
                return gate_result

            def summarize(result: Any) -> CacheDecisionMetrics:
                if state.exact is not None:
                    decision = state.exact
                elif state.semantic is not None:
                    decision = state.semantic
                else:
                    decision = None
                if decision is not None:
                    outcome = decision.gate.outcome.value
                    reasons = decision.gate.rejection_reasons
                elif result.retrieval is not None and result.retrieval.error is not None:
                    outcome = "FAILED"
                    reasons = ()
                elif any(
                    purpose == "QUERY_REWRITE" and not trace.succeeded
                    for purpose, trace in state.calls
                ):
                    outcome = "FAILED"
                    reasons = ()
                elif not state.query_resolved:
                    outcome = "SKIPPED"
                    reasons = ()
                elif not selection.semantic_cache_enabled:
                    outcome = "SKIPPED"
                    reasons = ()
                else:
                    outcome = "FAILED"
                    reasons = ()
                traces = tuple(
                    ModelCallEvidence.from_trace(purpose, trace)
                    for purpose, trace in state.calls
                )
                model_calls = len(state.calls)
                embedding_calls = sum(
                    1
                    for purpose, trace in state.calls
                    if purpose == "QUERY_EMBEDDING"
                )
                retrieval_candidates = tuple(
                    RetrievalCandidateEvidence(
                        chunk_id=item.chunk.chunk_id,
                        document_version_id=item.chunk.document_version_id,
                        index_version_id=item.chunk.index_version_id,
                        rank=item.final_rank,
                    )
                    for item in (() if state.search is None else state.search.fused_results)
                    if item.chunk.chunk_id is not None
                    and item.chunk.document_version_id is not None
                    and item.chunk.index_version_id is not None
                )
                subproblem_candidates = _subproblem_evidence(state.candidate_presentation)
                evidence = CacheRuntimeEvidence(
                    seed=state.seed,
                    retrieval_candidates=retrieval_candidates,
                    subproblem_candidates=subproblem_candidates,
                    selected_subproblem_id=state.selected_subproblem_id,
                    selected_canonical_answer_id=state.selected_canonical_answer_id,
                    gate_rejection_reasons=tuple(reasons),
                    exact_source_classification_id=state.exact_classification_id,
                    exact_fallthrough_reasons=state.exact_fallthrough_reasons,
                    model_calls=traces,
                )
                return CacheDecisionMetrics(
                    outcome=outcome,
                    next_stage="NOT_APPLICABLE",
                    model_calls=model_calls,
                    embedding_calls=embedding_calls,
                    evidence=evidence,
                )

            async def finalize() -> None:
                if state.closed:
                    return
                state.closed = True
                await manager.__aexit__(None, None, None)

            return CacheDecisionCallbacks(
                resolve_query=resolve_query,
                retrieve=retrieve,
                choose_source=choose_source,
                summarize=summarize,
                inspect_exact_cache=(inspect_exact if selection.exact_cache_enabled else None),
                handle_exact_cache=(handle_exact if selection.exact_cache_enabled else None),
                should_choose_source=should_choose_source,
                finalize=finalize,
            )
        except BaseException as error:
            if not state.closed:
                state.closed = True
                try:
                    await manager.__aexit__(type(error), error, error.__traceback__)
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "Per-turn cache snapshot cleanup also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
            raise

    def replay_context_for_turn(
        self,
        metadata: CachePathRunMetadata,
        selected_turn: Mapping[str, Any],
    ) -> ReplayContext:
        item_id, turn_no = _reference(selected_turn)
        context = ReplayContext(
            run_id=metadata.run_id,
            item_id=item_id,
            turn_no=turn_no,
            golden_version=metadata.golden_version,
            source_corpus_version=metadata.source_corpus_version,
            service_version=metadata.service_key,
            code_revision=metadata.app_build_commit,
            cache_policy_version=metadata.cache_policy_version,
            embedding_model_version=metadata.embedding_model_version,
            prompt_version=metadata.judge_prompt_version,
        )
        return context


def create_riido_chat_profile_factory(
    *,
    service_key: str,
    service_data: ReadOnlyRiidoServiceData,
    embedder: Any,
    query_rewrite_service: QueryRewriteService,
    judge_client: QuestionJudgeClient,
    golden_root: Path,
    analyzer_factory: Callable[[], KiwiAnalyzer] = KiwiAnalyzer,
) -> Callable[..., RiidoChatCachePathEvaluationProfile]:
    """Return an inert factory; all DB and model clients are caller supplied."""

    def factory(*, config: Mapping[str, Any], metadata: CachePathRunMetadata, manifest: Mapping[str, Any]):
        if metadata.service_key != service_key:
            raise ValueError("선택한 service key가 등록된 Riido 프로필과 다릅니다.")
        configured_root = config.get("goldenRoot")
        if configured_root is not None and Path(configured_root).resolve() != golden_root.resolve():
            raise ValueError("profileConfig goldenRoot가 주입된 frozen corpus와 다릅니다.")
        loader = FrozenCacheCorpusLoader(
            golden_root=golden_root,
            selection_manifest=manifest,
            metadata=metadata,
        )
        _validate_reference_manifest(manifest, metadata)
        return RiidoChatCachePathEvaluationProfile(
            service_data=service_data,
            input_loader=loader,
            metadata=metadata,
            embedder=embedder,
            query_rewrite_service=query_rewrite_service,
            judge_client=judge_client,
            analyzer_factory=analyzer_factory,
        )

    return factory


def register_riido_chat_profile(
    *,
    service_key: str,
    service_data: ReadOnlyRiidoServiceData,
    embedder: Any,
    query_rewrite_service: QueryRewriteService,
    judge_client: QuestionJudgeClient,
    golden_root: Path,
    analyzer_factory: Callable[[], KiwiAnalyzer] = KiwiAnalyzer,
) -> None:
    """Register for one real Riido service key without constructing clients."""

    from evaluation.cache_path_eval.profile_registry import register_profile
    from app.profiles.services import REGISTERED_SERVICES

    if service_key not in {route.profile_key for route in REGISTERED_SERVICES}:
        raise ValueError(f"등록되지 않은 Riido service key입니다: {service_key}")

    register_profile(
        service_key,
        create_riido_chat_profile_factory(
            service_key=service_key,
            service_data=service_data,
            embedder=embedder,
            query_rewrite_service=query_rewrite_service,
            judge_client=judge_client,
            golden_root=golden_root,
            analyzer_factory=analyzer_factory,
        ),
    )


def _reference(turn: Mapping[str, Any]) -> tuple[str, int]:
    if set(turn) != {"itemId", "turnNo"}:
        raise ValueError("runtime에는 itemId와 turnNo reference만 전달할 수 있습니다.")
    item_id, turn_no = turn["itemId"], turn["turnNo"]
    if not isinstance(item_id, str) or not item_id or isinstance(turn_no, bool) or not isinstance(turn_no, int) or turn_no < 1:
        raise ValueError("선택한 cache-path reference 형식이 잘못되었습니다.")
    return item_id, turn_no


def _validate_reference_manifest(manifest: Mapping[str, Any], metadata: CachePathRunMetadata) -> None:
    if manifest.get("group") != "cache-path" or manifest.get("goldenVersion") != metadata.golden_version:
        raise ValueError("factory reference manifest가 선택 run metadata와 다릅니다.")
    turns = manifest.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("factory reference manifest에 선택 턴이 없습니다.")
    for turn in turns:
        _reference(turn)


def _validate_snapshot_selection(selection: Any, metadata: CachePathRunMetadata) -> None:
    expected = {
        "service_key": metadata.service_key,
        "profile_revision_id": metadata.profile_revision_id,
        "document_group_id": metadata.document_group_id,
        "index_version_id": metadata.index_version_id,
        "embedding_config_id": metadata.embedding_config_id,
    }
    if any(getattr(selection, key) != value for key, value in expected.items()):
        raise ValueError("실제 서비스 snapshot이 선택된 run metadata와 다릅니다.")


def _validate_embedding_identity(identity: Any, metadata: CachePathRunMetadata) -> None:
    if identity.id != metadata.embedding_config_id:
        raise ValueError("embedding config가 선택 metadata와 다릅니다.")
    if (
        identity.provider != OPENAI_EMBEDDING_PROVIDER
        or identity.model_name != OPENAI_EMBEDDING_MODEL
        or identity.dimensions != OPENAI_EMBEDDING_DIMENSIONS
        or identity.input_template_version != EMBEDDING_INPUT_TEMPLATE_VERSION
    ):
        raise ValueError("선택 embedding config는 Riido의 현재 검색 adapter에서 지원하지 않습니다.")
    if not metadata.embedding_model_version or metadata.embedding_model_version != identity.model_name:
        raise ValueError("embedding model이 run provenance와 다릅니다.")


def _last_resolved_query(state: _TurnState, fallback: str) -> str:
    # Query rewrite runs immediately before retrieval. The hybrid call stores only
    # the expanded retrieval query; keep the resolved value separately on state.
    return state.resolved_query or fallback


def _subproblem_evidence(presentation: Any) -> tuple[SubproblemCandidateEvidence, ...]:
    if presentation is None:
        return ()
    return tuple(
        SubproblemCandidateEvidence(
            key=item.key,
            subproblem_id=str(item.subproblem_id),
            version=item.subproblem_version,
            canonical_answer_id=(
                None
                if item.canonical_answer_id is None
                else str(item.canonical_answer_id)
            ),
        )
        for item in presentation.subproblems
    )


class _NoGenerationService:
    async def generate_answer(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("cache-path 평가에서 답변 생성은 허용되지 않습니다.")


def create_riido_cache_decision_execution_adapter():
    """Create the shared AnswerQuestion adapter with an explicit no-generation stub."""

    from evaluation.cache_path_eval.execution import CacheDecisionExecutionAdapter

    return CacheDecisionExecutionAdapter(AnswerQuestion(_NoGenerationService()))
