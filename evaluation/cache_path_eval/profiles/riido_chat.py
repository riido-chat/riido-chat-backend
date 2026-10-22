"""Concrete read-only Riido cache-path callbacks over production components."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from app.answering.models import FinalWithheldReason
from app.answering.trace import ReplayContext
from app.answering.use_case import AnswerQuestion
from app.chat.query_rewrite import QueryRewriteService
from app.core.model_trace import ModelCallTrace
from app.core.stage_settings import StageSettings
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
from app import prompts
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
from evaluation.cache_path_eval.plan import PreparedCacheInputs, CachePathPlanRunSpec
from evaluation.cache_path_eval.readonly_service import ReadOnlyRiidoServiceData
from evaluation.cache_path_eval.run import CachePathEvaluationProfilePort
from evaluation.cache_path_eval.store import CachePathRunMetadata
from evaluation.core.plan import EvaluationVariant


class FrozenInputLoaderPort(Protocol):
    def load_turn(self, item_id: str, turn_no: int) -> Any: ...


class RiidoJudgeClientPort(Protocol):
    provider: str

    async def judge(self, payload, *, before_model_call, **kwargs) -> Any: ...


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


@dataclass(frozen=True)
class _PreparedRiidoTurn:
    """Private, in-memory candidate work shared by the child variants."""

    resolved: bool
    resolved_query: Optional[str]
    seed: str
    search: Optional[HybridSearchCall] = None
    presentation: Any = None
    scope: Any = None
    candidate_failure: Optional[BaseException] = None
    exact_checked: bool = False
    exact: Optional[CacheDecisionResult] = None
    exact_classification_id: Optional[int] = None
    exact_fallthrough_reasons: tuple[str, ...] = ()
    exact_subproblem_id: Optional[str] = None
    exact_canonical_answer_id: Optional[str] = None


class RiidoChatCachePathEvaluationProfile(CachePathEvaluationProfilePort):
    """Builds one isolated read snapshot and production-backed callback bundle per turn."""

    def __init__(
        self,
        *,
        service_data: ReadOnlyRiidoServiceData,
        input_loader: Optional[FrozenInputLoaderPort],
        metadata: CachePathRunMetadata,
        embedder: Any,
        query_rewrite_service: QueryRewriteService,
        judge_client: RiidoJudgeClientPort,
        analyzer_factory: Callable[[], KiwiAnalyzer] = KiwiAnalyzer,
        clock: Callable[[], float] = time.perf_counter,
        prepared_turns: Optional[Mapping[tuple[str, int], _PreparedRiidoTurn]] = None,
        prepared_golden_inputs: Optional[Mapping[tuple[str, int], Any]] = None,
        allow_judge_override: bool = False,
        judge_params_override: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._service_data = service_data
        self._input_loader = input_loader
        self._metadata = metadata
        self._embedder = embedder
        self._rewrite = query_rewrite_service
        self._judge = judge_client
        self._analyzer_factory = analyzer_factory
        self._clock = clock
        self._prepared_turns = prepared_turns
        self._prepared_golden_inputs = prepared_golden_inputs
        self._allow_judge_override = allow_judge_override
        self._judge_params_override = judge_params_override

    async def callbacks_for_turn(
        self, selected_turn: Mapping[str, Any]
    ) -> CacheDecisionCallbacks:
        item_id, turn_no = _reference(selected_turn)
        prepared_turn = (
            None
            if self._prepared_turns is None
            else self._prepared_turns.get((item_id, turn_no))
        )
        golden = (
            self._prepared_golden_inputs[(item_id, turn_no)]
            if self._prepared_golden_inputs is not None
            else self._load_turn(item_id, turn_no)
        )
        manager = self._service_data.open_snapshot(
            self._metadata, allow_judge_override=self._allow_judge_override
        ) if self._allow_judge_override else self._service_data.open_snapshot(self._metadata)
        snapshot = await manager.__aenter__()
        state = _TurnState(
            seed=presentation_seed(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{self._metadata.run_id}:{item_id}:{turn_no}")
            ),
            calls=[],
        )
        if prepared_turn is not None:
            state.seed = prepared_turn.seed
            state.query_resolved = prepared_turn.resolved
            state.resolved_query = prepared_turn.resolved_query
            state.search = prepared_turn.search
            state.candidate_presentation = prepared_turn.presentation
            state.candidate_failure = prepared_turn.candidate_failure
            state.exact = prepared_turn.exact
            state.exact_classification_id = prepared_turn.exact_classification_id
            state.exact_fallthrough_reasons = prepared_turn.exact_fallthrough_reasons
            state.selected_subproblem_id = prepared_turn.exact_subproblem_id
            state.selected_canonical_answer_id = prepared_turn.exact_canonical_answer_id
        try:
            selection = snapshot.selection
            _validate_snapshot_selection(selection, self._metadata)
            _validate_embedding_identity(selection.embedding_config, self._metadata)
            retriever: Optional[HybridRetriever] = None
            decision_service = CacheDecisionService(
                snapshot.catalog_reader, snapshot.outline_reader
            )

            async def resolve_query() -> Optional[str]:
                if prepared_turn is not None:
                    return prepared_turn.resolved_query
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
                if prepared_turn is not None:
                    if prepared_turn.search is None:
                        raise RuntimeError("공유 준비 결과에 검색 결과가 없습니다.")
                    return prepared_turn.search
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
                if prepared_turn is not None and prepared_turn.exact_checked:
                    return prepared_turn.exact
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
                if prepared_turn is not None and prepared_turn.candidate_failure is not None:
                    state.semantic = CacheDecisionResult(
                        gate=GateResult(outcome=CacheAttemptOutcome.FAILED)
                    )
                    return state.semantic
                assert state.search.retrieval_query is not None
                prepared_candidates = None
                if prepared_turn is not None:
                    presentation = prepared_turn.presentation
                    if presentation is None:
                        state.semantic = CacheDecisionResult(
                            gate=GateResult(outcome=CacheAttemptOutcome.FAILED)
                        )
                        return state.semantic
                    scope = prepared_turn.scope
                else:
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
                    prepared_candidates = await decision_service.prepare_candidates(
                        resolved_query=resolved_query,
                        embedding=vector.embedding,
                        index_version_id=selection.index_version_id,
                        document_group_id=selection.document_group_id,
                        retrieval=state.search,
                        seed=state.seed,
                    )
                    if not prepared_candidates.ready:
                        state.candidate_failure = prepared_candidates.error
                        state.semantic = CacheDecisionResult(
                            gate=GateResult(outcome=CacheAttemptOutcome.FAILED)
                        )
                        return state.semantic
                    presentation = prepared_candidates.presentation
                    scope = prepared_candidates.scope
                    state.candidate_presentation = presentation
                assert presentation is not None and scope is not None
                judge_stage = _selected_judge_stage(
                    selection.stage_settings.judge,
                    self._metadata,
                    allow_override=self._allow_judge_override,
                    params_override=self._judge_params_override,
                )
                decision = await decision_service.judge(
                    client=self._judge,
                    presentation=presentation,
                    model=judge_stage.model_name,
                    prompt_version=judge_stage.prompt_version,
                    stage=judge_stage,
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
                    scope=scope,
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

    def _load_turn(self, item_id: str, turn_no: int) -> Any:
        if self._input_loader is None:
            raise RuntimeError("선택 턴의 frozen source 입력이 준비되지 않았습니다.")
        return self._input_loader.load_turn(item_id, turn_no)

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
    judge_client: RiidoJudgeClientPort,
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


def create_riido_chat_plan_binding(
    *,
    service_key: str,
    base_metadata: CachePathRunMetadata,
    service_data: ReadOnlyRiidoServiceData,
    embedder: Any,
    query_rewrite_service: QueryRewriteService,
    judge_client: RiidoJudgeClientPort,
    golden_root: Path,
    judge_client_factory: Optional[
        Callable[[EvaluationVariant, CachePathPlanRunSpec], RiidoJudgeClientPort]
    ] = None,
    judge_client_validator: Optional[
        Callable[[EvaluationVariant, CachePathPlanRunSpec], None]
    ] = None,
    analyzer_factory: Callable[[], KiwiAnalyzer] = KiwiAnalyzer,
    evaluation_only_judge_override: bool = False,
):
    """Build the Riido-specific shared preparer and fixed-input child factory.

    The shared preparer performs rewrite, retrieval, question-vector resolution,
    and candidate construction once. Child runs reuse that private presentation
    while keeping the existing exact-cache and gate lifecycle per variant. When
    supplied, ``judge_client_factory`` is called once per variant during plan
    execution, after plan preflight and shared preparation; otherwise all child
    variants use the injected ``judge_client``.
    """
    if base_metadata.service_key != service_key:
        raise ValueError("base metadata service key does not match the Riido binding")

    async def prepare(reference_manifest: Mapping[str, Any]) -> PreparedCacheInputs:
        _validate_reference_manifest(reference_manifest, base_metadata)
        loader = FrozenCacheCorpusLoader(
            golden_root=golden_root,
            selection_manifest=reference_manifest,
            metadata=base_metadata,
        )
        references: list[tuple[str, int]] = []
        payloads: dict[tuple[str, int], Any] = {}
        prepared_turns: dict[tuple[str, int], _PreparedRiidoTurn] = {}
        golden_inputs: dict[tuple[str, int], Any] = {}
        preparation_traces: list[ModelCallEvidence] = []
        for selected in reference_manifest["turns"]:
            item_id, turn_no = _reference(selected)
            key = (item_id, turn_no)
            references.append(key)
            golden = loader.load_turn(item_id, turn_no)
            golden_inputs[key] = golden
            seed = presentation_seed(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{base_metadata.run_id}:{item_id}:{turn_no}")
            )
            manager = service_data.open_snapshot(
                base_metadata,
                allow_judge_override=evaluation_only_judge_override,
            ) if evaluation_only_judge_override else service_data.open_snapshot(base_metadata)
            async with manager as snapshot:
                selection = snapshot.selection
                _validate_snapshot_selection(selection, base_metadata)
                _validate_embedding_identity(selection.embedding_config, base_metadata)
                decision_service = CacheDecisionService(
                    snapshot.catalog_reader, snapshot.outline_reader
                )

                # A currently served exact hit keeps the zero-judge path and does
                # not incur shared rewrite, retrieval, or embedding work.
                exact_served = False
                exact_result = None
                exact_classification_id = None
                exact_fallthrough_reasons: tuple[str, ...] = ()
                exact_subproblem_id = None
                exact_canonical_answer_id = None
                if selection.exact_cache_enabled:
                    match = await snapshot.exact_reader.find_exact_question_log_match(
                        selection.document_group_id, golden.question
                    )
                    if match is not None:
                        exact_classification_id = match.classification_id
                        inspection = await decision_service.inspect_exact_source(
                            match,
                            index_version_id=selection.index_version_id,
                            document_group_id=selection.document_group_id,
                        )
                        if inspection.fallthrough_reasons:
                            exact_fallthrough_reasons = inspection.fallthrough_reasons
                        if inspection.scope_matches_document_group and not inspection.fallthrough_reasons:
                            presented = PresentedSubproblem(
                                key=match.key,
                                subproblem_id=match.subproblem_id,
                                subproblem_version=match.source_subproblem_version,
                                problem_group_id=match.problem_group_id,
                                document_source_id=match.document_source_id,
                                document_key=match.document_key or "",
                                canonical_answer_id=(
                                    None if inspection.inputs is None or inspection.inputs.canonical_answer is None
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
                            exact_result = (
                                CacheDecisionResult(
                                    gate=GateResult(
                                        outcome=CacheAttemptOutcome.REJECTED,
                                        rejection_reasons=(REJECT_CANONICAL_DATA_INVALID,),
                                    ),
                                    data_error=inspection.data_error,
                                )
                                if inspection.data_error is not None
                                else await decision_service.decide_gate(
                                    judgment,
                                    scope=inspection.scope,
                                    semantic_cache_enabled=selection.exact_cache_enabled,
                                    inputs=inspection.inputs,
                                )
                            )
                            exact_subproblem_id = str(match.subproblem_id)
                            canonical = (
                                None if exact_result.inputs is None
                                else exact_result.inputs.canonical_answer
                            )
                            exact_canonical_answer_id = (
                                None if canonical is None else str(canonical.canonical_answer_id)
                            )
                            exact_served = exact_result.gate.outcome == CacheAttemptOutcome.SERVED
                if exact_served:
                    payloads[key] = None
                    prepared_turns[key] = _PreparedRiidoTurn(
                        resolved=False,
                        resolved_query=None,
                        seed=seed,
                        exact_checked=selection.exact_cache_enabled,
                        exact=exact_result,
                        exact_classification_id=exact_classification_id,
                        exact_fallthrough_reasons=exact_fallthrough_reasons,
                        exact_subproblem_id=exact_subproblem_id,
                        exact_canonical_answer_id=exact_canonical_answer_id,
                    )
                    continue

                traces: list[tuple[str, ModelCallTrace]] = []
                if turn_no == 1:
                    resolved_query = golden.question
                    resolved = True
                elif not golden.candidates:
                    resolved_query, resolved = None, False
                else:
                    rewritten = await query_rewrite_service.rewrite(
                        golden.question,
                        golden.candidates,
                        stage=selection.stage_settings.rewrite,
                    )
                    traces.append(("QUERY_REWRITE", rewritten.trace))
                    resolved = (
                        rewritten.resolution is not None
                        and rewritten.resolution.should_retrieve
                    )
                    resolved_query = (
                        None if rewritten.resolution is None
                        else rewritten.resolution.resolved_query
                    )
                if resolved_query is None:
                    payloads[key] = None
                    prepared_turns[key] = _PreparedRiidoTurn(
                        resolved=False,
                        resolved_query=resolved_query,
                        seed=seed,
                        exact_checked=selection.exact_cache_enabled,
                        exact=exact_result,
                        exact_classification_id=exact_classification_id,
                        exact_fallthrough_reasons=exact_fallthrough_reasons,
                        exact_subproblem_id=exact_subproblem_id,
                        exact_canonical_answer_id=exact_canonical_answer_id,
                    )
                    preparation_traces.extend(
                        ModelCallEvidence.from_trace(purpose, trace)
                        for purpose, trace in traces
                    )
                    continue

                chunks = await snapshot.search_reader.load_active_chunks()
                if not chunks:
                    failure = ValueError("선택한 index snapshot에 검색 chunk가 없습니다.")
                    payloads[key] = None
                    prepared_turns[key] = _PreparedRiidoTurn(
                        resolved,
                        resolved_query,
                        seed,
                        candidate_failure=failure,
                        exact_checked=selection.exact_cache_enabled,
                        exact=exact_result,
                        exact_classification_id=exact_classification_id,
                        exact_fallthrough_reasons=exact_fallthrough_reasons,
                        exact_subproblem_id=exact_subproblem_id,
                        exact_canonical_answer_id=exact_canonical_answer_id,
                    )
                    preparation_traces.extend(
                        ModelCallEvidence.from_trace(purpose, trace)
                        for purpose, trace in traces
                    )
                    continue
                retriever = HybridRetriever(
                    BM25Retriever(chunks, analyzer=analyzer_factory()),
                    VectorRetriever(embedder, snapshot.search_reader),
                )
                search = await retriever.search_with_trace(resolved_query)
                if search.embedding_call is not None:
                    traces.append(("QUERY_EMBEDDING", search.embedding_call))
                if exact_result is not None:
                    payloads[key] = None
                    prepared_turns[key] = _PreparedRiidoTurn(
                        resolved=resolved,
                        resolved_query=resolved_query,
                        seed=seed,
                        search=search,
                        exact_checked=selection.exact_cache_enabled,
                        exact=exact_result,
                        exact_classification_id=exact_classification_id,
                        exact_fallthrough_reasons=exact_fallthrough_reasons,
                        exact_subproblem_id=exact_subproblem_id,
                        exact_canonical_answer_id=exact_canonical_answer_id,
                    )
                    preparation_traces.extend(
                        ModelCallEvidence.from_trace(purpose, trace)
                        for purpose, trace in traces
                    )
                    continue
                vector = await resolve_question_vector(
                    resolved_query, search, embedder
                )
                if vector.trace is not None:
                    traces.append(("QUERY_EMBEDDING", vector.trace))
                presentation = None
                scope = None
                failure = vector.error
                if failure is None and vector.embedding is not None:
                    candidates = await decision_service.prepare_candidates(
                        resolved_query=resolved_query,
                        embedding=vector.embedding,
                        index_version_id=selection.index_version_id,
                        document_group_id=selection.document_group_id,
                        retrieval=search,
                        seed=seed,
                    )
                    presentation = candidates.presentation
                    scope = candidates.scope
                    failure = candidates.error
                payload = None if presentation is None or failure is not None else presentation.payload
                payloads[key] = payload
                prepared_turns[key] = _PreparedRiidoTurn(
                    resolved=resolved,
                    resolved_query=resolved_query,
                    seed=seed,
                    search=search,
                    presentation=presentation,
                    scope=scope,
                    candidate_failure=failure,
                    exact_checked=selection.exact_cache_enabled,
                    exact=exact_result,
                    exact_classification_id=exact_classification_id,
                    exact_fallthrough_reasons=exact_fallthrough_reasons,
                    exact_subproblem_id=exact_subproblem_id,
                    exact_canonical_answer_id=exact_canonical_answer_id,
                )
                preparation_traces.extend(
                    ModelCallEvidence.from_trace(purpose, trace)
                    for purpose, trace in traces
                )

        by_purpose: dict[str, dict[str, int]] = {}
        for call in preparation_traces:
            totals = by_purpose.setdefault(
                call.purpose,
                {"calls": 0, "latencyMs": 0, "retryCount": 0, "inputTokens": 0, "outputTokens": 0},
            )
            totals["calls"] += 1
            totals["latencyMs"] += call.latency_ms
            totals["retryCount"] += call.retry_count
            totals["inputTokens"] += call.input_tokens or 0
            totals["outputTokens"] += call.output_tokens or 0
        evidence = {
            "sharedModelCalls": len(preparation_traces),
            "sharedEmbeddingCalls": by_purpose.get("QUERY_EMBEDDING", {}).get("calls", 0),
            "sharedCallsByPurpose": by_purpose,
        }
        return PreparedCacheInputs(
            references=tuple(references),
            payloads=payloads,
            opaque_context={"riidoTurns": prepared_turns, "goldenInputs": golden_inputs},
            preparation_evidence=evidence,
        )

    def profile_factory(
        *,
        variant: EvaluationVariant,
        spec: CachePathPlanRunSpec,
        prepared: PreparedCacheInputs,
        reference_manifest: Mapping[str, Any],
    ) -> RiidoChatCachePathEvaluationProfile:
        validate_variant(variant, spec)
        _validate_reference_manifest(reference_manifest, spec.metadata)
        context = prepared.opaque_context
        shared_turns = context.get("riidoTurns") if isinstance(context, Mapping) else None
        if not isinstance(shared_turns, Mapping):
            raise ValueError("prepared inputs do not contain Riido shared candidates")
        golden_inputs = context.get("goldenInputs")
        if not isinstance(golden_inputs, Mapping):
            raise ValueError("prepared inputs do not contain frozen Riido source inputs")
        # Each variant gets its own payload copy from the canonical snapshot so
        # one injected judge cannot mutate another variant's logical input.
        turns = {
            key: (
                replace(
                    turn,
                    presentation=replace(
                        turn.presentation,
                        payload=prepared.payload_for_turn(*key),
                    ),
                )
                if turn.presentation is not None
                else turn
            )
            for key, turn in shared_turns.items()
        }
        selected_judge_client = (
            judge_client
            if judge_client_factory is None
            else judge_client_factory(variant, spec)
        )
        if selected_judge_client is None or not callable(
            getattr(selected_judge_client, "judge", None)
        ):
            raise TypeError("judge_client_factory must return a judge client")
        return RiidoChatCachePathEvaluationProfile(
            service_data=service_data,
            input_loader=None,
            metadata=spec.metadata,
            embedder=embedder,
            query_rewrite_service=query_rewrite_service,
            judge_client=selected_judge_client,
            analyzer_factory=analyzer_factory,
            prepared_turns=turns,
            prepared_golden_inputs=golden_inputs,
            allow_judge_override=evaluation_only_judge_override,
            judge_params_override=_judge_params_from_config(spec.profile_config_dict()),
        )

    def validate_variant(variant: EvaluationVariant, spec: CachePathPlanRunSpec) -> None:
        """Validate all code-owned judge settings before shared model/embedding work."""
        if spec.metadata.service_key != service_key:
            raise ValueError("selected service key does not match the Riido plan binding")
        config = spec.profile_config_dict()
        configured_override = config.get("evaluationOnlyJudgeOverride", False)
        if type(configured_override) is not bool or configured_override is not evaluation_only_judge_override:
            raise ValueError("profileConfig evaluationOnlyJudgeOverride must match the explicit plan binding")
        model = spec.metadata.judge_model_version
        prompt_version = spec.metadata.judge_prompt_version
        if not isinstance(model, str) or not isinstance(prompt_version, str):
            raise ValueError("cache-path plan variants require judge model and prompt metadata")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", model):
            raise ValueError("judge model must be a safe model identifier")
        try:
            prompts.get("judge", prompt_version)
        except Exception as error:  # registry adapters may expose a typed lookup error
            raise ValueError("judge prompt version is not registered") from error
        _judge_params_from_config(config)
        if judge_client_validator is not None:
            judge_client_validator(variant, spec)

    profile_factory.validate_variant = validate_variant
    profile_factory.supports_fixed_prepared_inputs = True
    return prepare, profile_factory


def _selected_judge_stage(
    production_stage: StageSettings,
    metadata: CachePathRunMetadata,
    *,
    allow_override: bool,
    params_override: Optional[Mapping[str, Any]] = None,
) -> StageSettings:
    if allow_override and (
        metadata.judge_model_version is None
        or metadata.judge_prompt_version is None
    ):
        raise ValueError(
            "evaluation-only judge override requires model and prompt metadata"
        )
    model = metadata.judge_model_version or production_stage.model_name
    prompt_version = metadata.judge_prompt_version or production_stage.prompt_version
    params = production_stage.params if params_override is None else params_override
    if (
        (model, prompt_version) == (production_stage.model_name, production_stage.prompt_version)
        and params_override is None
    ):
        return production_stage
    if not allow_override:
        raise ValueError("selected judge model/prompt differs from the production profile")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", model):
        raise ValueError("evaluation-only judge model must be a safe model identifier")
    # Prompt content remains code-owned and registered; the plan metadata records
    # the selected target while all service/index/cache identities stay pinned.
    instructions = prompts.get("judge", prompt_version)
    return StageSettings(
        model_name=model,
        prompt_version=prompt_version,
        instructions=instructions,
        params=params,
    )


def _judge_params_from_config(config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Read an optional provider-neutral request parameter override safely."""
    if "judgeParams" not in config:
        return None
    params = config["judgeParams"]
    if not isinstance(params, Mapping):
        raise ValueError("profileConfig judgeParams must be an object")
    if set(params) - {"reasoning", "max_output_tokens"}:
        raise ValueError("profileConfig judgeParams contains unsupported request settings")
    if "max_output_tokens" in params:
        tokens = params["max_output_tokens"]
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("profileConfig judgeParams max_output_tokens must be a positive integer")
    if "reasoning" in params:
        reasoning = params["reasoning"]
        if not isinstance(reasoning, Mapping) or any(not isinstance(key, str) for key in reasoning):
            raise ValueError("profileConfig judgeParams reasoning must be an object")
    return dict(params)


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
