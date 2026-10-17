"""Read-only cache decisions shared by the online recorder and evaluation adapters.

This service owns no operational writes and its results do not require IDs allocated by
classification/cache-attempt inserts. The caller controls transaction boundaries and, for
recording, supplies a reader using the same transaction so gate inputs are freshly loaded.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Awaitable, Callable, Mapping, Optional, Protocol, Sequence, Tuple

from app.database.models import CacheAttemptOutcome, ClassificationDecision
from app.core.model_trace import ModelCallTrace
from app.core.stage_settings import StageSettings
from app.question_grouping.catalog_reader import (
    CatalogDataError,
    IndexScopeNotFoundError,
)
from app.question_grouping.constants import REJECT_CANONICAL_DATA_INVALID
from app.question_grouping.decision import build_turn_judgment, failed_turn_judgment
from app.question_grouping.document_candidates import (
    attach_outlines,
    rank_documents_from_search,
)
from app.question_grouping.gate import (
    evaluate_cache_gate,
    exact_source_fallthrough_reasons,
    resolve_citations,
)
from app.question_grouping.models import (
    CitationResolution,
    ExactQuestionLogMatch,
    GateInputs,
    GateResult,
    DocumentOutline,
    IndexScope,
    JudgeCall,
    JudgeFailure,
    JudgeFailureKind,
    JudgePresentation,
    SubproblemCatalog,
    TurnJudgment,
)
from app.question_grouping.payload import build_judge_presentation
from app.question_grouping.subproblem_search import rank_subproblems
from app.retrieval.models import HybridSearchCall


class CacheDecisionReader(Protocol):
    """Narrow read port needed to inspect current cache eligibility."""

    async def load_index_scope(self, index_version_id: int) -> IndexScope: ...

    async def load_gate_inputs(
        self, subproblem_id: uuid.UUID, scope: IndexScope
    ) -> GateInputs: ...

    async def load_subproblem_catalog(self, scope: IndexScope) -> SubproblemCatalog: ...


class OutlineReader(Protocol):
    async def load_outlines(
        self, document_version_ids: Sequence[int], *, chunking_config_id: int
    ) -> Mapping[int, DocumentOutline]: ...


class JudgeClient(Protocol):
    provider: str

    async def judge(self, payload, *, before_model_call, **kwargs) -> JudgeCall: ...


@dataclass(frozen=True)
class CacheDecisionResult:
    """ID-independent decision plus the read snapshot that explains it."""

    gate: GateResult
    inputs: GateInputs = field(
        default_factory=lambda: GateInputs(
            contexts_by_source_id=MappingProxyType({})
        )
    )
    citation_resolutions: Tuple[CitationResolution, ...] = ()
    data_error: Optional[CatalogDataError] = None


@dataclass(frozen=True)
class CandidatePreparation:
    """Operational-ID-free candidate payload result for one resolved query."""

    scope: Optional[IndexScope] = None
    catalog: Optional[SubproblemCatalog] = None
    presentation: Optional[JudgePresentation] = None
    failure: Optional[JudgeFailure] = None
    failure_stage: Optional[str] = None
    error: Optional[BaseException] = None

    @property
    def ready(self) -> bool:
        return self.failure is None and self.presentation is not None


@dataclass(frozen=True)
class JudgeDecision:
    """Provider call and normalized judgment, before operational row recording."""

    call: JudgeCall
    judgment: TurnJudgment
    latency_ms: int


@dataclass(frozen=True)
class ExactCacheInspection:
    """Fresh exact-source inspection; fallthrough remains distinct from gate rejection."""

    scope: IndexScope
    scope_matches_document_group: bool
    inputs: Optional[GateInputs]
    fallthrough_reasons: Tuple[str, ...] = ()
    data_error: Optional[CatalogDataError] = None


def _snapshot_inputs(inputs: GateInputs) -> GateInputs:
    """Keep the returned decision snapshot immutable even when a reader used a dict."""

    return GateInputs(
        subproblem=inputs.subproblem,
        canonical_answer=inputs.canonical_answer,
        citations=tuple(inputs.citations),
        contexts_by_source_id=MappingProxyType(dict(inputs.contexts_by_source_id)),
    )


class CacheDecisionService:
    """Build candidates, judge and decide cache eligibility through read-only ports."""

    def __init__(
        self,
        reader: CacheDecisionReader,
        outline_reader: Optional[OutlineReader] = None,
    ) -> None:
        self._reader = reader
        self._outline_reader = outline_reader

    @staticmethod
    def _now(clock: Callable[[], float], started: float) -> int:
        return max(0, int((clock() - started) * 1000))

    async def prepare_candidates(
        self,
        *,
        resolved_query: str,
        embedding: Sequence[float],
        index_version_id: int,
        document_group_id: int,
        retrieval: HybridSearchCall,
        seed: str,
    ) -> CandidatePreparation:
        """Read and assemble the candidates and presentation without operational writes."""

        scope: Optional[IndexScope] = None
        catalog: Optional[SubproblemCatalog] = None
        try:
            scope = await self._reader.load_index_scope(index_version_id)
            if scope.document_group_id != document_group_id:
                raise CatalogDataError(
                    "색인 판의 문서 그룹이 턴의 문서 그룹과 다릅니다: "
                    f"index={scope.document_group_id}, turn={document_group_id}"
                )
            catalog = await self._reader.load_subproblem_catalog(scope)
            subproblems = rank_subproblems(embedding, catalog.items)
            ranked_documents = rank_documents_from_search(retrieval)
            if self._outline_reader is None:
                raise RuntimeError("문서 outline reader가 필요합니다.")
            outlines = await self._outline_reader.load_outlines(
                [item.document_version_id for item in ranked_documents],
                chunking_config_id=scope.chunking_config_id,
            )
            documents = attach_outlines(ranked_documents, outlines)
            presentation = build_judge_presentation(
                resolved_query, subproblems, documents, seed=seed
            )
            return CandidatePreparation(
                scope=scope, catalog=catalog, presentation=presentation
            )
        except (CatalogDataError, IndexScopeNotFoundError, ValueError) as error:
            if isinstance(error, (CatalogDataError, IndexScopeNotFoundError)):
                safe_message = f"판별 후보 데이터 오류: {error}"
            else:
                safe_message = f"판별 후보를 만들지 못했습니다: {type(error).__name__}"
            return CandidatePreparation(
                scope=scope,
                catalog=catalog,
                failure=JudgeFailure(
                    kind=JudgeFailureKind.INTERNAL_ERROR,
                    safe_message=safe_message,
                ),
                failure_stage="CANDIDATES",
                error=error,
            )

    async def judge(
        self,
        *,
        client: JudgeClient,
        presentation: JudgePresentation,
        model: str,
        prompt_version: str,
        stage: Optional[StageSettings] = None,
        before_model_call: Optional[
            Callable[[str, str, Optional[str]], Awaitable[None]]
        ] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> JudgeDecision:
        """Run and normalize a judgment while leaving checkpoints to the caller."""

        checkpoint_failed = False

        async def checkpoint(provider: str, model_name: str, prompt: Optional[str]) -> None:
            nonlocal checkpoint_failed
            if before_model_call is None:
                return
            try:
                await before_model_call(provider, model_name, prompt)
            except BaseException:
                checkpoint_failed = True
                raise

        kwargs = {} if stage is None else {"stage": stage}
        started = clock()
        try:
            call = await client.judge(
                presentation.payload,
                before_model_call=checkpoint,
                **kwargs,
            )
        except Exception as error:  # provider errors fail open; checkpoint errors do not
            if checkpoint_failed:
                raise
            logging.getLogger(__name__).warning(
                "판별 호출 중 예상하지 못한 오류가 발생했습니다", exc_info=error
            )
            failure = JudgeFailure(
                kind=JudgeFailureKind.INTERNAL_ERROR,
                safe_message=f"판별 호출 중 오류: {type(error).__name__}",
            )
            call = JudgeCall(
                trace=ModelCallTrace(
                    provider=client.provider,
                    model_name=model,
                    succeeded=False,
                    latency_ms=self._now(clock, started),
                    prompt_version=prompt_version,
                    error_message=failure.safe_message,
                ),
                failure=failure,
            )
        try:
            judgment = build_turn_judgment(call, presentation)
        except Exception as error:  # invalid normalization implementation fails open
            logging.getLogger(__name__).exception("판별 응답을 정규화하지 못했습니다")
            failure = JudgeFailure(
                kind=JudgeFailureKind.INTERNAL_ERROR,
                safe_message=f"판별 응답 정규화 오류: {type(error).__name__}",
            )
            judgment = failed_turn_judgment(failure)
        return JudgeDecision(
            call=call,
            judgment=judgment,
            latency_ms=self._now(clock, started),
        )

    async def inspect_exact_source(
        self,
        match: ExactQuestionLogMatch,
        *,
        index_version_id: int,
        document_group_id: int,
    ) -> ExactCacheInspection:
        scope = await self._reader.load_index_scope(index_version_id)
        if scope.document_group_id != document_group_id:
            return ExactCacheInspection(
                scope=scope,
                scope_matches_document_group=False,
                inputs=None,
            )
        inputs: Optional[GateInputs] = None
        data_error: Optional[CatalogDataError] = None
        try:
            inputs = await self._reader.load_gate_inputs(match.subproblem_id, scope)
        except CatalogDataError as error:
            data_error = error
        if inputs is not None:
            inputs = _snapshot_inputs(inputs)
        reasons = exact_source_fallthrough_reasons(match, inputs)
        return ExactCacheInspection(
            scope=scope,
            scope_matches_document_group=True,
            inputs=inputs,
            fallthrough_reasons=reasons,
            data_error=data_error,
        )

    async def decide_gate(
        self,
        judgment: TurnJudgment,
        *,
        scope: IndexScope,
        semantic_cache_enabled: bool,
        inputs: Optional[GateInputs] = None,
    ) -> CacheDecisionResult:
        if (
            judgment.failed
            or judgment.decision != ClassificationDecision.CONNECT
            or judgment.subproblem is None
        ):
            return CacheDecisionResult(
                gate=evaluate_cache_gate(
                    judgment,
                    subproblem=None,
                    canonical_answer=None,
                    citation_resolutions=(),
                    semantic_cache_enabled=semantic_cache_enabled,
                )
            )

        if inputs is None:
            try:
                inputs = await self._reader.load_gate_inputs(
                    judgment.subproblem.subproblem_id, scope
                )
            except CatalogDataError as error:
                return CacheDecisionResult(
                    gate=GateResult(
                        outcome=CacheAttemptOutcome.REJECTED,
                        rejection_reasons=(REJECT_CANONICAL_DATA_INVALID,),
                    ),
                    data_error=error,
                )
        inputs = _snapshot_inputs(inputs)
        resolutions = resolve_citations(inputs.citations, inputs.contexts_by_source_id)
        gate = evaluate_cache_gate(
            judgment,
            subproblem=inputs.subproblem,
            canonical_answer=inputs.canonical_answer,
            citation_resolutions=resolutions,
            semantic_cache_enabled=semantic_cache_enabled,
        )
        return CacheDecisionResult(
            gate=gate,
            inputs=inputs,
            citation_resolutions=resolutions,
        )
