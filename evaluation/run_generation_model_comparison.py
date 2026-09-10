"""고정된 Retrieval 결과로 OpenAI 생성 모델의 답변 품질을 비교한다."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.answering.generator import (
    ANSWER_PROMPT_VERSION,
    GENERATION_PROMPT_VERSION,
    SOURCE_PLANNING_PROMPT_VERSION,
    OpenAIGenerator,
)
from app.answering.models import FinalGenerationResult
from app.answering.service import GenerationService
from app.core.config import get_settings
from app.retrieval.models import HybridRetrievalResult, RetrievalChunk


EXPERIMENT_VERSION = "openai-generation-model-comparison-v1"
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation/mvp_quality_100_cases.json"
DEFAULT_FIXTURE_SOURCE_PATH = PROJECT_ROOT / (
    "evaluation/baselines/"
    "mvp-quality-v2-rechecked-recheck-20260907T165121Z.json"
)
DEFAULT_SUPPLEMENTAL_FIXTURE_PATHS = (
    PROJECT_ROOT / "evaluation/baselines/mq063-query-rewrite-v7-targeted.json",
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / (
    "evaluation/baselines/openai-generation-model-comparison-v1.json"
)
EXCLUDED_CASE_IDS = frozenset({"MQ027", "MQ040", "MQ066"})
EVALUATION_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class CandidateConfig:
    id: str
    model: str
    input_price_per_million_usd: float
    output_price_per_million_usd: float


CANDIDATE_CONFIGS: Tuple[CandidateConfig, ...] = (
    CandidateConfig("A", "gpt-5.4-mini", 0.75, 4.50),
    CandidateConfig("B", "gpt-5.6-luna", 0.20, 1.20),
    CandidateConfig("C", "gpt-5.6-terra", 2.00, 12.00),
    CandidateConfig("D", "gpt-5.6-sol", 4.00, 20.00),
    CandidateConfig("E", "gpt-6-astra", 10.00, 50.00),
)


@dataclass(frozen=True)
class FrozenGenerationTurn:
    case_id: str
    turn_no: int
    description: str
    original_question: str
    generation_query: str
    expected: Dict[str, Any]
    retrieval_results: Tuple[HybridRetrievalResult, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _git_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_actual_cases(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    actual_by_id: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload.get("runs") or []
        if not runs:
            raise ValueError(f"평가 fixture 실행 결과가 없습니다: {path}")
        for result in runs[0].get("results", []):
            actual_by_id[result["id"]] = result
    return actual_by_id


def _retriever_rank(item: Dict[str, Any], retriever_type: str) -> Optional[int]:
    for retriever in item.get("retrievers", []):
        if retriever.get("type") == retriever_type:
            return retriever.get("rank")
    return None


def _to_retrieval_result(item: Dict[str, Any]) -> HybridRetrievalResult:
    content = item["content"]
    expected_hash = item.get("contentHash")
    if expected_hash and hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_hash:
        raise ValueError(f"고정 Chunk 본문 hash가 일치하지 않습니다: {item['chunkId']}")

    return HybridRetrievalResult(
        chunk=RetrievalChunk(
            document_id=item["documentId"],
            section_id=item["sectionId"],
            document_title=item["documentTitle"],
            section_path=tuple(item["sectionPath"]),
            source_url=item["sourceUrl"],
            category=None,
            content=content,
            chunk_id=item["chunkId"],
            document_version_id=item["documentVersionId"],
            index_version_id=None,
        ),
        rrf_score=float(item["fusedScore"]),
        final_rank=int(item["fusedRank"]),
        bm25_rank=_retriever_rank(item, "BM25"),
        vector_rank=_retriever_rank(item, "VECTOR"),
    )


def load_frozen_turns(
    cases_path: Path = DEFAULT_CASES_PATH,
    fixture_source_path: Path = DEFAULT_FIXTURE_SOURCE_PATH,
    supplemental_fixture_paths: Sequence[Path] = DEFAULT_SUPPLEMENTAL_FIXTURE_PATHS,
    excluded_case_ids: Iterable[str] = EXCLUDED_CASE_IDS,
) -> Tuple[List[FrozenGenerationTurn], List[Dict[str, Any]]]:
    """저장된 E2E trace에서 Generation 직전 입력만 복원한다."""

    cases_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    actual_by_id = _load_actual_cases(
        (fixture_source_path, *supplemental_fixture_paths)
    )
    excluded = set(excluded_case_ids)
    frozen_turns: List[FrozenGenerationTurn] = []
    skipped: List[Dict[str, Any]] = []

    for case in cases_payload["cases"]:
        case_id = case["id"]
        if case_id in excluded:
            skipped.append({"caseId": case_id, "reason": "KNOWN_MVP_LIMITATION"})
            continue
        actual_case = actual_by_id.get(case_id)
        if actual_case is None:
            raise ValueError(f"고정 평가 결과에 case가 없습니다: {case_id}")
        actual_turns = {
            int(turn["turnNo"]): turn for turn in actual_case.get("turns", [])
        }

        for turn_no, expected in enumerate(case["turns"], start=1):
            fixture_turn_no = int(expected.get("fixtureTurnNo", turn_no))
            actual = actual_turns.get(fixture_turn_no)
            if actual is None:
                raise ValueError(
                    "고정 평가 결과에 turn이 없습니다: "
                    f"{case_id}/{fixture_turn_no}"
                )
            if actual.get("question") != expected.get("question"):
                raise ValueError(
                    f"질문이 일치하지 않습니다: {case_id}/{fixture_turn_no}"
                )

            db = actual.get("db") or {}
            top_five = (
                db.get("stageTrace", {}).get("retrieval", {}).get("top5") or []
            )
            if not top_five:
                skipped.append(
                    {
                        "caseId": case_id,
                        "turnNo": fixture_turn_no,
                        "reason": "GENERATION_NOT_REACHED",
                    }
                )
                continue

            frozen_turns.append(
                FrozenGenerationTurn(
                    case_id=case_id,
                    turn_no=fixture_turn_no,
                    description=case["description"],
                    original_question=expected["question"],
                    generation_query=db.get("resolvedQuery") or expected["question"],
                    expected=expected,
                    retrieval_results=tuple(
                        _to_retrieval_result(item) for item in top_five
                    ),
                )
            )

    return frozen_turns, skipped


def _lead_paragraph(markdown: str) -> str:
    return markdown.strip().split("\n\n", 1)[0] if markdown.strip() else ""


def _lead_sentence(markdown: str) -> str:
    paragraph = _lead_paragraph(markdown)
    for marker in (". ", "다. ", "요. ", "습니다. "):
        if marker in paragraph:
            return paragraph.split(marker, 1)[0] + marker.rstrip()
    return paragraph


def evaluate_result(
    expected: Dict[str, Any],
    result: FinalGenerationResult,
) -> List[str]:
    """기존 MVP 평가의 Generation 관련 결정적 기준만 적용한다."""

    failures: List[str] = []
    expected_status = expected.get("expectedStatus")
    if result.status.value != expected_status:
        failures.append(
            f"status 불일치: expected={expected_status}, actual={result.status.value}"
        )

    expected_reason = expected.get("expectedWithheldReason")
    actual_reason = (
        result.withheld_reason.value if result.withheld_reason is not None else None
    )
    if expected_reason is not None and actual_reason != expected_reason:
        failures.append(
            "withheld reason 불일치: "
            f"expected={expected_reason}, actual={actual_reason}"
        )

    expected_planning_status = expected.get("expectedPlanningStatus")
    source_plan = (
        result.stage_trace.source_plan
        if result.stage_trace is not None
        else None
    )
    actual_planning_status = (
        source_plan.status.value if source_plan is not None else None
    )
    if (
        expected_planning_status is not None
        and actual_planning_status != expected_planning_status
    ):
        failures.append(
            "planning status 불일치: "
            f"expected={expected_planning_status}, "
            f"actual={actual_planning_status}"
        )

    answer = result.answer_markdown or ""
    for field, target, label in (
        ("expectedDefinitionSentenceConceptGroups", _lead_sentence(answer), "첫 문장"),
        ("expectedLeadConceptGroups", _lead_paragraph(answer), "첫 문단"),
        ("expectedAnswerConceptGroups", answer, "답변"),
    ):
        for alternatives in expected.get(field, []):
            if not any(keyword in target for keyword in alternatives):
                failures.append(
                    f"{label}에 기대 개념이 없음: alternatives={alternatives!r}"
                )

    citation_titles = [citation.document_title for citation in result.citations]
    minimum_citations = expected.get("minimumCitationCount")
    if minimum_citations is not None and len(citation_titles) < minimum_citations:
        failures.append(
            "Citation 개수 부족: "
            f"minimum={minimum_citations}, actual={len(citation_titles)}"
        )
    expected_titles = expected.get("expectedCitationDocumentTitlesAny", [])
    if expected_titles and not any(title in citation_titles for title in expected_titles):
        failures.append(
            "기대 Citation 문서가 없음: "
            f"expected_any={expected_titles!r}, actual={citation_titles!r}"
        )
    for phrase in expected.get("forbiddenAnswerPhrases", []):
        if phrase in answer:
            failures.append(f"답변에 금지 문구가 포함됨: {phrase!r}")
    return failures


def _serialize_stage_trace(result: FinalGenerationResult) -> Dict[str, Any]:
    trace = result.stage_trace
    if trace is None:
        return {}
    return {
        "sourcePlan": (
            trace.source_plan.model_dump(mode="json")
            if trace.source_plan is not None
            else None
        ),
        "selectedSourceIds": [source.source_id for source in trace.selected_sources],
        "selectedSectionIds": [
            source.chunk.section_id for source in trace.selected_sources
        ],
        "planningAttemptCount": trace.planning_attempt_count,
        "planningRegenerationCount": trace.planning_regeneration_count,
        "answerAttemptCount": trace.answer_attempt_count,
        "validationRegenerationCount": trace.validation_regeneration_count,
        "validationErrors": list(trace.validation_errors),
    }


def _serialize_result(
    fixture: FrozenGenerationTurn,
    result: FinalGenerationResult,
    elapsed_ms: int,
) -> Dict[str, Any]:
    failures = evaluate_result(fixture.expected, result)
    model_call = result.model_call
    return {
        "caseId": fixture.case_id,
        "turnNo": fixture.turn_no,
        "description": fixture.description,
        "originalQuestion": fixture.original_question,
        "generationQuery": fixture.generation_query,
        "expected": fixture.expected,
        "passed": not failures,
        "failures": failures,
        "status": result.status.value,
        "withheldReason": (
            result.withheld_reason.value
            if result.withheld_reason is not None
            else None
        ),
        "answerMarkdown": result.answer_markdown,
        "citations": [
            {
                "documentTitle": citation.document_title,
                "sectionPath": list(citation.section_path),
                "sourceUrl": citation.source_url,
            }
            for citation in result.citations
        ],
        "latencyMs": elapsed_ms,
        "inputTokens": model_call.input_tokens if model_call else None,
        "outputTokens": model_call.output_tokens if model_call else None,
        "retryCount": model_call.retry_count if model_call else None,
        "errorCode": result.error_code,
        "errorMessage": model_call.error_message if model_call else None,
        "stageTrace": _serialize_stage_trace(result),
    }


def _execution_error(
    fixture: FrozenGenerationTurn,
    error: Exception,
    elapsed_ms: int,
) -> Dict[str, Any]:
    return {
        "caseId": fixture.case_id,
        "turnNo": fixture.turn_no,
        "description": fixture.description,
        "originalQuestion": fixture.original_question,
        "generationQuery": fixture.generation_query,
        "passed": False,
        "failures": [f"실행 오류: {type(error).__name__}: {error}"],
        "status": "ERROR",
        "withheldReason": None,
        "answerMarkdown": None,
        "citations": [],
        "latencyMs": elapsed_ms,
        "inputTokens": None,
        "outputTokens": None,
        "retryCount": None,
        "errorCode": "EVALUATION_EXECUTION_ERROR",
        "errorMessage": str(error),
        "stageTrace": {},
    }


def calculate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    config: CandidateConfig,
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("토큰 수는 음수일 수 없습니다.")
    return (
        input_tokens * config.input_price_per_million_usd
        + output_tokens * config.output_price_per_million_usd
    ) / 1_000_000


def _percentile(values: Sequence[int], percentile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile + 0.999999) - 1))
    return ordered[index]


def summarize_candidate(
    config: CandidateConfig,
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    input_tokens = sum(item.get("inputTokens") or 0 for item in results)
    output_tokens = sum(item.get("outputTokens") or 0 for item in results)
    latencies = [item["latencyMs"] for item in results]
    return {
        "turnExecutionCount": len(results),
        "passedCount": sum(bool(item["passed"]) for item in results),
        "failedCount": sum(not bool(item["passed"]) for item in results),
        "errorCount": sum(item["status"] == "ERROR" for item in results),
        "completedCount": sum(item["status"] == "COMPLETED" for item in results),
        "withheldCount": sum(item["status"] == "WITHHELD" for item in results),
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "estimatedCostUsd": round(
            calculate_cost_usd(input_tokens, output_tokens, config), 6
        ),
        "averageLatencyMs": (
            round(sum(latencies) / len(latencies)) if latencies else None
        ),
        "p50LatencyMs": _percentile(latencies, 0.50),
        "p95LatencyMs": _percentile(latencies, 0.95),
        "planningRegenerationCount": sum(
            item.get("stageTrace", {}).get("planningRegenerationCount", 0)
            for item in results
        ),
        "validationRegenerationCount": sum(
            item.get("stageTrace", {}).get("validationRegenerationCount", 0)
            for item in results
        ),
    }


def select_candidates(model_ids: Optional[Sequence[str]]) -> List[CandidateConfig]:
    if not model_ids:
        return list(CANDIDATE_CONFIGS)
    by_id = {config.id: config for config in CANDIDATE_CONFIGS}
    by_model = {config.model: config for config in CANDIDATE_CONFIGS}
    selected = []
    for value in model_ids:
        config = by_id.get(value) or by_model.get(value)
        if config is None:
            raise ValueError(f"알 수 없는 후보입니다: {value}")
        if config not in selected:
            selected.append(config)
    return selected


def select_turns(
    turns: Sequence[FrozenGenerationTurn],
    case_ids: Optional[Sequence[str]],
) -> List[FrozenGenerationTurn]:
    if not case_ids:
        return list(turns)
    requested = set(case_ids)
    selected = [turn for turn in turns if turn.case_id in requested]
    missing = requested - {turn.case_id for turn in selected}
    if missing:
        raise ValueError(f"평가할 수 없는 case ID입니다: {sorted(missing)}")
    return selected


def _save(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def run_comparison(
    *,
    candidates: Sequence[CandidateConfig],
    turns: Sequence[FrozenGenerationTurn],
    repeat: int,
    output_path: Path,
    cases_path: Path = DEFAULT_CASES_PATH,
    fixture_source_path: Path = DEFAULT_FIXTURE_SOURCE_PATH,
    supplemental_fixture_paths: Sequence[
        Path
    ] = DEFAULT_SUPPLEMENTAL_FIXTURE_PATHS,
) -> Dict[str, Any]:
    if repeat <= 0:
        raise ValueError("repeat은 1 이상이어야 합니다.")
    if not turns:
        raise ValueError("평가할 Generation turn이 없습니다.")

    payload: Dict[str, Any] = {
        "experimentVersion": EXPERIMENT_VERSION,
        "executedAt": datetime.now(timezone.utc).isoformat(),
        "repositoryRevision": _git_revision(),
        "candidateConfigs": [asdict(config) for config in candidates],
        "prompts": {
            "generation": GENERATION_PROMPT_VERSION,
            "sourcePlanning": SOURCE_PLANNING_PROMPT_VERSION,
            "answer": ANSWER_PROMPT_VERSION,
        },
        "fixedInputs": {
            "casesPath": _project_relative(cases_path),
            "casesSha256": _sha256(cases_path),
            "fixtureSourcePath": _project_relative(fixture_source_path),
            "fixtureSourceSha256": _sha256(fixture_source_path),
            "supplementalFixtureSources": [
                {
                    "path": _project_relative(path),
                    "sha256": _sha256(path),
                }
                for path in supplemental_fixture_paths
            ],
            "turnCount": len(turns),
            "retrievalFrozen": True,
            "queryRewriteFrozen": True,
        },
        "excludedCaseIds": sorted(EXCLUDED_CASE_IDS),
        "runs": [],
        "summaries": [],
    }

    total = len(candidates) * repeat * len(turns)
    completed = 0
    for config in candidates:
        candidate_results: List[Dict[str, Any]] = []
        api_key = get_settings().openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
        client = AsyncOpenAI(
            api_key=api_key,
            max_retries=0,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )
        service = GenerationService(
            OpenAIGenerator(client=client, model_name=config.model)
        )
        for repeat_no in range(1, repeat + 1):
            run_results = []
            for fixture in turns:
                completed += 1
                print(
                    f"[{completed}/{total}] {config.id}/{config.model} "
                    f"repeat={repeat_no} {fixture.case_id}/{fixture.turn_no}",
                    flush=True,
                )
                started = time.perf_counter()
                try:
                    result = await service.generate_answer(
                        fixture.generation_query,
                        fixture.retrieval_results,
                    )
                    serialized = _serialize_result(
                        fixture,
                        result,
                        int((time.perf_counter() - started) * 1000),
                    )
                except Exception as error:
                    serialized = _execution_error(
                        fixture,
                        error,
                        int((time.perf_counter() - started) * 1000),
                    )
                run_results.append(serialized)
                candidate_results.append(serialized)
                payload["runs"] = [
                    run
                    for run in payload["runs"]
                    if not (
                        run["candidateId"] == config.id
                        and run["repeatNo"] == repeat_no
                    )
                ] + [
                    {
                        "candidateId": config.id,
                        "model": config.model,
                        "repeatNo": repeat_no,
                        "results": run_results,
                    }
                ]
                payload["summaries"] = [
                    summary
                    for summary in payload["summaries"]
                    if summary["candidateId"] != config.id
                ] + [
                    {
                        "candidateId": config.id,
                        "model": config.model,
                        **summarize_candidate(config, candidate_results),
                    }
                ]
                _save(payload, output_path)
        await client.close()

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--fixture-source",
        type=Path,
        default=DEFAULT_FIXTURE_SOURCE_PATH,
    )
    parser.add_argument(
        "--supplemental-fixtures",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SUPPLEMENTAL_FIXTURE_PATHS),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        help="후보 ID(A~E) 또는 정확한 model ID",
    )
    parser.add_argument("--case-ids", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    turns, skipped = load_frozen_turns(
        cases_path=args.cases,
        fixture_source_path=args.fixture_source,
        supplemental_fixture_paths=args.supplemental_fixtures,
    )
    selected_turns = select_turns(turns, args.case_ids)
    candidates = select_candidates(args.models)
    payload = asyncio.run(
        run_comparison(
            candidates=candidates,
            turns=selected_turns,
            repeat=args.repeat,
            output_path=args.output,
            cases_path=args.cases,
            fixture_source_path=args.fixture_source,
            supplemental_fixture_paths=args.supplemental_fixtures,
        )
    )
    print(f"제외/미도달: {len(skipped)}건", flush=True)
    for summary in payload["summaries"]:
        print(
            f"{summary['candidateId']} {summary['model']}: "
            f"{summary['passedCount']}/{summary['turnExecutionCount']} 통과, "
            f"${summary['estimatedCostUsd']:.6f}",
            flush=True,
        )
    print(f"저장: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
