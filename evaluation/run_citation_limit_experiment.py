"""P1 Generator에서 출처 상한과 Planning 규칙을 분리해 반복 평가한다."""

import argparse
import asyncio
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE_REVISION = "4ec0c8a"
EXPERIMENT_VERSION = "v7-no-citation-cap"
OUTPUT = ROOT / "evaluation/baselines/answer-consistency-p1-no-citation-cap.json"
DIRECT_SECTION_SELECTION_RULE = """

## Direct source selection
- 먼저 질문에 직접 답하는 핵심 SOURCE 하나를 찾으세요.
- 그 SOURCE만으로 사용자가 명시적으로 요청한 모든 정보 단위를 뒷받침할 수 있으면
  그 SOURCE만 선택하세요.
- 다른 SOURCE는 그것을 빼면 사용자가 명시적으로 요청한 정보 단위가 근거 없이 남을 때만
  추가하세요.
- 단지 관련이 있거나, 더 자세한 설명·예시·조건을 덧붙일 수 있다는 이유로는
  다른 SOURCE를 추가하지 마세요.
- 사용자가 하위 기능을 열거하지 않고 "어떤 기능", "무슨 기능", "무엇을 제공"처럼
  넓게 물으면 SUMMARY입니다. 기능이나 핵심 가치를 직접 열거한 SOURCE를 핵심 근거로
  선택하세요.
- 이런 SUMMARY 질문에서는 연동·설정 절차, FAQ, 예외·제한, 개요를 단지 기능과 관련
  있다는 이유로 추가하지 마세요. 사용자가 설정 방법, 조건, 제한, 예외를 직접 물었을
  때만 해당 SOURCE를 추가하세요.
- answer_scope가 SUMMARY이면 evidence_requirements는 질문 전체를 나타내는 하나만 만들고,
  source_ids에는 그 정보 단위를 충분히 뒷받침하는 가장 직접적인 SOURCE 하나만 작성하세요.
  같은 내용을 보충하거나 반복하는 SOURCE를 함께 선택하지 마세요.
- 사용자가 여러 정보 단위를 명시적으로 요청한 MULTI_DETAIL 질문에는 이 한 개 제한을
  적용하지 말고, 정보 단위별로 필요한 SOURCE를 선택하세요.
"""


def extract_string_constant(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"{name} 문자열 상수를 찾지 못했습니다.")


def install_experiment_generator(*, direct_section_selection: bool = False):
    """별도 프로세스의 모듈만 교체하고 제품 파일은 수정하지 않는다."""
    name = "app.answering.generator"
    if name in sys.modules:
        raise RuntimeError("Generator를 import하기 전에 실행해야 합니다.")
    original = subprocess.run(
        ["git", "show", f"{BASE_REVISION}:app/answering/generator.py"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout
    baseline = json.loads(
        (ROOT / "evaluation/baselines/answer-consistency-after-p1.json").read_text()
    )
    original_planning_prompt = extract_string_constant(
        original, "SOURCE_PLANNING_PROMPT_V6"
    )
    original_planning_hash = hashlib.sha256(
        original_planning_prompt.encode()
    ).hexdigest()
    assert original_planning_hash == baseline["evaluator"]["prompts"][
        "sourcePlanning"
    ]["sha256"]

    old_rule = (
        "- 최종 고유 Citation은 최대 3개입니다. 같은 원문 URL과 Section Path를 가진 여러 SOURCE는\n"
        "  Backend에서 하나의 Citation으로 병합되므로, SOURCE 개수 자체를 3개로 제한하지 마세요."
    )
    new_rule = (
        "- 같은 원문 URL과 Section Path를 가진 여러 SOURCE는\n"
        "  Backend에서 하나의 Citation으로 병합됩니다."
    )
    assert original.count(old_rule) == 1
    source = original.replace(old_rule, new_rule)
    planning_version = "v6"
    experiment_version = EXPERIMENT_VERSION
    if direct_section_selection:
        marker = "\n## Structured Output contract\n"
        assert source.count(marker) == 2
        source = source.replace(
            marker,
            DIRECT_SECTION_SELECTION_RULE + marker,
            1,
        )
        planning_version = "v6-direct-section-3"
        experiment_version = f"{EXPERIMENT_VERSION}-direct-section-3"
    source = source.replace(
        'GENERATION_PROMPT_VERSION = "v7"',
        f'GENERATION_PROMPT_VERSION = "{experiment_version}"',
    ).replace(
        'ANSWER_REPAIR_PROMPT_VERSION = "v7-repair-1"',
        f'ANSWER_REPAIR_PROMPT_VERSION = "{experiment_version}-repair-1"',
    )
    gate = "        if count_distinct_citations(selected_sources) > MAX_PLANNED_CITATIONS:"
    assert source.count(gate) == 1
    start = source.index(gate)
    end = source.index("        answer_call = await self._parse_with_retry(", start)
    source = source[:start] + source[end:]
    module = types.ModuleType(name)
    module.__file__ = f"git:{BASE_REVISION}:app/answering/generator.py"
    module.__package__ = "app.answering"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)

    # 현재 평가기의 import 이름과 연결하되 내용·버전·해시는 실제 P1 실험값을 쓴다.
    module.SOURCE_PLANNING_PROMPT_V10 = module.SOURCE_PLANNING_PROMPT_V6
    module.SOURCE_PLANNING_PROMPT_VERSION = planning_version
    module.SOURCE_PLANNING_REPAIR_PROMPT_V10 = module.SOURCE_PLANNING_PROMPT_V6
    module.SOURCE_PLANNING_REPAIR_PROMPT_VERSION = f"{planning_version}-repair-1"
    module.MAX_SOURCE_PLANNING_REGENERATIONS = 0
    module.ANSWER_PROMPT_V17 = module.ANSWER_PROMPT_V7
    module.ANSWER_PROMPT_VERSION = experiment_version
    module.ANSWER_REPAIR_PROMPT_V17 = module.ANSWER_REPAIR_PROMPT_V7
    planning_hash = hashlib.sha256(module.SOURCE_PLANNING_PROMPT_V6.encode()).hexdigest()
    return {
        "name": (
            "P1_WITHOUT_CITATION_COUNT_CAP_DIRECT_SECTION_SELECTION"
            if direct_section_selection
            else "P1_WITHOUT_CITATION_COUNT_CAP"
        ),
        "baseRevision": BASE_REVISION,
        "generationPromptVersion": experiment_version,
        "sourcePlanningPromptVersion": planning_version,
        "sourcePlanningPromptSha256": planning_hash,
        "generatorSourceSha256": hashlib.sha256(source.encode()).hexdigest(),
        "generatorDiffFromP1": "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile="P1/generator.py",
            tofile="experiment/generator.py",
        )),
        "runtime": "P1 generator loaded in memory; current uncapped service/log/DTO",
        "planningMatchesP1": planning_hash == original_planning_hash,
        "planningBaseMatchesP1": original_planning_hash
        == baseline["evaluator"]["prompts"]["sourcePlanning"]["sha256"],
        "planningIntervention": (
            "Select one direct core source first; add another only when an explicitly "
            "requested information unit would otherwise be unsupported. Broad feature "
            "questions use the source that directly enumerates features and exclude setup, "
            "FAQ, exceptions, and overview unless explicitly requested. SUMMARY uses exactly "
            "one evidence requirement with one most direct and sufficient source."
            if direct_section_selection
            else None
        ),
        "retrievalMode": "LIVE_CURRENT_INDEX",
        "baselineComparison": "Historical P1 runs; not a randomized paired trial",
        "grading": "Unchanged answer-consistency-v1 criteria; manual content review separately",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--direct-section-selection", action="store_true")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat은 1 이상이어야 합니다.")
    experiment = install_experiment_generator(
        direct_section_selection=args.direct_section_selection
    )
    from evaluation import run_chat_multiturn_evaluation as evaluator

    experiment_version = experiment["generationPromptVersion"]
    assert evaluator.GENERATION_PROMPT_VERSION == experiment_version
    assert evaluator.GenerationService.generate_answer.__globals__[
        "GENERATION_PROMPT_VERSION"
    ] == experiment_version
    if args.check_only:
        print(json.dumps(experiment, ensure_ascii=False, indent=2))
        return 0

    output = evaluator.unique_output_path(args.output)
    payload = asyncio.run(evaluator.run_evaluation_and_dispose(
        execution_mode="in-process",
        base_url=evaluator.DEFAULT_BASE_URL,
        timeout=evaluator.DEFAULT_TIMEOUT_SECONDS,
        cases_path=args.cases or evaluator.DEFAULT_ANSWER_CONSISTENCY_CASES_PATH,
        output_path=output,
        repository_revision=None,
        target_repository_revision=None,
        repeat=args.repeat,
        case_ids=args.case_ids or (None if args.cases else ["AC02", "AC07"]),
    ))
    payload["experiment"] = experiment
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"저장: {output}")
    return 0 if payload["summary"]["failedCaseExecutionCount"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
