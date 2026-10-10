"""판별 측정 실행기(evaluation/judge_eval) 시험. 실제 모델 API 는 부르지 않고 가짜 클라이언트만 쓴다.

평가 등록부는 시험마다 임시 폴더에 합성 문항으로 만든다(고객 문장 없음). 등록부의 묶음 생성 스크립트와
스키마 검사 스크립트도 그 계약(명령줄 입출력, make_schema_checker)만 흉내 낸 가짜다. 실제 등록부
(RIIDO_EVALS_ROOT 환경 변수 또는 evaluation/registry 서브모듈)가 있으면 실제 결과 스키마로 검사하고
실제 judge-official-A 묶음 추정도 확인한다. 없으면 그 시험만 건너뛴다.
"""

import asyncio
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest import mock

from app.database.models import QuestionSubproblemServingState
from app.question_grouping.judge_client import build_judge_request
from app.question_grouping.models import (
    CatalogCanonicalAnswer,
    DocumentCandidate,
    DocumentOutline,
    SubproblemCandidate,
    SubproblemCatalogItem,
)
from app.question_grouping.payload import build_judge_presentation, split_criteria
from evaluation.judge_eval import cli
from evaluation.judge_eval import golden as golden_mod
from evaluation.judge_eval.clients import (
    JEV_NONE_KEY,
    JevJudge,
    LunaJudge,
    Prices,
    build_jev_request,
)
from evaluation.judge_eval.golden import GoldenError, HoldoutRefused, Label
from evaluation.judge_eval.inputs import request_sha256
from evaluation.judge_eval.ledger import RunCapReached
from evaluation.judge_eval.scoring import (
    binom_cdf,
    clopper_pearson_upper,
    compare,
    mcnemar_exact_p,
    metrics,
    rate_with_bound,
    score_turn,
    summarize,
    zero_error_upper,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_VERSION = "9.9.9"
SUITE_A = "judge-official-A"
SUITE_B = "judge-official-B"
GIT = ["git", "-c", "user.name=judge-eval-test", "-c", "user.email=judge-eval@example.invalid", "-c", "commit.gpgsign=false"]

CATALOG = [
    # key, servingState, documentKey
    ("doc-a.serve", "SERVING", "folder/doc-a"),
    ("doc-a.shadow", "SHADOW", "folder/doc-a"),
    ("doc-b.other", "SHADOW", "doc-b"),
    ("doc-b.fourth", "SHADOW", "doc-b"),
    ("doc-c.fifth", "SHADOW", "doc-c"),
    ("doc-c.missing", "SHADOW", "doc-c"),
]
PRESENTED = ["doc-a.serve", "doc-a.shadow", "doc-b.other", "doc-b.fourth", "doc-c.fifth"]

# (문항 ID, 층, 라벨, 키, 출처, 상태, 판별했는지)
TURNS = [
    ("A-t1", "A", "CONNECT", ["doc-a.serve"], "model-panel", "confirmed", True),
    ("A-t2", "A", "SEPARATE", [], "audited", "confirmed", True),
    ("A-t3", "A", "UNCLASSIFIED", [], "human", "confirmed", True),
    ("A-t4", "A", "CONNECT", ["doc-c.missing"], "model-panel", "confirmed", True),
    ("A-t5", "A", "CONNECT", ["doc-a.shadow"], "model-panel", "confirmed", True),
    ("A-t6", "A", "SEPARATE", [], "model-draft", "draft", True),
    ("A-t7", "A", "SEPARATE", [], "model-panel", "confirmed", False),
    ("B-x1", "B", "CONNECT", ["doc-a.serve"], "model-panel", "confirmed", True),
    ("B-x2", "B", "SEPARATE", [], "human", "confirmed", True),
]

FAKE_BUILD_SUITES = textwrap.dedent(
    '''
    """가짜 build_suites.py: 등록부 명령줄 계약(--suite --layers --out)만 흉내 낸다."""
    import argparse, json
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--suite"); parser.add_argument("--layers"); parser.add_argument("--out", type=Path)
        args = parser.parse_args()
        rules = json.loads((ROOT / "golden/suites/rules.json").read_text(encoding="utf-8"))
        suite = rules["suites"][args.suite]
        layers = args.layers.split(",")
        entries = []
        for layer in layers:
            for line in (ROOT / f"golden/items/{layer}.jsonl").read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                turn_nos = [
                    no for no, turn in enumerate(item["turns"], 1)
                    if turn.get("expectedJudgeDecision")
                    and (not suite.get("requireConfirmed") or turn["review"]["judge"]["status"] == "confirmed")
                ]
                if turn_nos:
                    entries.append({"id": item["id"], "layer": layer, "turnNos": turn_nos})
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {"suite": args.suite, "kind": suite["kind"], "reportable": suite["reportable"], "stage": "judge",
                   "requireConfirmed": True, "layers": layers, "items": entries}
        (args.out / f"{args.suite}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


    if __name__ == "__main__":
        main()
    '''
)

FAKE_VALIDATE = textwrap.dedent(
    '''
    """가짜 validate.py: make_schema_checker 계약만 흉내 낸다."""
    def make_schema_checker(schema):
        return (lambda value: []), "fake"
    '''
)


def real_registry() -> Optional[Path]:
    candidates = [os.environ.get("RIIDO_EVALS_ROOT"), str(PROJECT_ROOT / "evaluation" / "registry")]
    for value in candidates:
        if value and (Path(value) / "schemas" / "eval-result.schema.json").exists():
            return Path(value)
    return None


def sha256(path: Path) -> str:
    return golden_mod.sha256_file(path)


def git(root: Path, *args: str) -> None:
    subprocess.run([*GIT, "-C", str(root), *args], check=True, capture_output=True)


def catalog_entries() -> List[Dict[str, Any]]:
    return [
        {
            "key": key,
            "name": f"{key} 이름",
            "documentTitle": f"{doc} 문서",
            "servingState": state,
            "currentRevision": 1,
            "inclusionCriteria": [f"- {key} 포함 기준 1", f"- {key} 포함 기준 2"],
            "exclusionCriteria": [f"- {key} 제외 기준"],
            "representativeQuestion": f"{key} 대표 질문",
            "canonical": {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, key)), "sha256": "x", "text": f"{key} 정본 본문"},
        }
        for key, state, doc in CATALOG
    ]


def frozen_row(item_id: str, layer: str, judged: bool, entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """운영 코드로 제시 순서와 요청 해시를 계산한 합성 동결 행 (freeze_judge_inputs.py 와 같은 칸)."""

    row: Dict[str, Any] = {"id": item_id, "turnNo": 1, "layer": layer, "resolvedQuery": f"합성 질의 {item_id}"}
    if not judged:
        row.update({"seed": None, "subproblemCandidates": [], "presentedSubproblemKeys": [], "documentCandidates": [],
                    "presentedDocumentIds": [], "payloadSha256": None, "notJudgedReason": "모호 후속 보류"})
        return row
    candidates, subproblems = [], []
    by_key = {key: doc for key, _, doc in CATALOG}
    for rank, key in enumerate(PRESENTED, 1):
        entry = entries[key]
        candidates.append({"key": key, "retrievalRank": rank, "similarity": 0.9 - rank / 100, "subproblemVersion": 1,
                           "documentKey": by_key[key], "canonicalAnswerId": entry["canonical"]["id"]})
        subproblems.append(
            SubproblemCandidate(
                item=SubproblemCatalogItem(
                    subproblem_id=uuid.uuid4(), key=key, name=entry["name"],
                    inclusion_criteria=split_criteria("\n".join(entry["inclusionCriteria"])),
                    exclusion_criteria=split_criteria("\n".join(entry["exclusionCriteria"])),
                    current_version=1, problem_group_id=uuid.uuid4(), document_source_id=None, document_key=by_key[key],
                    serving_state=QuestionSubproblemServingState(entry["servingState"]), document_title=entry["documentTitle"],
                    canonical_answer=CatalogCanonicalAnswer(
                        canonical_answer_id=uuid.UUID(entry["canonical"]["id"]), content_markdown=entry["canonical"]["text"]
                    ),
                ),
                similarity=0.5, retrieval_rank=rank,
            )
        )
    documents = [{"id": f"D{i}", "documentKey": key, "title": f"{key} 문서", "retrievalRank": i, "presentedOrder": i,
                  "score": 0.03, "documentVersionId": 100 + i} for i, key in enumerate(("folder/doc-a", "doc-b"), 1)]
    doc_candidates = [
        DocumentCandidate(outline=DocumentOutline(document_source_id=i, document_version_id=d["documentVersionId"],
                                                  document_key=d["documentKey"], title=d["title"],
                                                  parent_path=d["documentKey"].rpartition("/")[0], headings=()),
                          score=0.03, retrieval_rank=d["retrievalRank"])
        for i, d in enumerate(documents, 1)
    ]
    seed = f"seed-{item_id}"
    presentation = build_judge_presentation(row["resolvedQuery"], subproblems, doc_candidates, seed=seed)
    order = {item.key: item.presented_order for item in presentation.subproblems}
    doc_order = {doc.document_key: doc.presented_order for doc in presentation.documents}
    for candidate in candidates:
        candidate["presentedOrder"] = order[candidate["key"]]
    for doc in documents:
        doc["presentedOrder"] = doc_order[doc["documentKey"]]
    row.update({
        "seed": seed,
        "subproblemCandidates": candidates,
        "presentedSubproblemKeys": [item.key for item in presentation.subproblems],
        "documentCandidates": documents,
        "presentedDocumentIds": [doc.id for doc in presentation.documents],
        "payloadSha256": request_sha256(build_judge_request(presentation.payload)),
        "notJudgedReason": None,
    })
    return row


def build_registry(root: Path, *, schema_from: Optional[Path] = None) -> Path:
    for sub in ("golden/items", "golden/suites", "golden/frozen", "scripts", "schemas", "criteria", "evals"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # 스키마 검사 모듈 import 가 만든 캐시는 등록부 원본 변경이 아니다.
    # 전역 Git 제외 설정이 없는 CI 에서도 두 번째 실행이 같은 등록부를 읽는다.
    (root / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")
    entries = {entry["key"]: entry for entry in catalog_entries()}
    items: Dict[str, List[str]] = {"A": [], "B": []}
    rows = []
    for item_id, layer, decision, keys, source, status, judged in TURNS:
        turn = {"question": f"합성 질문 {item_id}", "expectedJudgeDecision": decision,
                "review": {"judge": {"source": source, "status": status, "reviewers": []}}}
        if keys:
            turn["expectedSubproblemKeysAny"] = keys
        items[layer].append(json.dumps({"id": item_id, "layer": layer, "turns": [turn]}, ensure_ascii=False))
        rows.append(frozen_row(item_id, layer, judged, entries))
    for layer, lines in items.items():
        (root / f"golden/items/{layer}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "golden/frozen/judge-inputs-20990101.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
    )
    catalog = {"meta": {"fingerprint": "fp-test"}, "entries": list(entries.values())}
    (root / "golden/frozen/catalog-HELP_CHATBOT-20990101.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    environment = {"mainRepo": {"commit": "0" * 40}, "catalog": {"fingerprint": "fp-test"},
                   "index": {"documentGroup": "HELP_CHATBOT", "indexVersionId": 10, "version": "idx-test"}}
    (root / "golden/frozen/environment-20990101.json").write_text(json.dumps(environment), encoding="utf-8")
    judge_suite = {"kind": "official", "reportable": True, "stage": "judge", "requireConfirmed": True,
                   "turnFilter": [{"field": "expectedJudgeDecision", "op": "exists"}]}
    rules = {"holdoutLayers": ["B"], "suites": {
        SUITE_A: dict(judge_suite, allowedLayers=["A"], defaultLayers=["A"]),
        SUITE_B: dict(judge_suite, holdout=True, allowedLayers=["B"], defaultLayers=["B"]),
        "rewrite": {"kind": "regression", "reportable": False, "stage": "rewrite", "allowedLayers": ["A"], "defaultLayers": ["A"]},
    }}
    (root / "golden/suites/rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (root / "scripts/build_suites.py").write_text(FAKE_BUILD_SUITES, encoding="utf-8")
    if schema_from is not None:
        shutil.copy(schema_from / "schemas" / "eval-result.schema.json", root / "schemas" / "eval-result.schema.json")
        # 등록부 validate.py 와 그것이 import 하는 모듈. build_suites.py 는 가짜를 그대로 둔다 (import 해도 실행되지 않음)
        for name in ("validate.py", "stages.py", "sources.py", "freeze_judge_inputs.py"):
            shutil.copy(schema_from / "scripts" / name, root / "scripts" / name)
    else:
        (root / "schemas/eval-result.schema.json").write_text("{}", encoding="utf-8")
        (root / "scripts/validate.py").write_text(FAKE_VALIDATE, encoding="utf-8")
    frozen_files = {}
    for name in ("judge-inputs-20990101.jsonl", "catalog-HELP_CHATBOT-20990101.json", "environment-20990101.json"):
        path = root / "golden/frozen" / name
        frozen_files[f"golden/frozen/{name}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "goldenVersion": GOLDEN_VERSION,
        "suiteRules": "golden/suites/rules.json",
        "layers": {layer: {"path": f"golden/items/{layer}.jsonl", "sha256": sha256(root / f"golden/items/{layer}.jsonl")}
                   for layer in ("A", "B")},
        "suiteCounts": {SUITE_A: {"A": {"items": 6, "turns": 6}}, SUITE_B: {"B": {"items": 2, "turns": 2}}},
        "frozen": {"files": frozen_files, "catalogFingerprint": {"file": "golden/frozen/catalog-HELP_CHATBOT-20990101.json",
                                                                "fingerprint": "fp-test"}},
    }
    (root / "golden/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture")
    return root


def commit_criteria(root: Path, name: str, *, final: bool = True, suite: str = SUITE_A, commit: bool = True) -> str:
    relative = f"criteria/{name}"
    status = "상태: 확정" if final else "상태: 초안 (소유자 확정 전)"
    (root / relative).write_text(f"# 기준\n\n- {status}\n- 묶음: {suite}\n- 골든셋: {GOLDEN_VERSION}\n", encoding="utf-8")
    if commit:
        git(root, "add", relative)
        git(root, "commit", "-q", "-m", f"criteria {name}")
    return relative


def write_baseline(root: Path, uses: int = 2) -> None:
    (root / "evals/run-ledger.jsonl").write_text(
        json.dumps({"kind": "baseline", "holdoutUses": uses, "source": "test"}) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------- 가짜 모델

LUNA_SCRIPT = {
    "A-t1": ("CONNECT", "doc-a.serve"),
    "A-t2": ("CONNECT", "doc-a.serve"),
    "A-t3": ("UNCLASSIFIED", None),
    "A-t4": ("SEPARATE", None),
    "A-t5": ("CONNECT", "doc-b.other"),
    "B-x1": ("CONNECT", "doc-a.serve"),
    "B-x2": ("SEPARATE", None),
}
JEV_SCRIPT = {
    "A-t1": ("doc-a.serve", 0.97),
    "A-t2": (JEV_NONE_KEY, 0.99),
    "A-t3": ("doc-b.other", 0.50),
    "A-t4": (JEV_NONE_KEY, 0.96),
    "A-t5": ("doc-a.shadow", 0.96),
    "B-x1": ("doc-a.serve", 0.98),
    "B-x2": (JEV_NONE_KEY, 0.97),
}
DOC_OF = {key: doc for key, _, doc in CATALOG}


def item_of(question: str) -> str:
    return question.rsplit(" ", 1)[-1]


class FakeResponses:
    def __init__(self, script, fail=()):
        self.script = script
        self.fail = set(fail)
        self.inputs: Dict[str, str] = {}

    async def create(self, **request):
        payload = json.loads(request["input"])
        item_id = item_of(payload["question"])
        self.inputs[item_id] = request["input"]
        if item_id in self.fail:
            raise RuntimeError("fake failure")
        decision, key = self.script[item_id]
        output = {
            "decision": decision, "groupId": DOC_OF.get(key) if key else None, "subproblemId": key, "confidence": 0.9,
            "rationaleCode": "TEST", "matchedCriteria": ["I1"] if key else [], "conflictingCriteria": [],
            "ambiguityReason": None, "documentDecision": "SUBPROBLEM_DOCUMENT" if key else "NONE",
            "documentCandidateId": None, "documentRationaleCode": None,
        }
        usage = SimpleNamespace(input_tokens=5000, output_tokens=300,
                                input_tokens_details=SimpleNamespace(cached_tokens=3000),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=200))
        return SimpleNamespace(status="completed", output_text=json.dumps(output), usage=usage)


class FakeOpenAI:
    def __init__(self, script=LUNA_SCRIPT, fail=()):
        self.responses = FakeResponses(script, fail)


class FakeJevTransport:
    def __init__(self, script=JEV_SCRIPT, statuses=None):
        self.script = script
        self.bodies: List[Dict[str, Any]] = []
        self.statuses = list(statuses or [])

    async def __call__(self, body):
        self.bodies.append(body)
        if self.statuses:
            return {"ok": False, "httpStatus": self.statuses.pop(0)}
        choice, confidence = self.script[item_of(body["state"]["질문"])]
        answer = {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": {choice: confidence}}
        return {"ok": True, "httpStatus": 200,
                "body": {"model": "jev-1.13.0", "answers": {"세부문제": answer}, "usage": {"input_tokens": 3000, "output_tokens": 4}}}


def luna_adapter(openai=None):
    return LunaJudge(openai_client=openai or FakeOpenAI())


def jev_adapter(transport=None):
    return JevJudge(transport or FakeJevTransport(), retry_delay=0.0)


class RegistryCase(unittest.TestCase):
    schema_from: Optional[Path] = None

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="judge-eval-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = build_registry(self.tmp / "registry", schema_from=self.schema_from)
        self.criteria_a = commit_criteria(self.root, "judge-a.md")

    def config(self, judge="luna", suite=SUITE_A, name="out", **kwargs) -> cli.RunConfig:
        prices = Prices(0.20, 1.20, None) if judge == "luna" else Prices(0.042, None, None)
        defaults = dict(registry=self.root, output=self.tmp / "out" / f"{name}.json", system_version="1.2.1",
                        criteria=self.criteria_a if suite == SUITE_A else None, prices=prices, concurrency=2)
        defaults.update(kwargs)
        return cli.RunConfig(suite=suite, judge=judge, golden_version=GOLDEN_VERSION, **defaults)

    def run_live(self, config, adapter):
        return asyncio.run(cli.execute_run(config, adapter))

    def ledger_runs(self):
        path = self.root / "evals/run-ledger.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if json.loads(line).get("kind") == "run"]


# ---------------------------------------------------------------- 입력 동일성


class InputIdentityTest(RegistryCase):
    def test_both_models_receive_the_same_payload_bytes(self):
        openai = FakeOpenAI()
        transport = FakeJevTransport()
        luna = self.run_live(self.config("luna", name="luna"), luna_adapter(openai))
        jev = self.run_live(self.config("jev", name="jev"), jev_adapter(transport))

        luna_raw = [json.loads(l) for l in (self.tmp / "out/luna.raw.jsonl").read_text().splitlines()]
        jev_raw = [json.loads(l) for l in (self.tmp / "out/jev.raw.jsonl").read_text().splitlines()]
        self.assertEqual([(l["itemId"], l["inputSha256"]) for l in luna_raw], [(l["itemId"], l["inputSha256"]) for l in jev_raw])
        self.assertEqual(luna["summary"]["inputs"]["inputSetSha256"], jev["summary"]["inputs"]["inputSetSha256"])

        # luna 가 받은 input 문자열의 해시가 기록한 입력 해시와 같고, 동결 요청 해시와도 같다
        for line in luna_raw:
            if line.get("inputSha256") is None:
                continue
            sent = openai.responses.inputs[line["itemId"]].encode("utf-8")
            self.assertEqual(golden_mod.sha256_bytes(sent), line["inputSha256"])
            self.assertTrue(line["payloadVerified"])
            # Jev 요청은 같은 바이트에서 결정적으로 만든다
            body = next(b for b in transport.bodies if item_of(b["state"]["질문"]) == line["itemId"])
            self.assertEqual(body, build_jev_request(json.loads(sent)))
            payload = json.loads(sent)
            self.assertEqual(list(body["questions"]["세부문제"]["criteria"])[:-1],
                             [c["subproblem"]["id"] for c in payload["candidates"]])
            self.assertIn("정본(사용자에게 그대로 전송될 답변)", body["questions"]["세부문제"]["criteria"][payload["candidates"][0]["subproblem"]["id"]])

        comparison = cli.compare_results(self.root, self.tmp / "out/luna.json", self.tmp / "out/jev.json")
        self.assertTrue(comparison["inputsIdentical"])

    def test_compare_refuses_different_inputs(self):
        self.run_live(self.config("luna", name="luna"), luna_adapter())
        self.run_live(self.config("jev", name="jev"), jev_adapter())
        raw = self.tmp / "out/jev.raw.jsonl"
        lines = [json.loads(l) for l in raw.read_text().splitlines()]
        lines[0]["inputSha256"] = "0" * 64
        raw.write_text("".join(json.dumps(l) + "\n" for l in lines))
        with self.assertRaisesRegex(GoldenError, "판별 입력이 다른 턴"):
            cli.compare_results(self.root, self.tmp / "out/luna.json", self.tmp / "out/jev.json")

    def test_strict_payload_check_refuses_before_ledger(self):
        path = self.root / "golden/frozen/judge-inputs-20990101.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[0])
        row["payloadSha256"] = "f" * 64
        lines[0] = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n")
        manifest = json.loads((self.root / "golden/manifest.json").read_text())
        manifest["frozen"]["files"]["golden/frozen/judge-inputs-20990101.jsonl"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        (self.root / "golden/manifest.json").write_text(json.dumps(manifest))
        git(self.root, "commit", "-qam", "tamper")
        with self.assertRaisesRegex(GoldenError, "1/5턴"):
            self.run_live(self.config("luna"), luna_adapter())
        self.assertEqual(self.ledger_runs(), [])
        result = self.run_live(self.config("luna", payload_check="record"), luna_adapter())
        self.assertEqual(result["summary"]["inputs"]["payloadCheck"]["unverified"], 1)

    def test_golden_version_and_hash_mismatch_refused(self):
        with self.assertRaisesRegex(GoldenError, "골든셋 버전이 다릅니다"):
            golden_mod.open_golden(self.root, "1.0.0")
        (self.root / "golden/frozen/catalog-HELP_CHATBOT-20990101.json").write_text("{}")
        with self.assertRaisesRegex(GoldenError, "해시 또는 크기 불일치"):
            golden_mod.open_golden(self.root, GOLDEN_VERSION)


# ---------------------------------------------------------------- 보류 시험셋


class HoldoutTest(RegistryCase):
    def test_holdout_suite_refused_without_flag_and_reads_nothing(self):
        with mock.patch.object(golden_mod, "load_labels", side_effect=AssertionError("B 라벨을 읽음")), \
                mock.patch.object(golden_mod, "load_frozen_rows", side_effect=AssertionError("B 동결 행을 읽음")), \
                mock.patch.object(golden_mod, "select_suite", side_effect=AssertionError("B 묶음을 뽑음")):
            with self.assertRaises(HoldoutRefused):
                cli.run_estimate(self.config("luna", suite=SUITE_B))
            with self.assertRaises(HoldoutRefused):
                self.run_live(self.config("luna", suite=SUITE_B), luna_adapter())
        with self.assertRaises(HoldoutRefused):
            golden_mod.select_suite(golden_mod.open_golden(self.root, GOLDEN_VERSION), SUITE_B)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli.main(["run", "--registry", str(self.root), "--golden-version", GOLDEN_VERSION,
                             "--suite", SUITE_B, "--judge", "luna", "--estimate"])
        self.assertEqual(code, 2)
        self.assertIn("보류 시험셋", stderr.getvalue())
        self.assertEqual(self.ledger_runs(), [])

    def test_holdout_needs_committed_final_criteria(self):
        with self.assertRaisesRegex(HoldoutRefused, "criteria"):
            self.run_live(self.config("luna", suite=SUITE_B, holdout=True), luna_adapter())
        uncommitted = commit_criteria(self.root, "b-uncommitted.md", suite=SUITE_B, commit=False)
        with self.assertRaisesRegex(GoldenError, "커밋돼 있지 않습니다"):
            self.run_live(self.config("luna", suite=SUITE_B, holdout=True, criteria=uncommitted), luna_adapter())
        (self.root / uncommitted).unlink()
        draft = commit_criteria(self.root, "b-draft.md", final=False, suite=SUITE_B)
        with self.assertRaisesRegex(GoldenError, "확정본이 아닙니다"):
            self.run_live(self.config("luna", suite=SUITE_B, holdout=True, criteria=draft), luna_adapter())
        wrong_suite = commit_criteria(self.root, "b-wrong.md", suite="judge-official-X")
        with self.assertRaisesRegex(GoldenError, SUITE_B):
            self.run_live(self.config("luna", suite=SUITE_B, holdout=True, criteria=wrong_suite), luna_adapter())
        final = commit_criteria(self.root, "b-final.md", suite=SUITE_B)
        with self.assertRaisesRegex(GoldenError, "baseline"):
            self.run_live(self.config("luna", suite=SUITE_B, holdout=True, criteria=final), luna_adapter())
        self.assertEqual(self.ledger_runs(), [])

        write_baseline(self.root, uses=2)
        result = self.run_live(self.config("luna", suite=SUITE_B, holdout=True, criteria=final, name="b1"), luna_adapter())
        self.assertEqual(result["holdoutRunNo"], 3)
        self.assertEqual(result["criteriaPath"], final)
        result = self.run_live(self.config("jev", suite=SUITE_B, holdout=True, criteria=final, name="b2"), jev_adapter())
        self.assertEqual(result["holdoutRunNo"], 4)
        self.assertEqual([run["holdoutRunNo"] for run in self.ledger_runs()], [3, 4])
        with self.assertRaises(HoldoutRefused):
            cli.rescore(self.root, self.tmp / "out/b1.json")

    def test_holdout_flag_on_non_holdout_suite_rejected(self):
        with self.assertRaisesRegex(GoldenError, "보류 시험셋 묶음이 아닙니다"):
            cli.run_estimate(self.config("luna", holdout=True))


# ---------------------------------------------------------------- 실행 상한


class RunCapTest(RegistryCase):
    def test_two_runs_per_suite_model_golden_version(self):
        first = self.run_live(self.config("luna", name="r1"), luna_adapter())
        second = self.run_live(self.config("luna", name="r2"), luna_adapter())
        self.assertEqual((first["runNo"], first["primary"]), (1, True))
        self.assertEqual((second["runNo"], second["primary"]), (2, False))
        with self.assertRaises(RunCapReached):
            self.run_live(self.config("luna", name="r3"), luna_adapter())
        self.assertFalse((self.tmp / "out/r3.json").exists())
        # 다른 모델은 따로 센다
        self.assertEqual(self.run_live(self.config("jev", name="j1"), jev_adapter())["runNo"], 1)
        self.assertEqual([(r["model"], r["runNo"]) for r in self.ledger_runs()],
                         [("gpt-5.6-luna", 1), ("gpt-5.6-luna", 2), ("jev-latest", 1)])

    def test_failed_run_still_counts_and_output_not_overwritten(self):
        class Boom:
            variant = LunaJudge.variant

            async def judge(self, turn):
                raise RuntimeError("중간 실패")

        with self.assertRaises(RuntimeError):
            self.run_live(self.config("luna", name="boom"), Boom())
        self.assertEqual(len(self.ledger_runs()), 1)
        self.run_live(self.config("luna", name="ok"), luna_adapter())
        with self.assertRaisesRegex(GoldenError, "이미 있습니다"):
            self.run_live(self.config("jev", name="ok"), jev_adapter())

    def test_live_needs_criteria_system_version_and_clean_registry(self):
        with self.assertRaisesRegex(GoldenError, "criteria"):
            self.run_live(self.config("luna", criteria=None), luna_adapter())
        with self.assertRaisesRegex(GoldenError, "system-version"):
            self.run_live(self.config("luna", system_version=None), luna_adapter())
        (self.root / "golden/suites/rules.json").write_text((self.root / "golden/suites/rules.json").read_text() + " ")
        with self.assertRaisesRegex(GoldenError, "커밋하지 않은 변경"):
            self.run_live(self.config("luna"), luna_adapter())
        self.assertEqual(self.ledger_runs(), [])


# ---------------------------------------------------------------- 추정


class EstimateTest(RegistryCase):
    def test_estimate_counts_calls_without_calling_or_logging(self):
        with mock.patch.object(LunaJudge, "judge", side_effect=AssertionError("모델을 부름")), \
                mock.patch.object(JevJudge, "judge", side_effect=AssertionError("모델을 부름")):
            luna = cli.run_estimate(self.config("luna"))
            jev = cli.run_estimate(self.config("jev"))
        self.assertEqual((luna["turns"], luna["calls"], luna["skippedTurns"], luna["callsUpperWithRetries"]), (6, 5, 1, 10))
        self.assertEqual(jev["calls"], 5)
        self.assertGreater(luna["estimatedInputTokens"], 0)
        self.assertGreater(luna["upperUsd"], luna["estimatedUsd"])
        self.assertEqual(luna["payloadCheck"], {"judgedTurns": 5, "verified": 5, "unverified": 0})
        unpriced = cli.run_estimate(self.config("jev", prices=Prices(None, None, None)))
        self.assertIsNone(unpriced["estimatedUsd"])
        self.assertFalse((self.root / "evals/run-ledger.jsonl").exists())

    def test_cli_requires_estimate_or_live(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(["run", "--registry", str(self.root), "--golden-version", GOLDEN_VERSION,
                             "--suite", SUITE_A, "--judge", "luna", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["calls"], 5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["run", "--registry", str(self.root), "--golden-version", GOLDEN_VERSION, "--suite", SUITE_A, "--judge", "luna"])


# ---------------------------------------------------------------- 채점


def label(decision, keys=(), source="model-panel"):
    return Label(decision=decision, keys=tuple(keys), source=source, status="confirmed")


SERVING = {"k.serve": "SERVING", "k.shadow": "SHADOW", "k.other": "SHADOW"}


class ScoringMathTest(unittest.TestCase):
    def test_zero_error_upper_bound(self):
        for n in (1, 10, 93, 132):
            self.assertAlmostEqual(zero_error_upper(n), 1 - 0.05 ** (1 / n), places=12)
            self.assertAlmostEqual(clopper_pearson_upper(0, n), zero_error_upper(n), places=9)
        bound = rate_with_bound(0, 132)
        self.assertEqual(bound["boundMethod"], "zero-error-exact")
        self.assertAlmostEqual(bound["upper95"], 0.022443, places=5)

    def test_clopper_pearson_upper(self):
        # 단측 95% Clopper-Pearson 상한 = Beta(0.95; k+1, n-k). 기대값은 분수 계산으로 P(X<=k) 가 0.05 를 지나는 점을 따로 확인함
        self.assertAlmostEqual(clopper_pearson_upper(1, 10), 0.39416, places=4)
        self.assertAlmostEqual(clopper_pearson_upper(5, 20), 0.45558, places=4)
        upper = clopper_pearson_upper(3, 132)
        self.assertAlmostEqual(binom_cdf(3, 132, upper), 0.05, places=9)
        self.assertEqual(clopper_pearson_upper(10, 10), 1.0)
        bound = rate_with_bound(1, 10)
        self.assertEqual((bound["count"], bound["n"], bound["rate"], bound["boundMethod"]), (1, 10, 0.1, "clopper-pearson"))
        self.assertIsNone(rate_with_bound(0, 0)["upper95"])

    def test_mcnemar_exact(self):
        self.assertIsNone(mcnemar_exact_p(0, 0))
        self.assertAlmostEqual(mcnemar_exact_p(0, 5), 0.0625)
        self.assertAlmostEqual(mcnemar_exact_p(3, 3), 1.0)

    def test_three_tier_rule_and_separate_denominators(self):
        presented = ["k.serve", "k.shadow", "k.other"]
        rows = [
            # 정답 키가 후보 밖: 연결하지 않으면 판별기 오류 아님
            score_turn(("m", 1), "A", label("CONNECT", ["k.missing"]), presented, "SEPARATE", None, False, SERVING),
            # 정답 키가 후보 밖인데 다른 키로 연결: 판별기 오류이자 잘못된 서빙
            score_turn(("m", 2), "A", label("CONNECT", ["k.missing"]), presented, "CONNECT", "k.serve", False, SERVING),
            # 맞는 서빙
            score_turn(("c", 1), "A", label("CONNECT", ["k.serve"]), presented, "CONNECT", "k.serve", False, SERVING),
            # 놓친 연결
            score_turn(("c", 2), "A", label("CONNECT", ["k.shadow"]), presented, "UNCLASSIFIED", None, False, SERVING),
            # 비연결 정답에 SHADOW 로 연결: 잘못된 연결이지만 서빙은 아님
            score_turn(("s", 1), "A", label("SEPARATE", source="human"), presented, "CONNECT", "k.other", False, SERVING),
            # 호출 실패는 UNCLASSIFIED 로 채점
            score_turn(("u", 1), "A", label("UNCLASSIFIED", source="audited"), presented, "UNCLASSIFIED", None, True, SERVING),
            # 판별하지 않은 턴은 분모에서 뺌
            score_turn(("k", 1), "A", label("SEPARATE"), presented, None, None, False, SERVING),
        ]
        self.assertTrue(rows[0].judge_correct)
        self.assertFalse(rows[0].semantic_correct)
        self.assertFalse(rows[1].judge_correct)
        self.assertTrue(rows[1].wrong_serve)
        result = metrics(rows)
        self.assertEqual((result["turns"], result["scored"], result["skipped"], result["errors"]), (7, 6, 1, 1))
        self.assertEqual((result["candidateMiss"]["count"], result["candidateMiss"]["n"]), (2, 4))
        self.assertEqual((result["missedConnect"]["count"], result["missedConnect"]["n"]), (1, 2))
        self.assertEqual((result["missedConnectSemantic"]["count"], result["missedConnectSemantic"]["n"]), (2, 4))
        self.assertEqual((result["wrongConnect"]["count"], result["wrongConnect"]["n"]), (1, 2))
        self.assertEqual((result["wrongServe"]["count"], result["wrongServe"]["n"]), (1, 2))
        self.assertEqual(result["wrongServe"]["boundMethod"], "clopper-pearson")
        self.assertEqual((result["wrongServePerTurn"]["count"], result["wrongServePerTurn"]["n"]), (1, 6))
        self.assertEqual((result["judgeAccuracy"]["count"], result["semanticAccuracy"]["count"]), (3, 2))
        self.assertEqual(result["decisionAccuracy"]["count"], 3)
        self.assertEqual(result["unclassified"]["count"], 2)
        self.assertEqual(result["labelSourceCounts"], {"audited": 1, "human": 1, "model-panel": 5})

        summary = summarize(rows)
        self.assertEqual(set(summary["byLayer"]["A"]["byLabelSource"]), {"audited", "human", "model-panel"})
        baseline = summary["baselines"]["neverConnect"]["A"]["all"]
        self.assertEqual((baseline["wrongConnect"]["count"], baseline["served"]["count"]), (0, 0))
        self.assertEqual(baseline["wrongServe"]["n"], 0)
        self.assertEqual(baseline["missedConnectSemantic"]["count"], 4)

    def test_paired_comparison(self):
        presented = ["k.serve", "k.shadow"]
        base = {("a", 1): score_turn(("a", 1), "A", label("SEPARATE"), presented, "CONNECT", "k.serve", False, SERVING),
                ("b", 1): score_turn(("b", 1), "A", label("CONNECT", ["k.serve"]), presented, "CONNECT", "k.serve", False, SERVING)}
        cand = {("a", 1): score_turn(("a", 1), "A", label("SEPARATE"), presented, "SEPARATE", None, False, SERVING),
                ("b", 1): score_turn(("b", 1), "A", label("CONNECT", ["k.serve"]), presented, "UNCLASSIFIED", None, False, SERVING)}
        paired = compare(base, cand)["A"]["all"]
        self.assertEqual(paired["pairs"], 2)
        self.assertEqual((paired["wrongServe"]["base"], paired["wrongServe"]["candidate"]), (1, 0))
        self.assertEqual((paired["judgeCorrect"]["baseOnly"], paired["judgeCorrect"]["candidateOnly"]), (1, 1))
        self.assertEqual(paired["decisionChanged"], 2)


class RunScoringTest(RegistryCase):
    def test_run_summary_by_layer_and_label_source(self):
        luna = self.run_live(self.config("luna", name="luna"), luna_adapter())
        jev = self.run_live(self.config("jev", name="jev"), jev_adapter())
        luna_a = luna["summary"]["scoring"]["byLayer"]["A"]["all"]
        self.assertEqual(luna_a["labelSourceCounts"], {"audited": 1, "human": 1, "model-panel": 4})
        self.assertEqual((luna_a["wrongServe"]["count"], luna_a["wrongServe"]["n"]), (1, 2))
        self.assertEqual((luna_a["wrongConnect"]["count"], luna_a["wrongConnect"]["n"]), (1, 2))
        self.assertEqual((luna_a["missedConnect"]["count"], luna_a["missedConnect"]["n"]), (0, 2))
        self.assertEqual(luna_a["candidateMiss"]["count"], 1)
        self.assertEqual(luna_a["skipped"], 1)
        jev_a = jev["summary"]["scoring"]["byLayer"]["A"]["all"]
        self.assertEqual((jev_a["wrongServe"]["count"], jev_a["wrongServe"]["n"]), (0, 1))
        self.assertEqual(jev_a["wrongServe"]["boundMethod"], "zero-error-exact")
        self.assertEqual(jev_a["judgeAccuracy"]["count"], 5)
        self.assertIn("model-panel", luna["summary"]["scoring"]["byLayer"]["A"]["byLabelSource"])
        # 다시 채점해도 같다
        _, rescored, _ = cli.rescore(self.root, self.tmp / "out/luna.json")
        self.assertEqual(summarize(list(rescored.values())), luna["summary"]["scoring"])
        # 결과 칸
        self.assertEqual(luna["models"]["judge"]["model"], "gpt-5.6-luna")
        self.assertEqual(luna["models"]["judge"]["promptVersion"], "question-grouping-v7-2")
        self.assertEqual(jev["models"]["judge"]["model"], "jev-latest")
        self.assertEqual(luna["goldenVersion"], GOLDEN_VERSION)
        self.assertRegex(luna["goldenCommit"], r"^[0-9a-f]{40}$")
        outcomes = {r["itemId"]: r["outcome"] for r in luna["results"]}
        self.assertEqual(outcomes["A-t7"], "SKIPPED")
        self.assertAlmostEqual(luna["cost"]["upperUsd"], 5 * (5000 * 0.2 + 300 * 1.2) / 1e6)
        self.assertEqual(luna["summary"]["latency"]["n"], 5)
        self.assertEqual(luna["summary"]["inputs"]["payloadCheck"], {"judgedTurns": 5, "verified": 5, "unverified": 0, "mode": "strict"})

    def test_luna_api_failure_is_error_and_invalid_output_is_unclassified(self):
        script = dict(LUNA_SCRIPT, **{"A-t1": ("CONNECT", "not-presented")})
        result = self.run_live(self.config("luna"), luna_adapter(FakeOpenAI(script=script, fail={"A-t2"})))
        by_item = {r["itemId"]: r for r in result["results"]}
        self.assertEqual(by_item["A-t2"]["outcome"], "ERROR")
        self.assertEqual(by_item["A-t2"]["actual"]["judgeDecision"], "UNCLASSIFIED")
        self.assertEqual(by_item["A-t1"]["outcome"], "SCORED")
        self.assertEqual(by_item["A-t1"]["actual"]["judgeDecision"], "UNCLASSIFIED")
        raw = [json.loads(l) for l in (self.tmp / "out/out.raw.jsonl").read_text().splitlines()]
        self.assertEqual(next(l for l in raw if l["itemId"] == "A-t1")["invalidReason"], "UNKNOWN_SUBPROBLEM_KEY")

    def test_jev_retries_transient_once(self):
        transport = FakeJevTransport(statuses=[503])
        result = self.run_live(self.config("jev", concurrency=1), jev_adapter(transport))
        self.assertEqual(len(transport.bodies), 6)
        self.assertEqual(result["summary"]["usage"]["calls"], 6)


# ---------------------------------------------------------------- 결과 스키마


REAL = real_registry()


@unittest.skipIf(REAL is None, "평가 등록부(RIIDO_EVALS_ROOT 또는 evaluation/registry 서브모듈)가 없어 실제 스키마 검사를 건너뜀")
class SchemaConformanceTest(RegistryCase):
    schema_from = REAL

    def setUp(self):
        schema = json.loads((REAL / "schemas/eval-result.schema.json").read_text(encoding="utf-8"))
        if SUITE_A not in schema["properties"]["suite"].get("enum", []) or "runNo" not in schema["properties"]:
            self.skipTest("등록부 결과 스키마가 판별 묶음 이름이나 runNo 를 모름 (riido-evals 스키마 갱신 전)")
        super().setUp()

    def test_results_conform_to_registry_schema(self):
        golden = golden_mod.open_golden(self.root, GOLDEN_VERSION)
        checker = cli.load_schema_checker(golden)
        for judge, adapter in (("luna", luna_adapter()), ("jev", jev_adapter())):
            result = self.run_live(self.config(judge, name=judge), adapter)
            self.assertEqual(checker(result), [])
        broken = dict(result, suite="judge-unknown")
        self.assertTrue(checker(broken))


@unittest.skipIf(REAL is None or not (REAL / "golden/frozen").exists(), "실제 골든셋이 없어 건너뜀")
class RealRegistryTest(unittest.TestCase):
    def test_judge_official_a_estimate_and_b_refusal(self):
        config = cli.RunConfig(suite=SUITE_A, judge="luna", golden_version="1.0.0", registry=REAL)
        estimate = cli.run_estimate(config)
        self.assertEqual((estimate["turns"], estimate["calls"]), (132, 132))
        with self.assertRaises(HoldoutRefused):
            cli.run_estimate(cli.RunConfig(suite=SUITE_B, judge="luna", golden_version="1.0.0", registry=REAL))


if __name__ == "__main__":
    unittest.main()
