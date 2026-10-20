import json
import unittest
import uuid
from types import SimpleNamespace
from typing import Any, Mapping, Optional
from unittest.mock import AsyncMock, patch

from app.database.models import ClassificationDecision
from app.core.stage_settings import StageSettings
from app.core.model_trace import canonical_request_sha256
from app.question_grouping.judge_client import QuestionJudgeClient
from app.question_grouping.decision import build_turn_judgment
from app.question_grouping.jev_judge_client import (
    JEV_MODEL,
    JEV_NONE_KEY,
    JEV_PROMPT_VERSION,
    JEV_CONTEXT_PROMPT,
    JEV_QUESTION_NAME,
    JEV_THRESHOLD,
    JevJudgeClient,
    build_jev_request,
    build_jev_request_legacy,
)
from app.question_grouping.models import JudgePresentation, PresentedSubproblem


def _payload() -> dict[str, Any]:
    return {
        "policyVersion": "question-grouping-v7-2",
        "question": "어떻게 설정하나요?",
        "contextSummary": None,
        "candidateCount": 2,
        "candidates": [
            {
                "rank": 1,
                "group": {"id": "settings", "name": "설정"},
                "subproblem": {
                    "id": "settings.install",
                    "name": "설정 방법",
                    "inclusionCriteria": {"I1": "설정 위치를 묻는다."},
                    "exclusionCriteria": {"E1": "설치 오류는 제외한다."},
                    "canonicalAnswer": {
                        "contentMarkdown": "설정에서 바꿉니다.",
                        "applicabilityRules": ["관리자만 변경 가능"],
                    },
                }
            },
            {
                "rank": 2,
                "group": {"id": "billing"},
                "subproblem": {
                    "id": "billing.plan",
                    "name": "요금제 변경",
                    "inclusionCriteria": {"I1": "요금제를 바꾼다."},
                    "exclusionCriteria": {"E1": "환불 문의는 제외한다."},
                },
            },
        ],
        "documentCandidateCount": 1,
        "documentCandidates": [
            {"id": "D1", "title": "설정 안내", "parentPath": "관리/설정", "headings": ["변경"]}
        ],
    }


def _presentation() -> JudgePresentation:
    subproblem = PresentedSubproblem(
        key="settings.install",
        subproblem_id=uuid.uuid4(),
        subproblem_version=1,
        problem_group_id=uuid.uuid4(),
        document_source_id=1,
        document_key="settings",
        canonical_answer_id=None,
        similarity=0.9,
        retrieval_rank=1,
        presented_order=1,
        inclusion_count=1,
        exclusion_count=1,
    )
    return JudgePresentation(
        payload=_payload(),
        subproblems=(subproblem,),
        documents=(),
        subproblem_seed="seed:s",
        document_seed="seed:d",
    )


class FakeTransport:
    def __init__(self, *outcomes: Mapping[str, Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests = []

    async def __call__(self, request: dict[str, Any]) -> Mapping[str, Any]:
        self.requests.append(request)
        return self.outcomes.pop(0)


def _success(choice: str, confidence: float) -> dict[str, Any]:
    return {
        "ok": True,
        "httpStatus": 200,
        "body": {
            "model": JEV_MODEL,
            "answers": {
                JEV_QUESTION_NAME: {"choice": choice, "confidence": confidence}
            },
            "usage": {"input_tokens": 77, "output_tokens": 4},
        },
    }


class JevRequestTest(unittest.TestCase):
    def test_request_matches_choice_set_contract(self) -> None:
        request = build_jev_request(_payload())
        question = request["questions"][JEV_QUESTION_NAME]

        self.assertEqual(JEV_MODEL, request["model"])
        self.assertEqual(_payload(), request["state"]["classifierInput"])
        self.assertEqual("choice", question["type"])
        self.assertEqual(
            "설정 방법 (state.classifierInput.candidates ID: settings.install)",
            question["criteria"]["settings.install"],
        )
        self.assertEqual(
            ["settings.install", "billing.plan", JEV_NONE_KEY],
            list(question["criteria"]),
        )
        self.assertNotIn("정본", question["criteria"]["settings.install"])

    def test_context_preserves_nested_payload_and_legacy_builder_reproduces_p(self) -> None:
        payload = _payload()
        request = build_jev_request(payload)
        request["state"]["classifierInput"]["candidates"][0]["subproblem"]["name"] = "edited"
        self.assertEqual("설정 방법", payload["candidates"][0]["subproblem"]["name"])
        legacy = build_jev_request_legacy(payload)
        self.assertEqual({"질문": payload["question"]}, legacy["state"])
        self.assertIn("정본(사용자에게 그대로 전송될 답변)", legacy["questions"][JEV_QUESTION_NAME]["criteria"]["settings.install"])

    def test_rejects_invalid_ids_and_counts(self) -> None:
        for edit in (
            lambda p: p["candidates"][0]["subproblem"].update(id=""),
            lambda p: p["candidates"][0]["subproblem"].update(id=JEV_NONE_KEY),
            lambda p: p["candidates"][1]["subproblem"].update(id="settings.install"),
            lambda p: p.update(candidateCount=3),
        ):
            payload = _payload()
            edit(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    build_jev_request(payload)

    def test_zero_candidates_yields_only_none_choice(self) -> None:
        payload = _payload()
        payload["candidates"] = []
        payload["candidateCount"] = 0
        request = build_jev_request(payload)
        self.assertEqual([JEV_NONE_KEY], list(request["questions"][JEV_QUESTION_NAME]["criteria"]))

    def test_prompt_identity_is_registered(self) -> None:
        from app.prompts.fingerprints import load_fingerprint_table, sha256_text

        self.assertEqual(
            sha256_text(JEV_CONTEXT_PROMPT),
            load_fingerprint_table()[f"judge@{JEV_PROMPT_VERSION}"],
        )


class JevJudgeClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_output_is_accepted_by_production_normalizer(self) -> None:
        transport = FakeTransport(_success("settings.install", JEV_THRESHOLD))
        hook_calls = []

        async def before_model_call(
            provider: str, model_name: str, prompt_version: Optional[str]
        ) -> None:
            hook_calls.append((provider, model_name, prompt_version))

        call = await JevJudgeClient(transport).judge(
            _payload(), before_model_call=before_model_call
        )
        judgment = build_turn_judgment(call, _presentation())

        self.assertEqual(
            [("typesafe", JEV_MODEL, JEV_PROMPT_VERSION)], hook_calls
        )
        self.assertEqual(
            "settings.install", json.loads(call.output_text)["subproblemId"]
        )
        self.assertEqual(ClassificationDecision.CONNECT, judgment.decision)
        self.assertEqual("settings.install", judgment.subproblem.key)
        self.assertEqual((77, 4), (call.trace.input_tokens, call.trace.output_tokens))

    async def test_none_and_below_threshold_follow_evaluation_mapping(self) -> None:
        cases = (
            (_success(JEV_NONE_KEY, JEV_THRESHOLD), ClassificationDecision.SEPARATE),
            (
                _success("settings.install", JEV_THRESHOLD - 0.01),
                ClassificationDecision.UNCLASSIFIED,
            ),
        )
        for outcome, expected in cases:
            with self.subTest(expected=expected):
                call = await JevJudgeClient(FakeTransport(outcome)).judge(_payload())
                judgment = build_turn_judgment(call, _presentation())
                self.assertEqual(expected, judgment.decision)

    async def test_unknown_choice_uses_r7_unknown_key_normalization(self) -> None:
        call = await JevJudgeClient(
            FakeTransport(_success("not-presented", 0.99))
        ).judge(_payload())

        judgment = build_turn_judgment(call, _presentation())

        self.assertEqual(ClassificationDecision.UNCLASSIFIED, judgment.decision)
        self.assertEqual("UNKNOWN_SUBPROBLEM_KEY", judgment.normalized.invalid_reason)

    async def test_rejects_openai_or_wrong_jev_stage_before_hook_and_transport(self) -> None:
        transport = FakeTransport(_success("settings.install", 0.99))
        stage = StageSettings(
            model_name="jev-profile-model",
            prompt_version="jev-profile-prompt",
            instructions="stage metadata is not Jev's choice rules",
        )
        hook_calls = []

        async def before_model_call(
            provider: str, model_name: str, prompt_version: Optional[str]
        ) -> None:
            hook_calls.append((provider, model_name, prompt_version))

        call = await JevJudgeClient(transport).judge(
            _payload(), before_model_call=before_model_call, stage=stage
        )

        self.assertFalse(call.succeeded)
        self.assertEqual([], transport.requests)
        self.assertEqual([], hook_calls)

    async def test_rejects_unsupported_stage_params(self) -> None:
        transport = FakeTransport(_success("settings.install", 0.99))
        stage = StageSettings(
            model_name="jev-profile-model",
            prompt_version=JEV_PROMPT_VERSION,
            instructions=JEV_CONTEXT_PROMPT,
            params={"unknown": "value"},
        )
        call = await JevJudgeClient(transport).judge(_payload(), stage=stage)
        self.assertFalse(call.succeeded)
        self.assertEqual(0, call.trace.retry_count)
        self.assertEqual([], transport.requests)

    async def test_invalid_common_input_fails_before_hook_and_transport(self) -> None:
        transport = FakeTransport(_success("settings.install", 0.99))
        hook = AsyncMock()
        payload = _payload()
        payload["candidates"][0]["subproblem"]["id"] = JEV_NONE_KEY
        call = await JevJudgeClient(transport).judge(payload, before_model_call=hook)
        self.assertFalse(call.succeeded)
        self.assertEqual([], transport.requests)
        hook.assert_not_awaited()
        self.assertGreaterEqual(call.trace.retry_count, 0)

    async def test_response_normalization_uses_the_frozen_request_choices(self) -> None:
        payload = _payload()
        transport = FakeTransport(_success("settings.install", 0.99))

        async def mutate_original(*_args: Any) -> None:
            payload["candidates"].clear()

        call = await JevJudgeClient(transport).judge(
            payload, before_model_call=mutate_original
        )
        self.assertEqual("settings.install", json.loads(call.output_text)["subproblemId"])

    async def test_context_overflow_status_is_safe_and_not_retried(self) -> None:
        for status in (400, 413):
            transport = FakeTransport({"ok": False, "httpStatus": status})
            with self.subTest(status=status), patch(
                "app.question_grouping.jev_judge_client.asyncio.sleep", new=AsyncMock()
            ) as sleep:
                call = await JevJudgeClient(transport, retry_delay=0).judge(_payload())
            self.assertFalse(call.succeeded)
            self.assertEqual(1, len(transport.requests))
            self.assertIn(str(status), call.failure.safe_message)
            self.assertEqual(_payload(), transport.requests[0]["state"]["classifierInput"])
            sleep.assert_not_awaited()

    async def test_accepts_matching_explicit_jev_stage(self) -> None:
        transport = FakeTransport(_success("settings.install", 0.99))
        stage = StageSettings(
            model_name="jev-profile-model",
            prompt_version=JEV_PROMPT_VERSION,
            instructions=JEV_CONTEXT_PROMPT,
        )
        call = await JevJudgeClient(transport).judge(_payload(), stage=stage)
        self.assertTrue(call.succeeded)
        self.assertEqual("jev-profile-model", transport.requests[0]["model"])
        self.assertEqual(JEV_PROMPT_VERSION, call.trace.prompt_version)
        self.assertEqual("jev-profile-model", call.trace.model_name)
        self.assertIsNotNone(call.trace.request_evidence)

    async def test_retries_only_transient_http_statuses_and_records_count(self) -> None:
        transport = FakeTransport(
            {"ok": False, "httpStatus": 503},
            _success("settings.install", 0.99),
        )
        with patch(
            "app.question_grouping.jev_judge_client.asyncio.sleep", new=AsyncMock()
        ) as sleep:
            call = await JevJudgeClient(transport, retry_delay=0).judge(_payload())

        self.assertEqual(2, len(transport.requests))
        self.assertEqual(1, call.trace.retry_count)
        self.assertEqual(transport.requests[0], transport.requests[1])
        sleep.assert_awaited_once_with(0)

    async def test_retry_transport_mutation_is_isolated_from_next_request(self) -> None:
        class MutatingTransport(FakeTransport):
            async def __call__(self, request: dict[str, Any]) -> Mapping[str, Any]:
                self.requests.append(request)
                if len(self.requests) == 1:
                    request["state"]["classifierInput"]["candidates"].clear()
                    return {"ok": False, "httpStatus": 503}
                return _success("settings.install", 0.99)

        transport = MutatingTransport()
        with patch("app.question_grouping.jev_judge_client.asyncio.sleep", new=AsyncMock()):
            call = await JevJudgeClient(transport, retry_delay=0).judge(_payload())
        self.assertTrue(call.succeeded)
        self.assertEqual(2, len(transport.requests[1]["state"]["classifierInput"]["candidates"]))
        self.assertEqual(2, len(_payload()["candidates"]))

    async def test_evidence_matches_common_input_across_providers(self) -> None:
        payload = _payload()
        transport = FakeTransport(_success("settings.install", 0.99))
        jev_call = await JevJudgeClient(transport).judge(payload)
        response = SimpleNamespace(
            output_text='{"decision":"CONNECT","subproblemId":"settings.install"}',
            status="completed",
            usage=None,
            model="gpt-5.6-luna",
            incomplete_details=None,
        )
        openai = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response)))
        luna_call = await QuestionJudgeClient(client=openai).judge(payload)

        jev_evidence = jev_call.trace.request_evidence
        luna_evidence = luna_call.trace.request_evidence
        self.assertEqual(canonical_request_sha256(payload), jev_evidence.common_input_sha256)
        self.assertEqual(jev_evidence.common_input_sha256, luna_evidence.common_input_sha256)
        self.assertNotEqual(jev_evidence.request_sha256, luna_evidence.request_sha256)
        self.assertEqual("gpt-5.6-luna", luna_evidence.response_model_name)
        self.assertEqual(JEV_MODEL, jev_evidence.response_model_name)

    async def test_untrusted_response_model_is_omitted_from_trace(self) -> None:
        outcome = _success("settings.install", 0.99)
        outcome["body"]["model"] = "secret-response-text\n"
        call = await JevJudgeClient(FakeTransport(outcome)).judge(_payload())
        self.assertIsNone(call.trace.request_evidence.response_model_name)

    async def test_http_failure_becomes_safe_judge_call_failure(self) -> None:
        call = await JevJudgeClient(
            FakeTransport({"ok": False, "httpStatus": 401, "error": "secret"})
        ).judge(_payload())

        self.assertFalse(call.succeeded)
        self.assertEqual("API_ERROR", call.failure.kind.value)
        self.assertNotIn("secret", call.failure.safe_message)
        self.assertIn("401", call.failure.safe_message)
        self.assertIsNotNone(call.trace.request_evidence)


if __name__ == "__main__":
    unittest.main()
