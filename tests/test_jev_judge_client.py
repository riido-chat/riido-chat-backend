import json
import unittest
import uuid
from typing import Any, Mapping, Optional
from unittest.mock import AsyncMock, patch

from app.database.models import ClassificationDecision
from app.core.stage_settings import StageSettings
from app.question_grouping.decision import build_turn_judgment
from app.question_grouping.jev_judge_client import (
    JEV_MODEL,
    JEV_NONE_KEY,
    JEV_PROMPT_VERSION,
    JEV_QUESTION_NAME,
    JEV_THRESHOLD,
    JevJudgeClient,
    build_jev_request,
)
from app.question_grouping.models import JudgePresentation, PresentedSubproblem


def _payload() -> dict[str, Any]:
    return {
        "question": "어떻게 설정하나요?",
        "candidates": [
            {
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
            }
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
        self.assertEqual({"질문": "어떻게 설정하나요?"}, request["state"])
        self.assertEqual("choice", question["type"])
        self.assertEqual(
            "설정 방법 / 포함: 설정 위치를 묻는다. / 제외: 설치 오류는 제외한다."
            " / 정본(사용자에게 그대로 전송될 답변): 설정에서 바꿉니다."
            " / 정본이 다루지 않는 것: 관리자만 변경 가능",
            question["criteria"]["settings.install"],
        )
        self.assertIn(JEV_NONE_KEY, question["criteria"])


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

    async def test_honors_per_turn_stage_model_and_prompt_identity(self) -> None:
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

        self.assertEqual("jev-profile-model", transport.requests[0]["model"])
        self.assertEqual(
            [("typesafe", "jev-profile-model", "jev-profile-prompt")], hook_calls
        )
        self.assertEqual("jev-profile-model", call.trace.model_name)
        self.assertEqual("jev-profile-prompt", call.trace.prompt_version)

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
        sleep.assert_awaited_once_with(0)

    async def test_http_failure_becomes_safe_judge_call_failure(self) -> None:
        call = await JevJudgeClient(
            FakeTransport({"ok": False, "httpStatus": 401, "error": "secret"})
        ).judge(_payload())

        self.assertFalse(call.succeeded)
        self.assertEqual("API_ERROR", call.failure.kind.value)
        self.assertNotIn("secret", call.failure.safe_message)
        self.assertIn("401", call.failure.safe_message)


if __name__ == "__main__":
    unittest.main()
