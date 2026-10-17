"""CacheDecisionService must only read snapshots and share production gate policy."""

import unittest
import uuid
from dataclasses import FrozenInstanceError

from app.database.models import (
    CacheAttemptOutcome,
    ClassificationDecision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.question_grouping.cache_decision import CacheDecisionService
from app.question_grouping.constants import EXACT_SOURCE_SUBPROBLEM_VERSION_STALE
from app.question_grouping.models import (
    ExactQuestionLogMatch,
    GateCanonicalAnswer,
    GateInputs,
    GateSubproblemState,
    IndexScope,
    JudgeCall,
    JudgeFailureKind,
    ModelCallTrace,
    PresentedSubproblem,
    SubproblemCatalog,
    TurnJudgment,
)
from app.question_grouping.attribution import subproblem_attribution
from app.retrieval.models import HybridSearchCall


SUBPROBLEM_ID = uuid.UUID(int=1)
CANONICAL_ID = uuid.UUID(int=2)
GROUP_ID = uuid.UUID(int=3)


class ReadOnlyReader:
    def __init__(self, *, scope=None, inputs=None):
        self.scope = scope or IndexScope(4, 9, 2, 7)
        self.inputs = inputs or GateInputs()
        self.calls = []
        self.writes = 0
        self.catalog = SubproblemCatalog(())

    async def load_index_scope(self, index_version_id):
        self.calls.append(("scope", index_version_id))
        return self.scope

    async def load_gate_inputs(self, subproblem_id, scope):
        self.calls.append(("inputs", subproblem_id, scope.index_version_id))
        return self.inputs

    async def load_subproblem_catalog(self, scope):
        self.calls.append(("catalog", scope.index_version_id))
        return self.catalog


class EmptyOutlineReader:
    async def load_outlines(self, document_version_ids, *, chunking_config_id):
        return {}


def connected_judgment(*, version=2):
    presented = PresentedSubproblem(
        key="billing.cancel",
        subproblem_id=SUBPROBLEM_ID,
        subproblem_version=version,
        problem_group_id=GROUP_ID,
        document_source_id=12,
        document_key="billing",
        canonical_answer_id=CANONICAL_ID,
        similarity=1.0,
        retrieval_rank=1,
        presented_order=1,
        inclusion_count=1,
        exclusion_count=0,
    )
    return TurnJudgment(
        decision=ClassificationDecision.CONNECT,
        attribution=subproblem_attribution(GROUP_ID),
        subproblem=presented,
    )


class CacheDecisionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_preparation_and_judging_share_read_only_logic(self):
        reader = ReadOnlyReader()
        decisions = CacheDecisionService(reader, EmptyOutlineReader())
        prepared = await decisions.prepare_candidates(
            resolved_query="question",
            embedding=(0.0,),
            index_version_id=4,
            document_group_id=9,
            retrieval=HybridSearchCall(),
            seed="seed",
        )
        self.assertTrue(prepared.ready)
        self.assertEqual("question", prepared.presentation.payload["question"])
        self.assertEqual(
            [("scope", 4), ("catalog", 4)], reader.calls
        )

        class Client:
            provider = "test"

            async def judge(self, payload, *, before_model_call, **kwargs):
                await before_model_call("test", "model", "prompt")
                return JudgeCall(
                    trace=ModelCallTrace(
                        provider="test", model_name="model", succeeded=True, latency_ms=1
                    ),
                    output_text="",
                )

        checkpoints = []

        async def checkpoint(provider, model, prompt):
            checkpoints.append((provider, model, prompt))

        judged = await decisions.judge(
            client=Client(),
            presentation=prepared.presentation,
            model="model",
            prompt_version="prompt",
            before_model_call=checkpoint,
        )
        self.assertEqual([("test", "model", "prompt")], checkpoints)
        self.assertEqual(JudgeFailureKind.INVALID_OUTPUT, judged.judgment.failure.kind)
        self.assertEqual(0, reader.writes)

    async def test_gate_reads_fresh_state_and_returns_immutable_id_free_result(self):
        current = GateInputs(
            subproblem=GateSubproblemState(
                subproblem_id=SUBPROBLEM_ID,
                status=QuestionSubproblemStatus.APPROVED,
                serving_state=QuestionSubproblemServingState.SERVING,
                current_version=3,
            ),
            canonical_answer=GateCanonicalAnswer(
                canonical_answer_id=CANONICAL_ID,
                subproblem_version=3,
                content_markdown="answer",
            ),
        )
        reader = ReadOnlyReader(inputs=current)
        decisions = CacheDecisionService(reader)

        result = await decisions.decide_gate(
            connected_judgment(version=2),
            scope=reader.scope,
            semantic_cache_enabled=True,
        )

        self.assertEqual(CacheAttemptOutcome.REJECTED, result.gate.outcome)
        self.assertEqual(("SUBPROBLEM_VERSION_MISMATCH", "CANONICAL_CITATION_MISSING"), result.gate.rejection_reasons)
        self.assertEqual(current, result.inputs)
        self.assertEqual([("inputs", SUBPROBLEM_ID, 4)], reader.calls)
        self.assertEqual(0, reader.writes)
        self.assertFalse(hasattr(result, "classification_id"))
        self.assertFalse(hasattr(result, "cache_attempt_id"))
        with self.assertRaises(FrozenInstanceError):
            result.gate = result.gate
        with self.assertRaises(TypeError):
            result.inputs.contexts_by_source_id[1] = None

    async def test_exact_source_staleness_is_fallthrough_not_gate_rejection(self):
        current = GateInputs(
            subproblem=GateSubproblemState(
                subproblem_id=SUBPROBLEM_ID,
                status=QuestionSubproblemStatus.APPROVED,
                serving_state=QuestionSubproblemServingState.SERVING,
                current_version=3,
            )
        )
        reader = ReadOnlyReader(inputs=current)
        match = ExactQuestionLogMatch(
            subproblem_id=SUBPROBLEM_ID,
            key="billing.cancel",
            problem_group_id=GROUP_ID,
            current_version=3,
            document_source_id=12,
            document_key="billing",
            normalized_question="cancel?",
            source_rag_run_id=uuid.UUID(int=4),
            classification_id=31,
            matched_count=1,
            source_subproblem_version=2,
        )

        inspection = await CacheDecisionService(reader).inspect_exact_source(
            match, index_version_id=4, document_group_id=9
        )

        self.assertEqual((EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,), inspection.fallthrough_reasons)
        self.assertIsNone(inspection.data_error)
        self.assertEqual([("scope", 4), ("inputs", SUBPROBLEM_ID, 4)], reader.calls)
        self.assertEqual(0, reader.writes)

