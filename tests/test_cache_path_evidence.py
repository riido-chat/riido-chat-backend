"""Validation and privacy guarantees for cache-path runtime evidence."""

import json
import unittest
from pathlib import Path

from app.core.model_trace import ModelCallTrace
from evaluation.cache_path_eval.evidence import (
    CacheRuntimeEvidence,
    ModelCallEvidence,
    RetrievalCandidateEvidence,
    SubproblemCandidateEvidence,
)


class CacheRuntimeEvidenceTests(unittest.TestCase):
    def test_model_trace_conversion_discards_error_text_and_roundtrips_allowlist(self):
        trace = ModelCallTrace(
            provider="openai",
            model_name="gpt-6-luna",
            succeeded=False,
            latency_ms=14,
            retry_count=2,
            input_tokens=20,
            output_tokens=3,
            cached_input_tokens=4,
            reasoning_tokens=1,
            prompt_version="rewrite-v2",
            error_message="private prompt and response must not persist",
        )
        evidence = ModelCallEvidence.from_trace("QUERY_REWRITE", trace)
        encoded = json.dumps(evidence.to_dict())
        decoded = json.loads(encoded)
        self.assertEqual("QUERY_REWRITE", decoded["purpose"])
        self.assertEqual("rewrite-v2", decoded["promptVersion"])
        self.assertNotIn("error", encoded)
        self.assertNotIn("private prompt", encoded)
        self.assertEqual(
            {
                "purpose", "provider", "modelName", "promptVersion", "succeeded",
                "latencyMs", "retryCount", "inputTokens", "outputTokens",
                "cachedInputTokens", "reasoningTokens",
            },
            set(decoded),
        )

    def test_runtime_evidence_is_canonical_and_schema_is_strict(self):
        evidence = CacheRuntimeEvidence(
            seed="seed-4",
            retrieval_candidates=(RetrievalCandidateEvidence(3, 5, 8, 1),),
            subproblem_candidates=(SubproblemCandidateEvidence("billing", "sp-2", 4, "answer-9"),),
            selected_subproblem_id="sp-2",
            selected_canonical_answer_id="answer-9",
            gate_rejection_reasons=("LOW_CONFIDENCE",),
            exact_source_classification_id=12,
            model_calls=(
                ModelCallEvidence.from_trace(
                    "QUESTION_CLASSIFICATION",
                    ModelCallTrace("openai", "model-x", True, 1),
                ),
            ),
        )
        data = json.loads(json.dumps(evidence.to_dict()))
        self.assertEqual(data, evidence.to_dict())

        for schema_name in ("run.schema.json", "checkpoint.schema.json"):
            schema = json.loads(
                (Path(__file__).parents[1] / "evaluation/cache_path_eval" / schema_name).read_text()
            )
            definition = schema["$defs"]["evidence"]
            self.assertIs(definition["additionalProperties"], False)
            self.assertEqual(set(definition["required"]), set(data))
            for field in ("retrievalCandidates", "subproblemCandidates", "modelCalls"):
                child = schema["$defs"][
                    {"retrievalCandidates": "retrievalCandidate", "subproblemCandidates": "subproblemCandidate", "modelCalls": "modelCallEvidence"}[field]
                ]
                self.assertIs(child["additionalProperties"], False)

    def test_rejects_unknown_or_malformed_runtime_fields(self):
        with self.assertRaises(ValueError):
            ModelCallEvidence.from_trace(
                "QUESTION_TEXT", ModelCallTrace("openai", "model-x", True, 1)
            )
        with self.assertRaises(ValueError):
            RetrievalCandidateEvidence(True, 1, 1, 1)
        with self.assertRaises(ValueError):
            ModelCallEvidence(
                "QUERY_REWRITE", "openai", "model-x", "prompt with private text", True,
                1, 0, None, None, None, None,
            )
        with self.assertRaises(ValueError):
            CacheRuntimeEvidence(retrieval_candidates=[{"chunkId": 4}])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CacheRuntimeEvidence(unknown_text="private question")  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
