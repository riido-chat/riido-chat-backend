import unittest

from app.core.model_trace import (
    ModelCallTrace,
    ModelRequestEvidence,
    canonical_request_sha256,
)


class ModelRequestEvidenceTest(unittest.TestCase):
    def test_validates_hashes_and_safe_identifiers(self) -> None:
        digest = canonical_request_sha256({"é": 1})
        evidence = ModelRequestEvidence(
            common_input_sha256=digest,
            request_sha256=digest,
            input_transform_version="jev-choice-context-v1",
            instructions_sha256=digest,
            response_model_name="jev-latest",
        )
        self.assertEqual("jev-latest", evidence.response_model_name)
        with self.assertRaises(ValueError):
            ModelRequestEvidence("bad", digest, "transform-v1", digest)
        with self.assertRaises(ValueError):
            ModelRequestEvidence(digest, digest, "bad\nsecret", digest)
        with self.assertRaises(ValueError):
            ModelRequestEvidence(digest, digest, "transform-v1", digest, "arbitrary response")

    def test_trace_evidence_is_optional_for_existing_callers(self) -> None:
        trace = ModelCallTrace("openai", "model-x", True, 1)
        self.assertIsNone(trace.request_evidence)

    def test_canonical_hash_ignores_mapping_order_and_rejects_nan(self) -> None:
        self.assertEqual(
            canonical_request_sha256({"b": 2, "a": 1}),
            canonical_request_sha256({"a": 1, "b": 2}),
        )
        with self.assertRaises(ValueError):
            canonical_request_sha256({"bad": float("nan")})
