import json
import tempfile
import unittest
from pathlib import Path

from evaluation.cache_path_eval.runner import CachePathEvalError, evaluate


def _manifest():
    return {
        "group": "cache-path",
        "goldenVersion": "1.0.1",
        "sourceCorpusVersion": "cache-path-v1",
        "inputs": {"draftsSha256": "a" * 64, "judgeLabelsSha256": "b" * 64},
        "turns": [
            {"itemId": "SYN-1", "path": "P1", "turnNo": 1, "kind": "exact_hit", "expectedServed": True},
            {"itemId": "SYN-2", "path": "P3", "turnNo": 1, "kind": "judge", "expectedServed": False},
            {"itemId": "SYN-3", "path": "P6", "turnNo": 1, "kind": "context_only", "expectedExactHit": False},
        ],
    }


def _observations():
    return [
        {"itemId": "SYN-1", "turnNo": 1, "outcome": "SERVED", "nextStage": "NOT_APPLICABLE", "latencyMs": 11, "modelCalls": 0, "embeddingCalls": 0},
        {"itemId": "SYN-2", "turnNo": 1, "outcome": "SERVED", "nextStage": "FAIL", "latencyMs": 22, "modelCalls": 1, "embeddingCalls": 0},
        {"itemId": "SYN-3", "turnNo": 1, "outcome": "SKIPPED", "nextStage": "NOT_APPLICABLE", "latencyMs": 0, "modelCalls": 0, "embeddingCalls": 0},
    ]


class CachePathEvalTest(unittest.TestCase):
    def test_aggregates_path_metrics_without_turn_level_output(self):
        result = evaluate(_manifest(), _observations(), "c" * 64)

        self.assertEqual(result["runMode"], "offline-observations")
        self.assertEqual(result["summary"]["all"]["turns"], 3)
        self.assertEqual(result["summary"]["all"]["outcomes"], {"SERVED": 2, "SKIPPED": 1})
        self.assertEqual(result["summary"]["all"]["ineligibleServed"], 1)
        self.assertEqual(result["summary"]["all"]["nextStage"]["denominator"], 1)
        self.assertEqual(result["summary"]["all"]["nextStage"]["passRate"], 0)
        self.assertEqual(result["summary"]["paths"]["P1"]["all"]["latencyMs"]["median"], 11)
        rendered = json.dumps(result)
        self.assertNotIn("SYN-1", rendered)
        self.assertNotIn("question", rendered)

    def test_rejects_missing_duplicate_unknown_and_text_fields(self):
        manifest = _manifest()
        observations = _observations()
        with self.assertRaisesRegex(CachePathEvalError, "빠진 턴"):
            evaluate(manifest, observations[:-1], "c" * 64)
        with self.assertRaisesRegex(CachePathEvalError, "중복 관측"):
            evaluate(manifest, observations + [observations[0]], "c" * 64)
        with self.assertRaisesRegex(CachePathEvalError, "manifest에 없는"):
            evaluate(manifest, [{**observations[0], "itemId": "UNKNOWN"}, *observations[1:]], "c" * 64)
        with tempfile.TemporaryDirectory() as directory:
            observations_path = Path(directory) / "observations.jsonl"
            row = {**observations[0], "question": "private text"}
            observations_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            from evaluation.cache_path_eval.runner import _load_observations
            with self.assertRaisesRegex(CachePathEvalError, "허용되지 않음"):
                _load_observations(observations_path)

    def test_rejects_invalid_counts_and_manifest(self):
        observations = _observations()
        observations[0]["modelCalls"] = True
        with self.assertRaisesRegex(CachePathEvalError, "정수"):
            evaluate(_manifest(), observations, "c" * 64)
        manifest = _manifest()
        manifest["turns"].append(manifest["turns"][0])
        with self.assertRaisesRegex(CachePathEvalError, "중복 턴"):
            evaluate(manifest, _observations(), "c" * 64)
        manifest = _manifest()
        manifest["turns"][0]["question"] = "private text"
        with self.assertRaisesRegex(CachePathEvalError, "허용되지 않은 필드"):
            evaluate(manifest, _observations(), "c" * 64)


if __name__ == "__main__":
    unittest.main()
