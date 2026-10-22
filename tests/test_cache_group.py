import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.judge_eval.cache_group import (
    _input_set_sha256,
    _merge_retry_result,
    _read_retry_source,
    _claude_spent_or_reserved,
)
from evaluation.judge_eval.clients import variant_by_name


def _turn(item_id, turn_no, input_hash, payload_hash):
    return SimpleNamespace(
        key=(item_id, turn_no),
        item_id=item_id,
        turn_no=turn_no,
        input_sha256=input_hash,
        frozen_payload_sha256=payload_hash,
    )


def _row(turn, *, error=None, decision="CONNECT", usage=None, cost=None):
    return {
        "itemId": turn.item_id,
        "turnNo": turn.turn_no,
        "path": "P3",
        "inputSha256": turn.input_sha256,
        "frozenPayloadSha256": turn.frozen_payload_sha256,
        "expectedDecision": "CONNECT",
        "actualDecision": decision,
        "error": error,
        "attempts": 1,
        "usage": usage or {"inputTokens": 10, "outputTokens": 5, "cachedInputTokens": 0, "reasoningTokens": 0},
        "costUsd": cost or {"cachedUsd": 0.1, "upperUsd": 0.2},
    }


class CacheGroupRetryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = Path(self.temp.name)
        self.turns = [_turn("item-a", 1, "input-a", "payload-a"), _turn("item-b", 2, "input-b", "payload-b")]
        self.group = {"groupId": "cache-path", "groupVersion": "2.0.0", "groupSha256": "group-hash"}
        self.context = SimpleNamespace(
            golden=SimpleNamespace(root=self.registry.resolve(), version="golden-v1"),
            group=self.group,
            turns=self.turns,
        )
        self.source = {
            "kind": "cache-model-comparison",
            "mode": "full",
            "runId": "original-run",
            "model": variant_by_name("claude-haiku").stage_model(),
            "group": {"id": "cache-path", "version": "2.0.0", "sha256": "group-hash", "goldenVersion": "golden-v1"},
            "inputSetSha256": _input_set_sha256(self.turns),
            "summary": {"usage": {"inputTokens": 20, "outputTokens": 10, "cachedInputTokens": 0, "reasoningTokens": 0, "calls": 2},
                        "cost": {"cachedUsd": 0.2, "upperUsd": 0.4}},
            "results": [_row(self.turns[0]), _row(self.turns[1], error="Claude response truncated", decision="UNCLASSIFIED")],
        }

    def tearDown(self):
        self.temp.cleanup()

    def _write_source(self):
        path = self.registry / "original.json"
        path.write_text(json.dumps(self.source), encoding="utf-8")
        return path

    def test_source_selects_only_error_turns_for_same_model_and_frozen_inputs(self):
        source, selected = _read_retry_source(self.context, "claude-haiku", self._write_source())

        self.assertEqual("original-run", source["runId"])
        self.assertEqual([("item-b", 2)], [turn.key for turn in selected])

    def test_source_rejects_changed_payload_hash_before_retry(self):
        self.source["results"][1]["frozenPayloadSha256"] = "changed"

        with self.assertRaisesRegex(Exception, "payload hash"):
            _read_retry_source(self.context, "claude-haiku", self._write_source())

    def test_merge_keeps_full_rows_and_records_retry_provenance_and_total_usage(self):
        retry_row = _row(
            self.turns[1],
            usage={"inputTokens": 12, "outputTokens": 8, "cachedInputTokens": 2, "reasoningTokens": 0},
            cost={"cachedUsd": 0.3, "upperUsd": 0.4},
        )
        retry = {
            "runId": "retry-run",
            "finishedAt": "2026-10-22T10:00:00+00:00",
            "summary": {"usage": {"inputTokens": 12, "outputTokens": 8, "cachedInputTokens": 2, "reasoningTokens": 0, "calls": 1}},
            "results": [retry_row],
        }
        original_bytes = json.dumps(self.source, sort_keys=True)

        merged = _merge_retry_result(self.context, self.source, retry)

        self.assertEqual(2, len(merged["results"]))
        self.assertEqual("original", merged["results"][0]["provenance"]["result"])
        self.assertEqual("retry_success", merged["results"][1]["provenance"]["result"])
        self.assertEqual(22, merged["results"][1]["usage"]["inputTokens"])
        self.assertAlmostEqual(0.6, merged["results"][1]["costUsd"]["upperUsd"])
        self.assertEqual(32, merged["summary"]["usage"]["inputTokens"])
        self.assertEqual(3, merged["summary"]["usage"]["calls"])
        self.assertEqual(original_bytes, json.dumps(self.source, sort_keys=True))

    def test_unfinished_retry_reservation_counts_against_shared_claude_budget(self):
        ledger = self.registry / "evals" / "cache-path" / "model-comparison-ledger.jsonl"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps({
            "kind": "cache-model-comparison-retry-started",
            "model": "claude-haiku-5-5",
            "runId": "reserved-retry",
            "reservedUpperUsd": 1.25,
        }) + "\n", encoding="utf-8")

        self.assertEqual(1.25, _claude_spent_or_reserved(self.registry.resolve()))


if __name__ == "__main__":
    unittest.main()
