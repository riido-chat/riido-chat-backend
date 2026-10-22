"""Private input reader for the frozen cache-path corpus.

The public selection manifest contains references and labels. This reader opens the
separate frozen text corpus, verifies all version/hash pins, and projects only the
current question plus explicit prior conversation inputs into runtime values.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from app.chat.query_rewrite import (
    MAX_QUERY_REWRITE_TURNS,
    QueryRewriteCandidateTurn,
    QueryRewriteTurnStatus,
)
from app.answering.models import FinalWithheldReason
from evaluation.cache_path_eval.store import CachePathRunMetadata


class FrozenCacheInputError(ValueError):
    """The selected private cache corpus does not match its immutable pins."""


def load_canonical_cache_path_manifest(
    *,
    golden_root: Path,
    group_path: Path,
    metadata: CachePathRunMetadata,
) -> dict[str, Any]:
    """Project the official versioned cache group into the runner's scoring manifest."""
    repository_root = golden_root.resolve()
    if metadata.source_corpus_version is not None:
        raise FrozenCacheInputError(
            "정본 A/C 그룹 실행에는 legacy sourceCorpusVersion을 지정하지 않아야 합니다."
        )
    try:
        actual_root = Path(
            subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        if actual_root != repository_root:
            raise FrozenCacheInputError("골든 입력은 registry Git 저장소 루트여야 합니다.")
        head = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != metadata.golden_commit:
            raise FrozenCacheInputError("골든 registry git commit이 실행 metadata와 다릅니다.")
        relative = group_path.resolve().relative_to(repository_root).as_posix()
        if not re.fullmatch(r"golden/groups/cache-path/[0-9]+\.[0-9]+\.[0-9]+\.json", relative):
            raise FrozenCacheInputError("공식 cache-path 그룹 파일만 사용할 수 있습니다.")

        def read_blob(path: str) -> bytes:
            return subprocess.run(
                ["git", "-C", str(repository_root), "show", f"HEAD:{path}"],
                check=True,
                capture_output=True,
            ).stdout

        group_bytes = read_blob(relative)
        if group_path.resolve().read_bytes() != group_bytes:
            raise FrozenCacheInputError("선택한 평가 그룹 파일이 registry HEAD와 다릅니다.")
        golden_bytes = read_blob("golden/manifest.json")
        a_bytes = read_blob("golden/items/A.jsonl")
        c_bytes = read_blob("golden/items/C.jsonl")
        source_bytes = read_blob("golden/sources/channeltalk-qa-reviewed-20260804.html")
        group = json.loads(group_bytes.decode("utf-8"))
        golden_manifest = json.loads(golden_bytes.decode("utf-8"))
        group_file_sha = hashlib.sha256(group_bytes).hexdigest()
        item_sha = hashlib.sha256(b"A\0" + a_bytes + b"\0C\0" + c_bytes).hexdigest()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
    except FrozenCacheInputError:
        raise
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FrozenCacheInputError("공식 cache-path 그룹을 읽을 수 없습니다.") from error

    content = {key: value for key, value in group.items() if key != "groupSha256"}
    group_digest = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    frozen_group = ((golden_manifest.get("frozen") or {}).get("files") or {}).get(relative)
    if not isinstance(frozen_group, Mapping) or frozen_group.get("sha256") != group_file_sha:
        raise FrozenCacheInputError("골든 manifest와 평가 그룹 파일 해시가 다릅니다.")
    if group.get("groupId") != "cache-path" or group.get("groupSha256") != group_digest:
        raise FrozenCacheInputError("공식 평가 그룹의 식별자 또는 내용 해시가 잘못되었습니다.")
    if group.get("sourceGoldenVersion") != metadata.golden_version or golden_manifest.get("goldenVersion") != metadata.golden_version:
        raise FrozenCacheInputError("평가 그룹과 실행 metadata의 골든 버전이 다릅니다.")

    a_items = {
        item["id"]: item
        for item in (json.loads(line) for line in a_bytes.decode("utf-8").splitlines() if line.strip())
    }
    c_items = {
        item["id"]: item
        for item in (json.loads(line) for line in c_bytes.decode("utf-8").splitlines() if line.strip())
    }
    turns = []
    for member in group.get("members", ()):
        if not isinstance(member, Mapping) or not isinstance(member.get("labels"), Mapping):
            raise FrozenCacheInputError("공식 평가 그룹 문항 형식이 잘못되었습니다.")
        item_id, turn_no, labels = member.get("itemId"), member.get("turnNo"), member["labels"]
        path = labels.get("path")
        if not isinstance(item_id, str) or not isinstance(turn_no, int) or path not in {"P1", "P3", "P4", "P6"}:
            raise FrozenCacheInputError("공식 평가 그룹 참조 형식이 잘못되었습니다.")
        item = a_items.get(item_id) or c_items.get(item_id)
        if item is None or turn_no < 1 or turn_no > len(item.get("turns", [])):
            raise FrozenCacheInputError("공식 평가 그룹 참조가 A/C 정본에 없습니다.")
        projected = {
            "itemId": item_id,
            "path": path,
            "turnNo": turn_no,
            "kind": "exact_hit" if path == "P1" else "judge",
            "expectedExactHit": labels.get("expectedExactHit"),
            "expectedServed": labels.get("expectedServed"),
        }
        if path != "P1":
            projected["expectedJudgeDecision"] = labels.get("expectedJudgeDecision")
            projected["expectedSubproblemKeysAny"] = labels.get("expectedSubproblemKeysAny", [])
        if path == "P6":
            projected["expectedSelectedTurnNos"] = labels.get("expectedSelectedTurnNos", [])
        turns.append(projected)

    return {
        "group": "cache-path",
        "goldenVersion": metadata.golden_version,
        "sourceCorpusVersion": None,
        "evaluationGroup": {
            "groupId": group["groupId"],
            "groupVersion": group["groupVersion"],
            "groupSha256": group_digest,
            "groupFileSha256": group_file_sha,
        },
        "inputs": {
            "draftsSha256": None,
            "judgeLabelsSha256": None,
            "canonicalItemsSha256": item_sha,
            "sourceHtmlSha256": source_sha,
        },
        "turns": turns,
    }


@dataclass(frozen=True)
class FrozenCacheTurnInput:
    question: str
    candidates: tuple[QueryRewriteCandidateTurn, ...]
    history_synthetic: bool


class FrozenCacheCorpusLoader:
    """Load only user turns and explicit history from a pinned cache corpus."""

    def __init__(
        self,
        *,
        golden_root: Path,
        selection_manifest: Mapping[str, Any],
        metadata: CachePathRunMetadata,
    ) -> None:
        self._golden_root = golden_root.resolve()
        self._selection_manifest = selection_manifest
        self._metadata = metadata
        self._payload: Optional[Mapping[str, Any]] = None
        self._canonical_payload: Optional[dict[str, Any]] = None
        self._allowed_refs = {
            (turn.get("itemId"), turn.get("turnNo"))
            for turn in selection_manifest.get("turns", ())
            if isinstance(turn, Mapping)
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_payload(self) -> Mapping[str, Any]:
        if self._payload is not None:
            return self._payload
        manifest_path = self._golden_root / "golden" / "manifest.json"
        drafts_path = self._golden_root / "golden" / "sources" / "cache-drafts.json"
        try:
            # The caller must supply the frozen registry repository root itself;
            # accepting an ancestor could silently read a different checkout.
            repository_root = self._golden_root
            git_root = Path(
                subprocess.run(
                    ["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ).resolve()
            if git_root != repository_root:
                raise FrozenCacheInputError(
                    "골든 입력은 지정한 registry Git 저장소 안에 있어야 합니다."
                )
            head = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != self._metadata.golden_commit:
                raise FrozenCacheInputError("골든 registry git commit이 실행 metadata와 다릅니다.")
            manifest_relative = manifest_path.relative_to(repository_root).as_posix()
            drafts_relative = drafts_path.relative_to(repository_root).as_posix()
            manifest_bytes = subprocess.run(
                ["git", "-C", str(repository_root), "show", f"HEAD:{manifest_relative}"],
                check=True,
                capture_output=True,
            ).stdout
            drafts_bytes = subprocess.run(
                ["git", "-C", str(repository_root), "show", f"HEAD:{drafts_relative}"],
                check=True,
                capture_output=True,
            ).stdout
            golden_manifest = json.loads(manifest_bytes.decode("utf-8"))
            payload = json.loads(drafts_bytes.decode("utf-8"))
        except FrozenCacheInputError:
            raise
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise FrozenCacheInputError("동결 캐시 입력을 읽을 수 없습니다.") from error

        if golden_manifest.get("goldenVersion") != self._metadata.golden_version:
            raise FrozenCacheInputError("골든 버전이 실행 metadata와 다릅니다.")
        if self._selection_manifest.get("goldenVersion") != self._metadata.golden_version:
            raise FrozenCacheInputError("선택 manifest의 골든 버전이 실행 metadata와 다릅니다.")
        if self._selection_manifest.get("sourceCorpusVersion") != self._metadata.source_corpus_version:
            raise FrozenCacheInputError("선택 corpus 버전이 실행 metadata와 다릅니다.")
        if payload.get("version") != self._metadata.source_corpus_version:
            raise FrozenCacheInputError("동결 corpus 버전이 실행 metadata와 다릅니다.")
        if not self._metadata.source_corpus_version:
            raise FrozenCacheInputError("실행 metadata에 source corpus version이 없습니다.")

        relative = "golden/sources/cache-drafts.json"
        expected = ((golden_manifest.get("frozen") or {}).get("files") or {}).get(relative)
        actual_hash = hashlib.sha256(drafts_bytes).hexdigest()
        if not isinstance(expected, dict) or expected.get("sha256") != actual_hash:
            raise FrozenCacheInputError("동결 manifest와 캐시 corpus 해시가 다릅니다.")
        selected_hash = (self._selection_manifest.get("inputs") or {}).get("draftsSha256")
        if selected_hash != actual_hash:
            raise FrozenCacheInputError("선택 manifest와 캐시 corpus 해시가 다릅니다.")
        if payload.get("status") != "draft":
            raise FrozenCacheInputError("캐시 corpus 진단 상태가 바뀌었습니다.")

        self._payload = payload
        return payload

    def _load_canonical_payload(self) -> Mapping[str, Any]:
        """Read the canonical A/C turns and their original ChannelTalk source."""
        if self._canonical_payload is not None:
            return self._canonical_payload
        group = self._selection_manifest.get("evaluationGroup")
        inputs = self._selection_manifest.get("inputs")
        if not isinstance(group, Mapping) or not isinstance(inputs, Mapping):
            raise FrozenCacheInputError("정본 평가 그룹 provenance가 없습니다.")
        if group.get("groupId") != "cache-path" or not isinstance(group.get("groupVersion"), str):
            raise FrozenCacheInputError("정본 평가 그룹 식별자가 잘못되었습니다.")
        try:
            repository_root = Path(
                subprocess.run(
                    ["git", "-C", str(self._golden_root), "rev-parse", "--show-toplevel"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ).resolve()
            if repository_root != self._golden_root:
                raise FrozenCacheInputError("골든 입력은 registry Git 저장소 루트여야 합니다.")
            head = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != self._metadata.golden_commit:
                raise FrozenCacheInputError("골든 registry git commit이 실행 metadata와 다릅니다.")

            def read_blob(relative: str) -> bytes:
                return subprocess.run(
                    ["git", "-C", str(repository_root), "show", f"HEAD:{relative}"],
                    check=True,
                    capture_output=True,
                ).stdout

            golden_bytes = read_blob("golden/manifest.json")
            group_relative = f"golden/groups/cache-path/{group['groupVersion']}.json"
            group_bytes = read_blob(group_relative)
            a_bytes = read_blob("golden/items/A.jsonl")
            c_bytes = read_blob("golden/items/C.jsonl")
            source_relative = "golden/sources/channeltalk-qa-reviewed-20260804.html"
            source_bytes = read_blob(source_relative)
            golden_manifest = json.loads(golden_bytes.decode("utf-8"))
            group_payload = json.loads(group_bytes.decode("utf-8"))
            a_items = [json.loads(line) for line in a_bytes.decode("utf-8").splitlines() if line.strip()]
            c_items = [json.loads(line) for line in c_bytes.decode("utf-8").splitlines() if line.strip()]
            source_match = re.search(
                rb"const\s+records\s*=\s*(\[.*?\])\s*;",
                source_bytes,
                re.DOTALL,
            )
            if source_match is None:
                raise FrozenCacheInputError("고정 ChannelTalk 원본에서 상담 기록을 찾지 못했습니다.")
            source_records = json.loads(source_match.group(1).decode("utf-8"))
        except FrozenCacheInputError:
            raise
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise FrozenCacheInputError("정본 평가 입력을 읽을 수 없습니다.") from error

        frozen_files = ((golden_manifest.get("frozen") or {}).get("files") or {})
        group_file_sha = hashlib.sha256(group_bytes).hexdigest()
        items_sha = hashlib.sha256(b"A\0" + a_bytes + b"\0C\0" + c_bytes).hexdigest()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        expected_group_file = frozen_files.get(group_relative)
        if not isinstance(expected_group_file, dict) or expected_group_file.get("sha256") != group_file_sha:
            raise FrozenCacheInputError("골든 manifest와 평가 그룹 파일 해시가 다릅니다.")
        if golden_manifest.get("goldenVersion") != self._metadata.golden_version:
            raise FrozenCacheInputError("정본 골든 버전이 실행 metadata와 다릅니다.")
        if group_payload.get("sourceGoldenVersion") != self._metadata.golden_version:
            raise FrozenCacheInputError("평가 그룹과 실행 metadata의 골든 버전이 다릅니다.")
        group_content = {key: value for key, value in group_payload.items() if key != "groupSha256"}
        group_digest = hashlib.sha256(
            json.dumps(group_content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if group_payload.get("groupSha256") != group_digest or group.get("groupSha256") != group_digest:
            raise FrozenCacheInputError("평가 그룹의 내용 해시가 실행 manifest와 다릅니다.")
        if group.get("groupFileSha256") != group_file_sha:
            raise FrozenCacheInputError("평가 그룹 파일이 실행 manifest와 다릅니다.")
        if inputs.get("canonicalItemsSha256") != items_sha or inputs.get("sourceHtmlSha256") != source_sha:
            raise FrozenCacheInputError("정본 문항 또는 원본 대화의 해시가 실행 manifest와 다릅니다.")
        group_refs = {
            (member.get("itemId"), member.get("turnNo"))
            for member in group_payload.get("members", ())
            if isinstance(member, Mapping)
        }
        if group_refs != self._allowed_refs:
            raise FrozenCacheInputError("실행 reference가 공식 평가 그룹과 다릅니다.")

        item_by_id = {item.get("id"): item for item in [*a_items, *c_items] if isinstance(item, dict)}
        source_by_id = {record.get("sample_id"): record for record in source_records if isinstance(record, dict)}
        self._canonical_payload = {
            "items": item_by_id,
            "sources": source_by_id,
            "group": group_payload,
            "groupFileSha256": group_file_sha,
            "canonicalItemsSha256": items_sha,
            "sourceHtmlSha256": source_sha,
        }
        return self._canonical_payload

    @staticmethod
    def _canonical_source_answer(item: Mapping[str, Any], turn_no: int, source: Mapping[str, Any]) -> Optional[str]:
        spans = [
            span for span in item.get("sourceSpans", ())
            if isinstance(span, Mapping) and span.get("turnNo") == turn_no
        ]
        message_numbers = [span.get("messageNo") for span in spans]
        if not message_numbers or any(isinstance(number, bool) or not isinstance(number, int) for number in message_numbers):
            return None
        conversation = source.get("conversation")
        if not isinstance(conversation, list):
            return None
        user_indices = [
            index for index, message in enumerate(conversation)
            if isinstance(message, Mapping) and message.get("role") == "user"
        ]
        last_user_ordinal = max(message_numbers)
        if last_user_ordinal < 1 or last_user_ordinal > len(user_indices):
            return None
        last_user_index = user_indices[last_user_ordinal - 1]
        next_user_index = (
            user_indices[last_user_ordinal]
            if last_user_ordinal < len(user_indices)
            else len(conversation)
        )
        replies = [
            message.get("text", "").strip()
            for message in conversation[last_user_index + 1 : next_user_index]
            if isinstance(message, Mapping)
            and message.get("role") in {"bot", "manager"}
            and isinstance(message.get("text"), str)
            and message.get("text", "").strip()
        ]
        return "\n\n".join(replies) if replies else None

    def _load_canonical_turn(self, item_id: str, turn_no: int) -> FrozenCacheTurnInput:
        payload = self._load_canonical_payload()
        item = payload["items"].get(item_id)
        if not isinstance(item, Mapping):
            raise FrozenCacheInputError("선택한 정본 문항이 A/C JSONL에 없습니다.")
        turns = item.get("turns")
        if not isinstance(turns, list) or turn_no > len(turns):
            raise FrozenCacheInputError("선택한 정본 턴이 없습니다.")
        current = turns[turn_no - 1]
        question = current.get("question") if isinstance(current, Mapping) else None
        if not isinstance(question, str) or not question.strip():
            raise FrozenCacheInputError("선택한 정본 턴의 질문이 없습니다.")

        history_synthetic = item.get("historySynthetic", False)
        if not isinstance(history_synthetic, bool):
            raise FrozenCacheInputError("historySynthetic 표식 형식이 잘못되었습니다.")
        source_id = item.get("sourceId")
        source = None
        if isinstance(source_id, str) and source_id.startswith("channeltalk:"):
            source = payload["sources"].get(source_id.partition(":")[2])

        candidates: list[QueryRewriteCandidateTurn] = []
        for prior_no, prior in enumerate(turns[: turn_no - 1], start=1):
            if not isinstance(prior, Mapping):
                continue
            prior_question = prior.get("question")
            if not isinstance(prior_question, str) or not prior_question.strip():
                continue
            answer = prior.get("historyAnswer")
            content: Optional[str] = None
            status = QueryRewriteTurnStatus.COMPLETED
            withheld_reason = None
            if isinstance(answer, Mapping):
                answer_status = answer.get("status")
                if answer_status == "COMPLETED":
                    content = answer.get("content")
                elif answer_status == "WITHHELD":
                    reason = answer.get("withheldReasonCode", answer.get("withheldReason"))
                    try:
                        withheld_reason = FinalWithheldReason(reason)
                    except (TypeError, ValueError) as error:
                        raise FrozenCacheInputError("고정된 이전 보류 턴의 사유가 잘못되었습니다.") from error
                    status = QueryRewriteTurnStatus.WITHHELD
                else:
                    raise FrozenCacheInputError("고정된 이전 턴 상태를 지원하지 않습니다.")
            elif source is not None:
                content = self._canonical_source_answer(item, prior_no, source)
            if status == QueryRewriteTurnStatus.COMPLETED and (not isinstance(content, str) or not content.strip()):
                # Production history contains only completed/withheld RAG runs.
                # Unanswered source turns are therefore omitted from candidates.
                continue
            candidates.append(
                QueryRewriteCandidateTurn(
                    rag_run_id=self._candidate_id(item_id, prior_no),
                    turn_no=prior_no,
                    status=status,
                    user_query=prior_question.strip(),
                    resolved_query=prior_question.strip() if prior_no == 1 else None,
                    answer_content=content.strip() if isinstance(content, str) else None,
                    withheld_reason_code=withheld_reason,
                )
            )
        candidates = candidates[-MAX_QUERY_REWRITE_TURNS:]
        return FrozenCacheTurnInput(
            question=question.strip(),
            candidates=tuple(candidates),
            history_synthetic=history_synthetic,
        )

    def load_turn(self, item_id: str, turn_no: int) -> FrozenCacheTurnInput:
        """Resolve one opaque reference; expected labels and route fields are ignored."""

        if not isinstance(item_id, str) or not item_id:
            raise FrozenCacheInputError("문항 ID가 필요합니다.")
        if isinstance(turn_no, bool) or not isinstance(turn_no, int) or turn_no < 1:
            raise FrozenCacheInputError("턴 번호가 잘못되었습니다.")
        if (item_id, turn_no) not in self._allowed_refs:
            raise FrozenCacheInputError("선택한 reference가 run manifest에 없습니다.")
        if isinstance(self._selection_manifest.get("evaluationGroup"), Mapping):
            return self._load_canonical_turn(item_id, turn_no)
        payload = self._load_payload()
        selected_item = next(
            (item for item in payload.get("items", ()) if item.get("id") == item_id),
            None,
        )
        if selected_item is None:
            raise FrozenCacheInputError("선택한 문항이 동결 corpus에 없습니다.")
        turns = selected_item.get("turns")
        if not isinstance(turns, list) or turn_no > len(turns):
            raise FrozenCacheInputError("선택한 턴이 동결 corpus에 없습니다.")

        history_synthetic = selected_item.get("historySynthetic", False)
        if not isinstance(history_synthetic, bool):
            raise FrozenCacheInputError("historySynthetic 표식 형식이 잘못되었습니다.")
        if history_synthetic and (
            not turns
            or not isinstance(turns[0], dict)
            or not isinstance(turns[0].get("historyAnswer"), dict)
        ):
            raise FrozenCacheInputError("synthetic history 표식에 대응하는 고정 답변이 없습니다.")
        current = turns[turn_no - 1]
        question = current.get("question") if isinstance(current, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise FrozenCacheInputError("선택한 턴의 질문 입력이 없습니다.")

        candidates: list[QueryRewriteCandidateTurn] = []
        for prior_no, prior in enumerate(turns[: turn_no - 1], start=1):
            if not isinstance(prior, dict):
                raise FrozenCacheInputError("이전 대화 턴 형식이 잘못되었습니다.")
            prior_question = prior.get("question")
            prior_resolved_query = prior.get("historyResolvedQuery")
            answer = prior.get("historyAnswer")
            if not isinstance(prior_question, str) or not prior_question.strip():
                raise FrozenCacheInputError("이전 대화 질문이 없어 문맥을 복원할 수 없습니다.")
            if prior_no == 1:
                # Production bypasses Query Rewrite on the first conversation turn.
                prior_resolved_query = prior_question
            elif not isinstance(prior_resolved_query, str) or not prior_resolved_query.strip():
                raise FrozenCacheInputError("이전 후속 턴의 확정 질의가 없어 문맥을 복원할 수 없습니다.")
            if not isinstance(answer, dict):
                raise FrozenCacheInputError("이전 대화 답변이 없어 후속 질문 문맥을 복원할 수 없습니다.")
            status = answer.get("status")
            content = answer.get("content")
            if status == "COMPLETED":
                if not isinstance(content, str) or not content.strip():
                    raise FrozenCacheInputError("완료된 이전 턴에 답변 본문이 없습니다.")
                candidates.append(
                    QueryRewriteCandidateTurn(
                        rag_run_id=self._candidate_id(item_id, prior_no),
                        turn_no=prior_no,
                        status=QueryRewriteTurnStatus.COMPLETED,
                        user_query=prior_question,
                        resolved_query=prior_resolved_query.strip(),
                        answer_content=content,
                        withheld_reason_code=None,
                    )
                )
            elif status == "WITHHELD":
                reason = answer.get("withheldReasonCode")
                try:
                    withheld_reason = FinalWithheldReason(reason)
                except (TypeError, ValueError) as error:
                    raise FrozenCacheInputError("보류된 이전 턴의 사유가 없습니다.") from error
                candidates.append(
                    QueryRewriteCandidateTurn(
                        rag_run_id=self._candidate_id(item_id, prior_no),
                        turn_no=prior_no,
                        status=QueryRewriteTurnStatus.WITHHELD,
                        user_query=prior_question,
                        resolved_query=prior_resolved_query.strip(),
                        answer_content=None,
                        withheld_reason_code=withheld_reason,
                    )
                )
            else:
                raise FrozenCacheInputError("이전 대화 턴 상태를 지원하지 않습니다.")

        # ChatLogStore selects the most recent five valid prior turns and returns
        # them in chronological order. Match that window without moving private
        # corpus rows into the public run manifest.
        candidates = candidates[-MAX_QUERY_REWRITE_TURNS:]

        return FrozenCacheTurnInput(
            question=question.strip(),
            candidates=tuple(candidates),
            history_synthetic=history_synthetic,
        )

    @staticmethod
    def _candidate_id(item_id: str, turn_no: int):
        import uuid

        return uuid.uuid5(uuid.NAMESPACE_URL, f"riido-cache-golden:{item_id}:{turn_no}")
