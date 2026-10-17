"""Private input reader for the frozen cache-path corpus.

The public selection manifest contains references and labels. This reader opens the
separate frozen text corpus, verifies all version/hash pins, and projects only the
current question plus explicit prior conversation inputs into runtime values.
"""

from __future__ import annotations

import hashlib
import json
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

    def load_turn(self, item_id: str, turn_no: int) -> FrozenCacheTurnInput:
        """Resolve one opaque reference; expected labels and route fields are ignored."""

        if not isinstance(item_id, str) or not item_id:
            raise FrozenCacheInputError("문항 ID가 필요합니다.")
        if isinstance(turn_no, bool) or not isinstance(turn_no, int) or turn_no < 1:
            raise FrozenCacheInputError("턴 번호가 잘못되었습니다.")
        if (item_id, turn_no) not in self._allowed_refs:
            raise FrozenCacheInputError("선택한 reference가 run manifest에 없습니다.")
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
