"""과거 첫 턴 SERVED 질문을 정확 일치 매핑으로 준비/적용한다.

기본은 dry-run이다. ``--apply``를 지정해야만 매핑을 쓰고 커밋한다.
답변 본문은 복사하지 않고 기존 canonical_answers 행만 참조한다.
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    CacheAttemptOutcome,
    ClassificationDecision,
    ClassificationRun,
    QuestionCacheAttempt,
    QuestionClassification,
    RagRun,
)
from app.question_grouping.store import QuestionGroupingStore


@dataclass(frozen=True)
class HistoricalCandidate:
    rag_run_id: object
    document_group_id: int
    question: str
    subproblem_id: object
    subproblem_version: int
    canonical_answer_id: object


async def load_candidates(session: AsyncSession) -> Sequence[HistoricalCandidate]:
    rows = (
        await session.execute(
            select(
                RagRun.id,
                ClassificationRun.document_group_id,
                RagRun.user_query,
                QuestionClassification.subproblem_id,
                QuestionClassification.subproblem_version,
                QuestionCacheAttempt.canonical_answer_id,
            )
            .join(
                QuestionClassification,
                QuestionClassification.rag_run_id == RagRun.id,
            )
            .join(
                ClassificationRun,
                ClassificationRun.id == QuestionClassification.run_id,
            )
            .join(
                QuestionCacheAttempt,
                QuestionCacheAttempt.rag_run_id == RagRun.id,
            )
            .where(
                RagRun.turn_no == 1,
                QuestionClassification.effective_to.is_(None),
                QuestionClassification.decision == ClassificationDecision.CONNECT,
                QuestionCacheAttempt.outcome == CacheAttemptOutcome.SERVED,
                QuestionCacheAttempt.canonical_answer_id.is_not(None),
                QuestionClassification.subproblem_id.is_not(None),
                QuestionClassification.subproblem_version.is_not(None),
            )
            .order_by(RagRun.created_at, RagRun.id)
        )
    ).all()
    return tuple(
        HistoricalCandidate(
            rag_run_id=row.id,
            document_group_id=row.document_group_id,
            question=row.user_query,
            subproblem_id=row.subproblem_id,
            subproblem_version=row.subproblem_version,
            canonical_answer_id=row.canonical_answer_id,
        )
        for row in rows
    )


async def run(*, apply: bool) -> int:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            candidates = await load_candidates(session)
            if not apply:
                print(f"historical exact candidates: {len(candidates)} (dry-run)")
                return 0
            store = QuestionGroupingStore(session)
            changed = 0
            for candidate in candidates:
                if await store.materialize_historical_exact(
                    document_group_id=candidate.document_group_id,
                    question=candidate.question,
                    subproblem_id=candidate.subproblem_id,
                    subproblem_version=candidate.subproblem_version,
                    canonical_answer_id=candidate.canonical_answer_id,
                    source_rag_run_id=candidate.rag_run_id,
                ):
                    changed += 1
            await session.commit()
            print(f"historical exact rows changed: {changed}/{len(candidates)}")
            return 0
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="매핑을 쓰고 커밋합니다")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(apply=args.apply)))


if __name__ == "__main__":
    main()
