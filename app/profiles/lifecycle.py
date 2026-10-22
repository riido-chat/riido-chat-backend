"""프로필 버전의 생성, 검증, 게시와 복원. 구성은 생성 트랜잭션에서 확정한다.

TESTING 전환은 등록된 프롬프트와 지문, 출력 스키마의 호환성 검증만 수행한다.
실제 모델 호출이나 품질 평가를 실행하지 않으며 챗봇 주소의 서빙 상태를 바꾸지 않는다.
"""

from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChatProfile, ChatProfileRevision, ChatProfileRevisionStatus as Status,
    Conversation, ProfileRevisionComponent, ProfileRevisionComponentRecordedBy,
    ProfileRevisionComponentStage as Stage, RagRun,
)
from app.profiles.stage_settings import (
    ComponentSpec, STAGE_ORDER, code_components, component_problems,
    read_profile_components,
)


class ProfileRevisionError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class RevisionDetail:
    revision: ChatProfileRevision
    components: Sequence[ProfileRevisionComponent]


def _conflict(message: str) -> ProfileRevisionError:
    return ProfileRevisionError("PROFILE_REVISION_CONFLICT", message, 409)


class ProfileRevisionService:
    """요청 단위 세션을 사용하고 모든 쓰기를 성공 시 commit, 실패 시 rollback 한다."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _profile(self, profile_key: str, *, lock: bool = False) -> ChatProfile:
        statement = select(ChatProfile).where(ChatProfile.profile_key == profile_key)
        if lock:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        profile = await self.session.scalar(statement)
        if profile is None:
            raise ProfileRevisionError("NOT_FOUND", "서비스를 찾을 수 없습니다.", 404)
        return profile

    async def _revision(self, profile: ChatProfile, revision_id: int) -> ChatProfileRevision:
        revision = await self.session.scalar(
            select(ChatProfileRevision).where(
                ChatProfileRevision.profile_id == profile.id,
                ChatProfileRevision.id == revision_id,
            ).execution_options(populate_existing=True)
        )
        if revision is None:
            raise ProfileRevisionError("NOT_FOUND", "프로필 버전을 찾을 수 없습니다.", 404)
        return revision

    async def list(self, profile_key: str) -> Sequence[ChatProfileRevision]:
        profile = await self._profile(profile_key)
        return list((await self.session.scalars(
            select(ChatProfileRevision).where(ChatProfileRevision.profile_id == profile.id)
            .order_by(ChatProfileRevision.version.desc())
        )).all())

    async def detail(self, profile_key: str, revision_id: int) -> RevisionDetail:
        profile = await self._profile(profile_key)
        revision = await self._revision(profile, revision_id)
        return RevisionDetail(revision, await read_profile_components(self.session, revision.id))

    @staticmethod
    def _validate(revision: ChatProfileRevision, rows: Sequence) -> None:
        stages = [row.stage for row in rows]
        if len(stages) != len(STAGE_ORDER) or set(stages) != set(STAGE_ORDER):
            raise ProfileRevisionError(
                "INVALID_PROFILE_CONFIGURATION", "일곱 단계의 구성을 각각 한 번 지정해야 합니다.", 400
            )
        problems = component_problems(revision, rows)
        if problems:
            raise ProfileRevisionError(
                "INVALID_PROFILE_CONFIGURATION", "구성을 서빙할 수 없습니다: " + "; ".join(problems), 400
            )

    async def create(
        self, profile_key: str, *, components: Sequence[ComponentSpec] | None = None,
        semantic_cache_enabled: bool | None = None, exact_cache_enabled: bool | None = None,
    ) -> RevisionDetail:
        try:
            profile = await self._profile(profile_key, lock=True)
            specs = tuple(code_components() if components is None else components)
            by_stage = {spec.stage: spec for spec in specs}
            if len(specs) != len(STAGE_ORDER) or set(by_stage) != set(STAGE_ORDER):
                raise ProfileRevisionError(
                    "INVALID_PROFILE_CONFIGURATION", "일곱 단계의 구성을 각각 한 번 지정해야 합니다.", 400
                )
            current = await self.session.scalar(select(ChatProfileRevision).where(
                ChatProfileRevision.profile_id == profile.id,
                ChatProfileRevision.status == Status.PUBLISHED,
            ))
            version = await self.session.scalar(select(func.max(ChatProfileRevision.version)).where(
                ChatProfileRevision.profile_id == profile.id,
            ))
            generation, rewrite = by_stage[Stage.GENERATION], by_stage[Stage.REWRITE]
            revision = ChatProfileRevision(
                profile_id=profile.id, document_group_id=profile.document_group_id,
                version=(version or 0) + 1, status=Status.DRAFT,
                generation_model_name=generation.model_name,
                generation_prompt_version=generation.prompt_version,
                query_rewrite_model_name=rewrite.model_name,
                query_rewrite_prompt_version=rewrite.prompt_version,
                semantic_cache_enabled=(getattr(current, "semantic_cache_enabled", False)
                    if semantic_cache_enabled is None else semantic_cache_enabled),
                exact_cache_enabled=(getattr(current, "exact_cache_enabled", False)
                    if exact_cache_enabled is None else exact_cache_enabled),
            )
            # 저장 전에 검증하여 잘못된 구성을 남기지 않는다.
            self._validate(revision, specs)
            self.session.add(revision)
            await self.session.flush()
            rows = [ProfileRevisionComponent(
                revision_id=revision.id, stage=spec.stage, model_name=spec.model_name,
                prompt_key=spec.prompt_key, prompt_version=spec.prompt_version,
                prompt_sha256=spec.prompt_sha256, params=spec.params,
                recorded_by=ProfileRevisionComponentRecordedBy.PUBLISH,
            ) for spec in specs]
            self.session.add_all(rows)
            await self.session.flush()
            await self.session.commit()
            return RevisionDetail(revision, rows)
        except IntegrityError as error:
            await self.session.rollback()
            raise _conflict("프로필 버전 생성이 데이터베이스 규칙과 충돌합니다.") from error
        except BaseException:
            await self.session.rollback()
            raise

    async def _transition(
        self, profile_key: str, revision_id: int, *, before: Status, after: Status,
    ) -> RevisionDetail:
        try:
            profile = await self._profile(profile_key, lock=True)
            revision = await self._revision(profile, revision_id)
            if revision.status != before:
                raise _conflict(f"{before.value} 프로필 버전만 {after.value} 상태로 바꿀 수 있습니다.")
            rows = await read_profile_components(self.session, revision.id)
            self._validate(revision, rows)
            previous = await self.session.scalar(select(ChatProfileRevision).where(
                ChatProfileRevision.profile_id == profile.id,
                ChatProfileRevision.status == after,
            ).execution_options(populate_existing=True))
            if previous is not None:
                previous.status = Status.RETIRED
                # 부분 unique 인덱스 때문에 이전 활성 버전을 먼저 내린다.
                await self.session.flush()
            revision.status = after
            await self.session.flush()
            await self.session.commit()
            return RevisionDetail(revision, rows)
        except IntegrityError as error:
            await self.session.rollback()
            raise _conflict("프로필 버전 상태 변경이 데이터베이스 규칙과 충돌합니다.") from error
        except BaseException:
            await self.session.rollback()
            raise

    async def testing(self, profile_key: str, revision_id: int) -> RevisionDetail:
        """구성 호환성만 검증한다. 실제 모델 시험이나 서비스 B 연결은 하지 않는다."""
        return await self._transition(profile_key, revision_id, before=Status.DRAFT, after=Status.TESTING)

    async def publish(self, profile_key: str, revision_id: int) -> RevisionDetail:
        return await self._transition(profile_key, revision_id, before=Status.TESTING, after=Status.PUBLISHED)

    async def rollback(self, profile_key: str, revision_id: int) -> RevisionDetail:
        return await self._transition(profile_key, revision_id, before=Status.RETIRED, after=Status.PUBLISHED)

    async def delete(self, profile_key: str, revision_id: int) -> None:
        try:
            profile = await self._profile(profile_key, lock=True)
            revision = await self._revision(profile, revision_id)
            if revision.status != Status.DRAFT:
                raise _conflict("사용되지 않은 DRAFT 프로필 버전만 지울 수 있습니다.")
            for model, column in (
                (Conversation, Conversation.chat_profile_revision_id),
                (RagRun, RagRun.profile_revision_id),
            ):
                if await self.session.scalar(select(model.id).where(column == revision.id).limit(1)) is not None:
                    raise _conflict("서빙 기록이 있는 프로필 버전은 지울 수 없습니다.")
            await self.session.execute(delete(ProfileRevisionComponent).where(
                ProfileRevisionComponent.revision_id == revision.id,
            ))
            await self.session.delete(revision)
            await self.session.flush()
            await self.session.commit()
        except IntegrityError as error:
            await self.session.rollback()
            raise _conflict("프로필 버전 삭제가 데이터베이스 규칙과 충돌합니다.") from error
        except BaseException:
            await self.session.rollback()
            raise
