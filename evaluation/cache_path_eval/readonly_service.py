"""Read-only, explicitly selected service data for cache-path evaluation.

Each opened snapshot is one PostgreSQL repeatable-read/read-only transaction.
It is consistent for one turn. Mutable approval rows can still differ between
turn snapshots in the same evaluation run.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    EmbeddingConfig,
    IndexVersion,
)
from app.profiles.stage_settings import (
    ProfileStageSettings,
    ProfileStageSettingsError,
    load_profile_stage_settings,
)
from app.question_grouping.catalog_reader import QuestionCatalogReader
from app.question_grouping.exact_question_reader import ExactQuestionReader
from app.question_grouping.outline_reader import DocumentOutlineReader
from app.retrieval.search_reader import SearchReader
from evaluation.cache_path_eval.store import CachePathRunMetadata


class ReadOnlyServiceSnapshotError(ValueError):
    """The requested metadata does not identify a coherent service snapshot."""


@dataclass(frozen=True)
class EmbeddingConfigIdentity:
    id: int
    version: str
    provider: str
    model_name: str
    dimensions: int
    input_template_version: str


@dataclass(frozen=True)
class SelectedServiceSettings:
    service_key: str
    profile_revision_id: int
    document_group_id: int
    index_version_id: int
    embedding_config_id: int
    embedding_config: EmbeddingConfigIdentity
    exact_cache_enabled: bool
    semantic_cache_enabled: bool
    stage_settings: ProfileStageSettings


@dataclass(frozen=True)
class ReadOnlyServiceSnapshot:
    selection: SelectedServiceSettings
    catalog_reader: QuestionCatalogReader
    outline_reader: DocumentOutlineReader
    search_reader: SearchReader
    exact_reader: ExactQuestionReader


class ReadOnlyRiidoServiceData:
    """Create read-only per-turn snapshots from an injected async session factory.

    Construction is inert; it neither opens a session nor connects to the DB.
    The supplied factory should be an ``async_sessionmaker`` or compatible
    zero-argument factory. Sessions are forced to disable autoflush.
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def open_snapshot(
        self, metadata: CachePathRunMetadata, *, allow_judge_override: bool = False
    ) -> AsyncIterator[ReadOnlyServiceSnapshot]:
        session = self._session_factory()
        session.autoflush = False
        transaction_started = False
        try:
            await session.begin()
            transaction_started = True
            # This must be the first SQL statement in the transaction, before
            # any profile or index data is read.
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            selection, revision = await self._select_settings(
                session, metadata, allow_judge_override=allow_judge_override
            )
            snapshot = ReadOnlyServiceSnapshot(
                selection=selection,
                catalog_reader=QuestionCatalogReader(session),
                outline_reader=DocumentOutlineReader(session),
                search_reader=SearchReader(
                    session,
                    document_group_id=selection.document_group_id,
                    index_version_id=selection.index_version_id,
                ),
                exact_reader=ExactQuestionReader(session),
            )
            yield snapshot
        finally:
            # Always discard the transaction, including normal completion and
            # cancellation. Shield cleanup so task cancellation cannot skip it.
            try:
                if transaction_started and session.in_transaction():
                    await asyncio.shield(session.rollback())
            finally:
                await asyncio.shield(session.close())

    @staticmethod
    async def _select_settings(
        session: AsyncSession,
        metadata: CachePathRunMetadata,
        *,
        allow_judge_override: bool = False,
    ) -> tuple[SelectedServiceSettings, ChatProfileRevision]:
        selected = (
            await session.execute(
                select(ChatProfileRevision, ChatProfile.profile_key)
                .join(ChatProfile, ChatProfile.id == ChatProfileRevision.profile_id)
                .where(
                    ChatProfileRevision.id == metadata.profile_revision_id,
                    ChatProfile.profile_key == metadata.service_key,
                    ChatProfileRevision.document_group_id == metadata.document_group_id,
                    ChatProfile.document_group_id == metadata.document_group_id,
                )
            )
        ).one_or_none()
        if selected is None:
            raise ReadOnlyServiceSnapshotError(
                "Selected profile revision does not belong to the requested service "
                "and document group."
            )
        revision, _service_key = selected

        index = await session.get(IndexVersion, metadata.index_version_id)
        if index is None:
            raise ReadOnlyServiceSnapshotError(
                f"Selected index version {metadata.index_version_id} does not exist."
            )
        if (
            index.document_group_id != metadata.document_group_id
            or index.embedding_config_id != metadata.embedding_config_id
        ):
            raise ReadOnlyServiceSnapshotError(
                "Selected index version does not match the requested document group "
                "and embedding configuration."
            )

        embedding = await session.get(EmbeddingConfig, metadata.embedding_config_id)
        if embedding is None:
            raise ReadOnlyServiceSnapshotError(
                f"Selected embedding configuration {metadata.embedding_config_id} does not exist."
            )
        if metadata.embedding_model_version is not None and (
            metadata.embedding_model_version != embedding.model_name
        ):
            raise ReadOnlyServiceSnapshotError(
                "Selected embedding model does not match run metadata."
            )

        try:
            stage_settings = await load_profile_stage_settings(session, revision)
        except ProfileStageSettingsError as error:
            raise ReadOnlyServiceSnapshotError(
                f"Selected profile revision is not supported by this application: {error}"
            ) from error

        if not allow_judge_override and metadata.judge_model_version is not None and (
            metadata.judge_model_version != stage_settings.judge.model_name
        ):
            raise ReadOnlyServiceSnapshotError(
                "Selected judge model does not match run metadata."
            )
        if not allow_judge_override and metadata.judge_prompt_version is not None and (
            metadata.judge_prompt_version != stage_settings.judge.prompt_version
        ):
            raise ReadOnlyServiceSnapshotError(
                "Selected judge prompt does not match run metadata."
            )

        identity = EmbeddingConfigIdentity(
            id=embedding.id,
            version=embedding.version,
            provider=embedding.provider,
            model_name=embedding.model_name,
            dimensions=embedding.dimensions,
            input_template_version=embedding.input_template_version,
        )
        return (
            SelectedServiceSettings(
                service_key=metadata.service_key,
                profile_revision_id=revision.id,
                document_group_id=revision.document_group_id,
                index_version_id=index.id,
                embedding_config_id=embedding.id,
                embedding_config=identity,
                exact_cache_enabled=revision.exact_cache_enabled,
                semantic_cache_enabled=revision.semantic_cache_enabled,
                stage_settings=stage_settings,
            ),
            revision,
        )
