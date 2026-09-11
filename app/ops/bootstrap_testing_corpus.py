"""Copy the production search corpus into HELP_CHATBOT_TEST.

This is deliberately an application command rather than an Alembic migration.
The operation copies rows with generated identities and shared-PK relationships
and is intended to run once against a remote dev database after migrations.
It is idempotent: a second run reuses matching target rows and the target's
existing ACTIVE index.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from sqlalchemy import func, select

from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ChunkEmbedding,
    ContentNode,
    DocumentChunk,
    DocumentGroup,
    DocumentGroupSource,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    IndexDocument,
    IndexVersion,
    IndexVersionStatus,
)
from app.database.session import dispose_engine, get_session_factory


PRODUCTION_GROUP_KEY = "HELP_CHATBOT"
TEST_GROUP_KEY = "HELP_CHATBOT_TEST"
TESTING_STATUS = ChatProfileRevisionStatus.TESTING.value
EMBEDDING_MODEL = "gpt-5.6-terra"
GENERATION_PROMPT_VERSION = "v24"
REWRITE_MODEL = "gpt-5.4-mini"
REWRITE_PROMPT_VERSION = "v7"
TEST_INDEX_VERSION = "bootstrap-help-chatbot-test-v1"


async def _one_group(session, key: str) -> DocumentGroup:
    groups = list(
        (
            await session.scalars(
                select(DocumentGroup).where(DocumentGroup.group_key == key)
            )
        ).all()
    )
    if len(groups) != 1:
        raise RuntimeError(f"{key} 문서 그룹이 정확히 하나가 아닙니다: {len(groups)}")
    return groups[0]


async def _copy_version_rows(
    session,
    production_version: DocumentVersion,
    target_version: DocumentVersion,
) -> None:
    """Copy nodes, chunks, and only referentially valid embeddings."""

    production_nodes = list(
        (
            await session.scalars(
                select(ContentNode)
                .where(ContentNode.document_version_id == production_version.id)
                .order_by(ContentNode.node_order, ContentNode.id)
            )
        ).all()
    )
    target_nodes = list(
        (
            await session.scalars(
                select(ContentNode)
                .where(ContentNode.document_version_id == target_version.id)
                .order_by(ContentNode.node_order, ContentNode.id)
            )
        ).all()
    )
    target_keys = [(node.node_order, node.content_hash) for node in target_nodes]
    if len(target_keys) != len(set(target_keys)):
        raise RuntimeError(
            f"target ContentNode 매핑 키가 중복됩니다: document_version_id={target_version.id}"
        )
    source_keys = [
        (node.node_order, node.content_hash) for node in production_nodes
    ]
    if len(source_keys) != len(set(source_keys)):
        raise RuntimeError(
            f"production ContentNode 매핑 키가 중복됩니다: document_version_id={production_version.id}"
        )
    target_by_key = dict(zip(target_keys, target_nodes))
    node_map: dict[int, ContentNode] = {}
    for source_node in production_nodes:
        target_node = target_by_key.get(
            (source_node.node_order, source_node.content_hash)
        )
        if target_node is None:
            target_node = ContentNode(
                document_version_id=target_version.id,
                parent_node_id=None,
                node_type=source_node.node_type,
                node_path=source_node.node_path,
                node_order=source_node.node_order,
                title=source_node.title,
                normalized_content=source_node.normalized_content,
                source_locator=source_node.source_locator,
                content_hash=source_node.content_hash,
                node_identity_hash=source_node.node_identity_hash,
                node_identity_kind=source_node.node_identity_kind,
                metadata_=source_node.metadata_,
                created_at=source_node.created_at,
            )
            session.add(target_node)
            await session.flush()
        node_map[source_node.id] = target_node

    # Parent IDs reference generated target IDs, so assign them only after every
    # node has been mapped. This also makes reruns safe for nested sections.
    for source_node in production_nodes:
        if source_node.parent_node_id is not None:
            parent = node_map.get(source_node.parent_node_id)
            if parent is None:
                raise RuntimeError(
                    f"ContentNode parent를 매핑하지 못했습니다: {source_node.id}"
                )
            node_map[source_node.id].parent_node_id = parent.id

    for source_node in production_nodes:
        source_chunk = await session.get(DocumentChunk, source_node.id)
        if source_chunk is None:
            raise RuntimeError(f"DocumentChunk가 없습니다: {source_node.id}")
        target_node = node_map[source_node.id]
        target_chunk = await session.get(DocumentChunk, target_node.id)
        if target_chunk is None:
            target_chunk = DocumentChunk(
                id=target_node.id,
                chunking_config_id=source_chunk.chunking_config_id,
                chunk_index=source_chunk.chunk_index,
                token_count=source_chunk.token_count,
                embedding_input_hash=source_chunk.embedding_input_hash,
                keyword_search_text=source_chunk.keyword_search_text,
                created_at=source_chunk.created_at,
            )
            session.add(target_chunk)
            await session.flush()
        elif (
            target_chunk.embedding_input_hash != source_chunk.embedding_input_hash
        ):
            raise RuntimeError(
                f"기존 target Chunk의 embedding 입력이 다릅니다: {target_chunk.id}"
            )

        source_embeddings = list(
            (
                await session.scalars(
                    select(ChunkEmbedding).where(
                        ChunkEmbedding.chunk_id == source_chunk.id
                    )
                )
            ).all()
        )
        for source_embedding in source_embeddings:
            target_embedding = await session.scalar(
                select(ChunkEmbedding).where(
                    ChunkEmbedding.chunk_id == target_chunk.id,
                    ChunkEmbedding.embedding_config_id
                    == source_embedding.embedding_config_id,
                )
            )
            if target_embedding is None:
                session.add(
                    ChunkEmbedding(
                        chunk_id=target_chunk.id,
                        embedding_config_id=source_embedding.embedding_config_id,
                        embedding=list(source_embedding.embedding),
                        embedding_input_hash=source_embedding.embedding_input_hash,
                        created_at=source_embedding.created_at,
                    )
                )
            elif (
                target_embedding.embedding_input_hash
                != source_embedding.embedding_input_hash
            ):
                raise RuntimeError(
                    f"기존 target embedding의 입력이 다릅니다: {target_chunk.id}"
                )


async def _copy_source_and_version(
    session,
    production_source: DocumentSource,
    production_version: DocumentVersion,
    target_group: DocumentGroup,
    target_group_source: DocumentGroupSource,
) -> DocumentVersion:
    target_source = await session.scalar(
        select(DocumentSource).where(
            DocumentSource.document_group_id == target_group.id,
            DocumentSource.document_key == production_source.document_key,
        )
    )
    if target_source is None:
        target_source = DocumentSource(
            document_group_id=target_group.id,
            group_source_id=target_group_source.id,
            document_key=production_source.document_key,
            source_type=production_source.source_type,
            canonical_uri=production_source.canonical_uri,
            title=production_source.title,
            metadata_=production_source.metadata_,
            enabled=True,
            created_at=production_source.created_at,
            updated_at=production_source.updated_at,
        )
        session.add(target_source)
        await session.flush()

    target_version = await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_source_id == target_source.id,
            DocumentVersion.normalized_content_hash
            == production_version.normalized_content_hash,
            DocumentVersion.status == DocumentVersionStatus.READY,
        )
    )
    if target_version is None:
        next_version = await session.scalar(
            select(func.max(DocumentVersion.version_no)).where(
                DocumentVersion.document_source_id == target_source.id
            )
        )
        target_version = DocumentVersion(
            document_source_id=target_source.id,
            version_no=(next_version or 0) + 1,
            raw_content_uri=production_version.raw_content_uri,
            raw_content=production_version.raw_content,
            mime_type=production_version.mime_type,
            raw_content_hash=production_version.raw_content_hash,
            normalized_content_hash=production_version.normalized_content_hash,
            parser_name=production_version.parser_name,
            parser_version=production_version.parser_version,
            status=DocumentVersionStatus.READY,
            source_updated_at=production_version.source_updated_at,
            collected_at=production_version.collected_at,
            created_at=production_version.created_at,
        )
        session.add(target_version)
        await session.flush()

    await _copy_version_rows(session, production_version, target_version)
    return target_version


async def bootstrap() -> tuple[int, int, int]:
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            production_group = await _one_group(session, PRODUCTION_GROUP_KEY)
            target_group = await _one_group(session, TEST_GROUP_KEY)
            profile = await session.scalar(
                select(ChatProfile).where(
                    ChatProfile.profile_key == PRODUCTION_GROUP_KEY
                )
            )
            if profile is None:
                raise RuntimeError("HELP_CHATBOT chat profile이 없습니다.")

            testing_revisions = list(
                (
                    await session.scalars(
                        select(ChatProfileRevision).where(
                            ChatProfileRevision.profile_id == profile.id,
                            ChatProfileRevision.status == TESTING_STATUS,
                        )
                    )
                ).all()
            )
            if len(testing_revisions) > 1:
                raise RuntimeError("HELP_CHATBOT TESTING revision이 둘 이상입니다.")
            if testing_revisions:
                testing_revision = testing_revisions[0]
                testing_revision.document_group_id = target_group.id
                testing_revision.generation_model_name = EMBEDDING_MODEL
                testing_revision.generation_prompt_version = GENERATION_PROMPT_VERSION
                testing_revision.query_rewrite_model_name = REWRITE_MODEL
                testing_revision.query_rewrite_prompt_version = REWRITE_PROMPT_VERSION
                testing_revision.semantic_cache_enabled = False
            else:
                max_version = await session.scalar(
                    select(func.max(ChatProfileRevision.version)).where(
                        ChatProfileRevision.profile_id == profile.id
                    )
                )
                testing_revision = ChatProfileRevision(
                    profile_id=profile.id,
                    version=(max_version or 0) + 1,
                    document_group_id=target_group.id,
                    status=ChatProfileRevisionStatus.TESTING,
                    generation_model_name=EMBEDDING_MODEL,
                    generation_prompt_version=GENERATION_PROMPT_VERSION,
                    query_rewrite_model_name=REWRITE_MODEL,
                    query_rewrite_prompt_version=REWRITE_PROMPT_VERSION,
                    semantic_cache_enabled=False,
                )
                session.add(testing_revision)
                await session.flush()

            target_group_source = await session.scalar(
                select(DocumentGroupSource)
                .where(
                    DocumentGroupSource.document_group_id == target_group.id,
                    DocumentGroupSource.enabled.is_(True),
                )
                .order_by(DocumentGroupSource.id)
            )
            if target_group_source is None:
                raise RuntimeError(
                    "HELP_CHATBOT_TEST의 활성 document_group_source가 없습니다. "
                    "이 명령은 document group/source를 생성하지 않습니다."
                )

            production_index = await session.scalar(
                select(IndexVersion)
                .where(
                    IndexVersion.document_group_id == production_group.id,
                    IndexVersion.status == IndexVersionStatus.ACTIVE,
                )
                .order_by(IndexVersion.id.desc())
            )
            if production_index is None:
                raise RuntimeError("HELP_CHATBOT의 ACTIVE index가 없습니다.")

            production_pairs = list(
                (
                    await session.execute(
                        select(DocumentSource, DocumentVersion)
                        .join(
                            DocumentVersion,
                            DocumentVersion.document_source_id == DocumentSource.id,
                        )
                        .join(
                            IndexDocument,
                            IndexDocument.document_version_id == DocumentVersion.id,
                        )
                        .where(
                            IndexDocument.index_version_id == production_index.id,
                            DocumentVersion.status == DocumentVersionStatus.READY,
                        )
                        .order_by(DocumentSource.id, DocumentVersion.id)
                    )
                ).all()
            )
            if not production_pairs:
                raise RuntimeError("HELP_CHATBOT ACTIVE index에 문서가 없습니다.")

            target_versions = []
            for source, version in production_pairs:
                target_versions.append(
                    await _copy_source_and_version(
                        session,
                        source,
                        version,
                        target_group,
                        target_group_source,
                    )
                )

            target_index = await session.scalar(
                select(IndexVersion).where(
                    IndexVersion.document_group_id == target_group.id,
                    IndexVersion.status == IndexVersionStatus.ACTIVE,
                )
            )
            if target_index is None:
                target_index = await session.scalar(
                    select(IndexVersion).where(
                        IndexVersion.version == TEST_INDEX_VERSION
                    )
                )
                if target_index is None:
                    next_no = await session.scalar(
                        select(func.max(IndexVersion.version_no)).where(
                            IndexVersion.document_group_id == target_group.id
                        )
                    )
                    target_index = IndexVersion(
                        document_group_id=target_group.id,
                        version=TEST_INDEX_VERSION,
                        version_no=(next_no or 0) + 1,
                        status=IndexVersionStatus.ACTIVE,
                        chunking_config_id=production_index.chunking_config_id,
                        embedding_config_id=production_index.embedding_config_id,
                        keyword_config=production_index.keyword_config,
                        fusion_config=production_index.fusion_config,
                        created_at=datetime.now(timezone.utc),
                        activated_at=datetime.now(timezone.utc),
                    )
                    session.add(target_index)
                    await session.flush()
                elif target_index.document_group_id != target_group.id:
                    raise RuntimeError("bootstrap index version이 다른 그룹에 속합니다.")
                else:
                    target_index.status = IndexVersionStatus.ACTIVE
                    target_index.activated_at = datetime.now(timezone.utc)

            for version in target_versions:
                existing = await session.scalar(
                    select(IndexDocument).where(
                        IndexDocument.index_version_id == target_index.id,
                        IndexDocument.document_version_id == version.id,
                    )
                )
                if existing is None:
                    session.add(
                        IndexDocument(
                            index_version_id=target_index.id,
                            document_version_id=version.id,
                        )
                    )

            return profile.id, testing_revision.id, target_index.id


async def _main() -> None:
    try:
        profile_id, revision_id, index_id = await bootstrap()
        print(
            "HELP_CHATBOT_TEST bootstrap 완료: "
            f"profile={profile_id}, testing_revision={revision_id}, index={index_id}"
        )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(_main())
