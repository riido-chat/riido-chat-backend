"""DocumentVersion 원문 저장 방식의 로컬 DB 통합 테스트."""

import asyncio
from datetime import datetime, timezone
import hashlib
from typing import Optional
import unittest
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
)


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class DocumentVersionRawContentDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 원문 저장 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(bind=self.connection, expire_on_commit=False)

        suffix = uuid.uuid4().hex
        self.source = DocumentSource(
            source_type="TEST_MARKDOWN",
            canonical_uri=f"https://example.com/{suffix}",
            title="테스트 문서",
        )
        self.session.add(self.source)
        await self.session.flush()

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def test_accepts_uri_only_raw_content_storage(self) -> None:
        version = self._version(raw_content_uri="raw/test.md")

        self.session.add(version)
        await self.session.flush()

        self.assertIsNotNone(version.id)
        self.assertIsNone(version.raw_content)

    async def test_accepts_inline_only_raw_content_storage(self) -> None:
        version = self._version(raw_content="# 관리자 업로드 원문")

        self.session.add(version)
        await self.session.flush()

        self.assertIsNotNone(version.id)
        self.assertIsNone(version.raw_content_uri)

    async def test_accepts_uri_and_inline_raw_content_storage_together(self) -> None:
        version = self._version(
            raw_content_uri="raw/test.md",
            raw_content="# 함께 보존한 원문",
        )

        self.session.add(version)
        await self.session.flush()

        self.assertIsNotNone(version.id)

    async def test_rejects_document_version_without_raw_content_storage(self) -> None:
        version = self._version()

        with self.assertRaises(IntegrityError):
            async with self.session.begin_nested():
                self.session.add(version)
                await self.session.flush()

    def _version(
        self,
        *,
        raw_content_uri: Optional[str] = None,
        raw_content: Optional[str] = None,
    ) -> DocumentVersion:
        raw = raw_content or "URI 원문"
        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        return DocumentVersion(
            document_source_id=self.source.id,
            version_no=1,
            raw_content_uri=raw_content_uri,
            raw_content=raw_content,
            mime_type="text/markdown",
            raw_content_hash=raw_hash,
            normalized_content_hash=raw_hash,
            parser_name="test-parser",
            parser_version="v1",
            status=DocumentVersionStatus.READY,
            collected_at=now,
            created_at=now,
        )


if __name__ == "__main__":
    unittest.main()
