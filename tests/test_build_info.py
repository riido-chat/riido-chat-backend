import unittest

from app.core.build_info import (
    BUILD_VERSION_ENV,
    MAX_BUILD_VERSION_LENGTH,
    UNKNOWN_BUILD_VERSION,
    read_build_version,
)
from app.database.models import RagRun


class ReadBuildVersionTest(unittest.TestCase):
    def test_reads_trimmed_value(self) -> None:
        self.assertEqual("abc123", read_build_version({BUILD_VERSION_ENV: " abc123\n"}))

    def test_missing_or_blank_is_unknown(self) -> None:
        for environ in ({}, {BUILD_VERSION_ENV: ""}, {BUILD_VERSION_ENV: "   "}):
            with self.subTest(environ=environ):
                self.assertEqual(UNKNOWN_BUILD_VERSION, read_build_version(environ))

    def test_long_value_is_cut_to_column_length(self) -> None:
        value = read_build_version({BUILD_VERSION_ENV: "x" * (MAX_BUILD_VERSION_LENGTH + 5)})

        self.assertEqual(MAX_BUILD_VERSION_LENGTH, len(value))
        self.assertEqual(MAX_BUILD_VERSION_LENGTH, RagRun.__table__.c.build_version.type.length)


if __name__ == "__main__":
    unittest.main()
