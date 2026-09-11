"""Regression checks for the one-time testing corpus bootstrap boundary."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestingCorpusBootstrapTest(unittest.TestCase):
    def test_sql_is_transactional_and_refuses_to_create_test_group(self):
        sql = (ROOT / "ops" / "bootstrap_testing_corpus.sql").read_text()
        self.assertIn("after Alembic 20260912_12", sql)
        self.assertIn("BEGIN;", sql)
        self.assertIn("COMMIT;", sql)
        self.assertIn("refusing to create it", sql)
        self.assertNotIn("INSERT INTO document_groups", sql)
        self.assertIn("generation_prompt_version = 'v38'", sql)
        self.assertIn("query_rewrite_prompt_version = 'v8'", sql)

    def test_application_command_has_no_alembic_dependency_and_checks_target_source(self):
        source = (ROOT / "app" / "ops" / "bootstrap_testing_corpus.py").read_text()
        tree = ast.parse(source)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("alembic", imports)
        self.assertIn("HELP_CHATBOT_TEST", source)
        self.assertIn("활성 document_group_source가 없습니다", source)
        self.assertIn("IndexVersionStatus.ACTIVE", source)
        self.assertIn('GENERATION_PROMPT_VERSION = "v38"', source)
        self.assertIn('REWRITE_PROMPT_VERSION = "v8"', source)


if __name__ == "__main__":
    unittest.main()
