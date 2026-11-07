"""Tokenisation, detokenisation and the inference-time input adapters. Run: python tests/test_preprocessing.py"""
import unittest
import _paths
import pandas as pd
from preprocessing import (tokenize_source, tokenize_target, detokenize_target, ensure_traceback_header, exception_name, HEADER,
                           normalize_source)


class TestPreprocessing(unittest.TestCase):
    def test_target_tokenisation_is_exactly_reversible_on_every_dataset_file(self):
        for name in ("train", "validation", "test", "test_novel_entities", "test_unseen_families"):
            df = pd.read_csv(f"{_paths.DATASET}/{name}.csv")
            bad = [t for t in df.target_text if detokenize_target(tokenize_target(t)) != t]
            self.assertEqual(bad, [], f"{name}: {len(bad)} targets are not reversible, e.g. {bad[:1]}")

    def test_line_numbers_become_placeholder_but_message_numbers_stay(self):
        toks = tokenize_source('Traceback (most recent call last):\n  File "a.py", line 123, in f\n    x\nValueError: expected 3, got 2')
        self.assertIn("<LN>", toks)
        self.assertNotIn("123", toks)
        self.assertIn("3", toks)

    def test_dotted_names_are_split_and_newlines_marked(self):
        self.assertEqual(tokenize_source("a\nb.c"), ["a", "<NL>", "b", ".", "c"])
        self.assertEqual(tokenize_target("Could not read customers.csv."), ["Could", "not", "read", "customers", ".", "csv", "."])

    def test_header_adapter_wraps_bare_lines_only(self):
        self.assertEqual(ensure_traceback_header("KeyError: 'x'"), HEADER + "\nKeyError: 'x'")
        full = HEADER + '\n  File "a.py", line 1, in f\n    g()\nKeyError: \'x\''
        self.assertEqual(ensure_traceback_header(full), full)

    def test_exception_name(self):
        self.assertEqual(exception_name("Traceback...\nsqlite3.IntegrityError: UNIQUE constraint failed"), "sqlite3.IntegrityError")
        self.assertEqual(exception_name("KeyboardInterrupt"), "KeyboardInterrupt")
        self.assertIsNone(exception_name("hello world, nothing failed"))
        self.assertIsNone(exception_name("   "))

    def test_indentation_is_ignored(self):
        self.assertEqual(normalize_source("  a\n      b"), "a <NL> b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
