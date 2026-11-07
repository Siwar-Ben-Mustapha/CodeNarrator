"""Saved checkpoint -> real inference: loading, eval mode, determinism, error handling, copy slots. Run: python tests/test_inference.py"""
import os, tempfile, unittest
import _paths
import torch
from inference import Summarizer, load_checkpoint

TB = 'Traceback (most recent call last):\n  File "a.py", line 3, in f\n    {code}\n{exc}'


@unittest.skipUnless(os.path.exists(_paths.CHECKPOINT), "checkpoints/codenarrator_best.pt not found")
class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = Summarizer(_paths.CHECKPOINT, "cpu")

    def test_checkpoint_is_the_attention_model_with_matching_vocab_and_config(self):
        ck = self.S.ck
        self.assertTrue(ck["use_attention"])
        self.assertFalse(self.S.model.training)
        self.assertEqual(len(self.S.sv), len(ck["src_vocab"]))
        self.assertEqual(len(self.S.tv), len(ck["tgt_vocab"]))
        self.assertEqual([self.S.sv.itos[i] for i in range(4)], ["<PAD>", "<UNK>", "<SOS>", "<EOS>"])
        self.assertTrue(ck["config"]["use_slots"])
        self.assertIn("<S1>", self.S.sv.stoi)
        self.assertIn("<S1>", self.S.tv.stoi)
        self.assertEqual(self.S.model.num_parameters(), ck["extra"]["params"])

    def test_deterministic(self):
        t = TB.format(code='row["email"]', exc="KeyError: 'email'")
        self.assertEqual(len({self.S(t) for _ in range(10)}), 1)

    def test_known_example(self):
        t = TB.format(code='open("customers.csv")', exc="FileNotFoundError: [Errno 2] No such file or directory: 'customers.csv'")
        self.assertEqual(self.S(t), "The application could not read customers.csv because the file does not exist.")

    def test_copy_slots_preserve_unseen_identifiers(self):
        for ident in ("qty_in_stock_2024", "zzTotal9", "revenue_eur"):
            self.assertEqual(self.S.sv.stoi.get(ident, self.S.sv.unk_id), self.S.sv.unk_id, "identifier must be unknown to the vocabulary")
            out = self.S(TB.format(code=f'row["{ident}"]', exc=f"KeyError: '{ident}'"))
            self.assertIn(ident, out)

    def test_empty_input_is_rejected(self):
        for bad in ("", "   ", "\n\n"):
            with self.assertRaises(ValueError):
                self.S(bad)

    def test_output_is_never_empty_for_error_lines(self):
        for line in ("ZeroDivisionError: division by zero", "KeyError: 'username'", "IndexError: list index out of range"):
            self.assertTrue(self.S(line).strip())

    def test_incompatible_or_foreign_checkpoints_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "foreign.pt")
            torch.save({"hello": 1}, p)
            with self.assertRaisesRegex(ValueError, "not a CodeNarrator checkpoint"):
                load_checkpoint(p)
            ck = torch.load(_paths.CHECKPOINT, map_location="cpu", weights_only=False)
            ck["config"] = dict(ck["config"], enc_hidden=64)               # architecture no longer matches the weights
            p2 = os.path.join(d, "mismatch.pt")
            torch.save(ck, p2)
            with self.assertRaises(RuntimeError):
                load_checkpoint(p2)

    def test_warnings(self):
        self.assertEqual(self.S.warnings("KeyError: 'x'"), [])
        self.assertTrue(self.S.warnings("PaymentError: card declined"))        # exception class never seen in training
        self.assertTrue(self.S.warnings("hello world, nothing failed here"))   # not an exception line


if __name__ == "__main__":
    unittest.main(verbosity=2)
