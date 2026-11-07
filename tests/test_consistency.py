"""Consistency: the training/evaluation pipeline, the direct inference function, the FastAPI backend and the final notebook must
all give IDENTICAL predictions for the same checkpoint. Run: python tests/test_consistency.py"""
import json, os, re, unittest
import _paths
import torch
from fastapi.testclient import TestClient
from webapp.backend import main
from inference import Summarizer
from train import DEFAULT_CONFIG
from pipeline import prepare_data
from evaluation import generate_dataset
from inference_cases import CASES

NOTEBOOK = os.path.join(_paths.ROOT, "academic", "CodeNarrator_Final.ipynb")


@unittest.skipUnless(os.path.exists(_paths.CHECKPOINT), "checkpoints/codenarrator_best.pt not found")
class TestConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = Summarizer(_paths.CHECKPOINT, "cpu")
        cls.data = prepare_data(_paths.DATASET, dict(DEFAULT_CONFIG, **cls.S.cfg))

    def test_batched_evaluation_pipeline_equals_single_request_inference(self):
        """generate_dataset (padded batches, Dataset preprocessing) vs summarize (one request, raw text) on 150 rows per split."""
        for split, ds in self.data["ds"].items():
            if split == "train":
                continue
            batched = generate_dataset(self.S.model, ds, self.S.tv, "cpu")
            for i in range(0, len(ds), max(1, len(ds) // 150)):
                self.assertEqual(batched[i], self.S(ds.df.input_text[i]), f"{split} row {i} differs between the two pipelines")

    def test_vocabularies_equal_those_rebuilt_from_train_csv(self):
        self.assertEqual(self.data["src_vocab"].itos, self.S.sv.itos)
        self.assertEqual(self.data["tgt_vocab"].itos, self.S.tv.itos)

    def test_fastapi_equals_direct_inference_on_all_inference_cases(self):
        with TestClient(main.app) as client:
            for name, kind, text, _ in CASES:
                r = client.post("/api/summarize", json={"traceback": text})
                self.assertEqual(r.status_code, 200, name)
                self.assertEqual(r.json()["summary"], self.S(text), f"{name}: API and direct inference differ")

    @unittest.skipUnless(os.path.exists(NOTEBOOK), "final notebook not built yet")
    def test_notebook_contains_exactly_the_current_source_files(self):
        nb = json.load(open(NOTEBOOK, encoding="utf-8"))
        nb_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        norm = lambda s: re.sub(r"\s+", " ", s)
        for f in ("preprocessing", "vocab", "copy_slots", "dataset", "encoder", "attention", "decoder", "seq2seq", "train", "evaluation", "pipeline", "inference"):
            src = open(os.path.join(_paths.ROOT, "src", f + ".py"), encoding="utf-8").read()
            body = [ln for ln in src.splitlines() if not re.match(r"^(from|import) (preprocessing|vocab|copy_slots|dataset|encoder|attention|decoder|seq2seq|train|evaluation|pipeline|inference)\b", ln)]
            body = "\n".join(body).split('if __name__ == "__main__":')[0]
            code_lines = [ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith(("#", '"""'))]
            for ln in code_lines[:400]:
                self.assertIn(norm(ln.strip()), norm(nb_code), f"notebook is out of sync with src/{f}.py: {ln.strip()[:80]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
