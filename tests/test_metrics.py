"""ROUGE and BLEU implementations. Cross-checked against rouge-score / NLTK when installed. Run: python tests/test_metrics.py"""
import re, unittest
import _paths
from evaluation import rouge_n, rouge_l, bleu

PAIRS = [("The application could not read data.csv because the file does not exist.", "The application could not read customers.csv because the file does not exist."),
         ("The calculation failed because the divisor was zero.", "The calculation failed because a float value was added to a string."),
         ("JSON parsing failed because a property name was not correctly quoted.", "JSON parsing failed because a property name was not correctly quoted.")]


class TestMetrics(unittest.TestCase):
    def test_identical_and_disjoint(self):
        self.assertAlmostEqual(rouge_l("a b c d", "a b c d"), 1.0)
        self.assertEqual(rouge_l("a b", "c d"), 0.0)
        self.assertAlmostEqual(bleu(["the cat sat on the mat today"], ["the cat sat on the mat today"]), 100.0)
        self.assertEqual(bleu(["x y z w"], ["a b c d"]), 0.0)

    def test_rouge_matches_google_rouge_score(self):
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            self.skipTest("rouge-score not installed")
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        for ref, pred in PAIRS:
            ref_scores = sc.score(ref, pred)
            self.assertAlmostEqual(rouge_n(pred, ref, 1), ref_scores["rouge1"].fmeasure, places=6)
            self.assertAlmostEqual(rouge_n(pred, ref, 2), ref_scores["rouge2"].fmeasure, places=6)
            self.assertAlmostEqual(rouge_l(pred, ref), ref_scores["rougeL"].fmeasure, places=6)

    def test_bleu_matches_nltk_corpus_bleu(self):
        try:
            from nltk.translate.bleu_score import corpus_bleu
        except ImportError:
            self.skipTest("nltk not installed")
        refs, preds = [r for r, _ in PAIRS], [p for _, p in PAIRS]
        tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        expected = 100 * corpus_bleu([[tok(r)] for r in refs], [tok(p) for p in preds])
        self.assertAlmostEqual(bleu(preds, refs), expected, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
