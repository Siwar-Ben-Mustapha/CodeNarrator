"""FastAPI backend: endpoints, validation, error handling. Uses the real saved model. Run: python tests/test_api.py"""
import os, tempfile, unittest
import _paths
from fastapi.testclient import TestClient
from webapp.backend import main

TB = 'Traceback (most recent call last):\n  File "pipeline.py", line 82, in run\n    open("customers.csv")\nFileNotFoundError: [Errno 2] No such file or directory: \'customers.csv\''


@unittest.skipUnless(os.path.exists(_paths.CHECKPOINT), "checkpoints/codenarrator_best.pt not found")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()                       # runs the lifespan: loads the model once

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["status"], "ok")
        self.assertIn("attention", j["model"])
        self.assertGreater(j["parameters"], 1_000_000)

    def test_summarize_real_model_output(self):
        r = self.client.post("/api/summarize", json={"traceback": TB})
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["summary"], "The application could not read customers.csv because the file does not exist.")
        self.assertEqual(len(j["attention"]), len(j["source_tokens"]))
        self.assertEqual(j["warnings"], [])
        self.assertGreater(j["latency_ms"], 0)

    def test_empty_and_whitespace_input_gives_422(self):
        for bad in ("", "   \n  "):
            r = self.client.post("/api/summarize", json={"traceback": bad})
            self.assertEqual(r.status_code, 422)
            self.assertIn("paste", r.json()["detail"].lower())

    def test_invalid_payloads_give_422(self):
        self.assertEqual(self.client.post("/api/summarize", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/summarize", json={"traceback": 123}).status_code, 422)
        self.assertEqual(self.client.post("/api/summarize", content="not json", headers={"Content-Type": "application/json"}).status_code, 422)

    def test_too_long_input_gives_413(self):
        r = self.client.post("/api/summarize", json={"traceback": "x" * (main.MAX_CHARS + 1)})
        self.assertEqual(r.status_code, 413)

    def test_long_but_allowed_input_works(self):
        frames = "".join(f'  File "m{i}.py", line {i}, in f{i}\n    f{i+1}()\n' for i in range(400))
        r = self.client.post("/api/summarize", json={"traceback": "Traceback (most recent call last):\n" + frames + "KeyError: 'email'"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["truncated"])
        self.assertTrue(r.json()["summary"])

    def test_not_a_traceback_returns_a_warning(self):
        j = self.client.post("/api/summarize", json={"traceback": "hello world, nothing failed here"}).json()
        self.assertTrue(j["warnings"])

    def test_repeated_requests_are_identical(self):
        outs = {self.client.post("/api/summarize", json={"traceback": TB}).json()["summary"] for _ in range(5)}
        self.assertEqual(len(outs), 1)

    def test_missing_or_broken_checkpoint_is_reported_clearly(self):
        saved, saved_state = main.CHECKPOINT, dict(main.state)
        try:
            main.CHECKPOINT = main.Path(tempfile.gettempdir()) / "does_not_exist.pt"
            main.load_model()
            r = self.client.get("/api/health")
            self.assertEqual(r.status_code, 503)
            self.assertIn("checkpoint not found", r.json()["detail"])
            r = self.client.post("/api/summarize", json={"traceback": TB})
            self.assertEqual(r.status_code, 503)
        finally:
            main.CHECKPOINT = saved
            main.load_model()
        self.assertEqual(self.client.get("/api/health").status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
