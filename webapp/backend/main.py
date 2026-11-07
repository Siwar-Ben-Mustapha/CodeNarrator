"""FastAPI backend for CodeNarrator: real inference with the saved checkpoint (no retraining, no explanation lookup, no LLM API).

Run from the CodeNarrator folder:
    python -m uvicorn webapp.backend.main:app --port 8000
Environment variables (optional): CODENARRATOR_CHECKPOINT (path to a .pt file), CODENARRATOR_DEVICE (cpu | cuda).
The model, vocabularies and copy-slot logic come from src/ (the same code as the notebook and the tests).
"""
import os, sys, time, logging
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                     # .../CodeNarrator
sys.path.insert(0, str(ROOT / "src"))

import torch                                                    # noqa: E402  (import torch first: avoids a Windows DLL-order issue)
from fastapi import FastAPI, HTTPException                      # noqa: E402
from fastapi.middleware.cors import CORSMiddleware              # noqa: E402
from fastapi.staticfiles import StaticFiles                     # noqa: E402
from pydantic import BaseModel                                  # noqa: E402

from inference import Summarizer                                # noqa: E402

log = logging.getLogger("codenarrator")
CHECKPOINT = Path(os.environ.get("CODENARRATOR_CHECKPOINT", ROOT / "checkpoints" / "codenarrator_best.pt"))
DEVICE = os.environ.get("CODENARRATOR_DEVICE", "cpu")
MAX_CHARS = 20000
state = {"summarizer": None, "error": None}


def load_model():
    """Load the checkpoint once. A missing or incompatible file is reported by the API instead of crashing the server."""
    state["summarizer"], state["error"] = None, None
    try:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT}")
        state["summarizer"] = Summarizer(CHECKPOINT, DEVICE)
    except Exception as e:                                      # noqa: BLE001
        state["error"] = f"{type(e).__name__}: {e}"
        log.error("could not load the model: %s", state["error"])


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(title="CodeNarrator API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class SummarizeRequest(BaseModel):
    traceback: str


@app.get("/api/health")
def health():
    s = state["summarizer"]
    if s is None:
        raise HTTPException(503, f"Model not loaded - {state['error']}")
    return {"status": "ok", "device": DEVICE,
            "model": "BiLSTM encoder-decoder + Bahdanau attention" if s.ck["use_attention"] else "BiLSTM encoder-decoder (no attention)",
            "parameters": s.model.num_parameters(), "src_vocab": len(s.sv), "tgt_vocab": len(s.tv),
            "checkpoint": CHECKPOINT.name, "seed": s.ck.get("extra", {}).get("seed")}


@app.post("/api/summarize")
def summarize_endpoint(req: SummarizeRequest):
    s = state["summarizer"]
    if s is None:
        raise HTTPException(503, f"Model not loaded - {state['error']}")
    text = req.traceback.strip()
    if not text:
        raise HTTPException(422, "Please paste a Python error or traceback.")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"Input too long ({len(text):,} characters, maximum {MAX_CHARS:,}).")
    try:
        t0 = time.perf_counter()
        summary, att, source_tokens, n_unk = s(text, return_attention=True)
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:                                      # noqa: BLE001
        log.exception("inference failed")
        raise HTTPException(500, f"Inference failed ({type(e).__name__}).") from e
    weights = att[0].mean(0).tolist() if att is not None else []   # attention averaged over output steps, one value per source token
    top = max(weights) if weights else 0
    weights = [w / top for w in weights] if top > 0 else weights
    return {"summary": summary, "latency_ms": round(latency, 1), "warnings": s.warnings(text), "source_tokens": source_tokens,
            "attention": weights, "n_unk": n_unk, "truncated": len(source_tokens) >= s.cfg["max_src_len"]}


# Optional: serve the built React app from the same server (after `npm run build` in webapp/frontend) -> one URL for everything.
DIST = ROOT / "webapp" / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
