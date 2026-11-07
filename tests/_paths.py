"""Shared path setup for the tests (imported first by every test file)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tests"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402,F401  (import torch before pandas/matplotlib: avoids a Windows DLL-order issue)
CHECKPOINT = os.path.join(ROOT, "checkpoints", "codenarrator_best.pt")
DATASET = os.path.join(ROOT, "academic", "dataset")
