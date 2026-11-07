"""Run every test file in its own process and print a pass/fail/skip summary (also saved to results/test_report.txt).
    python tests/run_all.py            (about 2 minutes on CPU; needs the checkpoint and the dataset CSVs, not the GPU)"""
import os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = ["test_preprocessing.py", "test_metrics.py", "test_slots.py", "test_sanity.py", "test_inference.py", "test_api.py", "test_consistency.py"]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
rows, ok_all = [], True
for f in FILES:
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, f)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
    out = p.stdout + p.stderr
    ran = re.search(r"Ran (\d+) tests?", out)
    skipped = re.search(r"skipped=(\d+)", out)
    n = int(ran.group(1)) if ran else (out.count("PASS ") or (1 if p.returncode == 0 else 0))
    status = "PASSED" if p.returncode == 0 else "FAILED"
    ok_all &= p.returncode == 0
    rows.append((f, status, n, int(skipped.group(1)) if skipped else 0, time.time() - t0))
    if p.returncode != 0:
        print(out[-3000:])
lines = [f"{f:26s} {s:7s} checks/tests: {n:3d}  skipped: {k}  ({sec:.0f}s)" for f, s, n, k, sec in rows]
lines.append("ALL TEST FILES PASSED" if ok_all else "SOME TESTS FAILED")
print("\n".join(lines))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
open(os.path.join(ROOT, "results", "test_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
sys.exit(0 if ok_all else 1)
