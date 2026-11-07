"""Re-evaluate saved checkpoints on all test sets (real greedy generation, no reference tokens) and write metric tables.

    python tools/evaluate_checkpoints.py [--ckpt checkpoints] [--out results/main_run]
Uses <model>_seed<k>.pt files found in the checkpoint folder. Nothing is retrained. Metrics: ROUGE-1/2/L, BLEU-4, exact match,
token accuracy, entity recall (share of copyable details such as file/column names that the summary reproduces)."""
import argparse, glob, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch
import pandas as pd
from train import DEFAULT_CONFIG
from pipeline import prepare_data, load_run

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "main_run"))
ap.add_argument("--data", default=os.path.join(ROOT, "academic", "dataset"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

device = torch.device("cpu")
files = sorted(glob.glob(f"{args.ckpt}/*_seed*.pt"))
assert files, f"no <model>_seed<k>.pt in {args.ckpt}"
rows = []
for f in files:
    name, seed = re.match(r"(.+)_seed(\d+)\.pt", os.path.basename(f)).groups()
    ck = torch.load(f, map_location="cpu", weights_only=False)
    cfg = dict(DEFAULT_CONFIG, **ck["config"])                                  # exactly the settings the checkpoint was trained with
    data = prepare_data(args.data, cfg)
    model, hist, info, preds = load_run(f, data, cfg, device)
    print(f"{name} seed {seed}: " + " | ".join(f"{s} ROUGE-L {m['rougeL']:.1f}" for s, m in info["results"].items()), flush=True)
    for split, m in info["results"].items():
        rows.append(dict(model=name, seed=int(seed), split=split, params=info["params"], best_epoch=info["best_epoch"], **m))
    for split, p in preds.items():
        data["frames"][split].assign(prediction=p).to_csv(f"{args.out}/predictions_{name}_seed{seed}_{split}.csv", index=False, encoding="utf-8")

allruns = pd.DataFrame(rows)
allruns.to_csv(f"{args.out}/metrics_all_runs.csv", index=False)
metrics = ["rouge1", "rouge2", "rougeL", "bleu", "exact_match", "token_acc", "entity_recall"]
summary = allruns.groupby(["split", "model"])[metrics].agg(["mean", "std"]).round(1)
summary.to_csv(f"{args.out}/metrics_summary.csv")
pd.set_option("display.width", 250)
print(allruns.groupby(["split", "model"])[metrics].agg(lambda x: f"{x.mean():.1f}+-{x.std():.1f}" if len(x) > 1 else f"{x.mean():.1f}").to_string())
