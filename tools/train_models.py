"""Train baseline + attention (local GPU or CPU). Outputs go to checkpoints/<tag>/ and results/<tag>/ so existing files are never overwritten.

    python tools/train_models.py --tag my_run [--seeds 42 43 44] [--epochs 30] [--patience 5] [--only baseline]
The best model of the run is chosen on VALIDATION only (never on a test set) and also saved as checkpoints/<tag>/codenarrator_best.pt."""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import torch
import pandas as pd
from train import DEFAULT_CONFIG, save_checkpoint
from pipeline import prepare_data, run_model
from evaluation import rouge_l
from inference import summarize
from probe_cases import PROBES

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tag", required=True, help="name of this experiment (folder name under results/ and checkpoints/)")
ap.add_argument("--data", default=os.path.join(ROOT, "academic", "dataset"))
ap.add_argument("--only", choices=["baseline", "attention"], help="train just one architecture")
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--max_src_len", type=int, default=192)
ap.add_argument("--no_slots", action="store_true", help="ablation: disable copy slots + augmentation")
args = ap.parse_args()
args.out, args.ckpt = (os.path.join(ROOT, d, args.tag) for d in ("results", "checkpoints"))
assert not os.path.exists(args.ckpt), f"{args.ckpt} already exists - choose another --tag (existing checkpoints are never overwritten)"
os.makedirs(args.out, exist_ok=True); os.makedirs(args.ckpt, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = dict(DEFAULT_CONFIG, epochs=args.epochs, patience=args.patience, max_src_len=args.max_src_len, use_slots=not args.no_slots, p_unk=0.0 if args.no_slots else 0.15)
print("device", device, "| config", cfg)
data = prepare_data(args.data, cfg)
print("vocab src/tgt", len(data["src_vocab"]), len(data["tgt_vocab"]))

runs, hists, preds_all = [], {}, {}
for seed in args.seeds:
    for name, att in [("baseline", False), ("attention", True)]:
        if args.only and name != args.only: continue
        model, hist, info, preds = run_model(name, att, seed, data, cfg, device, ckpt_dir=args.ckpt)
        print(f"==> {name} seed {seed}: {info['params']:,} params, {info['train_time_s']:.0f}s, best epoch {info['best_epoch']}")
        runs.append(info); hists[(name, seed)] = hist; preds_all[(name, seed)] = preds
        probe = [summarize(model, data["src_vocab"], data["tgt_vocab"], cfg, p[1], device) for p in PROBES]
        info["probe_rougeL"] = 100 * sum(rouge_l(a, p[2]) for a, p in zip(probe, PROBES)) / len(PROBES)
        info["probe_preds"] = probe

# ---------- tables
rows = [dict(model=r["name"], seed=r["seed"], split=sp, params=r["params"], train_time_s=r["train_time_s"],
             best_epoch=r["best_epoch"], **m) for r in runs for sp, m in r["results"].items()]
allruns = pd.DataFrame(rows); allruns.to_csv(f"{args.out}/metrics_all_runs.csv", index=False)
summ = allruns.groupby(["split", "model"])[["rouge1", "rouge2", "rougeL", "exact_match", "token_acc", "entity_recall"]].agg(["mean", "std"]).round(2)
summ.to_csv(f"{args.out}/metrics_summary.csv"); print(summ.to_string())
pd.DataFrame([dict(model=r["name"], seed=r["seed"], probe_rougeL=r["probe_rougeL"]) for r in runs]).to_csv(f"{args.out}/probe_scores.csv", index=False)
pd.DataFrame([dict(model=r["name"], seed=r["seed"], case=p[0], reference=p[2], prediction=pred)
              for r in runs for p, pred in zip(PROBES, r["probe_preds"])]).to_csv(f"{args.out}/probe_predictions.csv", index=False)
pd.concat([pd.DataFrame(h).assign(model=k[0], seed=k[1]) for k, h in hists.items()]).to_csv(f"{args.out}/training_history.csv", index=False)
for (name, seed), pr in preds_all.items():
    for sp in ("test_familiar", "test_novel", "test_unseen"):
        data["frames"][sp].assign(pred=pr[sp]).to_csv(f"{args.out}/pred_{name}_s{seed}_{sp}.csv", index=False)

# ---------- best model on VALIDATION ROUGE-L (tie -> lower validation loss), exported as one portable file
def val_key(r):
    best_ep = [h for h in hists[(r["name"], r["seed"])] if h.get("best")][-1]
    return (-r["results"]["val_familiar"]["rougeL"], best_ep["val_familiar_loss"])
best = min(runs, key=val_key)
src = f"{args.ckpt}/{best['name']}_seed{best['seed']}.pt"
ck = torch.load(src, map_location="cpu", weights_only=False)
torch.save(ck, f"{args.ckpt}/codenarrator_best.pt")
json.dump(dict(config=cfg, seeds=args.seeds, best=dict(name=best["name"], seed=best["seed"])), open(f"{args.out}/config_and_best.json", "w"), indent=1)
print("BEST (validation):", best["name"], best["seed"])
