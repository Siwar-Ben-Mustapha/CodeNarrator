"""Glue code: data preparation, training one model, loading a finished run. Shared by the notebook, the scripts and the tests."""
import time
import torch
from dataset import load_splits, build_vocabs, Seq2SeqDataset, make_loader
from seq2seq import Seq2Seq
from train import train_model, set_seed, save_checkpoint
from evaluation import generate_dataset, score, common_target_words
from inference import load_checkpoint


def prepare_data(data_dir, cfg):
    """CSV files -> dataframes, vocabularies (from train.csv ONLY) and datasets for every split."""
    frames = load_splits(data_dir)
    use_slots = cfg.get("use_slots", False)
    sv, tv = build_vocabs(frames["train"], cfg["min_freq_src"], cfg["min_freq_tgt"], cfg["normalize_line_numbers"], use_slots)
    make = lambda split, df: Seq2SeqDataset(df, sv, tv, cfg["max_src_len"], cfg["max_tgt_len"], cfg["normalize_line_numbers"],
                                            use_slots=use_slots, train=(split == "train"),
                                            p_mask=cfg.get("p_mask", .5), p_unk=cfg.get("p_unk", .15))
    ds = {split: make(split, df) for split, df in frames.items()}
    return dict(frames=frames, ds=ds, src_vocab=sv, tgt_vocab=tv, common=common_target_words(frames["train"]))


def make_loaders(data, cfg, seed):
    return {"train": make_loader(data["ds"]["train"], cfg["batch_size"], True, seed),
            "val_familiar": make_loader(data["ds"]["val_familiar"], 256, False)}


def evaluate_model(model, data, device):
    """Greedy generation + metrics on every evaluation split. Returns (results, predictions)."""
    results, preds = {}, {}
    for split in [k for k in data["ds"] if k != "train"]:
        preds[split] = generate_dataset(model, data["ds"][split], data["tgt_vocab"], device)
        results[split] = score(preds[split], data["frames"][split], data["common"])
    return results, preds


def run_model(name, use_attention, seed, data, cfg, device, ckpt_dir=None, log_fn=print):
    """Train one model, keep the best epoch (validation loss), evaluate on all splits, optionally save a checkpoint."""
    set_seed(seed)
    model = Seq2Seq(len(data["src_vocab"]), len(data["tgt_vocab"]), cfg, use_attention).to(device)
    n_params = model.num_parameters()
    loaders = make_loaders(data, cfg, seed)

    def quick(m):                                    # cheap per-epoch generation check (validation set, real greedy decoding)
        p = generate_dataset(m, data["ds"]["val_familiar"], data["tgt_vocab"], device)
        s = score(p, data["frames"]["val_familiar"], data["common"])
        return {"val_familiar_em": s["exact_match"], "val_familiar_rougeL": s["rougeL"]}

    t0 = time.time()
    model, hist = train_model(model, loaders, cfg, device, tag=f"{name}-s{seed}", log_fn=log_fn, quick_eval=quick)
    train_time = time.time() - t0
    results, preds = evaluate_model(model, data, device)
    info = dict(name=name, seed=seed, use_attention=use_attention, params=n_params, train_time_s=train_time,
                epochs_run=len(hist), best_epoch=[r["epoch"] for r in hist if r.get("best")][-1], results=results)
    if ckpt_dir:
        path = f"{ckpt_dir}/{name}_seed{seed}.pt"
        save_checkpoint(path, model, cfg, data["src_vocab"], data["tgt_vocab"], use_attention, hist, extra=info)
        info["checkpoint"] = path
    return model, hist, info, preds


def load_run(path, data, cfg, device):
    """Reuse a finished run: weights from the checkpoint; metrics/predictions are recomputed (greedy decoding is deterministic)."""
    model, sv, tv, ck = load_checkpoint(path, device)
    assert sv.itos == data["src_vocab"].itos and tv.itos == data["tgt_vocab"].itos, \
        f"{path}: vocabulary differs from the current dataset/config (retrain or fix the config)"
    assert ck["config"].get("use_slots") == cfg.get("use_slots"), f"{path}: trained with a different copy-slot setting"
    info = dict(ck["extra"]); info["checkpoint"] = path
    info["results"], preds = evaluate_model(model, data, device)
    return model, ck["history"], info, preds
