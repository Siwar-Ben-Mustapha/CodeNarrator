"""Training loop: cross-entropy (PAD ignored), Adam, gradient clipping, early stopping, best checkpoint."""
import time, random, math, copy
import numpy as np, torch, torch.nn as nn

DEFAULT_CONFIG = dict(
    emb_dim=128, enc_hidden=256, dec_hidden=256, num_layers=1, bidirectional=True, dropout=0.3, attn_dim=128,
    batch_size=64, lr=1e-3, epochs=30, patience=5, clip=1.0, teacher_forcing=1.0,
    max_src_len=192, max_tgt_len=32, min_freq_src=2, min_freq_tgt=1, normalize_line_numbers=True,
    use_slots=True, p_mask=0.5, p_unk=0.15,      # copy slots / UNK-dropout on frame tokens (see slots.py, dataset.py)
    select_on="familiar",          # which validation loss picks the best epoch: "familiar" | "heldout"
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    """Teacher-forced token-level cross-entropy (mean over non-PAD tokens)."""
    model.eval()
    tot, n = 0.0, 0
    for src, src_len, tgt, _ in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        logits = model(src, src_len, tgt, teacher_forcing_ratio=1.0)
        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        k = (gold != 0).sum().item()
        tot += loss.item() * k; n += k
    return tot / n


def train_model(model, loaders, cfg, device, tag="model", log_fn=print, quick_eval=None):
    """loaders: dict with 'train', 'val_familiar', 'val_heldout'. quick_eval(model)->dict (optional, per epoch).
    Returns (model with best weights loaded, history list)."""
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")      # PAD ignored
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    key = "val_familiar_loss" if cfg["select_on"] == "familiar" else "val_heldout_loss"
    best, best_state, bad, hist, t0 = float("inf"), None, 0, [], time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for src, src_len, tgt, _ in loaders["train"]:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            logits = model(src, src_len, tgt, teacher_forcing_ratio=cfg["teacher_forcing"])
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            k = (gold != 0).sum().item(); tot += loss.item() * k; n += k
        row = {"epoch": ep, "train_loss": tot / n,
               "val_familiar_loss": eval_loss(model, loaders["val_familiar"], criterion, device),
               "val_heldout_loss": eval_loss(model, loaders["val_heldout"], criterion, device) if loaders.get("val_heldout") else 0.0,
               "time_s": time.time() - t0}
        if quick_eval:
            row.update(quick_eval(model))
        hist.append(row)
        improved = row[key] < best - 1e-4
        if improved:
            best, bad, best_state = row[key], 0, copy.deepcopy(model.state_dict())
            row["best"] = True
        else:
            bad += 1
        log_fn(f"[{tag}] ep {ep:02d} train {row['train_loss']:.4f} | val_fam {row['val_familiar_loss']:.4f} "
               f"| val_heldout {row['val_heldout_loss']:.4f}" + (f" | fam_EM {row['val_familiar_em']:.1f}" if quick_eval else "")
               + (" *" if improved else ""))
        if bad >= cfg["patience"]:
            log_fn(f"[{tag}] early stopping at epoch {ep}")
            break
    model.load_state_dict(best_state)
    return model, hist


def save_checkpoint(path, model, cfg, src_vocab, tgt_vocab, use_attention, history=None, extra=None):
    """Self-contained checkpoint: weights + both vocabularies + config (loadable without retraining)."""
    torch.save({"model_state": model.state_dict(), "config": dict(cfg), "use_attention": use_attention,
                "src_vocab": src_vocab.state(), "tgt_vocab": tgt_vocab.state(),
                "history": history, "extra": extra or {}}, path)
