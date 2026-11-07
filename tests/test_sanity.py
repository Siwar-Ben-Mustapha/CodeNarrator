"""Model/pipeline sanity checks (shapes, masks, teacher forcing, loss, overfit, save/load). Run:  python tests/test_sanity.py [DATA_DIR]"""
import sys, tempfile, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, torch.nn as nn
from train import DEFAULT_CONFIG, set_seed, save_checkpoint
from pipeline import prepare_data
from seq2seq import Seq2Seq
from inference import load_checkpoint, summarize
from preprocessing import tokenize_target, detokenize_target
from dataset import collate

DATA = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).resolve().parents[1] / "academic" / "dataset")
cfg = dict(DEFAULT_CONFIG, emb_dim=32, enc_hidden=48, dec_hidden=48, attn_dim=32, epochs=1)
dev = torch.device("cpu")
set_seed(0)
data = prepare_data(DATA, cfg)
sv, tv = data["src_vocab"], data["tgt_vocab"]
ok = lambda m: print("PASS", m)

# 1. vocabularies: special ids, train-only
assert [sv.itos[i] for i in range(4)] == ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] == [tv.itos[i] for i in range(4)]
assert "FileNotFoundError" in sv.stoi
def words(df): return set(w for t in df.target_text for w in tokenize_target(t))
held_only = (words(data["frames"]["test_unseen"]) | words(data["frames"]["test_novel"])) - words(data["frames"]["train"])
assert held_only and all(w not in tv.stoi for w in held_only), "test-only words leaked into target vocab"
ok(f"vocab ids / train-only (src={len(sv)}, tgt={len(tv)}, {len(held_only)} test-only target words absent from the vocabulary)")
for k in data["ds"]:
    d = data["ds"][k]
    print(f"   {k:14s} n={len(d):5d} sources longer than max_src_len={d.src_trunc} targets truncated={d.tgt_trunc}")
assert all(d.tgt_trunc == 0 for d in data["ds"].values())
# no leakage: no input or (input, target) pair of an evaluation split occurs in train
tr_inputs = set(data["frames"]["train"].input_text)
for k in ["val_familiar", "test_familiar", "test_novel", "test_unseen"]:
    assert not (set(data["frames"][k].input_text) & tr_inputs), k
fam = lambda k: set(data["frames"][k].scenario_id)
assert not (fam("test_unseen") & fam("train")), "unseen families must not occur in train"
assert fam("val_familiar") <= fam("train") and fam("test_familiar") <= fam("train")
ok("no input overlap between train and any evaluation split; unseen families disjoint from train")

# 2. batch shapes + decoder shifting
batch = collate([data["ds"]["train"][i] for i in range(8)])
src, src_len, tgt, _ = batch
assert src.shape[0] == tgt.shape[0] == 8 and (src_len == (src != 0).sum(1)).all()
assert (tgt[:, 0] == 2).all()
for row in tgt:
    nz = row[row != 0]; assert nz[-1] == 3 and (nz[:-1] != 3).all()
ok(f"batch shapes src{tuple(src.shape)} tgt{tuple(tgt.shape)}; target starts <SOS>, ends <EOS>, padded with 0")

for att in [False, True]:
    name = "attention" if att else "baseline"
    set_seed(1)
    m = Seq2Seq(len(sv), len(tv), cfg, att).to(dev)
    logits = m(src, src_len, tgt, 1.0)
    assert logits.shape == (8, tgt.size(1) - 1, len(tv))
    ok(f"[{name}] forward shape {tuple(logits.shape)} = (B, T-1, V); decoder input t uses tgt[:, t-1] -> predicts tgt[:, t]")

    # encoder shapes
    enc_out, (h, c), mask = m.encode(src, src_len)
    assert enc_out.shape == (8, src.size(1), m.encoder.enc_dim) and h.shape == (cfg["num_layers"], 8, cfg["dec_hidden"]) == c.shape
    assert torch.equal(mask, src != 0)
    ok(f"[{name}] encoder outputs {tuple(enc_out.shape)}, h/c {tuple(h.shape)}")

    # 3. padding invariance: extra PAD columns must not change the result
    m.eval()
    s2 = torch.cat([src, torch.zeros(8, 10, dtype=torch.long)], 1)
    a = m(src, src_len, tgt, 1.0); b = m(s2, src_len, tgt, 1.0)
    assert torch.allclose(a, b, atol=1e-5), "padding changes outputs!"
    ok(f"[{name}] extra padding does not change logits (max diff {(a-b).abs().max():.2e})")

    # 4. teacher forcing: ratio 0 must differ from ratio 1 and must not read gold tokens after step 0
    random_state = torch.get_rng_state()
    l1 = m(src, src_len, tgt, 1.0); l0 = m(src, src_len, tgt, 0.0)
    tgt_corrupt = tgt.clone(); tgt_corrupt[:, 1:] = 5
    l0c = m(src, src_len, tgt_corrupt, 0.0)
    assert not torch.allclose(l1, l0) and torch.allclose(l0, l0c, atol=1e-6)
    ok(f"[{name}] tf=0 ignores gold tokens after <SOS> (identical logits when gold is corrupted); tf=1 differs")

    # 5. greedy generation is independent of any reference & ends at EOS / max_len
    g = m.generate(src, src_len, max_len=12)
    assert g.shape[0] == 8 and g.shape[1] <= 12
    ok(f"[{name}] greedy generation shape {tuple(g.shape)} (no target argument exists)")

    # 6. loss ignores PAD
    crit = nn.CrossEntropyLoss(ignore_index=0)
    gold = tgt[:, 1:]
    full = crit(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
    keep = gold != 0
    manual = nn.functional.cross_entropy(logits[keep], gold[keep])
    assert torch.allclose(full, manual, atol=1e-6)
    ok(f"[{name}] cross-entropy ignores PAD (loss {full.item():.3f} == manual non-pad mean)")

    # 7. attention masking
    if att:
        w = m.decoder.attention
        enc_out, (h, c), mask = m.encode(src, src_len)
        ctx, weights = w(h[-1], enc_out, mask)
        assert weights.shape == src.shape and ctx.shape == (8, m.encoder.enc_dim)
        assert (weights[~mask] == 0).all() and torch.allclose(weights.sum(1), torch.ones(8), atol=1e-5)
        manual_ctx = (weights.unsqueeze(-1) * enc_out).sum(1)
        assert torch.allclose(ctx, manual_ctx, atol=1e-5)
        ok("[attention] weights sum to 1, exactly 0 on PAD positions, context = weighted sum of encoder states")

    # 8. overfit a tiny batch (should reach ~0 loss, and greedy output == reference)
    set_seed(2)
    m = Seq2Seq(len(sv), len(tv), dict(cfg, dropout=0.0), att).to(dev)
    tiny = collate([data["ds"]["train"][i] for i in range(16)])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for step in range(600):
        m.train(); lg = m(tiny[0], tiny[1], tiny[2], 1.0)
        loss = crit(lg.reshape(-1, lg.size(-1)), tiny[2][:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    ids = m.generate(tiny[0], tiny[1], max_len=32)
    exact = sum(detokenize_target(tv.decode(r.tolist())) == detokenize_target(tv.decode(t.tolist()))
                for r, t in zip(ids, tiny[2]))
    assert loss.item() < 0.1 and exact >= 13, (loss.item(), exact)
    ok(f"[{name}] tiny-batch overfit: loss {loss.item():.4f}, greedy exact matches {exact}/16")

    # 9. save / load round trip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        save_checkpoint(p, m, dict(cfg, dropout=0.0), sv, tv, att)
        m2, sv2, tv2, ck = load_checkpoint(p)
        txt = data["frames"]["train"].input_text.iloc[0]
        assert summarize(m2, sv2, tv2, ck["config"], txt) == summarize(m.cpu(), sv, tv, dict(cfg), txt)
        assert sv2.itos == sv.itos and tv2.itos == tv.itos
    ok(f"[{name}] checkpoint save/load reproduces identical inference ({m.num_parameters():,} params)")
print("ALL SANITY CHECKS PASSED")
