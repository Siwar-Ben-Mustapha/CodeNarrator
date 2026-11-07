"""Evaluation: real autoregressive (greedy) generation + ROUGE-1/2/L, BLEU-4, Exact Match, token accuracy.

ROUGE (F1 on lower-cased word tokens, LCS for ROUGE-L) and BLEU are implemented here, so the project has no extra
dependency; tests/test_metrics.py cross-checks them against Google's `rouge-score` and NLTK when these are installed.
"""
import re, math, collections
import numpy as np, torch
from preprocessing import tokenize_source, tokenize_target, detokenize_target
from copy_slots import restore
from dataset import make_loader

_W = re.compile(r"[a-z0-9]+")
_rtok = lambda s: _W.findall(s.lower())          # same tokenisation as rouge-score (default)


def _f1(overlap, n_pred, n_ref):
    if overlap == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def _ngrams(toks, n):
    return collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def rouge_n(pred, ref, n):
    p, r = _ngrams(_rtok(pred), n), _ngrams(_rtok(ref), n)
    return _f1(sum((p & r).values()), sum(p.values()), sum(r.values())) if p and r else 0.0


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = _rtok(pred), _rtok(ref)
    return _f1(_lcs(p, r), len(p), len(r)) if p and r else 0.0


def bleu(preds, refs, max_n=4):
    """Corpus BLEU-4 (0-100): clipped n-gram precisions on lower-cased word tokens x brevity penalty, no smoothing."""
    clipped, total, plen, rlen = [0] * max_n, [0] * max_n, 0, 0
    for p, r in zip(preds, refs):
        p, r = _rtok(p), _rtok(r)
        plen, rlen = plen + len(p), rlen + len(r)
        for n in range(1, max_n + 1):
            pg, rg = _ngrams(p, n), _ngrams(r, n)
            clipped[n - 1] += sum((pg & rg).values())
            total[n - 1] += sum(pg.values())
    if plen == 0 or min(clipped) == 0:
        return 0.0
    brevity = 1.0 if plen > rlen else math.exp(1 - rlen / plen)
    return 100 * brevity * math.exp(sum(math.log(c / t) for c, t in zip(clipped, total)) / max_n)


@torch.no_grad()
def generate_dataset(model, ds, tgt_vocab, device, batch_size=128, max_len=40):
    """Greedy-decode every example of `ds`. Returns list of predicted strings (detokenised)."""
    model.eval()
    preds = [None] * len(ds)
    for src, src_len, _, idx in make_loader(ds, batch_size, shuffle=False):
        ids = model.generate(src.to(device), src_len.to(device), max_len=max_len).cpu()
        for k, row in zip(idx.tolist(), ids):
            toks = tgt_vocab.decode(row.tolist())
            if ds.maps is not None and ds.maps[k]:
                toks = restore(toks, ds.maps[k])               # put the copied words back
            preds[k] = detokenize_target(toks)
    return preds


def entity_recall(pred, ref, source, common):
    """Share of the reference's 'copyable details' (reference tokens that occur in the input
    traceback and are not very common summary words) that the prediction also contains."""
    src = set(w.lower() for w in tokenize_source(source))
    ents = [w.lower() for w in tokenize_target(ref) if w.lower() in src and w.lower() not in common and w.isalnum()]
    if not ents:
        return None
    pset = set(w.lower() for w in tokenize_target(pred))
    return float(np.mean([e in pset for e in ents]))


def score(preds, df, common=frozenset()):
    refs, srcs = df.target_text.tolist(), df.input_text.tolist()
    r1 = [rouge_n(p, r, 1) for p, r in zip(preds, refs)]
    r2 = [rouge_n(p, r, 2) for p, r in zip(preds, refs)]
    rl = [rouge_l(p, r) for p, r in zip(preds, refs)]
    em = [float(p.strip() == r.strip()) for p, r in zip(preds, refs)]
    tacc = []                                         # position-wise token accuracy (diagnostic only)
    for p, r in zip(preds, refs):
        pt, rt = tokenize_target(p), tokenize_target(r)
        tacc.append(sum(a == b for a, b in zip(pt, rt)) / max(len(rt), 1))
    er = [entity_recall(p, r, s, common) for p, r, s in zip(preds, refs, srcs)]
    er = [x for x in er if x is not None]
    return {"rouge1": 100 * np.mean(r1), "rouge2": 100 * np.mean(r2), "rougeL": 100 * np.mean(rl), "bleu": bleu(preds, refs),
            "exact_match": 100 * np.mean(em), "token_acc": 100 * np.mean(tacc),
            "entity_recall": 100 * np.mean(er) if er else float("nan"),
            "avg_pred_len": float(np.mean([len(tokenize_target(p)) for p in preds]))}


def common_target_words(train_df, top=40):
    c = collections.Counter(w.lower() for t in train_df.target_text for w in tokenize_target(t))
    return frozenset(w for w, _ in c.most_common(top))


def per_group(preds, df, col):
    """ROUGE-L / EM per group (scenario_id, category, length bucket...)."""
    rows = []
    d = df.assign(pred=preds)
    for key, g in d.groupby(col):
        rows.append({col: key, "n": len(g),
                     "rougeL": 100 * np.mean([rouge_l(p, r) for p, r in zip(g.pred, g.target_text)]),
                     "exact_match": 100 * np.mean([p.strip() == r.strip() for p, r in zip(g.pred, g.target_text)])})
    return rows


def template_regurgitation_rate(preds, train_df):
    """% of predictions that are verbatim copies of some TRAINING summary (template recall)."""
    known = set(train_df.target_text.str.strip())
    return 100 * float(np.mean([p.strip() in known for p in preds]))


def oracle_template_rougeL(refs, train_df):
    """Upper bound for any model that can only output a summary seen in training: for every
    reference, the best ROUGE-L achievable by picking the best training summary (retrieval oracle)."""
    cands = sorted(set(train_df.target_text.str.strip()))
    cache = {}
    for r in set(refs):
        cache[r] = max(rouge_l(c, r) for c in cands)
    return 100 * float(np.mean([cache[r] for r in refs]))
