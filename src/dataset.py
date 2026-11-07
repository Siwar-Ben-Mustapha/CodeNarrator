"""Reading the CSV files, PyTorch Dataset (with training-time augmentation) and DataLoader utilities."""
import random
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import tokenize_source, tokenize_target, clean_dataframe
from vocab import Vocabulary, UNK
from copy_slots import SLOT_TOKENS, slot_map, apply_map, last_line_start

CSV_FILES = {"train": "train", "val_familiar": "validation", "test_familiar": "test",
             "test_novel": "test_novel_entities", "test_unseen": "test_unseen_families"}


def load_splits(data_dir):
    """Read the five CSV files (never modified on disk). Cleaning in memory: missing/empty/duplicate rows only."""
    return {key: clean_dataframe(pd.read_csv(f"{data_dir}/{name}.csv"))[0] for key, name in CSV_FILES.items()}


class Seq2SeqDataset(Dataset):
    """Tokenised examples -> id lists.
    use_slots : unknown message words become copy slots <S1>.. (copy_slots.py).
    train     : ids are rebuilt every time an example is drawn, with two augmentations
                * entity words (column `entities`) are randomly treated as unknown -> the model learns to copy slots (prob p_mask)
                * tokens outside the exception line are randomly replaced by <UNK> (prob p_unk) -> robust to unseen frame names
    Evaluation datasets (train=False) are built once, deterministically; `maps` keeps word<->slot for each example."""

    def __init__(self, df, src_vocab, tgt_vocab, max_src_len, max_tgt_len, normalize_line_numbers=True,
                 use_slots=False, train=False, p_mask=0.5, p_unk=0.15, seed=0):
        self.df = df.reset_index(drop=True)
        self.sv, self.tv, self.max_src, self.max_tgt = src_vocab, tgt_vocab, max_src_len, max_tgt_len
        self.use_slots, self.train, self.p_mask, self.p_unk = use_slots, train, p_mask, p_unk
        self.rng = random.Random(seed)
        self.src_tokens = [tokenize_source(x, normalize_line_numbers) for x in self.df.input_text]
        self.tgt_tokens = [tokenize_target(x) for x in self.df.target_text]
        ents = self.df["entities"].fillna("") if "entities" in self.df else [""] * len(self.df)
        self.entity_words = [{w for e in str(x).split("|") if e for w in tokenize_source(e)} for x in ents]
        self.src_trunc = sum(len(t) > max_src_len for t in self.src_tokens)
        self.tgt_trunc = sum(len(t) + 2 > max_tgt_len for t in self.tgt_tokens)
        self.cache = None if train else [self._build(i, False) for i in range(len(self.df))]
        self.maps = None if train else [c[2] for c in self.cache]

    def _build(self, i, augment):
        toks, tgt = self.src_tokens[i], self.tgt_tokens[i]
        mapping = {}
        if self.use_slots:
            masked = {w for w in self.entity_words[i] if augment and self.rng.random() < self.p_mask}
            mapping = slot_map(toks, self.sv, masked)
            toks, tgt = apply_map(toks, mapping), apply_map(tgt, mapping)
        if augment and self.p_unk:
            start = last_line_start(toks)
            toks = [UNK if j < start and t not in SLOT_TOKENS and self.rng.random() < self.p_unk else t for j, t in enumerate(toks)]
        toks = toks[-self.max_src:]                      # too long: keep the END (it holds the exception line)
        src_ids, _ = self.sv.encode(toks)
        tgt_ids, _ = self.tv.encode(tgt, add_sos=True, add_eos=True, max_len=self.max_tgt)
        return src_ids, tgt_ids, mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        src_ids, tgt_ids, _ = self._build(i, True) if self.train else self.cache[i]
        return src_ids, tgt_ids, i


def collate(batch):
    """Pad to the longest sequence of the batch (PAD id = 0)."""
    srcs, tgts, idx = zip(*batch)
    src_len = torch.tensor([len(s) for s in srcs])
    src = torch.zeros(len(srcs), int(src_len.max()), dtype=torch.long)
    tgt = torch.zeros(len(tgts), max(len(t) for t in tgts), dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, src_len, tgt, torch.tensor(idx)


def make_loader(ds, batch_size, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


def build_vocabs(train_df, min_freq_src=2, min_freq_tgt=1, normalize_line_numbers=True, use_slots=False):
    """Vocabularies from the training rows ONLY (slot tokens are added when use_slots)."""
    extra = tuple(SLOT_TOKENS) if use_slots else ()
    sv = Vocabulary.build([tokenize_source(x, normalize_line_numbers) for x in train_df.input_text], min_freq_src, extra=extra)
    tv = Vocabulary.build([tokenize_target(x) for x in train_df.target_text], min_freq_tgt, extra=extra)
    return sv, tv
