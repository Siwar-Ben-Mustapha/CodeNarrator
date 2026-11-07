"""Vocabulary with the four mandatory special tokens. Built from TRAINING data only."""
from collections import Counter

PAD, UNK, SOS, EOS = "<PAD>", "<UNK>", "<SOS>", "<EOS>"
SPECIALS = [PAD, UNK, SOS, EOS]          # ids 0, 1, 2, 3 (PAD must be 0)


class Vocabulary:
    def __init__(self, itos=None):
        self.itos = list(itos) if itos else list(SPECIALS)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=None, extra=()):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [t for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq and t not in SPECIALS and t not in extra]
        if max_size:
            words = words[: max_size - len(SPECIALS)]
        return cls(SPECIALS + list(extra) + words)      # extra = copy-slot tokens, ids 4.. (after the 4 mandatory specials)

    def __len__(self):
        return len(self.itos)

    pad_id = property(lambda s: 0)
    unk_id = property(lambda s: 1)
    sos_id = property(lambda s: 2)
    eos_id = property(lambda s: 3)

    def encode(self, tokens, add_sos=False, add_eos=False, max_len=None):
        """tokens -> ids. Unknown tokens -> <UNK>. Truncation keeps room for SOS/EOS.
        Returns (ids, was_truncated)."""
        budget = None if max_len is None else max_len - int(add_sos) - int(add_eos)
        truncated = budget is not None and len(tokens) > budget
        if truncated:
            tokens = tokens[:budget]
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids, truncated

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id and stop_at_eos:
                break
            if i in (self.pad_id, self.sos_id):
                continue
            out.append(self.itos[i])
        return out

    def state(self):
        return list(self.itos)
