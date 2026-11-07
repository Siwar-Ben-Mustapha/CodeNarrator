"""Copy slots: lets a plain LSTM seq2seq reproduce words it has never seen (e.g. a new column name).

Problem: the target often repeats a word from the traceback message ("... the key revenue is missing").
If `revenue` is not in the training vocabulary, both encoder and decoder only see <UNK>.
Idea (positional-UNK, Luong et al. 2015): inside the exception message, every unknown word is replaced by a
numbered slot <S1>, <S2>, ... (order of first appearance). The decoder may output <S1>; afterwards we put the
original word back. During training we pretend that random entity words (given in the `entities` column of the
dataset) are unknown, so the model learns to use the slots. Only classical RNN components are involved.
"""
import re

N_SLOTS = 5
SLOT_TOKENS = [f"<S{i}>" for i in range(1, N_SLOTS + 1)]
_CANDIDATE = re.compile(r"(?=.*[A-Za-z])\w+$")            # a word with at least one letter


def last_line_start(tokens):
    """Index of the first token of the last line (the exception line)."""
    return len(tokens) - tokens[::-1].index("<NL>") if "<NL>" in tokens else 0


def message_region(tokens):
    """Indices of the tokens after the first ':' of the last line (= the error message)."""
    start = last_line_start(tokens)
    last = tokens[start:]
    return range(start + last.index(":") + 1, len(tokens)) if ":" in last else range(0)


def slot_map(tokens, vocab, mask_words=()):
    """{word: slot}. A message word gets a slot if it is unknown to the vocabulary or listed in mask_words."""
    mapping = {}
    for i in message_region(tokens):
        w = tokens[i]
        if w in mapping or not _CANDIDATE.match(w) or len(mapping) >= N_SLOTS:
            continue
        if w in mask_words or w not in vocab.stoi:
            mapping[w] = SLOT_TOKENS[len(mapping)]
    return mapping


def apply_map(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


def restore(tokens, mapping):
    inverse = {s: w for w, s in mapping.items()}
    return [inverse.get(t, "<UNK>" if t in SLOT_TOKENS else t) for t in tokens]
