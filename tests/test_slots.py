"""Checks for the copy-slot mechanism. Run:  python tests/test_slots.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vocab import Vocabulary
from copy_slots import slot_map, apply_map, restore, SLOT_TOKENS, last_line_start
from preprocessing import tokenize_source, tokenize_target, detokenize_target

vocab = Vocabulary(["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + SLOT_TOKENS + ["KeyError", ":", "'", "The", "key", "is", "missing", "."])
toks = tokenize_source("Traceback (most recent call last):\n  File \"a.py\", line 3, in f\n    row[\"revenue\"]\nKeyError: 'revenue'")

m = slot_map(toks, vocab)
assert m == {"revenue": "<S1>"}, m                      # unknown word of the message -> slot; frame-only words are not
mapped = apply_map(toks, m)
assert mapped.count("<S1>") == 2 and "revenue" not in mapped    # every occurrence is replaced (frame + message)
assert "a" not in m and "f" not in m                    # unknown words outside the message are not slotted
assert last_line_start(toks) > 0

out = ["The", "key", "<S1>", "is", "missing", "."]       # what a decoder could emit
assert detokenize_target(restore(out, m)) == "The key revenue is missing."
assert restore(["<S2>"], m) == ["<UNK>"]                # a slot that was never assigned does not leak through

# exception name before ':' is never slotted, even when unknown
assert slot_map(tokenize_source("MyOddError: boom"), vocab) == {"boom": "<S1>"}
# known words stay untouched (no slots at all)
assert slot_map(tokenize_source("KeyError: key"), vocab) == {}
# masking (training augmentation) can force a known word into a slot
assert slot_map(tokenize_source("KeyError: key"), vocab, mask_words={"key"}) == {"key": "<S1>"}
print("ALL SLOT CHECKS PASSED")
