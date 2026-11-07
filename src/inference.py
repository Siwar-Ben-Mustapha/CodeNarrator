"""Load a portable checkpoint and summarise tracebacks. No training data needed.

    python src/inference.py checkpoints/codenarrator_best.pt traceback.txt
    python src/inference.py checkpoints/codenarrator_best.pt            # reads the traceback from stdin

The same code path is used by the notebook, the tests and the FastAPI backend.
"""
import sys, torch
from vocab import Vocabulary
from preprocessing import tokenize_source, detokenize_target, ensure_traceback_header, exception_name
from seq2seq import Seq2Seq
from copy_slots import slot_map, apply_map, restore


def load_checkpoint(path, device="cpu"):
    """Portable checkpoint = weights + both vocabularies + config. Returns (model in eval mode, src_vocab, tgt_vocab, checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_state", "config", "use_attention", "src_vocab", "tgt_vocab"):
        if key not in ck:
            raise ValueError(f"{path} is not a CodeNarrator checkpoint (missing '{key}')")
    sv, tv = Vocabulary(ck["src_vocab"]), Vocabulary(ck["tgt_vocab"])
    model = Seq2Seq(len(sv), len(tv), ck["config"], ck["use_attention"]).to(device)
    model.load_state_dict(ck["model_state"])           # raises if the architecture does not match the weights
    model.eval()
    return model, sv, tv, ck


@torch.no_grad()
def summarize(model, sv, tv, cfg, traceback_text, device="cpu", return_attention=False, add_header=True):
    """Greedy summary of one raw traceback string (same preprocessing as in training)."""
    if not str(traceback_text).strip():
        raise ValueError("empty traceback")
    if add_header:
        traceback_text = ensure_traceback_header(traceback_text)       # bare exception lines -> minimal full traceback
    toks = tokenize_source(traceback_text, cfg.get("normalize_line_numbers", True))
    mapping = slot_map(toks, sv) if cfg.get("use_slots") else {}       # unknown message words -> copy slots
    shown = toks[-cfg["max_src_len"]:]                                 # too long: keep the end (exception line)
    ids, _ = sv.encode(apply_map(shown, mapping))
    src = torch.tensor([ids], device=device)
    out = model.generate(src, torch.tensor([len(ids)], device=device), max_len=cfg["max_tgt_len"] + 8,
                         return_attention=return_attention)
    if return_attention:
        out, att = out
    text = detokenize_target(restore(tv.decode(out[0].tolist()), mapping))
    n_unk = sum(i == sv.unk_id for i in ids)
    return (text, att, shown, n_unk) if return_attention else text


def input_warnings(text, sv):
    """Plain-language reasons why the summary of this input may be unreliable (rules on the input, not on the model output)."""
    name = exception_name(text)
    if name is None:
        return ["The last line does not look like a Python exception line (expected 'ExceptionName: message')."]
    if name.split(".")[-1] not in sv.stoi:
        return [f"The model never saw the exception type {name} during training; the summary is probably wrong."]
    return []


class Summarizer:
    """Loads a checkpoint once and answers many requests (used by the FastAPI backend)."""

    def __init__(self, path, device="cpu"):
        self.path, self.device = str(path), device
        self.model, self.sv, self.tv, self.ck = load_checkpoint(self.path, device)
        self.cfg = self.ck["config"]

    def __call__(self, text, return_attention=False):
        return summarize(self.model, self.sv, self.tv, self.cfg, text, self.device, return_attention)

    def warnings(self, text):
        return input_warnings(text, self.sv)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    summarizer = Summarizer(sys.argv[1], dev)
    text = open(sys.argv[2], encoding="utf-8").read() if len(sys.argv) > 2 else sys.stdin.read()
    print(summarizer(text))
