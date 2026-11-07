"""Text preprocessing for CodeNarrator (traceback -> summary).

Design choices (all explained in the report):
  * Source (traceback) tokenizer keeps case: exception names are CamelCase and meaningful.
    Dotted names such as ``customers.csv`` or ``requests.exceptions.HTTPError`` are split at the dot
    (``customers . csv``) in BOTH source and target. Reason: combinations like ``events.age`` are
    unseen as whole tokens, but their parts (``events``, ``age``) are not.
  * Newlines become the explicit token ``<NL>`` (the last line of a traceback holds the exception).
  * Leading indentation is dropped (no information in this dataset).
  * Line numbers inside ``File "x.py", line 123, in f`` are replaced by ``<LN>``: they are random
    noise, never appear in a target, and would only inflate the vocabulary.
    Numbers in the error *message* (HTTP 503, shapes, byte positions) are kept.
  * Target tokenizer: whitespace split, then each chunk is split at '.', and a trailing ',' is
    detached. It is exactly reversible (verified on the whole dataset), needed for Exact Match.
    Detokenisation rule: a '.' that is the LAST token ends the sentence; any other '.' glues its
    two neighbours together (``customers . csv`` -> ``customers.csv``).
"""
import re

LINE_NO_RE = re.compile(r'(File "[^"]*", line )\d+')
SRC_TOKEN_RE = re.compile(r"<NL>|<LN>|[\w’]+|[^\w\s]")
FRAME_PREFIX_RE = re.compile(r"[ \t]+")


def normalize_source(text: str, normalize_line_numbers: bool = True) -> str:
    """Normalise a raw traceback/log string (no tokenisation yet)."""
    text = str(text).replace("\r\n", "\n").strip()
    if normalize_line_numbers:
        text = LINE_NO_RE.sub(r"\1<LN>", text)
    lines = [FRAME_PREFIX_RE.sub(" ", ln.strip()) for ln in text.split("\n")]
    return " <NL> ".join(ln for ln in lines if ln)


def tokenize_source(text: str, normalize_line_numbers: bool = True) -> list:
    return SRC_TOKEN_RE.findall(normalize_source(text, normalize_line_numbers))


def tokenize_target(text: str) -> list:
    """Whitespace split; chunks are split at '.', trailing ',' detached."""
    tokens = []
    for chunk in str(text).strip().split():
        trailing_comma = chunk.endswith(",") and len(chunk) > 1
        if trailing_comma:
            chunk = chunk[:-1]
        parts = re.split(r"(\.)", chunk)          # keeps the dots as separate items
        tokens.extend(p for p in parts if p != "")
        if trailing_comma:
            tokens.append(",")
    return tokens


def detokenize_target(tokens: list) -> str:
    """Inverse of :func:`tokenize_target`."""
    out, glue_next = "", False
    for i, tok in enumerate(tokens):
        last = i == len(tokens) - 1
        if tok == "," or (tok == "." and out):
            out += tok
            glue_next = tok == "." and not last
        else:
            out += ("" if (glue_next or not out) else " ") + tok
            glue_next = False
    return out


HEADER = "Traceback (most recent call last):"
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")


def ensure_traceback_header(text):
    """Inference-time input adapter. The model was trained on full tracebacks (every training input contains the
    standard header). A bare exception line such as "KeyError: 'x'" is wrapped in that header; see tools/check_input_format.py
    for the measured effect (bare lines are summarised much worse without it)."""
    text = str(text).strip()
    return text if HEADER in text else HEADER + "\n" + text


def exception_name(text):
    """Exception class named on the last line ("sqlite3.IntegrityError: ..." -> "sqlite3.IntegrityError"), or None if the
    last line does not look like an exception line."""
    lines = [ln.strip() for ln in str(text).strip().split("\n") if ln.strip()]
    m = _EXC_LINE.match(lines[-1]) if lines else None
    return m.group(1) if m else None


def clean_dataframe(df):
    """Missing-value handling + duplicate removal. Returns (clean_df, report dict)."""
    report = {"rows_in": len(df)}
    df = df.dropna(subset=["input_text", "target_text"])
    df = df[(df["input_text"].str.strip() != "") & (df["target_text"].str.strip() != "")]
    report["after_missing_or_empty"] = len(df)
    df = df.drop_duplicates(subset=["input_text", "target_text"])
    report["after_exact_duplicates"] = len(df)
    return df.reset_index(drop=True), report
