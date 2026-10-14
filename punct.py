"""Punctuation and casing restoration that can never change a word.

Whisper output here has no punctuation. A small ONNX model (punctuators, pcs_en) predicts punctuation
and case, and we copy ONLY those onto the original tokens, matched by alignment. Words, their order and
their timestamps are untouched; anything the model cannot match (spelled-out letters, odd tokens) is kept
exactly as it was. If the check fails for any reason the original text is returned.

    PUNCTUATE=0   disables it (default: on when the package and model are available)
"""
import difflib
import os
import re
import warnings

_model = None
_failed = False


def _key(tok: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", tok.lower())


def _load():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    try:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from punctuators.models import PunctCapSegModelONNX

        _model = PunctCapSegModelONNX.from_pretrained("pcs_en")
    except Exception:  # noqa: BLE001 - optional dependency: fall back to the original text
        _failed = True
    return _model


def enabled() -> bool:
    return os.environ.get("PUNCTUATE", "1") != "0"


def merge(original: str, predicted: str) -> str:
    """Apply the case and trailing punctuation of `predicted` onto the tokens of `original`."""
    orig = original.split()
    pred = [t.replace("<unk>", "") for t in predicted.split()]
    pred = [t for t in pred if _key(t)]
    sm = difflib.SequenceMatcher(a=[_key(t) for t in orig], b=[_key(t) for t in pred], autojunk=False)
    out = list(orig)
    matched = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            matched[blk.a + k] = pred[blk.b + k]
    for i, o in enumerate(orig):
        p = matched.get(i)
        if p is None or re.search(r"[^A-Za-z0-9'’]", o):  # unmatched, or has its own punctuation ("-t", ".tech", "$11,500")
            continue
        trailing = re.search(r"[.,?!;:]+$", p)
        word = o
        # case: keep the transcript's own casing, except sentence starts and the pronoun "I"
        prev_end = i == 0 or bool(re.search(r"[.?!]$", out[i - 1]))
        if prev_end and word[:1].isalpha():
            word = word[:1].upper() + word[1:]
        if _key(word) == "i" or re.match(r"^i['’]", word.lower()):
            word = "I" + word[1:]
        out[i] = word + (trailing.group(0) if trailing else "")
    return " ".join(out)


def restore(text: str) -> str:
    """Punctuated and cased version of `text` with exactly the same words, or `text` unchanged."""
    if not enabled() or len(text.split()) < 4:
        return text
    m = _load()
    if m is None:
        return text
    try:
        predicted = " ".join(m.infer([text.lower()])[0])
        merged = merge(text, predicted)
    except Exception:  # noqa: BLE001
        return text
    # hard guarantee: same tokens in the same order, ignoring case and punctuation
    if [_key(t) for t in merged.split()] != [_key(t) for t in text.split()]:
        return text
    return merged
