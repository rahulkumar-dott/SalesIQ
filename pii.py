"""Mask personal identifiers in call transcripts before they go to the LLM or are stored.

ASR writes phone numbers and emails as spoken words ("two one two five five five zero one eight two",
"steve dot richards at omnicore dot com", "s -t -e -v -e"), so this works on tokens, not just regexes.

Masks (replaced by a label):
  [PHONE] 10 digits   [ID_NUMBER] 12 digits (Aadhaar-like)   [CARD_NUMBER] 13-19 digits
  [NUMBER] any other run of 7+ digits   [EMAIL]   [SPELLED] spelled-out letters (4+), which spell
  emails, names and codes   [ID_CODE] mixed letter+digit runs (PAN-like)

NOT masked: person and company names, money amounts, dates, short numbers (seat counts, years).
Names are kept because the analysis needs them; set PII_MASK_NAMES=1 is NOT implemented.

    MASK_PII=0         disable masking everywhere
    PII_STORE_RAW=1    keep raw words in the stored transcript (output/<name>.json); the LLM still sees masked text
"""
import os
import re
from collections import Counter

DIGIT_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
MULTIPLIERS = {"double": 2, "triple": 3}
TLDS = {"com", "ai", "net", "org", "io", "tech", "co", "in", "edu", "gov", "biz", "info", "me", "dev", "app", "us", "uk"}
CONNECTORS = {"dot", "at", "dash", "underscore", "hyphen"}
_STRIP = ".,!?;:\"'()[]"


def enabled() -> bool:
    return os.environ.get("MASK_PII", "1") != "0"


def _clean(tok: str) -> str:
    return tok.lower().strip(_STRIP)


def _digit_count(tok: str):
    """Number of digits this token contributes to a spoken/numeric run, or None if it is not a digit token."""
    c = _clean(tok)
    if c in DIGIT_WORDS:
        return 1
    if c == "oh":
        return 1
    if c in MULTIPLIERS:
        return 0  # only counts when followed by a digit word (handled by the run logic)
    if "$" in tok or "%" in tok:
        return None
    if re.fullmatch(r"\+?\(?\d[\d\-().\s]*", c):
        return sum(ch.isdigit() for ch in c)
    return None


def _single_letter(tok: str) -> bool:
    return bool(re.fullmatch(r"-?[a-zA-Z]\.?", tok.strip(",;:!?")))


def _label_for_digits(n: int) -> str:
    if n == 10:
        return "[PHONE]"
    if n == 12:
        return "[ID_NUMBER]"
    if 13 <= n <= 19:
        return "[CARD_NUMBER]"
    return "[NUMBER]"


def find_spans(tokens: list) -> list:
    """Non-overlapping (start, end_exclusive, label) spans to mask."""
    spans = []
    taken = [False] * len(tokens)

    def add(a, b, label):
        if any(taken[a:b]) or a >= b:
            return
        for k in range(a, b):
            taken[k] = True
        spans.append((a, b, label))

    n = len(tokens)
    # 1) emails: written (x@y.z) and spoken (x dot y at z dot com)
    for i, t in enumerate(tokens):
        if "@" in t and re.search(r"[A-Za-z0-9]", t) and "." in t:
            add(i, i + 1, "[EMAIL]")
    for i, t in enumerate(tokens):
        if _clean(t) != "at":
            continue
        # the TLD must follow within a few tokens: "... at omnicore dot com" / "... at omnicore .tech"
        for j in range(i + 1, min(i + 7, n)):
            c = _clean(tokens[j])
            ends = c in TLDS and j > i + 1 and _clean(tokens[j - 1]) == "dot"
            glued = bool(re.fullmatch(r"\.[a-z]{2,4}", c)) and c[1:] in TLDS
            if ends or glued:
                left = i - 1
                while left - 2 >= 0 and _clean(tokens[left - 1]) == "dot":
                    left -= 2
                if left >= 0:
                    add(left, j + 1, "[EMAIL]")
                break

    # 1b) mixed letter + digit codes read out character by character (PAN-like): a b c d e one two three four f
    def _char(t: str) -> bool:
        c = _clean(t)
        return _single_letter(t) or c in DIGIT_WORDS or bool(re.fullmatch(r"\d", c))

    i = 0
    while i < n:
        if taken[i] or not _char(tokens[i]):
            i += 1
            continue
        j = i
        while j < n and not taken[j] and _char(tokens[j]):
            j += 1
        run = tokens[i:j]
        has_letter = any(_single_letter(t) for t in run)
        has_digit = any(_clean(t) in DIGIT_WORDS or re.fullmatch(r"\d", _clean(t)) for t in run)
        if len(run) >= 6 and has_letter and has_digit:
            add(i, j, "[ID_CODE]")
        i = max(j, i + 1)

    # 2) digit runs (spoken digits and numerals)
    i = 0
    while i < n:
        if taken[i]:
            i += 1
            continue
        total, j, last_digit = 0, i, i
        while j < n and not taken[j]:
            c = _clean(tokens[j])
            d = _digit_count(tokens[j])
            if c in MULTIPLIERS and j + 1 < n and _clean(tokens[j + 1]) in DIGIT_WORDS:
                total += MULTIPLIERS[c] - 1  # the following digit word adds the last 1
                j += 1
                continue
            if d is None or (c == "oh" and total == 0):
                break
            total += d
            last_digit = j
            j += 1
        if total >= 7:
            add(i, last_digit + 1, _label_for_digits(total))
            i = last_digit + 1
        else:
            i = max(j, i + 1)

    # 3) spelled-out letters: s -t -e -v -e (allow dot/at between letters)
    i = 0
    while i < n:
        if taken[i] or not _single_letter(tokens[i]):
            i += 1
            continue
        j, letters, last_letter = i, 0, i
        while j < n and not taken[j]:
            if _single_letter(tokens[j]):
                letters += 1
                last_letter = j
                j += 1
            elif _clean(tokens[j]) in CONNECTORS and j + 1 < n and _single_letter(tokens[j + 1]):
                j += 1
            else:
                break
        # a lone "a"/"I" next to words is normal English; require 4+ letters in the run
        if letters >= 4:
            run = " ".join(tokens[i:last_letter + 1])
            has_digit = bool(re.search(r"\d", run))
            add(i, last_letter + 1, "[ID_CODE]" if has_digit else "[SPELLED]")
            i = last_letter + 1
        else:
            i = max(j, i + 1)

    # 4) letter+digit codes mixed (PAN-like), e.g. "a b c d e one two three four f"
    return sorted(spans)


def mask_tokens(tokens: list):
    """Returns (masked_tokens, findings). Each span becomes its label on the first token, '' on the rest."""
    spans = find_spans(tokens)
    out = list(tokens)
    for a, b, label in spans:
        trail = re.search(r"[.,!?;:]+$", tokens[b - 1])
        out[a] = label + (trail.group(0) if trail else "")
        for k in range(a + 1, b):
            out[k] = ""
    return out, Counter(label for _, _, label in spans)


def mask_text(text: str):
    """Returns (masked_text, findings)."""
    if not enabled():
        return text, Counter()
    toks = text.split()
    out, found = mask_tokens(toks)
    return " ".join(t for t in out if t), found


def scrub_segments(segments: list) -> Counter:
    """Mask PII in a stored transcript IN PLACE (word text only; timestamps and speakers are untouched).

    Works across segment boundaries, because a phone number can be split between two segments.
    Masked words keep their timestamps; the label sits on the first word and the rest become ''.
    """
    flat = [w for s in segments for w in s.get("words", [])]
    masked, found = mask_tokens([w["word"] for w in flat])
    for w, m in zip(flat, masked):
        w["word"] = m
    for s in segments:
        s["text"] = " ".join(w["word"] for w in s.get("words", []) if w["word"])
    return found


if __name__ == "__main__":  # python pii.py output/<name>.json [...]  -> scrub stored transcripts in place
    import json
    import sys

    for path in sys.argv[1:]:
        d = json.load(open(path, encoding="utf-8"))
        found = scrub_segments(d["segments"])
        json.dump(d, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"{path}: masked {dict(found) or 'nothing'}")
