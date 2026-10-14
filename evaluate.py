"""Score our output against the dataset's reference transcript.

Usage:
    python evaluate.py output/sample-1.json --ref data/reference/train.jsonl --row 0

Reports:
  - WER                      word error rate vs the reference text
  - timestamp error          our word times vs reference segment start/end (seconds)
  - speaker accuracy         share of matched words given the right speaker
  - speaker count            ours vs reference
"""
import argparse
import difflib
import json
import re
import statistics
from pathlib import Path

SEG = re.compile(
    r"Speaker (\d+) \[(\d+):(\d+):(\d+\.\d+) - (\d+):(\d+):(\d+\.\d+)\]:\s*(.*?)(?=\n\s*Speaker \d+ \[|\Z)",
    re.S,
)


def norm(text: str):
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9$%\s]", " ", text)
    text = text.replace("$", " ")
    return text.split()


def parse_reference(transcript: str):
    segs = []
    for m in SEG.finditer(transcript):
        spk, h1, m1, s1, h2, m2, s2, text = m.groups()
        segs.append(
            {
                "speaker": f"Speaker {spk}",
                "start": int(h1) * 3600 + int(m1) * 60 + float(s1),
                "end": int(h2) * 3600 + int(m2) * 60 + float(s2),
                "words": norm(text),
            }
        )
    return segs


def wer(ref, hyp):
    """Word error rate via edit distance."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(len(ref), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--ref", default="data/reference/train.jsonl")
    p.add_argument("--row", type=int, default=0)
    a = p.parse_args()

    ours = json.loads(Path(a.output).read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in Path(a.ref).read_text(encoding="utf-8").splitlines() if l.strip()]
    ref_segs = parse_reference(rows[a.row]["transcript"])

    duration = ours.get("duration") or max(w["end"] for s in ours["segments"] for w in s["words"])
    # the audio clip can be shorter than the reference; only score what the clip covers
    ref_segs = [s for s in ref_segs if s["start"] < duration + 1]

    our_words = [
        {"w": n, "start": w["start"], "end": w["end"], "speaker": w.get("speaker")}
        for s in ours["segments"]
        for w in s["words"]
        for n in norm(w["word"])
    ]
    ref_words = [
        {"w": n, "seg": i, "speaker": s["speaker"]}
        for i, s in enumerate(ref_segs)
        for n in s["words"]
    ]

    ref_t = [r["w"] for r in ref_words]
    our_t = [o["w"] for o in our_words]
    print(f"reference: {len(ref_segs)} segments, {len(ref_t)} words (audio {duration:.0f}s)")
    print(f"ours:      {len(ours['segments'])} segments, {len(our_t)} words\n")
    print(f"WER: {wer(ref_t, our_t) * 100:.1f}%   (numbers/spelling differences count as errors)")

    # align words, then compare timestamps and speakers on the matched ones
    sm = difflib.SequenceMatcher(a=ref_t, b=our_t, autojunk=False)
    pairs = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs.append((ref_words[blk.a + k], our_words[blk.b + k]))
    print(f"matched words: {len(pairs)} / {len(ref_t)} ({len(pairs) / max(len(ref_t), 1) * 100:.0f}%)")

    # timestamp error: first matched word in each reference segment vs segment start
    start_err, end_err = [], []
    by_seg = {}
    for r, o in pairs:
        by_seg.setdefault(r["seg"], []).append(o)
    for i, outs in by_seg.items():
        start_err.append(outs[0]["start"] - ref_segs[i]["start"])
        end_err.append(outs[-1]["end"] - ref_segs[i]["end"])
    if start_err:
        ae = [abs(x) for x in start_err]
        be = [abs(x) for x in end_err]
        print(
            f"\ntimestamps (per reference segment, n={len(ae)}):\n"
            f"  start error  mean {statistics.mean(ae):.2f}s  median {statistics.median(ae):.2f}s  max {max(ae):.2f}s\n"
            f"  end error    mean {statistics.mean(be):.2f}s  median {statistics.median(be):.2f}s  max {max(be):.2f}s\n"
            f"  within 0.5s: start {sum(x <= 0.5 for x in ae) / len(ae) * 100:.0f}%  end {sum(x <= 0.5 for x in be) / len(be) * 100:.0f}%"
        )

    # speaker accuracy: labels are arbitrary, so try both mappings and keep the best
    ours_spk = sorted({o["speaker"] for _, o in pairs if o["speaker"]})
    ref_spk = sorted({r["speaker"] for r, _ in pairs})
    best = (0.0, None)
    if ours_spk and ref_spk:
        from itertools import permutations

        for perm in permutations(ref_spk, min(len(ours_spk), len(ref_spk))):
            mapping = dict(zip(ours_spk, perm))
            ok = sum(mapping.get(o["speaker"]) == r["speaker"] for r, o in pairs)
            acc = ok / len(pairs)
            if acc > best[0]:
                best = (acc, mapping)
    print(f"\nspeakers: ours {ours.get('num_speakers')}  reference {len({s['speaker'] for s in ref_segs})}")
    print(f"speaker accuracy (matched words): {best[0] * 100:.1f}%   mapping: {best[1]}")

    # words wrong by speaker, to show where errors cluster
    if best[1]:
        wrong = [(r, o) for r, o in pairs if best[1].get(o["speaker"]) != r["speaker"]]
        print(f"wrong-speaker words: {len(wrong)}")
        for r, o in wrong[:10]:
            print(f"   t={o['start']:7.2f}s  '{o['w']}'  ours={best[1].get(o['speaker'])}  ref={r['speaker']}")


if __name__ == "__main__":
    main()
