"""Score SalesIQ against the golden labels (golden/labels.json).

    python eval_golden.py                      # analysis layer on all 10 reference transcripts (LLM calls)
    python eval_golden.py --only sample-1,sample-2 --repeat 3     # stability across repeated runs
    python eval_golden.py --asr                # ASR WER on calls that have output/<name>.json (no LLM)

What each mode measures
  analysis : the reference transcript (clean text + true speakers) is fed to the LLM step, so this
             isolates the analysis from ASR/diarization errors. Fields: sales gate, deal type,
             who the seller is, outcome, deal total + currency, and (when labelled) pitch/close counts.
  --repeat : runs each call N times and reports how often the answer is identical (run-to-run stability).
  --asr    : WER vs the reference, raw and number-normalised, for transcribed calls.

LLM calls are billed and counted by budget.py (hard cap). Golden labels are a DRAFT until reviewed by a human.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import config  # noqa: F401  (loads .env)
import evaluate as ev

ROOT = Path(__file__).parent
GOLDEN = ROOT / "golden" / "labels.json"
REF = ROOT / "data" / "reference" / "train.jsonl"


def load_reference_utts(row: int) -> list:
    rows = [json.loads(l) for l in REF.read_text(encoding="utf-8").splitlines() if l.strip()]
    utts = []
    for m in ev.SEG.finditer(rows[row]["transcript"]):
        text = m.group(8).strip()
        if text:
            utts.append({"speaker": f"Speaker_{m.group(1)}", "text": text})
    return utts


def analyse(utts: list, samples: int | None = None) -> dict:
    import segment

    res = segment.analyze(utts, n=samples)  # same path as production: consensus over samples
    roles = res["roles_final"]
    sales = res["metrics"]
    deal = sales.get("deal") or {}
    return {
        "is_sales": res["conversation"]["verdict"] == "sales",
        "verdict": res["conversation"]["verdict"],
        "category": res["category"],
        "roles": roles,
        "outcome": (res["insights"] or {}).get("outcome"),
        "amount": deal.get("amount"),
        "currency": deal.get("currency"),
        "amount_source": deal.get("amount_source"),
        "pitch": sales.get("pitch_attempts"),
        "close": sales.get("close_attempts"),
    }


def score(g: dict, r: dict) -> dict:
    """Per-field pass/fail; None means the golden file has no label for that field."""
    out = {
        "sales_gate": r["is_sales"] == g["is_sales"],
        "category": r["category"] == g["category"],
        "seller": r["roles"].get(g["seller_ref_speaker"]) == "salesperson",
        "outcome": r["outcome"] in g["outcome_ok"],
    }
    gt = g.get("total_contract_value")
    if gt is None:
        out["deal_total"] = r["amount"] is None  # inventing a deal where none was discussed is a miss
    else:
        out["deal_total"] = r["amount"] is not None and abs(float(r["amount"]) - gt) <= 1
    if g.get("currency"):
        out["currency"] = r["currency"] == g["currency"]
    for k, key in (("pitch_attempts", "pitch"), ("close_attempts", "close")):
        if g.get(k) is not None:
            out[k] = r[key] == g[k]
    return out


def cmd_analysis(a) -> None:
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    names = [n for n in gold["calls"] if not a.only or n in a.only.split(",")]
    results, t0 = {}, time.time()
    for n in names:
        g = gold["calls"][n]
        utts = load_reference_utts(g["reference_row"])
        runs = [analyse(utts, a.samples) for _ in range(a.repeat)]
        results[n] = {"golden": g, "runs": runs, "scores": [score(g, r) for r in runs]}
        r0, s0 = runs[0], results[n]["scores"][0]
        bad = [k for k, v in s0.items() if not v]
        print(f"{n:10s} {'OK ' if not bad else 'MISS'} gate={r0['verdict']:9s} cat={r0['category']:11s} "
              f"seller={[k for k, v in r0['roles'].items() if v == 'salesperson']} outcome={r0['outcome']:9s} "
              f"deal={r0['amount']} {r0['currency'] or ''} ({r0['amount_source'] or 'stated'}) "
              f"pitch={r0['pitch']} close={r0['close']}" + (f"   <- wrong: {bad}" if bad else ""), flush=True)

    # per-field accuracy over every run
    fields = {}
    for n, d in results.items():
        for s in d["scores"]:
            for k, v in s.items():
                fields.setdefault(k, []).append(bool(v))
    print("\nField accuracy (all runs):")
    for k, v in fields.items():
        print(f"  {k:16s} {sum(v):3d}/{len(v):<3d} {sum(v) / len(v) * 100:5.0f}%")

    if a.repeat > 1:
        keys = ("category", "outcome", "amount", "pitch", "close", "verdict")
        print(f"\nStability over {a.repeat} runs (share of calls with an identical answer every time):")
        for k in keys:
            same = sum(len({str(r[k]) for r in d["runs"]}) == 1 for d in results.values())
            print(f"  {k:10s} {same}/{len(results)}")
        seller_same = sum(len({str(sorted(k for k, v in r['roles'].items() if v == 'salesperson')) for r in d['runs']}) == 1
                          for d in results.values())
        print(f"  {'seller':10s} {seller_same}/{len(results)}")

    out = ROOT / "golden" / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"labels_by": gold.get("labelled_by"), "repeat": a.repeat, "results": results},
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    import budget
    s = budget.load()
    print(f"\nsaved {out.name}  |  {time.time() - t0:.0f}s  |  LLM spend so far ${s['usd']:.4f} of ${budget.limit():.2f}")
    print("Note: golden labels are a draft until a human reviews them (golden/labels.json).")


def norm_numbers(text: str) -> list:
    """Lowercase, drop punctuation, and spell digits out so '$11,500' and 'eleven thousand five hundred' agree."""
    from num2words import num2words

    text = text.lower().replace("$", " ").replace("-", " ")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: " " + num2words(int(float(m.group(0)))) + " ", text)
    return re.sub(r"[^a-z\s]", " ", text.replace("-", " ")).split()


def cmd_asr(a) -> None:
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in REF.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{'call':10s} {'WER raw':>8s} {'WER norm':>9s}  words(ref/ours)")
    for n, g in gold["calls"].items():
        p = ROOT / "output" / f"{n}.json"
        if not p.exists():
            continue
        ours = json.loads(p.read_text(encoding="utf-8"))
        dur = ours.get("duration") or max(w["end"] for s in ours["segments"] for w in s["words"])
        segs = [s for s in ev.parse_reference(rows[g["reference_row"]]["transcript"]) if s["start"] < dur + 1]
        ref_text = " ".join(ev_text for ev_text in (
            m.group(8) for m in ev.SEG.finditer(rows[g["reference_row"]]["transcript"])))
        # only the part of the reference that the audio clip covers
        keep = len(segs)
        ref_texts = [m.group(8) for m in ev.SEG.finditer(rows[g["reference_row"]]["transcript"])][:keep]
        ref_text = " ".join(ref_texts)
        our_text = " ".join(w["word"] for s in ours["segments"] for w in s["words"])
        raw = ev.wer(ev.norm(ref_text), ev.norm(our_text))
        nrm = ev.wer(norm_numbers(ref_text), norm_numbers(our_text))
        print(f"{n:10s} {raw * 100:7.1f}% {nrm * 100:8.1f}%  {len(ev.norm(ref_text))}/{len(ev.norm(our_text))}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated call names")
    p.add_argument("--repeat", type=int, default=1, help="runs per call (stability)")
    p.add_argument("--samples", type=int, default=None, help="LLM samples per analysis (default ANALYSIS_SAMPLES or 3)")
    p.add_argument("--asr", action="store_true", help="score ASR WER instead of the LLM analysis")
    a = p.parse_args()
    (cmd_asr if a.asr else cmd_analysis)(a)


if __name__ == "__main__":
    sys.exit(main())
