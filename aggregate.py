"""Roll per-call metrics up across calls.

Usage:
    python aggregate.py [output_dir]      # default: output/

Reads every *.utterances.json (written by segment.py), prints a summary and
writes <output_dir>/aggregate.json. Reps are grouped by voice-profile name when
known, otherwise by file ("unnamed").
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 2) if xs else None


def deal_of(s: dict) -> dict:
    """The deal object, with a fallback for results written before `deal` existed."""
    d = s.get("deal") or {}
    if d.get("amount") is None and s.get("deal_value_usd") is not None:
        return {"amount": s["deal_value_usd"], "currency": "USD", "basis": "unknown"}
    return d


def _won_amounts(calls) -> dict:
    out = {}
    for c in calls:
        d = deal_of(c["s"])
        if c["i"].get("outcome") == "won" and isinstance(d.get("amount"), (int, float)):
            out.setdefault(d.get("currency") or "?", []).append(d["amount"])
    return out


def build(out_dir: Path) -> dict:
    """Aggregate dict over every analysed call in out_dir; ValueError if there are none."""
    # one call may have several result files (<n>.utterances.json and <n>.speakers.utterances.json): newest wins
    newest = {}
    for f in out_dir.glob("*.utterances.json"):
        name = f.name.replace(".speakers.utterances.json", "").replace(".utterances.json", "")
        if name not in newest or f.stat().st_mtime > newest[name].stat().st_mtime:
            newest[name] = f
    calls, not_sales = [], []
    for name, f in sorted(newest.items()):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d.get("metrics") or {}
        if not m.get("sales"):
            continue
        if (d.get("conversation") or {}).get("verdict") == "not_sales":
            not_sales.append({"file": name, "kind": d["conversation"].get("kind")})
            continue  # not a sales conversation: would pollute win rate and averages
        calls.append({"file": name, "d": d, "a": m["audio"], "s": m["sales"], "i": d.get("insights") or {}})
    if not calls:
        raise ValueError(f"no analysed calls in {out_dir} (run segment.py first)")

    outcomes = Counter((c["i"].get("outcome") or "unclear") for c in calls)
    decided = outcomes["won"] + outcomes["lost"]
    stages = Counter(c["s"].get("stage_reached", "unclear") for c in calls)
    obj = Counter()
    for c in calls:
        for k, v in (c["s"].get("objections_by_type") or {}).items():
            obj[k] += v or 0

    # per rep: the speaker whose role is salesperson
    reps = defaultdict(list)
    for c in calls:
        spk = next((s for s, r in c["d"]["roles"].items() if r == "salesperson"), None)
        name = next((u.get("name") for u in c["d"]["utterances"] if u["speaker"] == spk and u.get("name")), None)
        reps[name or "unnamed"].append((c, spk))

    def rep_row(items):
        cs = [c for c, _ in items]
        return {
            "calls": len(cs),
            "win_rate": round(sum(c["i"].get("outcome") == "won" for c in cs) / len(cs), 2),
            "avg_pitch_attempts": avg([c["s"].get("pitch_attempts") for c in cs]),
            "avg_close_attempts": avg([c["s"].get("close_attempts") for c in cs]),
            "avg_talk_ratio": avg([c["a"]["talk_ratio"].get(spk) for c, spk in items]),
            "asked_for_close_pct": round(sum(bool(c["s"].get("asked_for_close")) for c in cs) / len(cs), 2),
        }

    res = {
        "calls": len(calls),
        "outcomes": dict(outcomes),
        "win_rate": round(outcomes["won"] / decided, 2) if decided else None,
        # currencies are never mixed: average of won deals per currency
        "avg_deal_won_by_currency": {
            cur: avg(amts) for cur, amts in _won_amounts(calls).items()
        },
        "avg_duration_s": avg([c["a"]["duration_s"] for c in calls]),
        "avg_pitch_attempts": avg([c["s"].get("pitch_attempts") for c in calls]),
        "avg_close_attempts": avg([c["s"].get("close_attempts") for c in calls]),
        "excluded_not_sales": not_sales,
        "stage_reached": dict(stages),
        "by_category": dict(Counter((c["d"].get("category_label") or "unclassified") for c in calls)),
        "objections_by_type": dict(obj.most_common()),
        "by_rep": {name: rep_row(items) for name, items in reps.items()},
        "per_call": [
            {"file": c["file"], "outcome": c["i"].get("outcome"), "pitch": c["s"].get("pitch_attempts"),
             "close": c["s"].get("close_attempts"), "stage": c["s"].get("stage_reached")}
            for c in calls
        ],
    }
    return res


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    try:
        res = build(out_dir)
    except ValueError as e:
        sys.exit(str(e))
    out = out_dir / "aggregate.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
