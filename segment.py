"""Turn transcribe.py output into speaker utterances with roles.

Usage:
    python segment.py output/sample-1.speakers.json [--gap 1.5] [--roles auto|profile|llm|heuristic]

Writes output/<name>.utterances.json and prints a readable conversation.

Steps:
  1. flatten words, group consecutive words by speaker into utterances
     (a long silence inside one speaker's turn also starts a new utterance)
  2. identify which speaker is the salesperson vs the customer
       profile    : match voices against enrolled reps (profiles.py); most reliable
       llm        : LLM reads the whole call, returns role + confidence + quote (see config.py)
       heuristic  : keyword scoring, no API; least reliable, only on request
     auto tries profile, then llm.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import budget
import config
import pii
import playbooks
import punct

LLM_MODEL = "claude-haiku-5-5"

SELLER_CUES = [
    r"\bour (platform|product|solution|tool|software|team|pricing|plan)\b", r"\bwe (offer|provide|can|help|have)\b",
    r"\bi understand\b", r"\binvestment\b", r"\broi\b", r"\bdemo\b", r"\bpackage\b", r"\bdiscount\b",
    r"\bsave\b", r"\bsaved\b", r"\bbenefit", r"\bcase study\b", r"\bnext steps?\b", r"\bsign\b",
    r"\bimplementation\b", r"\bonboarding\b", r"\bcontract\b", r"\bthank you for\b",
]
BUYER_CUES = [
    r"\bhow much\b", r"\btoo (expensive|much|high)\b", r"\bbudget\b", r"\bhurdle\b", r"\bwe currently\b",
    r"\bour (current|existing)\b", r"\bi('m| am) not sure\b", r"\bthink about it\b", r"\bcompetitor",
    r"\bdoes it (work|integrate|support)\b", r"\bcan you (send|explain)\b", r"\bmy team\b",
]


def build_utterances(data: dict, gap: float):
    words = []
    for seg in data["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "speaker" in w:
                words.append(w)
    words.sort(key=lambda w: w["start"])

    utts, cur = [], None
    for w in words:
        if cur and w["speaker"] == cur["speaker"] and w["start"] - cur["end"] <= gap:
            cur["words"].append(w)
            cur["end"] = w["end"]
        else:
            cur = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": [w]}
            utts.append(cur)
    for u in utts:
        # words/timestamps stay as they are; only the text gets punctuation (punct.py), then PII masking (pii.py)
        text = punct.restore(" ".join(x["word"] for x in u["words"] if x["word"]))
        u["text"], found = pii.mask_text(text)
        u["pii"] = dict(found)
    return utts


def heuristic_roles(utts):
    score = {}
    for u in utts:
        t = u["text"].lower()
        s = score.setdefault(u["speaker"], {"seller": 0, "buyer": 0, "words": 0})
        s["words"] += len(u["words"])
        s["seller"] += sum(len(re.findall(p, t)) for p in SELLER_CUES)
        s["buyer"] += sum(len(re.findall(p, t)) for p in BUYER_CUES)
    if len(score) < 2:
        return {k: "unknown" for k in score}, score
    # net seller-ness per speaker; the highest is the salesperson, others customers
    net = {k: v["seller"] - v["buyer"] for k, v in score.items()}
    ranked = sorted(net, key=lambda k: (net[k], score[k]["words"]), reverse=True)
    roles = {k: "customer" for k in score}
    roles[ranked[0]] = "salesperson"
    return roles, score


def ask_llm(prompt: str, max_tokens: int = 600) -> str:
    """One text completion via whichever backend config.llm_backend() selects."""
    backend = config.llm_backend()
    budget.check(prompt, max(max_tokens, 2000))  # raises BudgetExceeded before any spend
    if backend == "azure_openai":
        from openai import AzureOpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or "https://{}.openai.azure.com".format(
            os.environ.get("AZURE_OPENAI_RESOURCE") or os.environ["ANTHROPIC_FOUNDRY_RESOURCE"]
        )
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ["ANTHROPIC_FOUNDRY_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        kw = {}
        if os.environ.get("OPENAI_REASONING_EFFORT"):  # only for reasoning models
            kw["reasoning"] = {"effort": os.environ["OPENAI_REASONING_EFFORT"]}
        # Responses API; max_output_tokens also covers reasoning tokens, so leave headroom
        r = client.responses.create(
            model=os.environ["OPENAI_DEPLOYMENT"],
            input=prompt,
            max_output_tokens=max(max_tokens, 2000),
            text={"format": {"type": "json_object"}},
            **kw,
        )
        budget.record(r.usage.input_tokens, r.usage.output_tokens)
        if getattr(r, "status", None) == "incomplete":  # reasoning + answer hit the cap: retrying won't help
            raise RuntimeError(
                f"model output truncated at {max(max_tokens, 2000)} tokens "
                f"(reasoning used {getattr(r.usage.output_tokens_details, 'reasoning_tokens', '?')}); raise max tokens"
            )
        return r.output_text
    if backend in ("foundry", "anthropic"):
        import anthropic

        if backend == "foundry":
            client = anthropic.AnthropicFoundry(
                api_key=os.environ["ANTHROPIC_FOUNDRY_API_KEY"], resource=os.environ["ANTHROPIC_FOUNDRY_RESOURCE"]
            )
            model = os.environ.get("CLAUDE_DEPLOYMENT") or LLM_MODEL
        else:
            client, model = anthropic.Anthropic(), LLM_MODEL
        m = client.messages.create(
            model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        budget.record(m.usage.input_tokens, m.usage.output_tokens)
        return next(b.text for b in m.content if b.type == "text")
    raise RuntimeError("no LLM configured (see .env.example)")


INTENTS = [
    "greeting", "rapport", "discovery", "pitch", "demo", "objection",
    "pricing", "negotiation", "next_steps", "closing", "other",
]


def llm_analyze(utts, max_chars=60000):
    """One LLM call over the whole call. Returns
    {"roles": {speaker: {role, confidence, evidence}}, "intents": {index: label}, "summary": str}.

    Role is decided once per speaker (not per utterance); low confidence -> "unknown".
    """
    lines, n = [], 0
    for i, u in enumerate(utts):
        line = f"[{i}] {u['speaker']}: {u['text']}"
        n += len(line)
        if n > max_chars:
            break
        lines.append(line)
    speakers = sorted({u["speaker"] for u in utts})
    prompt = (
        "Below is a diarized transcript of a sales call (it may start mid-conversation); "
        "each line is [index] speaker: text.\n"
        "Tasks:\n"
        "0. conversation: FIRST decide whether this is a SALES conversation, meaning someone is trying to "
        "sell, quote, demo, negotiate or close a product or service (follow-ups and renewals count). "
        "Return {kind: \"sales_call\"|\"customer_support\"|\"internal_meeting\"|\"interview\"|"
        "\"personal_chat\"|\"other\"|\"unclear\", is_sales: bool, confidence: 0.0-1.0, reason: str, "
        "evidence: short verbatim quote}. Casual talk, support or complaint calls, meetings and interviews are "
        "NOT sales. If is_sales is false, still answer the other tasks but briefly: roles \"unknown\", all counts 0, "
        "other fields null or empty.\n"
        "1. roles: for each speaker decide \"salesperson\" (the one selling) or \"customer\" "
        "(the one being sold to), or \"unknown\" if the evidence is not clear.\n"
        f"2. intents: label every line index with one of {INTENTS}.\n"
        "3. summary: 2-3 sentences on what happened and the outcome.\n"
        "4. insights: facts a sales manager wants, based only on what is said (use null or [] when absent): "
        "outcome (won|lost|follow_up|undecided|unclear), deal_value, objections "
        "[{objection, resolved: true|false, how}], buying_signals [str], competitors [str], "
        "decision_maker (who decides / approval process), next_steps [str], "
        "customer_sentiment (positive|neutral|negative) and its trend, "
        "rep_performance {strengths: [str], improvements: [str]}.\n"
        "5. metrics (about what the SALESPERSON did; 0 or [] if none; do NOT do arithmetic or totals yourself): "
        "{attempts: [{i: line index, kind: \"pitch\"|\"close\", what: short description}] (list EVERY "
        "salesperson line that presents a product/feature/benefit as a pitch, or asks for a "
        "commitment/decision/signature as a close; one entry per line per kind; the code counts them), "
        "stage_reached (discovery|demo|pricing|negotiation|close|unclear), "
        "discovery_questions: {open: int, closed: int}, "
        "objections_by_type: {price: int, timing: int, authority: int, competitor: int, other: int}, "
        "buying_signal_count: int, price_quoted: str|null, "
        "product_sold: str|null (what is being sold, e.g. a software plan, a car, an insurance policy), "
        "deal: {pricing: {recurring_amount: number|null (the recurring price per period exactly as stated), "
        "recurring_period: \"monthly\"|\"yearly\"|null, term_months: int|null (length of the deal or pilot "
        "discussed), free_months: number (months given free, 0 if none), one_time_fees: [numbers actually "
        "charged once; waived fees are NOT included]}, "
        "amount: number|null (ONLY when there is no recurring price: vehicle price, loan principal, TOTAL "
        "premium per period including riders and NOT the sum assured, quoted rent, or a flat price; leave null "
        "when recurring_amount is set, the total is computed from pricing), "
        "currency: ISO code like USD|INR|EUR|null, "
        "basis: one_time|monthly|yearly|total_contract|premium_monthly|premium_yearly|principal|rent_monthly|unknown, "
        "description: str|null (e.g. '2-year plan, $11,500/yr' or 'sum assured 50 lakh, premium 12,000/yr')}, "
        "discount_offered: bool, negotiation_rounds: int, "
        "next_step_set: {set: bool, detail: str|null, date: str|null}, "
        "decision_maker_identified: bool, uncovered_needs: bool}.\n"
        + playbooks.prompt_section()
        + "\n"
        + "\n".join(lines)
        + "\n\nReply with only JSON of the form "
        '{"conversation": {"kind": "...", "is_sales": true, "confidence": 0.0, "reason": "...", "evidence": "..."}, '
        f'"roles": {{"{speakers[0]}": {{"role": "salesperson|customer|unknown", "confidence": 0.0-1.0, '
        '"evidence": "short verbatim quote from that speaker"}}, '
        '"intents": {"0": "greeting", "1": "discovery"}, "summary": "...", "insights": {...}, "metrics": {...}, '
        + playbooks.form_hint() + "}. "
        f"Speakers: {speakers}"
    )
    raw = None
    for attempt in range(3):  # models occasionally emit malformed JSON; retry
        text = ask_llm(prompt, max_tokens=8000 + 25 * len(lines))
        try:
            raw = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            break
        except (json.JSONDecodeError, AttributeError):
            continue
    if raw is None:
        raise RuntimeError("LLM returned invalid JSON 3 times")
    rr = {str(k).strip().lower(): v for k, v in (raw.get("roles") or {}).items()}
    roles = {}
    for spk in speakers:
        r = rr.get(spk.lower()) or {}
        role = r.get("role", "unknown")
        conf = float(r.get("confidence", 0) or 0)
        if role not in ("salesperson", "customer") or conf < 0.6:
            role = "unknown"
        roles[spk] = {"role": role, "confidence": round(conf, 2), "evidence": r.get("evidence", "")}
    intents = {}
    for k, v in (raw.get("intents") or {}).items():
        try:
            intents[int(k)] = v if v in INTENTS else "other"
        except (TypeError, ValueError):
            pass
    return {"roles": roles, "intents": intents, "summary": raw.get("summary", ""),
            "insights": raw.get("insights") or {}, "metrics": raw.get("metrics") or {},
            "conversation": _conversation(raw.get("conversation")),
            **playbooks.parse(raw), "labelled": len(lines)}


def _conversation(c) -> dict:
    """Is this a sales conversation at all? Unclear/low confidence is kept as 'unclear', not forced to a side."""
    kinds = {"sales_call", "customer_support", "internal_meeting", "interview", "personal_chat", "other", "unclear"}
    c = c if isinstance(c, dict) else {}
    kind = c.get("kind") if c.get("kind") in kinds else "unclear"
    try:
        conf = round(float(c.get("confidence") or 0), 2)
    except (TypeError, ValueError):
        conf = 0.0
    is_sales = c.get("is_sales")
    if not isinstance(is_sales, bool) or conf < 0.6:
        verdict = "unclear"
    else:
        verdict = "sales" if is_sales else "not_sales"
    return {"verdict": verdict, "kind": kind, "confidence": conf, "reason": c.get("reason"), "evidence": c.get("evidence")}


def _num(x):
    try:
        return float(x) if x is not None and not isinstance(x, bool) else None
    except (TypeError, ValueError):
        return None


def compute_deal(pricing) -> dict | None:
    """Deterministic deal arithmetic from the raw pricing pieces the model extracted.

    Returns None when there is no recurring price + term (the model's flat `amount` is then used).
    Free months are assumed to fall in the first year.
    """
    p = pricing if isinstance(pricing, dict) else {}
    amount, period = _num(p.get("recurring_amount")), p.get("recurring_period")
    term = _num(p.get("term_months"))
    assumed = False
    if (not term or term <= 0) and period == "yearly":
        term, assumed = 12.0, True  # a yearly price with no stated term: the priced period is one year
    if not amount or period not in ("monthly", "yearly") or not term or term <= 0:
        return None
    free = min(max(_num(p.get("free_months")) or 0.0, 0.0), term)
    fees = [f for f in (_num(x) for x in (p.get("one_time_fees") or [])) if f and f > 0]
    monthly = amount if period == "monthly" else amount / 12
    fee_total = sum(fees)
    r = lambda v: round(v, 2)
    return {
        "monthly_value": r(monthly),
        "annual_value": r(monthly * 12),
        "first_year_value": r(monthly * (min(12.0, term) - min(free, 12.0)) + fee_total),
        "total_contract_value": r(monthly * (term - free) + fee_total),
        "term_months": term,
        "term_assumed": assumed,
        "free_months": free,
        "one_time_fees_total": r(fee_total),
    }


def finalize_sales(sales: dict, utts: list, roles: dict) -> dict:
    """Numbers the code owns: deal totals from pricing pieces, attempt counts from tagged lines.

    roles: {speaker: "salesperson"|"customer"|...}. Counts only lines spoken by the salesperson.
    """
    sales = sales if isinstance(sales, dict) else {}
    deal = sales.get("deal") if isinstance(sales.get("deal"), dict) else {}
    comp = compute_deal(deal.get("pricing"))
    if comp:
        deal.update(amount=comp["total_contract_value"], basis="total_contract", computed=comp, amount_source="computed")
    sales["deal"] = deal

    sellers = {s for s, r in roles.items() if r == "salesperson"}
    seen, kept = set(), []
    for a in sales.get("attempts") or []:
        if not isinstance(a, dict) or a.get("kind") not in ("pitch", "close"):
            continue
        try:
            i = int(a.get("i"))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(utts) or utts[i]["speaker"] not in sellers or (i, a["kind"]) in seen:
            continue
        seen.add((i, a["kind"]))
        kept.append({"i": i, "kind": a["kind"], "what": a.get("what")})
    sales["attempts"] = kept
    for kind in ("pitch", "close"):
        items = [a for a in kept if a["kind"] == kind]
        sales[f"{kind}_attempts"] = len(items)
        sales[f"{kind}_examples"] = [a["what"] for a in items if a.get("what")][:5]
    sales["asked_for_close"] = sales["close_attempts"] > 0
    return sales


def audio_metrics(utts, data: dict, dead_air: float = 2.0, overlap_min: float = 0.3) -> dict:
    """Exact metrics from timestamps and diarization (no LLM)."""
    if not utts:
        return {}
    start, end = min(u["start"] for u in utts), max(u["end"] for u in utts)
    # merge consecutive utterances of one speaker into turns
    turns = []
    for u in utts:
        if turns and turns[-1]["speaker"] == u["speaker"]:
            turns[-1]["end"] = u["end"]
        else:
            turns.append({"speaker": u["speaker"], "start": u["start"], "end": u["end"]})
    talk, longest, n_turns = {}, {}, {}
    for t in turns:
        d = t["end"] - t["start"]
        n_turns[t["speaker"]] = n_turns.get(t["speaker"], 0) + 1
        longest[t["speaker"]] = max(longest.get(t["speaker"], 0.0), d)
    for u in utts:
        talk[u["speaker"]] = talk.get(u["speaker"], 0.0) + (u["end"] - u["start"])
    total = sum(talk.values()) or 1.0
    # silence between consecutive words/utterances
    gaps = [b["start"] - a["end"] for a, b in zip(utts, utts[1:]) if b["start"] - a["end"] >= dead_air]
    # overlaps from raw diarization turns; the later starter is counted as the interrupter
    inter, ov_secs = {}, 0.0
    st = sorted(data.get("speaker_turns") or [], key=lambda t: t["start"])
    for i, a in enumerate(st):
        for b in st[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if a["speaker"] != b["speaker"]:
                ov = min(a["end"], b["end"]) - b["start"]
                if ov >= overlap_min:
                    inter[b["speaker"]] = inter.get(b["speaker"], 0) + 1
                    ov_secs += ov
    r = lambda x: round(x, 2)
    return {
        "duration_s": r(end - start),
        "speakers": len(talk),
        "talk_seconds": {k: r(v) for k, v in talk.items()},
        "talk_ratio": {k: r(v / total) for k, v in talk.items()},
        "turns": n_turns,
        "avg_turn_s": {k: r(talk[k] / n_turns[k]) for k in talk},
        "longest_monologue_s": {k: r(v) for k, v in longest.items()},
        "interruptions": inter,
        "overlap_s": r(ov_secs),
        "dead_air_count": len(gaps),
        "dead_air_s": r(sum(gaps)),
    }


def _pick(values: list):
    """Most common value; ties prefer a non-null value, then the earliest."""
    from collections import Counter

    def key(v):  # 12000 and 12000.0 are the same amount
        return round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    c = Counter(key(v) for v in values)
    top = max(c.values())
    for v in values:  # first value with the top count, preferring non-null
        if c[key(v)] == top and v is not None:
            return v
    return values[0]


def _median_low(xs: list):
    xs = sorted(x for x in xs if x is not None)
    return xs[(len(xs) - 1) // 2] if xs else None


def analyze(utts: list, roles_override: dict | None = None, n: int | None = None) -> dict:
    """One analysis with a consensus over n parallel LLM samples (ANALYSIS_SAMPLES, default 3).

    Judgement fields are noisy run to run, so: counts = median, categorical fields = majority, and the
    deal comes from the majority amount. `consistency` exposes the per-sample values (disagreement = low
    trust). roles_override (e.g. from voice profiles) wins over the LLM's roles.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = max(1, int(n or os.environ.get("ANALYSIS_SAMPLES") or 3))
    errors, runs = [], []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(llm_analyze, utts) for _ in range(n)]
        for f in futs:
            try:
                runs.append(f.result())
            except Exception as e:  # noqa: BLE001 - keep the samples that worked
                errors.append(f"{type(e).__name__}: {e}")
    if not runs:
        raise RuntimeError("; ".join(errors) or "analysis failed")

    # final roles: override, else majority per speaker across samples
    speakers = sorted({sp for r in runs for sp in r["roles"]})
    roles = dict(roles_override) if roles_override else {
        sp: _pick([r["roles"].get(sp, {}).get("role", "unknown") for r in runs]) for sp in speakers
    }
    for r in runs:
        r["metrics"] = finalize_sales(r.get("metrics"), utts, roles)

    outcomes = [(r["insights"] or {}).get("outcome") for r in runs]
    cats = [r["category"] for r in runs]
    amounts = [(r["metrics"].get("deal") or {}).get("amount") for r in runs]
    cat, outcome, amount = _pick(cats), _pick(outcomes), _pick(amounts)
    # primary sample agrees with the consensus on type and amount, so details and deal come from one run
    primary = next((r for r in runs if r["category"] == cat and (r["metrics"].get("deal") or {}).get("amount") == amount),
                   next((r for r in runs if r["category"] == cat), runs[0]))
    m = primary["metrics"]
    pitch_all = [r["metrics"]["pitch_attempts"] for r in runs]  # per-sample values, before the consensus overwrites one
    close_all = [r["metrics"]["close_attempts"] for r in runs]
    pitch, close = _median_low(pitch_all), _median_low(close_all)
    m["pitch_attempts"], m["close_attempts"] = pitch, close
    m["asked_for_close"] = close > 0
    m["stage_reached"] = _pick([r["metrics"].get("stage_reached") for r in runs])
    if isinstance(primary.get("insights"), dict):
        primary["insights"]["outcome"] = outcome
    primary["roles_final"] = roles
    primary["consistency"] = {
        "samples": len(runs),
        "failed_samples": errors,
        "outcome": outcomes,
        "category": cats,
        "deal_amount": amounts,
        "pitch_attempts": pitch_all,
        "close_attempts": close_all,
    }
    return primary


def llm_roles(utts, max_chars=60000):
    return llm_analyze(utts, max_chars)["roles"]


def profile_roles(data: dict, threshold: float):
    """Roles from enrolled voice profiles.

    A speaker that matches a profile gets that profile's role (and name). In a
    two-party call, a non-matching speaker is the customer. Returns
    (None, None, {}) when no profiles exist or none matched, so callers can fall back.
    """
    import profiles as P

    if not P.load_profiles() or not data.get("audio") or not data.get("speaker_turns"):
        return None, None, {}
    res = P.identify_speakers(data["audio"], data["speaker_turns"], threshold)
    if not any(r["matched"] for r in res.values()):
        return None, None, {}
    roles = {s: (r["role"] if r["matched"] else "customer") for s, r in res.items()}
    names = {s: r["name"] for s, r in res.items() if r["matched"]}
    return roles, res, names


def fmt(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sum_counts(dicts) -> dict:
    total = {}
    for d in dicts:
        for k, v in (d or {}).items():
            total[k] = total.get(k, 0) + v
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("transcript")
    p.add_argument("--gap", type=float, default=1.5, help="silence (s) that splits one speaker's turn")
    p.add_argument("--roles", choices=["auto", "profile", "llm", "heuristic"], default="auto")
    p.add_argument("--threshold", type=float, default=0.5, help="voice-profile match threshold")
    p.add_argument("--no-intents", action="store_true", help="skip the LLM intent/summary pass")
    a = p.parse_args()

    data = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    utts = build_utterances(data, a.gap)

    method = a.roles
    roles, evidence, names = None, None, {}
    if method in ("auto", "profile"):
        roles, evidence, names = profile_roles(data, a.threshold)
        if roles is not None:
            method = "profile"
        elif method == "profile":
            sys.exit("no enrolled voice profile matched any speaker (see profiles.py)")
        else:
            method = "llm"  # no regex fallback: unreliable (see --roles heuristic to force)
    analysis = None
    if method == "llm" or (not a.no_intents and config.llm_backend() != "none" and method != "heuristic"):
        # consensus over parallel samples; voice-profile roles (if any) override the LLM's
        analysis = analyze(utts, roles_override=roles if method == "profile" else None)
    if method == "llm":
        roles = analysis["roles_final"]
        evidence = analysis["roles"]
    elif method == "heuristic":
        roles, evidence = heuristic_roles(utts)
    intents = (analysis or {}).get("intents", {})
    # analyze() already finalized the metrics with the final roles; do not recompute (it would undo the consensus)

    for i, u in enumerate(utts):
        u["intent"] = intents.get(i)
        u["role"] = roles.get(u["speaker"], "unknown")
        u["name"] = names.get(u["speaker"])
        u["start"], u["end"] = round(u["start"], 2), round(u["end"], 2)

    speaking = {}
    for u in utts:
        speaking[u["speaker"]] = speaking.get(u["speaker"], 0.0) + (u["end"] - u["start"])

    out = Path(a.transcript).with_suffix(".utterances.json")
    out.write_text(
        json.dumps(
            {
                "audio": data.get("audio"),
                "num_speakers": data.get("num_speakers"),
                "role_method": method,
                "roles": roles,
                "role_evidence": evidence,
                "summary": (analysis or {}).get("summary"),
                "insights": (analysis or {}).get("insights"),
                "conversation": (analysis or {}).get("conversation"),
                "consistency": (analysis or {}).get("consistency"),
                "pii_masked": _sum_counts(u.get("pii") for u in utts),
                "category": (analysis or {}).get("category"),
                "category_label": (analysis or {}).get("category_label"),
                "details": (analysis or {}).get("details"),
                "qualification": (analysis or {}).get("qualification"),
                "compliance": (analysis or {}).get("compliance"),
                "metrics": {
                    "audio": audio_metrics(utts, data),
                    "sales": (analysis or {}).get("metrics"),
                },
                "talk_seconds": {k: round(v, 1) for k, v in speaking.items()},
                "utterances": [
                    {k: u[k] for k in ("start", "end", "speaker", "role", "name", "intent", "text")} for u in utts
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"speakers: {data.get('num_speakers')}   roles ({method}): {roles}")
    print(f"talk time: { {k: round(v) for k, v in speaking.items()} }  utterances: {len(utts)}\n")
    for u in utts:
        tag = f"({u['intent']})" if u.get("intent") else ""
        print(f"[{fmt(u['start'])}] {u['role'].upper():11s} {tag:14s} {u['text']}")
    if analysis and analysis.get("summary"):
        print(f"\nsummary: {analysis['summary']}")
    if analysis and analysis.get("insights"):
        print("\ninsights: " + json.dumps(analysis["insights"], indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    sys.exit(main())
