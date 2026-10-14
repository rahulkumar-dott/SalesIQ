"""Per-call-type field templates (playbooks.json) for the LLM analysis.

prompt_section()  -> text to append to the analysis prompt
parse(raw)        -> validated {category, category_label, details, qualification, compliance}

Layers:
  common     : qualification (need/budget/authority/timeline), deal, sales metrics - every call
  per type   : the 'fields' of the category the model picks
  regulated  : compliance checks, only for categories with "regulated": true
"""
import json
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).parent / "playbooks.json"
STATUS = {"confirmed", "likely", "unknown"}
RESULTS = {"yes", "no", "not_applicable", "unclear"}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(FILE.read_text(encoding="utf-8"))


def prompt_section() -> str:
    pb = load()
    cats = pb["categories"]
    lines = [
        "6. category: the ONE type of product being sold: "
        + ", ".join(f'"{k}" ({v["describe"]})' for k, v in cats.items())
        + '. Use "other" if none fits.',
        "7. details: fill the fields of the chosen category only, using null when not discussed. Fields per category:",
    ]
    for k, v in cats.items():
        if v["fields"]:
            lines.append(f'   - {k}: ' + "; ".join(f"{f} ({d})" for f, d in v["fields"].items()))
        else:
            lines.append(f"   - {k}: free-form: up to 8 short snake_case keys of your own for the key facts")
    q = pb["qualification"]
    lines.append(
        "8. qualification (every call): for each of "
        + ", ".join(f"{k} ({d})" for k, d in q.items())
        + ' return {status: "confirmed"|"likely"|"unknown", detail: str|null}.'
    )
    regulated = [k for k, v in cats.items() if v.get("regulated")]
    cc = pb["compliance_checks"]
    lines.append(
        f"9. compliance: ONLY if category is one of {regulated}, else {{}}. For each check return "
        '{result: "yes"|"no"|"not_applicable"|"unclear", evidence: short verbatim quote or null}: '
        + "; ".join(f"{k} ({d})" for k, d in cc.items())
        + ". Use \"no\" only when the transcript shows it clearly did not happen; "
        'use "unclear" when the clip may simply not include it.'
    )
    return "\n".join(lines) + "\n"


def form_hint() -> str:
    return '"category": "...", "details": {...}, "qualification": {"need": {...}, ...}, "compliance": {...}'


def parse(raw: dict) -> dict:
    pb = load()
    cats = pb["categories"]
    cat = raw.get("category")
    cat = cat if cat in cats else "other"
    spec = cats[cat]

    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    if spec["fields"]:
        details = {k: details.get(k) for k in spec["fields"]}
    else:  # free-form fallback: keep a few keys, scalars/lists only
        details = {str(k)[:40]: v for k, v in list(details.items())[:8]}

    q_in = raw.get("qualification") if isinstance(raw.get("qualification"), dict) else {}
    qualification = {}
    for k in pb["qualification"]:
        v = q_in.get(k) if isinstance(q_in.get(k), dict) else {}
        status = v.get("status") if v.get("status") in STATUS else "unknown"
        qualification[k] = {"status": status, "detail": v.get("detail")}

    compliance = {}
    if spec.get("regulated"):
        c_in = raw.get("compliance") if isinstance(raw.get("compliance"), dict) else {}
        for k in pb["compliance_checks"]:
            v = c_in.get(k) if isinstance(c_in.get(k), dict) else {}
            res = v.get("result") if v.get("result") in RESULTS else "unclear"
            compliance[k] = {"result": res, "evidence": v.get("evidence")}
    return {
        "category": cat,
        "category_label": spec["label"],
        "details": details,
        "qualification": qualification,
        "compliance": compliance,
    }
