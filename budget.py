"""Hard spend cap for LLM calls, persisted across runs.

Settings (.env): LLM_BUDGET_USD (default 0.20), LLM_PRICE_IN_PER_M (default 0.10),
LLM_PRICE_OUT_PER_M (default 0.50). Cost is estimated from the token usage the API
reports, so it is only as accurate as those prices.

    python budget.py            # show spent / remaining
    python budget.py reset      # zero the counter
"""
import json
import os
import sys
import threading
from pathlib import Path

import config  # noqa: F401  (loads .env)

FILE = Path(__file__).parent / ".llm_spend.json"
_lock = threading.Lock()  # analysis samples run in parallel threads


class BudgetExceeded(RuntimeError):
    pass


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def limit() -> float:
    return _f("LLM_BUDGET_USD", 0.20)


def price() -> tuple:
    return _f("LLM_PRICE_IN_PER_M", 0.10), _f("LLM_PRICE_OUT_PER_M", 0.50)


def load() -> dict:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}


def cost(tokens_in: int, tokens_out: int) -> float:
    pin, pout = price()
    return (tokens_in * pin + tokens_out * pout) / 1_000_000


def check(prompt: str, max_out_tokens: int) -> None:
    """Refuse before the call if even a worst-case reply would pass the cap."""
    worst = cost(len(prompt) // 3 + 1, max_out_tokens)  # ~3 chars/token: deliberately pessimistic
    s = load()
    if s["usd"] + worst > limit():
        raise BudgetExceeded(
            f"LLM budget ${limit():.2f} reached (spent ${s['usd']:.4f}, next call up to ${worst:.4f}). "
            "Raise LLM_BUDGET_USD in .env or run `python budget.py reset`."
        )


def record(tokens_in: int, tokens_out: int) -> None:
    with _lock:
        s = load()
        s["usd"] = round(s["usd"] + cost(tokens_in, tokens_out), 6)
        s["calls"] += 1
        s["input_tokens"] += tokens_in
        s["output_tokens"] += tokens_out
        FILE.write_text(json.dumps(s), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        FILE.unlink(missing_ok=True)
        print("reset")
    s = load()
    print(f"spent ${s['usd']:.4f} of ${limit():.2f}  ({s['calls']} calls, "
          f"{s['input_tokens']} in / {s['output_tokens']} out tokens)  remaining ${limit() - s['usd']:.4f}")
