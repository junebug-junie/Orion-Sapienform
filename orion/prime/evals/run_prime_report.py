"""Periodic eval: is Prime a cause or a narrator, and does its cadence follow price? (design section 9/10, 2A)

Reads the live ledgers (read-only):

* ``grammar_acts`` where ``actor = 'prime'``: thoughts (verb ``infer``) and questions (verb ``ask``, arm in
  ``test_before = holdback:<arm>``);
* ``want_ledger`` where ``family = 'prime'``: every bid and what the market decided, with the prices it saw.

Reports, whatever the result:

1. **Narrator test.** For closed questions: share whose graded curiosity run wrote something, treated vs
   control, with Fisher's exact two-sided p. A difference is called only at p < 0.05; with fewer than
   ``MIN_PER_ARM`` closed per arm the verdict is ``too_few``, never a lean.
2. **Thoughts.** Opened, produced, preempted, failed.
3. **Concerns** (verb ``predict``, Prime step 1): opened, settled (produced 1), and how the rest ended,
   read from each close note's first word (dropped / faded / lapsed) or the ledger's ``expired``.
4. **Cadence vs price.** Funded share of bids when Prime's lane and heat prices were zero vs above zero.
   2A's proof needs "cadence follows price on cool vs warm days": funded share lower when priced.

Run: ``PYTHONPATH=. python orion/prime/evals/run_prime_report.py --dsn postgresql://... [--days 14]``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

MIN_PER_ARM = 10
ALPHA = 0.05


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """p for the 2x2 table [[a, b], [c, d]] (rows: arms; cols: produced, not)."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    lo, hi = max(0, k - n2), min(k, n1)
    observed = pmf(a)
    return min(1.0, sum(p for p in (pmf(x) for x in range(lo, hi + 1)) if p <= observed * (1 + 1e-9)))


@dataclass
class ArmCount:
    closed: int = 0
    produced: int = 0
    open: int = 0

    @property
    def rate(self) -> Optional[float]:
        return self.produced / self.closed if self.closed else None


def narrator(acts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    arms = {"treated": ArmCount(), "control": ArmCount()}
    for act in acts:
        if act.get("verb") != "ask":
            continue
        arm = str(act.get("test_before") or "").removeprefix("holdback:")
        if arm not in arms:
            continue
        if act.get("outcome_value") is None:
            arms[arm].open += 1
            continue
        arms[arm].closed += 1
        arms[arm].produced += int(float(act["outcome_value"]) == 1.0)
    t, c = arms["treated"], arms["control"]
    out: dict[str, Any] = {arm: vars(v) | {"rate": v.rate} for arm, v in arms.items()}
    if min(t.closed, c.closed) < MIN_PER_ARM:
        out["verdict"] = "too_few"
        out["p"] = None
        return out
    p = fisher_exact_two_sided(t.produced, t.closed - t.produced, c.produced, c.closed - c.produced)
    out["p"] = p
    out["verdict"] = "no_difference" if p >= ALPHA else ("cause_up" if t.rate > c.rate else "cause_down")
    return out


def thoughts(acts: Iterable[dict[str, Any]]) -> dict[str, int]:
    out = {"opened": 0, "produced": 0, "preempted": 0, "failed": 0, "open": 0}
    for act in acts:
        if act.get("verb") != "infer":
            continue
        out["opened"] += 1
        note = str(act.get("outcome_note") or "")
        if act.get("outcome_value") is None:
            out["open"] += 1
        elif float(act["outcome_value"]) == 1.0:
            out["produced"] += 1
        elif note.startswith("preempted"):
            out["preempted"] += 1
        else:
            out["failed"] += 1
    return out


def concerns(acts: Iterable[dict[str, Any]]) -> dict[str, int]:
    out = {"opened": 0, "open": 0, "settled": 0, "dropped": 0, "faded": 0, "lapsed": 0, "expired": 0, "other": 0}
    for act in acts:
        if act.get("verb") != "predict":
            continue
        out["opened"] += 1
        if act.get("outcome_value") is None:
            out["open"] += 1
            continue
        if float(act["outcome_value"]) == 1.0:
            out["settled"] += 1
            continue
        word = str(act.get("outcome_note") or "").split(":", 1)[0].split(" ", 1)[0]
        key = "expired" if act.get("outcome_source") == "expired" else word
        out[key if key in out and key not in ("opened", "open", "settled") else "other"] += 1
    return out


def cadence(decisions: Iterable[dict[str, Any]], *, lane: str = "chat") -> dict[str, Any]:
    buckets = {"unpriced": [0, 0], "priced": [0, 0]}   # [funded, total]
    for row in decisions:
        prices = ((row.get("decision") or {}).get("prices_seen") or {})
        priced = float(prices.get(f"lane:{lane}") or 0) > 0 or float(prices.get("heat") or 0) > 0
        b = buckets["priced" if priced else "unpriced"]
        b[1] += 1
        b[0] += int(row.get("outcome") == "funded")
    return {k: {"funded": f, "bids": n, "funded_share": (f / n if n else None)} for k, (f, n) in buckets.items()}


ACTS_SQL = """
SELECT verb, test_before, outcome_value, outcome_source, outcome_note, opened_at, closed_at
  FROM grammar_acts WHERE actor = 'prime' AND opened_at > now() - make_interval(days => %s)
"""
BIDS_SQL = """
SELECT outcome, reason, decision FROM want_ledger
 WHERE family = 'prime' AND decided_at > now() - make_interval(days => %s)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    import psycopg2
    import psycopg2.extras

    with psycopg2.connect(args.dsn) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(ACTS_SQL, (args.days,))
        acts = [dict(r) for r in cur.fetchall()]
        cur.execute(BIDS_SQL, (args.days,))
        bids = [dict(r) for r in cur.fetchall()]
    report = {"days": args.days, "narrator": narrator(acts), "thoughts": thoughts(acts), "concerns": concerns(acts),
              "cadence": cadence(bids)}
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
