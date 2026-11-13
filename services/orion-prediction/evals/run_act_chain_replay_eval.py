"""Replay the act chain (app/act_chain.py) over Orion's real acts, bucket by bucket (read-only).

Every 15-minute bucket after a 7-day warm-up is forecast from acts landed strictly before it, with the
same code a live forecaster would run, then graded: log loss vs the three baselines by local day,
excess surprise (rest point 0 by theory), calibration, the rolling-1h rest check, and the most
surprising buckets (to check spikes sit on real events).

Run in the orion-prediction image, POSTGRES_URI and network from orion-athena-prediction:
    python evals/run_act_chain_replay_eval.py [--since 2026-10-23] [--warmup-days 7]
Exit 0 always: the verdict is a report, not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]

import numpy as np  # noqa: E402

from app import act_chain as C  # noqa: E402
from app.store import PostgresStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    # 2026-10-23: first day with >= 6 of today's 9 producing verbs (act producers rolled out 10-15..10-30).
    ap.add_argument("--since", default="2026-10-23")
    ap.add_argument("--warmup-days", type=float, default=7.0)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    start = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    store = PostgresStore(os.environ["POSTGRES_URI"])
    acts = store.grammar_acts(start, end)
    store.close()
    rows = C.replay(acts, start, end, timedelta(days=a.warmup_days))
    v = C.verdict(rows)
    top = sorted(rows, key=lambda r: -r["scores"][C.MODEL]["surprise"])[: a.top]
    v["most_surprising"] = [{"origin": r["origin"].isoformat(), "block": r["block"], "prev": r["prev_state"],
                             "observed": r["observed"], "p_observed": float(r["probs"][C.MODEL][r["observed_i"]]),
                             "surprise": r["scores"][C.MODEL]["surprise"]} for r in top]
    v["chosen_k_median"] = {m: float(np.median([r["k"][m] for r in rows if m in r["k"]])) for m in rows[-1]["k"]} if rows else {}
    v["daily_excess"] = {str(d): x for d, x in sorted(C._by_day(rows, C.MODEL, "excess").items())}
    v["replay"] = {"from": start.isoformat(), "to": end.isoformat(), "acts": len(acts),
                   "acts_landed_35min_plus_after_open": sum(1 for o, c, _ in acts if c - o > C.SETTLE_GRACE + C.GRAIN)}
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main()
