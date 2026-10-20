"""Live event_pair pairs over vision_events, read-only (nothing is stored).

Each pair prints its counts, rate ratio and placebo p-value, or the refusal the endpoint would
return. A refusal is a correct answer when the data is not there (e.g. no doorbell rings yet).

    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_event_pair_live_eval.py --days 14

Pairs:
- doorbell_ring -> walkway person_presence, 0-300 s   (the spec's first pair; refuses until rings exist)
- doorbell person_presence -> walkway person_presence, 0-300 s   (same people, two cameras)
- walkway person_presence -> walkway camera_caption, 0-60 s   (sanity: captions follow presence)
- room cam0 person_presence -> walkway attention_worthy, 0-600 s   (placebo-ish: indoor vs street)

Each pair runs under both placebos. A pair significant only under ``circular`` shares the daily
rhythm rather than following A.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app.event_pair import EventPairRefused, event_pair_effect  # noqa: E402
from app.store import PostgresStore  # noqa: E402

PAIRS = [
    ("doorbell_ring", None, "person_presence", "walkway", 0, 300),
    ("person_presence", "doorbell", "person_presence", "walkway", 0, 300),
    ("person_presence", "walkway", "camera_caption", "walkway", 0, 60),
    ("person_presence", "cam0", "attention_worthy", "walkway", 0, 600),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--n-placebo", type=int, default=999)
    args = ap.parse_args()
    store = PostgresStore(os.environ["POSTGRES_URI"])
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    span = (end - start).total_seconds()
    for a_type, a_stream, b_type, b_stream, lo, hi in PAIRS:
        secs = []
        for et, st in ((a_type, a_stream), (b_type, b_stream)):
            rows = store.event_times("vision_events", et, st, start, end, 200_000)
            secs.append(np.array([(t - start).total_seconds() for t in rows]))
        label = f"{a_type}@{a_stream or '*'} -> {b_type}@{b_stream or '*'} [{lo},{hi}]s"
        for placebo in ("circular", "same_time_of_day"):
            print(f"{label} placebo={placebo}", json.dumps(_one(secs, span, lo, hi, args.n_placebo, placebo)))
    store.close()
    return 0


def _one(secs, span, lo, hi, n_placebo, placebo) -> dict:
    try:
        r = event_pair_effect(secs[0], secs[1], span, lo, hi, n_placebo=n_placebo, seed=0, placebo=placebo)
        return {"n_a": r.n_a, "n_b": r.n_b, "observed": r.observed_b_in_windows,
               "expected": round(r.expected_b_in_windows, 2), "rate_ratio": round(r.rate_ratio, 2),
               "p_value": round(r.p_value, 4), "placebo_q": {k: round(v, 2) for k, v in
                                                            r.placebo_ratio_quantiles.items()}}
    except EventPairRefused as exc:
        return {"refused": exc.reason, **exc.detail}


if __name__ == "__main__":
    raise SystemExit(main())
