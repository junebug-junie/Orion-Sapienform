"""Live check for rung 6 (Orion tunes their own surprise settings).

Run after the substrate runtime has recorded raw readings for a while. Read-only.

Per domain (execution, chat) it reports and gates:

- ``coverage``: readings in the tuner window; the tuner needs
  WARMUP + MIN_FIT + MIN_HOLDOUT before it can propose anything.
- ``not_flat``: more than one distinct raw value.
- ``can_rest``: share of readings at the minimum value. Reported, not gated:
  a domain that never returns to a calm floor is worth a look, not a failure.
- ``assessment``: what the tuner would propose right now against the live
  setting (or why nothing), with the fit gain and its lower bound.

Plus the rung's own trail: the live control-surface value per domain and every
prediction_expectation_patch proposal/trial/decision/adoption/rollback in the
window. Exit status 1 if a domain with coverage is flat.

Run:
  POSTGRES_URI=postgresql://... python services/orion-substrate-runtime/evals/run_prediction_expectation_live_check.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.substrate.prediction_expectation import (  # noqa: E402
    MIN_FIT_READINGS,
    MIN_HOLDOUT_READINGS,
    WARMUP_READINGS,
    assess,
    domain_specs,
    expectation_from_payload,
)


def summarize(domain: str, readings: list[tuple[datetime, float]], live_payload: dict | None) -> dict:
    spec = domain_specs()[domain]
    live = expectation_from_payload(live_payload, default=spec.default)
    values = [v for _, v in readings]
    needed = WARMUP_READINGS + MIN_FIT_READINGS + MIN_HOLDOUT_READINGS
    out: dict = {
        "domain": domain,
        "live": live.as_patch(),
        "live_is_default": live == spec.default,
        "readings": len(values),
        "readings_needed": needed,
        "coverage": len(values) >= needed,
        "not_flat": len(set(values)) > 1,
    }
    if values:
        floor = min(values)
        out["min"] = floor
        out["max"] = max(values)
        out["share_at_min"] = round(sum(1 for v in values if v == floor) / len(values), 3)
    a = assess(spec, readings, current=live) if out["coverage"] else None
    out["assessment"] = (
        {
            "candidate": a.candidate.as_patch(),
            "fit_gain_mean": round(a.fit_gain.mean, 5),
            "fit_gain_lower_bound": round(a.fit_gain.lower_bound, 5),
            "fit_readings": a.fit_gain.count,
        }
        if a is not None
        else ("no_confident_improvement" if out["coverage"] else "insufficient_readings")
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=float, default=72.0)
    args = parser.parse_args()
    url = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not url:
        print("POSTGRES_URI or DATABASE_URL required", file=sys.stderr)
        return 2
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    since = datetime.now(timezone.utc) - timedelta(hours=args.window_hours)
    failed = False
    with engine.connect() as conn:
        for domain, spec in domain_specs().items():
            rows = conn.execute(
                text(
                    "SELECT observed_at, raw_value FROM substrate_prediction_raw_reading "
                    "WHERE domain=:d AND observed_at >= :s ORDER BY observed_at"
                ),
                {"d": domain, "s": since},
            ).fetchall()
            live_row = conn.execute(
                text("SELECT value_json FROM substrate_runtime_control_surface WHERE surface_key=:k"),
                {"k": spec.control_surface_key},
            ).fetchone()
            summary = summarize(domain, [(r[0], float(r[1])) for r in rows], live_row[0] if live_row else None)
            print(summary)
            if summary["coverage"] and not summary["not_flat"]:
                failed = True
        trail = Counter()
        for table in ("proposal", "trial", "decision"):
            for row in conn.execute(
                text(
                    f"SELECT payload_json->>'action', payload_json->>'reason', payload_json->>'status' "
                    f"FROM substrate_mutation_{table} WHERE created_at >= :s "
                    "AND (payload_json->>'mutation_class' = 'prediction_expectation_patch' "
                    "OR payload_json->>'proposal_id' IN (SELECT proposal_id FROM substrate_mutation_proposal "
                    "WHERE payload_json->>'mutation_class' = 'prediction_expectation_patch'))"
                ),
                {"s": since},
            ).fetchall():
                trail[(table, row[0] or row[2] or "-", row[1] or "-")] += 1
        for table in ("adoption", "rollback"):
            n = conn.execute(
                text(
                    f"SELECT count(*) FROM substrate_mutation_{table} WHERE created_at >= :s "
                    "AND payload_json->>'target_surface' LIKE 'prediction_expectation:%'"
                    if table == "adoption"
                    else f"SELECT count(*) FROM substrate_mutation_{table} WHERE created_at >= :s"
                ),
                {"s": since},
            ).scalar()
            trail[(table, "count", "-")] = int(n or 0)
    for key, n in sorted(trail.items()):
        print("trail", key, n)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
