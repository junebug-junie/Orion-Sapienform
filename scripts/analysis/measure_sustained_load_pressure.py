#!/usr/bin/env python3
"""Measure `orion.field.significance.sustained_load_pressure` over real history.

CLAUDE.md 0A's metric-quality-gate items 2 and 4 for this metric: is it
independent of `deviation_pressure` (already in `PRESSURE_DIMENSIONS`), and
is its real distribution non-degenerate (real variance, reaches a genuine
rest state)? Both need real numbers, not architectural reasoning alone --
this script produces them.

Read-only: opens a read-only Postgres session, writes nothing, publishes
nothing. Same contract as `measure_field_tension_admission.py`.

Usage:
    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_sustained_load_pressure.py --hours 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.attention.tension.competition import FieldTensionCompetition, deviation_pressure  # noqa: E402
from orion.field.significance import compute_tick, sustained_load_pressure  # noqa: E402

logger = logging.getLogger("measure_sustained_load_pressure")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


def _fetch_rows(engine, hours: float):
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT generated_at, field_json
                FROM substrate_field_state
                -- secs, not hours: make_interval's hours arg is int, so a
                -- float --hours raises UndefinedFunction. Same pitfall fixed
                -- live in services/orion-hub/scripts/tension_outreach_
                -- trigger.py, 2026-08-18 -- see that fix's own commit for
                -- the full account (it had been silently swallowed there by
                -- a broad except; caught here before this script ever ran
                -- for real).
                WHERE generated_at > now() - make_interval(secs => :secs)
                ORDER BY generated_at ASC
                """
            ),
            {"secs": hours * 3600.0},
        ).fetchall()
    out = []
    for generated_at, field_json in rows:
        payload = json.loads(field_json) if isinstance(field_json, str) else field_json
        out.append((generated_at, payload))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--window-seconds", type=float, default=900.0, help="sustained_load_pressure's own window")
    parser.add_argument("--step-seconds", type=float, default=30.0, help="how often a live producer would recompute")
    parser.add_argument("--postgres-uri", default=os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI))
    args = parser.parse_args()

    from sqlalchemy import create_engine

    engine = create_engine(args.postgres_uri, pool_pre_ping=True)
    rows = _fetch_rows(engine, args.hours)
    if not rows:
        logger.error("no rows in the requested window")
        return 1
    logger.info("fetched %d rows spanning %.2fh", len(rows), args.hours)

    # deviation_pressure, replayed chronologically once across the whole
    # history -- same shape measure_field_tension_admission.py already uses,
    # needed here only as the second series for the independence check.
    competition = FieldTensionCompetition()
    dp_series: list[tuple[object, float]] = []
    for generated_at, payload in rows:
        tick = competition.observe_tick({"node_vectors": payload.get("node_vectors", {})})
        dp_series.append((generated_at, deviation_pressure(tick)))

    # sustained_load_pressure, computed at a slow cadence over a sliding
    # window of the SAME real payloads -- exactly what the live producer
    # will do, just replayed offline.
    slp_series: list[tuple[object, float]] = []
    window = timedelta(seconds=args.window_seconds)
    step = timedelta(seconds=args.step_seconds)
    next_check = rows[0][0] + window
    lo = 0
    for hi in range(len(rows)):
        generated_at, _ = rows[hi]
        if generated_at < next_check:
            continue
        window_start = generated_at - window
        while lo < hi and rows[lo][0] < window_start:
            lo += 1
        payloads = [p for _, p in rows[lo : hi + 1]]
        tick = compute_tick(payloads, window_seconds=args.window_seconds)
        slp_series.append((generated_at, sustained_load_pressure(tick)))
        next_check = generated_at + step

    if not slp_series:
        logger.error("history shorter than --window-seconds; no sustained_load_pressure points computed")
        return 1

    values = [v for _, v in slp_series]
    n = len(values)
    nonzero = sum(1 for v in values if v > 0.0)
    distinct = len(set(values))
    mean = statistics.fmean(values)
    median = statistics.median(values)
    variance = statistics.pvariance(values) if n > 1 else 0.0

    logger.info("")
    logger.info("=== sustained_load_pressure, %d points, step=%.0fs window=%.0fs ===", n, args.step_seconds, args.window_seconds)
    logger.info("nonzero: %d/%d (%.1f%%)", nonzero, n, 100.0 * nonzero / n)
    logger.info("distinct values: %d", distinct)
    logger.info("mean=%.6f median=%.6f pvariance=%.6e", mean, median, variance)
    logger.info("min=%.6f max=%.6f", min(values), max(values))

    # Independence check vs deviation_pressure: nearest-timestamp alignment
    # (the two series are on different cadences/step grids), then Pearson r.
    aligned_dp = []
    aligned_slp = []
    dp_idx = 0
    for ts, slp_v in slp_series:
        while dp_idx + 1 < len(dp_series) and dp_series[dp_idx + 1][0] <= ts:
            dp_idx += 1
        aligned_dp.append(dp_series[dp_idx][1])
        aligned_slp.append(slp_v)

    if len(set(aligned_dp)) > 1 and len(set(aligned_slp)) > 1:
        r = statistics.correlation(aligned_dp, aligned_slp)
    else:
        r = float("nan")
    logger.info("")
    logger.info("=== independence check vs deviation_pressure ===")
    logger.info("pearson r (nearest-timestamp aligned, n=%d): %.4f", len(aligned_dp), r)

    logger.info("")
    logger.info("=== suggested DIMENSION_PRECISION_MIN_VARIANCE entry (1/10th measured pvariance) ===")
    logger.info("sustained_load_pressure: %.6e", variance / 10.0 if variance > 0 else 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
