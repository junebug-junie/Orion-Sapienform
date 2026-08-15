#!/usr/bin/env python3
"""Measure whether Layer 5 attention has anything to attend to.

The next question on the tension arc, and the one that decides whether wiring
tension into attention is worth doing at all: **is the attention architecture
currently choosing between competing inputs, or is it idling for lack of any?**

If attention already selects richly, a new tension signal has to prove it adds
information. If attention is starved, tension is not competing with an incumbent
selector -- it is filling a vacuum, which is a different (and much weaker) claim
to have to defend, but also a much more useful thing to know first.

Deliberately kept separate from `measure_field_tension_admission.py`: different
table, different cadence, different question. **The two rates are NOT directly
comparable** -- `substrate_field_state` ticks ~42k/day while
`substrate_attention_broadcast_log` writes ~3.3k/day, and a tension admission is
not the same kind of event as an action selection. This script reports per-day
counts alongside percentages so the cadence difference stays visible instead of
being laundered into a misleading ratio.

Read-only: opens a read-only Postgres session, writes nothing, publishes nothing.

Usage:
    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_attention_input_starvation.py --hours 72
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger("measure_attention_input_starvation")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


def open_readonly_connection(dsn: str):
    """Same connection contract as `measure_ast_hot_reducer.py` -- refuses a
    session that is not read-only."""
    try:
        import psycopg2
    except Exception:
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only (got %r)", value)
            conn.close()
            return None
    except Exception:
        logger.error("failed to enforce read-only session", exc_info=True)
        conn.close()
        return None
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = open_readonly_connection(os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI))
    if conn is None:
        return 2

    frames = 0
    action_types: Counter[str] = Counter()
    signal_counts: Counter[int] = Counter()
    open_loop_counts: Counter[int] = Counter()
    candidate_counts: Counter[int] = Counter()
    selected_with_signal = 0
    selected_without_signal = 0
    span_first = span_last = None

    try:
        conn.autocommit = False
        with conn.cursor(name="attention_frames") as cur:
            cur.itersize = 2000
            cur.execute(
                "SELECT generated_at, projection_json FROM substrate_attention_broadcast_log "
                "WHERE generated_at > now() - make_interval(hours => %s) "
                "ORDER BY generated_at ASC",
                [args.hours],
            )
            for generated_at, payload in cur:
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        continue
                frame = (payload or {}).get("frame")
                if not isinstance(frame, dict):
                    continue
                frames += 1
                span_first = span_first or generated_at
                span_last = generated_at

                selected = frame.get("selected_action") or {}
                action_type = str(selected.get("action_type") or "none")
                action_types[action_type] += 1

                debug = frame.get("debug") or {}
                try:
                    n_signals = int(debug.get("signal_count") or 0)
                except (TypeError, ValueError):
                    n_signals = 0
                signal_counts[n_signals] += 1
                open_loop_counts[len(frame.get("open_loops") or [])] += 1
                candidate_counts[len(frame.get("candidate_actions") or [])] += 1

                if action_type != "none":
                    if n_signals > 0:
                        selected_with_signal += 1
                    else:
                        selected_without_signal += 1
    finally:
        conn.close()

    if frames == 0:
        logger.error("no attention frames in the last %sh", args.hours)
        return 2

    selecting = frames - action_types.get("none", 0)
    with_signal = frames - signal_counts.get(0, 0)
    with_open_loop = frames - open_loop_counts.get(0, 0)
    with_candidate = frames - candidate_counts.get(0, 0)

    span_days = None
    if span_first and span_last:
        span_days = max((span_last - span_first).total_seconds() / 86400.0, 1e-9)

    report = {
        "window_hours": args.hours,
        "frames": frames,
        "span": {"first": str(span_first), "last": str(span_last), "days": round(span_days or 0, 3)},
        "input": {
            "frames_with_any_signal": with_signal,
            "frames_with_any_signal_pct": round(with_signal / frames, 4),
            "frames_with_any_open_loop": with_open_loop,
            "frames_with_any_open_loop_pct": round(with_open_loop / frames, 4),
            "frames_with_any_candidate_action": with_candidate,
            "frames_with_any_candidate_action_pct": round(with_candidate / frames, 4),
            "signal_count_histogram": dict(sorted(signal_counts.items())),
        },
        "selection": {
            "frames_selecting_an_action": selecting,
            "frames_selecting_an_action_pct": round(selecting / frames, 4),
            "action_type_counts": dict(action_types.most_common()),
            "selections_per_day": round(selecting / span_days, 1) if span_days else None,
            "frames_per_day": round(frames / span_days, 1) if span_days else None,
            "selected_with_signal": selected_with_signal,
            "selected_without_signal": selected_without_signal,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    i, s = report["input"], report["selection"]
    print(f"\n=== Layer 5 attention input, last {args.hours}h ===")
    print(f"frames                        {frames}  ({span_first} -> {span_last})")
    print(f"frames/day                    {s['frames_per_day']}")
    print(f"\n--- does it have anything to attend to? ---")
    print(f"frames with >=1 signal        {with_signal:>7}  ({i['frames_with_any_signal_pct']:.1%})")
    print(f"frames with >=1 open loop     {with_open_loop:>7}  "
          f"({i['frames_with_any_open_loop_pct']:.1%})")
    print(f"frames with >=1 candidate     {with_candidate:>7}  "
          f"({i['frames_with_any_candidate_action_pct']:.1%})")
    print(f"signal_count histogram        {i['signal_count_histogram']}")
    print(f"\n--- does it choose anything? ---")
    print(f"frames selecting an action    {selecting:>7}  "
          f"({s['frames_selecting_an_action_pct']:.1%})")
    print(f"selections/day                {s['selections_per_day']}")
    print(f"action types                  {s['action_type_counts']}")
    print(f"\nNOTE: these rates are NOT comparable to field-tension admission rate --")
    print(f"      different table, different cadence, different kind of event.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
