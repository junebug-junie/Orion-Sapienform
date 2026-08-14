#!/usr/bin/env python3
"""Measure attention-loop outcome-label coverage, and derive the missing implicit labels.

The outcome measure the drives program never had, and the piece that must exist
*before* anything supplies attention candidates from interoceptive state -- the
whole failure mode being instruments built with no way to tell whether they
helped.

`AttentionLoopOutcomeV1` has always specified three verdicts. Two are human
clicks in the Hub (`resolved`, `dismissed`). The third, `decayed_unattended`, is
documented as implicit and machine-derived, and **has never been written once**:
all live rows are `resolved`/`juniper`.

Default mode is read-only measurement. `--emit` writes derived labels, and
follows CLAUDE.md 14: snapshots what it will change before changing anything,
never overwrites a human verdict, and is idempotent on `outcome_id`.

Usage:
    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_attention_outcome_coverage.py
    ... --sweep                 # horizon sensitivity instead of one defended constant
    ... --emit --yes            # actually write the derived labels
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.substrate.attention.implicit_outcome import (  # noqa: E402
    DEFAULT_MIN_SILENCE,
    DEFAULT_SILENCE_CADENCE_MULTIPLE,
    LoopObservation,
    derive_implicit_verdicts,
)

logger = logging.getLogger("measure_attention_outcome_coverage")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
SNAPSHOT_DIR = Path("/tmp/attention-outcome-backfill")


def open_connection(dsn: str, *, read_only: bool):
    try:
        import psycopg2
    except Exception:
        logger.error("psycopg2 unavailable")
        return None
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    if read_only:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            v = cur.fetchone()
        if not v or str(v[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only")
            conn.close()
            return None
    return conn


def load_loops(conn) -> list[LoopObservation]:
    """One `LoopObservation` per loop that has ever been scored."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.loop_id,
                   max(t.theme_key)                                   AS theme_key,
                   array_agg(t.created_at ORDER BY t.created_at)       AS trace_times,
                   (array_agg(t.salience ORDER BY t.created_at DESC))[1] AS last_salience,
                   (array_agg(t.features ORDER BY t.created_at DESC))[1] AS last_features,
                   (SELECT o.verdict FROM attention_loop_outcome o
                     WHERE o.loop_id = t.loop_id
                     ORDER BY o.created_at DESC LIMIT 1)               AS existing_verdict
              FROM attention_salience_trace t
             GROUP BY t.loop_id
            """
        )
        rows = cur.fetchall()

    out: list[LoopObservation] = []
    for loop_id, theme_key, times, last_sal, last_feat, verdict in rows:
        if isinstance(last_feat, str):
            try:
                last_feat = json.loads(last_feat)
            except Exception:
                last_feat = {}
        out.append(
            LoopObservation(
                loop_id=str(loop_id),
                theme_key=str(theme_key or ""),
                trace_times=list(times or []),
                last_salience=float(last_sal or 0.0),
                last_features=last_feat or {},
                existing_verdict=str(verdict) if verdict else None,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-silence-hours", type=float,
                        default=DEFAULT_MIN_SILENCE.total_seconds() / 3600.0)
    parser.add_argument("--cadence-multiple", type=float,
                        default=DEFAULT_SILENCE_CADENCE_MULTIPLE)
    parser.add_argument("--sweep", action="store_true",
                        help="report horizon sensitivity instead of one setting")
    parser.add_argument("--emit", action="store_true", help="write the derived labels")
    parser.add_argument("--yes", action="store_true", help="required alongside --emit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --emit needs a writable session; everything else is hard read-only.
    conn = open_connection(
        os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI), read_only=not args.emit
    )
    if conn is None:
        return 2

    try:
        loops = load_loops(conn)
        now = datetime.now(timezone.utc)

        labelled = sum(1 for lo in loops if lo.existing_verdict)
        human = sum(1 for lo in loops if lo.existing_verdict in {"resolved", "dismissed"})
        implicit_existing = sum(1 for lo in loops if lo.existing_verdict == "decayed_unattended")

        derived = derive_implicit_verdicts(
            loops,
            now=now,
            min_silence=timedelta(hours=args.min_silence_hours),
            cadence_multiple=args.cadence_multiple,
        )

        report = {
            "loops_ever_scored": len(loops),
            "coverage": {
                "labelled": labelled,
                "human_labelled": human,
                "implicit_labelled": implicit_existing,
                "unlabelled": len(loops) - labelled,
                "label_coverage_pct": round(labelled / len(loops), 4) if loops else 0.0,
            },
            "derivable_now": [
                {
                    "loop_id": v.loop_id,
                    "silence_hours": round(v.silence.total_seconds() / 3600, 1),
                    "median_gap_hours": (
                        round(v.median_gap.total_seconds() / 3600, 2) if v.median_gap else None
                    ),
                    "salience_at_close": v.salience_at_close,
                    "reason": v.reason,
                }
                for v in derived
            ],
        }

        if args.sweep:
            sweep = []
            for hours in (6, 12, 24, 48, 96, 168, 336):
                for mult in (1.0, 3.0, 10.0):
                    n = len(derive_implicit_verdicts(
                        loops, now=now,
                        min_silence=timedelta(hours=hours), cadence_multiple=mult))
                    sweep.append({"min_silence_hours": hours, "cadence_multiple": mult,
                                  "labels": n})
            report["sweep"] = sweep

        if args.emit:
            if not args.yes:
                logger.error("--emit requires --yes")
                return 2
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            snap = SNAPSHOT_DIR / "before_after.json"
            snap.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            logger.info("snapshot written: %s", snap)

            from orion.core.ids import stable_hash_id

            written = 0
            with conn.cursor() as cur:
                for v in derived:
                    cur.execute(
                        """
                        INSERT INTO attention_loop_outcome
                            (outcome_id, loop_id, theme_key, verdict, actor, note,
                             salience_at_close, weights_version, features_at_close, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (outcome_id) DO NOTHING
                        """,
                        (
                            stable_hash_id("loopoutcome", [v.loop_id, v.verdict, "implicit"]),
                            v.loop_id, v.theme_key, v.verdict, "implicit", v.reason[:500],
                            v.salience_at_close, "gwt-coalition-v1",
                            json.dumps(v.features_at_close), now,
                        ),
                    )
                    written += cur.rowcount
            report["emitted"] = written
            logger.info("emitted %d implicit labels", written)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    c = report["coverage"]
    print(f"\n=== Attention-loop outcome coverage ===")
    print(f"loops ever scored         {report['loops_ever_scored']}")
    print(f"  labelled                {c['labelled']}  ({c['label_coverage_pct']:.1%})")
    print(f"    human (resolve/dismiss) {c['human_labelled']}")
    print(f"    implicit (decayed)      {c['implicit_labelled']}")
    print(f"  unlabelled              {c['unlabelled']}")
    print(f"\n=== Derivable implicit labels now "
          f"(floor {args.min_silence_hours:g}h, {args.cadence_multiple:g}x own cadence) ===")
    if report["derivable_now"]:
        for d in report["derivable_now"]:
            print(f"  {d['loop_id']:<28} silent={d['silence_hours']:>7.1f}h  "
                  f"sal={d['salience_at_close']:.3f}")
            print(f"    {d['reason']}")
    else:
        print("  none")
    if args.sweep:
        print(f"\n=== Horizon sensitivity (labels derived) ===")
        print(f"  {'floor_h':>8} {'x cadence':>10} {'labels':>7}")
        for row in report["sweep"]:
            print(f"  {row['min_silence_hours']:>8} {row['cadence_multiple']:>10g} "
                  f"{row['labels']:>7}")
    if "emitted" in report:
        print(f"\nemitted {report['emitted']} labels")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
