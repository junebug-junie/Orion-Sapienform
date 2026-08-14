#!/usr/bin/env python3
"""Measure field-tension admission rate and rank discrimination over real history.

This is the deliverable of `docs/superpowers/specs/2026-08-14-field-deviation-
tension-sensing-design.md`. It answers the two questions that months of drive
work never had an instrument for:

1. **Admission rate** -- what fraction of ticks admit any tension at all? The
   number to beat is the 0.064% (284 of 444,943) measured in the 2026-07-07
   spec, which is why every taxonomy built on top of it failed.
2. **Rank discrimination** -- does the winner actually change, or is one target
   structurally pinned at the top? `top1_share` is the instrument that would
   have caught the 96% `relational` monoculture on day one instead of two weeks
   in.

Read-only: opens a read-only Postgres session (same contract as
`measure_ast_hot_reducer.py`), writes nothing, publishes nothing.

Usage:
    # From the host, against the live DB:
    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_field_tension_admission.py --hours 24
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.attention.tension.competition import FieldTensionCompetition  # noqa: E402
from orion.attention.tension.deviation_gate import DeviationGate  # noqa: E402
from orion.attention.tension.direction_map import load_direction_map  # noqa: E402
from orion.attention.tension.field_observations import (  # noqa: E402
    geometric_decay_ratio,
    iter_observations,
)

logger = logging.getLogger("measure_field_tension_admission")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# How many recent samples of one (node, channel) series to test for the
# constant-ratio decay artifact.
DECAY_PROBE_SAMPLES = 12


def open_readonly_connection(dsn: str):
    """Mirrors `measure_ast_hot_reducer.py`'s connection contract exactly --
    refuses to proceed on a session that is not read-only."""
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


def fetch_ticks(conn, hours: int, limit: int | None):
    sql = (
        "SELECT generated_at, field_json FROM substrate_field_state "
        "WHERE generated_at > now() - make_interval(hours => %s) "
        "ORDER BY generated_at ASC"
    )
    params: list = [hours]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for generated_at, field_json in cur:
            if isinstance(field_json, str):
                try:
                    field_json = json.loads(field_json)
                except Exception:
                    continue
            yield generated_at, field_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--z-threshold", type=float, default=1.5)
    parser.add_argument("--sigma-floor", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    conn = open_readonly_connection(dsn)
    if conn is None:
        return 2

    comp = FieldTensionCompetition(
        gate=DeviationGate(
            alpha=args.alpha,
            z_threshold=args.z_threshold,
            sigma_floor=args.sigma_floor,
            warmup=args.warmup,
        ),
        directions=load_direction_map(),
    )

    ticks = 0
    admitting_ticks = 0
    total_admissions = 0
    subnormal_total = 0
    winners: Counter[str] = Counter()
    disagreements = 0
    channel_admissions: Counter[str] = Counter()
    channels_seen: set[str] = set()
    first_ts = last_ts = None
    # Tail of each (node, channel) series, for the decay-artifact probe.
    tails: dict[tuple[str, str], list[float]] = defaultdict(list)

    try:
        for generated_at, field_json in fetch_ticks(conn, args.hours, args.limit):
            if not isinstance(field_json, dict):
                continue
            ticks += 1
            first_ts = first_ts or generated_at
            last_ts = generated_at

            for obs in iter_observations(field_json):
                channels_seen.add(obs.channel)
                tail = tails[(obs.node_id, obs.channel)]
                tail.append(obs.value)
                if len(tail) > DECAY_PROBE_SAMPLES:
                    tail.pop(0)

            result = comp.observe_tick(field_json)
            subnormal_total += result.subnormal_count
            if result.any_admitted:
                admitting_ticks += 1
                total_admissions += result.admitted_count
                for channel, per_node in result.admitted.items():
                    channel_admissions[channel] += len(per_node)
                if result.borda is not None:
                    if result.borda.winner:
                        winners[result.borda.winner] += 1
                    if result.borda.disagreement:
                        disagreements += 1
    finally:
        conn.close()

    if ticks == 0:
        logger.error("no ticks found in the last %sh -- nothing to measure", args.hours)
        return 2

    # Warm-up ticks structurally cannot admit; report against both denominators
    # rather than quietly flattering the headline number.
    post_warmup = max(0, ticks - args.warmup)
    admission_rate = admitting_ticks / ticks
    admission_rate_post_warmup = (admitting_ticks / post_warmup) if post_warmup else 0.0

    top1_share = (winners.most_common(1)[0][1] / admitting_ticks) if admitting_ticks else 0.0

    mapped = {c for c in channels_seen if comp.directions.worse_for(c) is not None}
    never_admitted = sorted(mapped - set(channel_admissions))
    unmapped_seen = sorted(channels_seen - mapped)

    decay_suspect = []
    for (node_id, channel), tail in sorted(tails.items()):
        ratio = geometric_decay_ratio(tail)
        if ratio is not None:
            decay_suspect.append({"node": node_id, "channel": channel, "ratio": round(ratio, 6)})

    report = {
        "window_hours": args.hours,
        "params": {
            "alpha": args.alpha,
            "z_threshold": args.z_threshold,
            "sigma_floor": args.sigma_floor,
            "warmup": args.warmup,
        },
        "ticks": ticks,
        "span": {"first": str(first_ts), "last": str(last_ts)},
        "admission": {
            "admitting_ticks": admitting_ticks,
            "admission_rate": round(admission_rate, 6),
            "admission_rate_post_warmup": round(admission_rate_post_warmup, 6),
            "mean_admissions_per_admitting_tick": (
                round(total_admissions / admitting_ticks, 3) if admitting_ticks else 0.0
            ),
            "baseline_to_beat": "0.00064 (284/444943, 2026-07-07 spec)",
        },
        "discrimination": {
            "distinct_winners": len(winners),
            "top1_share": round(top1_share, 4),
            "winner_counts": dict(winners.most_common()),
            "scorer_disagreement_rate": (
                round(disagreements / admitting_ticks, 4) if admitting_ticks else 0.0
            ),
        },
        "channels": {
            "mapped_and_seen": len(mapped),
            "admissions_by_channel": dict(channel_admissions.most_common()),
            "never_admitted": never_admitted,
            "unmapped_seen": unmapped_seen,
        },
        "data_quality": {
            "subnormal_coercions": subnormal_total,
            "decay_suspect_series": decay_suspect,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    a = report["admission"]
    d = report["discrimination"]
    print(f"\n=== Field tension admission, last {args.hours}h ===")
    print(f"ticks                     {ticks}  ({first_ts} -> {last_ts})")
    print(f"admitting ticks           {admitting_ticks}")
    print(f"ADMISSION RATE            {a['admission_rate']:.4%}   (post-warmup "
          f"{a['admission_rate_post_warmup']:.4%})")
    print(f"  baseline to beat        0.0640%  (284/444943, drives-era)")
    print(f"mean admissions/tick      {a['mean_admissions_per_admitting_tick']}")
    print(f"\n=== Rank discrimination ===")
    print(f"distinct winners          {d['distinct_winners']}")
    print(f"TOP-1 SHARE               {d['top1_share']:.2%}   (monoculture if -> 1.0)")
    print(f"winner counts             {d['winner_counts']}")
    print(f"scorer disagreement rate  {d['scorer_disagreement_rate']:.2%}")
    print(f"\n=== Channels ===")
    print(f"mapped and seen           {report['channels']['mapped_and_seen']}")
    for channel, count in list(channel_admissions.most_common(15)):
        print(f"  {channel:<34} {count}")
    if never_admitted:
        print(f"never admitted ({len(never_admitted)}): {', '.join(never_admitted)}")
    if unmapped_seen:
        print(f"unmapped (inert by design): {', '.join(unmapped_seen)}")
    print(f"\n=== Data quality ===")
    print(f"subnormal coercions       {subnormal_total}")
    if decay_suspect:
        print(f"decay-suspect series ({len(decay_suspect)}) -- NOT calm, unmaintained:")
        for entry in decay_suspect[:10]:
            print(f"  {entry['node']:<22} {entry['channel']:<30} ratio={entry['ratio']}")
    else:
        print("decay-suspect series      none")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
