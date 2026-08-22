#!/usr/bin/env python3
"""Liveness gate for `scripts/attention_loop_decay_digest.py`.

Context: the digest closes the implicit-decay half of the attention-loop label
stream (see that script's own docstring) -- but it is a standalone script run on
demand or via cron, not a live service loop. Nothing detects if the cron entry
dies, is silently dropped on a host migration, or the job starts failing. This
script is that detector, mirroring
`scripts/check_concept_relation_digest_liveness.py`'s shape.

Unlike that sibling gate, `attention_salience_trace` has no `digested` boolean --
decay-eligibility is derived, not stored. So "backlog age" here means: run the
same derivation the digest itself runs (read-only, nothing written), and measure
how far PAST its own decay threshold the most-overdue still-open loop is. If the
digest is running on schedule, no loop should exceed `min_silence` by more than
about one cron interval before getting labelled. If it stops running, that
overshoot grows without bound -- a real, unambiguous symptom, not a heartbeat
file that can go stale independently of the thing it claims to represent.

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/check_attention_loop_decay_liveness.py
    python scripts/check_attention_loop_decay_liveness.py --max-overshoot-hours 3 --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orion.substrate.attention.implicit_outcome import DEFAULT_MIN_SILENCE  # noqa: E402

# Shared grouping + eligibility logic with the digest this gate watches -- NOT
# reimplemented here. Review caught that a hand-reimplemented copy of the
# trace-row -> LoopObservation grouping had a real bug (theme_key=loop_id
# instead of the row's actual theme_key column, dormant only because every
# current producer happens to set them equal) and that a second independent
# copy of the "exclude already-decayed loops" filter is exactly the kind of
# duplication that let this gate falsely report STALE right after a successful
# digest run (confirmed live 2026-08-21) -- a fix applied in one copy and not
# the other would silently desync the gate from the thing it watches.
from scripts.attention_loop_decay_digest import (  # noqa: E402
    _SELECT_LATEST_VERDICTS_SQL,
    _SELECT_TRACES_SQL,
    build_observations,
    eligible_verdicts,
)


async def _query_overshoot(postgres_uri: str, *, min_silence: timedelta) -> tuple[int, float, str | None]:
    import asyncpg

    conn = await asyncpg.connect(postgres_uri)
    try:
        trace_rows = [dict(r) for r in await conn.fetch(_SELECT_TRACES_SQL)]
        verdict_rows = [dict(r) for r in await conn.fetch(_SELECT_LATEST_VERDICTS_SQL)]
    finally:
        await conn.close()

    observations = build_observations(trace_rows, verdict_rows)
    now = datetime.now(timezone.utc)
    verdicts = eligible_verdicts(observations, now=now, min_silence=min_silence)
    if not verdicts:
        return (0, 0.0, None)

    worst = max(verdicts, key=lambda v: v.silence)
    overshoot_hours = (worst.silence - min_silence).total_seconds() / 3600.0
    return (len(verdicts), max(0.0, overshoot_hours), worst.theme_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument(
        "--min-silence-hours",
        type=float,
        default=DEFAULT_MIN_SILENCE.total_seconds() / 3600.0,
        help="must match the digest's own --min-silence-hours or this gate measures the wrong threshold.",
    )
    parser.add_argument(
        "--max-overshoot-hours",
        type=float,
        default=3.0,
        help="fail if the most-overdue eligible loop exceeds min_silence by more than this (default: 3.0, "
        "generous headroom over a 30-60 minute cron cadence).",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "check_attention_loop_decay_liveness: no --postgres-uri given and "
            "$POSTGRES_URI is unset. Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    try:
        backlog, overshoot_hours, worst_theme = asyncio.run(
            _query_overshoot(args.postgres_uri, min_silence=timedelta(hours=args.min_silence_hours))
        )
    except Exception as exc:
        print(f"check_attention_loop_decay_liveness: query failed -- {exc}", file=sys.stderr)
        return 2

    stale = overshoot_hours > args.max_overshoot_hours

    if args.json:
        print(json.dumps({
            "backlog": backlog,
            "worst_overshoot_hours": overshoot_hours,
            "worst_theme_key": worst_theme,
            "max_overshoot_hours": args.max_overshoot_hours,
            "stale": stale,
        }))
    else:
        if backlog == 0:
            print("check_attention_loop_decay_liveness: OK -- no loop is currently decay-eligible.")
        else:
            print(
                f"check_attention_loop_decay_liveness: {backlog} loop(s) decay-eligible right now, "
                f"worst overshoot {overshoot_hours:.2f}h past its own threshold (theme={worst_theme}, "
                f"max allowed overshoot {args.max_overshoot_hours:.2f}h)"
            )
        if stale:
            print(
                "STALE: the attention-loop decay digest does not appear to be running on "
                "schedule. Check the cron entry (crontab -l) and whether this host still has "
                "the crontab installed after any migration.",
                file=sys.stderr,
            )

    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
