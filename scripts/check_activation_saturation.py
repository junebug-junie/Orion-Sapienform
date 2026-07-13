#!/usr/bin/env python3
"""Standing saturation gate for `memory_crystallizations.dynamics.activation`.

Context: live-data check on 2026-07-13 found 55% of active crystallizations already
sitting at the activation ceiling (1.00) with zero recall-time reinforcement having
ever fired -- see docs/superpowers/specs/2026-07-13-memory-recall-reinforcement-decay-
wiring-spec.md. Wiring `recall_boost()` (orion/memory/crystallization/dynamics.py)
without `decay()` in the same patch would only grow that ceiling-pinned fraction over
time. This codebase has already shipped exactly this saturation bug once before
(homeostatic drives pinned flat because a leaky integrator's soft-saturate went stale).

This script is the acceptance-check-1 gate from that spec: it is meant to be re-run by
anyone, anytime, before and after a real usage window (not a synthetic smoke burst) --
it is not a one-time manual query. If the ceiling-saturated fraction *increases* across
two runs, that is a fail; revert the patch, don't ship as-is.

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/check_activation_saturation.py
    python scripts/check_activation_saturation.py --postgres-uri postgresql://...
    python scripts/check_activation_saturation.py --ceiling 0.99 --json
    python scripts/check_activation_saturation.py --fail-above 0.55   # exit 1 if fraction exceeds this

Note on --fail-above: this repo has no persisted baseline to auto-diff two runs against
(and no cross-run state store this script should invent -- see the spec's own framing:
compare "before" vs. "after a real usage window" by hand, or pass the "before" run's
fraction here for a one-shot regression check). Without --fail-above this script only
reports; it always exits 0 on a successful query, matching "standing check you re-run
and read", not "stateful CI gate".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_activation_saturation.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_single_consumer_channels.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_QUERY = """
select
    count(*) filter (where (dynamics->>'activation')::float >= $1) as ceiling,
    count(*) as total
from memory_crystallizations
where status = 'active'
"""


async def _query_saturation(postgres_uri: str, *, ceiling_threshold: float) -> tuple[int, int]:
    import asyncpg

    conn = await asyncpg.connect(postgres_uri)
    try:
        row = await conn.fetchrow(_QUERY, ceiling_threshold)
    finally:
        await conn.close()
    return int(row["ceiling"]), int(row["total"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=0.99,
        help="activation value counted as 'ceiling-pinned' (default: 0.99).",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "exit 1 if the ceiling-pinned fraction exceeds this value (e.g. the fraction "
            "from a prior 'before' run). Omit to run in report-only mode (always exit 0 "
            "on a successful query)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "check_activation_saturation: no --postgres-uri given and $POSTGRES_URI is unset. "
            "Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    try:
        ceiling_count, total_count = asyncio.run(
            _query_saturation(args.postgres_uri, ceiling_threshold=args.ceiling)
        )
    except Exception as exc:
        print(f"check_activation_saturation: query failed -- {exc}", file=sys.stderr)
        return 2

    fraction = (ceiling_count / total_count) if total_count else 0.0

    if args.json:
        print(json.dumps({
            "ceiling_threshold": args.ceiling,
            "ceiling_count": ceiling_count,
            "total_active": total_count,
            "ceiling_fraction": fraction,
        }))
    else:
        print(
            f"activation_saturation: {ceiling_count}/{total_count} active crystallizations "
            f"at or above activation={args.ceiling} ({fraction:.1%} ceiling-pinned)"
        )
        if args.fail_above is None:
            print(
                "Compare this fraction against a prior run from before real usage of "
                "recall_boost()+decay() -- an INCREASE is a fail, revert the patch. "
                "(Pass --fail-above <prior-fraction> to make this check exit 1 automatically.)"
            )

    if args.fail_above is not None and fraction > args.fail_above:
        print(
            f"check_activation_saturation FAILED: {fraction:.1%} exceeds --fail-above "
            f"{args.fail_above:.1%}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
