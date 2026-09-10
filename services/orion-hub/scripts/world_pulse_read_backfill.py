#!/usr/bin/env python3
"""Enqueue world-pulse Stage 1 seeds from recent digests. No FCC.

Usage (from services/orion-hub, PYTHONPATH=.:../..):

    python3 scripts/world_pulse_read_backfill.py --dry-run
    python3 scripts/world_pulse_read_backfill.py --findings-only --limit-digests 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enqueue world-pulse read seeds from recent digests (no FCC)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count seeds that would be inserted; write nothing.",
    )
    parser.add_argument(
        "--findings-only",
        action="store_true",
        help="Enqueue curiosity-followup finding URLs only (skip digest_item).",
    )
    parser.add_argument(
        "--limit-digests",
        type=int,
        default=None,
        metavar="N",
        help="Newest N digests. Omit to scan all (capped at 10000).",
    )
    return parser.parse_args(argv)


def resolve_dsn() -> str:
    for key in ("RECALL_PG_DSN", "POSTGRES_URI", "DATABASE_URL"):
        value = str(os.getenv(key, "")).strip()
        if value:
            return value
    raise SystemExit(
        "No Postgres DSN. Set RECALL_PG_DSN, POSTGRES_URI, or DATABASE_URL."
    )


async def run_backfill(
    conn: Any,
    *,
    dry_run: bool,
    findings_only: bool,
    limit_digests: int,
    bus: Any = None,
) -> int:
    from orion.world_pulse_read.queue import enqueue_from_recent_digests

    return await enqueue_from_recent_digests(
        conn,
        limit_digests=limit_digests,
        findings_only=findings_only,
        dry_run=dry_run,
        bus=bus,
    )


async def _amain(args: argparse.Namespace) -> int:
    import asyncpg  # type: ignore

    dsn = resolve_dsn()
    limit = args.limit_digests if args.limit_digests is not None else 10000
    if limit < 1:
        raise SystemExit("--limit-digests must be >= 1")
    bus = None
    bus_url = os.getenv("ORION_BUS_URL", "").strip()
    if not args.dry_run and bus_url:
        from orion.core.bus.async_service import OrionBusAsync
        bus = OrionBusAsync(bus_url)
    conn = await asyncpg.connect(dsn)
    try:
        if bus is not None:
            try:
                await bus.connect()
            except Exception:
                # Persistence remains usable through a bus outage.
                print("reading accepted-event transport unavailable; queue writes remain durable", file=sys.stderr)
        n = await run_backfill(
            conn,
            dry_run=args.dry_run,
            findings_only=args.findings_only,
            limit_digests=limit,
            bus=bus,
        )
    finally:
        await conn.close()
        if bus is not None:
            await bus.close()
    mode = "dry-run" if args.dry_run else "inserted"
    extra = " findings-only" if args.findings_only else ""
    print(f"world_pulse_read_backfill {mode}{extra} count={n} limit_digests={limit}")
    return n


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
