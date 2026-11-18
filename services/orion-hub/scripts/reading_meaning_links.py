#!/usr/bin/env python3
"""Count or withdraw the reading links one acceptance policy made.

Revert path for ``reading_meaning_target_v1`` (orion/world_pulse_read/meaning_links.py).
Set HUB_WORLD_PULSE_READ_MEANING_TARGET_LINKS_ENABLED=false first to stop new ones, then:

    # from services/orion-hub, PYTHONPATH=.:../..
    python3 scripts/reading_meaning_links.py list
    python3 scripts/reading_meaning_links.py withdraw                 # dry run: what would go
    python3 scripts/reading_meaning_links.py withdraw --yes           # append the rejections
    python3 scripts/reading_meaning_links.py withdraw --yes --seed-file /tmp/reading-reread-backfill/seeds.txt \\
        --since 2026-10-15T00:00:00Z --policy reading_meaning_target_v1 --policy reading_quote_rule_v1

Writes only ``rejected`` decisions to substrate_graph_journal (append-only; nothing is
deleted). Hub's AssertionProjector closes each link on its next Stage 2 tick.
Exit 0 ok, 2 database unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.world_pulse_read.assertions import MEANING_POLICY  # noqa: E402


def resolve_dsn() -> str:
    for key in ("RECALL_PG_DSN", "POSTGRES_URI", "DATABASE_URL"):
        value = str(os.getenv(key, "")).strip()
        if value:
            return value
    raise SystemExit("No Postgres DSN. Set RECALL_PG_DSN, POSTGRES_URI, or DATABASE_URL.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("list", "withdraw"))
    ap.add_argument("--policy", action="append", default=None,
                    help=f"acceptance policy (repeatable; default {MEANING_POLICY})")
    ap.add_argument("--since", default=None, help="only decisions recorded at/after this ISO time")
    ap.add_argument("--seed-file", default=None, help="only decisions for these reading seed ids (one per line)")
    ap.add_argument("--reason", default="", help="note written into each withdrawal's rationale")
    ap.add_argument("--yes", action="store_true", help="withdraw: actually append the rejections")
    ap.add_argument("--dsn", default=None)
    return ap.parse_args(argv)


async def run(conn, args: argparse.Namespace, *, journal=None, out=sys.stdout) -> dict:
    from orion.substrate.graph_journal import RevisionConflict
    from orion.world_pulse_read.meaning_links import list_policy_links, withdrawal_for

    since = datetime.fromisoformat(args.since.replace("Z", "+00:00")) if args.since else None
    seeds = None
    if args.seed_file:
        seeds = [ln.strip() for ln in Path(args.seed_file).read_text().splitlines() if ln.strip()]
    links = await list_policy_links(conn, policies=args.policy or [MEANING_POLICY], since=since, seed_ids=seeds)
    counts: dict[str, int] = {}
    written = 0
    now = datetime.now(timezone.utc)
    for link in links:
        counts[link.status] = counts.get(link.status, 0) + 1
        row = {"assertion_id": link.decided.target_id, "policy": link.decided.policy,
               "decision_id": link.decided.decision_id, "status": link.status,
               "recorded_at": link.decided.recorded_at.isoformat(), "seeds": link.seed_ids()}
        if args.action == "withdraw":
            decision = withdrawal_for(link, recorded_at=now, reason=args.reason)
            row["withdrawal"] = decision.decision_id if decision else None
            if decision is not None and args.yes:
                try:
                    row["written"] = bool(await journal.append(decision))
                except RevisionConflict:
                    row["written"] = False
                    row["status"] = "superseded_during_withdraw"
                written += bool(row.get("written"))
        print(json.dumps(row), file=out)
    summary = {"action": args.action, "dry_run": args.action == "withdraw" and not args.yes,
               "links": len(links), "by_status": counts, "withdrawals_written": written}
    print(json.dumps(summary), file=out)
    return summary


async def _amain(args: argparse.Namespace) -> int:
    import asyncpg

    from orion.substrate.graph_journal import SubstrateGraphJournal

    try:
        pool = await asyncpg.create_pool(args.dsn or resolve_dsn(), min_size=1, max_size=2)
    except Exception as exc:  # noqa: BLE001
        print(f"UNKNOWN: database unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        async with pool.acquire() as conn:
            await run(conn, args, journal=SubstrateGraphJournal(pool))
    finally:
        await pool.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
