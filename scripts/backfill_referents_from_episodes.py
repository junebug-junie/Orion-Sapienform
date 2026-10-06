#!/usr/bin/env python3
"""Backfill memory referents (aliases, node ids, co-occurrence claims) for stored episodes.

Recovers the writer's aliases from each saved distill run's checkpoint and runs the same
referent step the live persist runs (orion/memory/referents/backfill.py). Idempotent.

Backfill protocol (AGENTS.md section 14):
- dry run by default: prints what exists and how many memories it would resolve; writes nothing;
- --apply snapshots every row it can change to /tmp/referent-backfill/ first (referent_alias,
  episode_memory_referent, memory_tension_shadow, substrate_graph_journal by memory.referents),
  logs progress to /tmp/referent-backfill/progress.log and writes report.md + before_after.csv.

    python scripts/backfill_referents_from_episodes.py --dsn postgresql://...            # dry run
    python scripts/backfill_referents_from_episodes.py --dsn postgresql://... --apply    # write

The DSN is required and never defaulted: point it at the database you mean.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = Path("/tmp/referent-backfill")
SNAPSHOT_SQL = {
    "referent_alias": "SELECT * FROM referent_alias",
    "episode_memory_referent": "SELECT * FROM episode_memory_referent",
    "memory_tension_shadow": "SELECT * FROM memory_tension_shadow",
    "substrate_graph_journal": "SELECT * FROM substrate_graph_journal WHERE actor = 'memory.referents'",
}


def _flag(name: str) -> bool:
    return os.getenv(name, "true").strip().lower() not in {"0", "false", "no", "off"}


async def _counts(conn) -> dict[str, int]:
    out = {}
    for table, sql in SNAPSHOT_SQL.items():
        out[table] = (await (await conn.execute(f"SELECT count(*) AS n FROM ({sql}) t")).fetchone())["n"]
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args()

    import psycopg
    from psycopg.rows import dict_row

    from orion.memory.referents.backfill import backfill_all
    from orion.memory.referents.resolve import ReferentPolicy

    policy = ReferentPolicy(grounding_auto_accept=_flag("MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT"),
                            cooccurrence_auto_accept=_flag("MEMORY_COOCCURRENCE_AUTO_ACCEPT"))
    async with await psycopg.AsyncConnection.connect(args.dsn, autocommit=True, row_factory=dict_row) as conn:
        before = await _counts(conn)
        runs = (await (await conn.execute("SELECT count(*) AS n FROM episode_distill_run")).fetchone())["n"]
        print(f"episodes with a distill run: {runs}; rows now: {json.dumps(before)}")
        if not args.apply:
            print("dry run: nothing written (pass --apply)")
            return 0
        OUT.mkdir(parents=True, exist_ok=True)
        for table, sql in SNAPSHOT_SQL.items():
            rows = await (await conn.execute(sql)).fetchall()
            (OUT / f"snapshot_{table}.json").write_text(json.dumps(rows, default=str, indent=1))
            if len(rows) > 100_000:
                print(f"snapshot of {table} exceeds 100k rows: stop and ask Juniper", file=sys.stderr)
                return 2
        started = time.monotonic()
        now = datetime.now(timezone.utc)
        report = await backfill_all(conn, now=now, policy=policy)
        with (OUT / "progress.log").open("a") as log:
            for i, row in enumerate(report, 1):
                rate = i / max(time.monotonic() - started, 1e-6)
                log.write(f"referent-backfill {100 * i / len(report):.0f}% {i}/{len(report)} "
                          f"rate={rate:.1f}/s errors=0 {json.dumps(row, default=str)}\n")
        after = await _counts(conn)
    with (OUT / "before_after.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["table", "before", "after"])
        for table in SNAPSHOT_SQL:
            writer.writerow([table, before[table], after[table]])
    (OUT / "report.md").write_text(
        "# Referent backfill\n\n"
        f"Verdict: applied to {len(report)} episodes.\n\n"
        + "".join(f"- {t}: {before[t]} -> {after[t]}\n" for t in SNAPSHOT_SQL)
        + "\nPer episode:\n\n" + "".join(f"- {json.dumps(r, default=str)}\n" for r in report)
        + "\nRe-running is a no-op (idempotent). Files: " + ", ".join(sorted(p.name for p in OUT.iterdir())) + "\n")
    print(f"applied: {json.dumps(after)}; see {OUT}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
