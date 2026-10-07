#!/usr/bin/env python3
"""Backfill memory referents (aliases, node ids, co-occurrence claims) for stored episodes.

Recovers the writer's aliases from each saved distill run's checkpoint and runs the same
referent step the live persist runs (orion/memory/referents/backfill.py). Idempotent.

Backfill protocol (AGENTS.md section 14), all under /tmp/referent-backfill/:
- dry run by default: does all the work in a transaction and ROLLS IT BACK, so report.md and
  before_after.csv show exactly what WOULD change; nothing is written to the database;
- --apply first snapshots every row it can change (snapshot_*.json; stops if any table holds
  more than 100k rows), then writes;
- both modes write progress.log DURING the run, one line per episode: percent, ETA,
  episodes processed/total, rate, error count, and that episode's counts or error. A failing
  episode is rolled back alone and counted; the run continues.

    python scripts/backfill_referents_from_episodes.py --dsn postgresql://...            # dry run
    python scripts/backfill_referents_from_episodes.py --dsn postgresql://... --apply    # write
    tail -f /tmp/referent-backfill/progress.log

The DSN is required and never defaulted: point it at the database you mean.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = Path(os.getenv("REFERENT_BACKFILL_OUT", "/tmp/referent-backfill"))
SNAPSHOT_SQL = {
    "referent_alias": "SELECT * FROM referent_alias",
    "episode_memory_referent": "SELECT * FROM episode_memory_referent",
    "memory_tension_shadow": "SELECT * FROM memory_tension_shadow",
    "substrate_graph_journal": "SELECT * FROM substrate_graph_journal WHERE actor = 'memory.referents'",
}
SNAPSHOT_ROW_LIMIT = 100_000


def _flag(name: str) -> bool:
    return os.getenv(name, "true").strip().lower() not in {"0", "false", "no", "off"}


async def _counts(conn) -> dict[str, int]:
    out = {}
    for table, sql in SNAPSHOT_SQL.items():
        out[table] = (await (await conn.execute(f"SELECT count(*) AS n FROM ({sql}) t")).fetchone())["n"]
    return out


def progress_line(p: dict) -> str:
    row = p["row"]
    eta = f"{p['eta_sec']:.0f}s" if p["eta_sec"] is not None else "n/a"
    return (f"referent-backfill {100 * p['index'] / max(p['total'], 1):.0f}% ETA {eta} "
            f"episodes {p['processed']}/{p['total']} rate {p['rate_per_sec']}/s errors {p['errors']} "
            f"{json.dumps(row, default=str)}")


async def run(dsn: str, apply: bool, out: Path) -> int:
    import psycopg
    from psycopg.rows import dict_row

    from orion.memory.referents.backfill import backfill_all
    from orion.memory.referents.resolve import ReferentPolicy

    policy = ReferentPolicy(grounding_auto_accept=_flag("MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT"),
                            cooccurrence_auto_accept=_flag("MEMORY_COOCCURRENCE_AUTO_ACCEPT"))
    out.mkdir(parents=True, exist_ok=True)
    mode = "apply" if apply else "dry-run"
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        before = await _counts(conn)
        if apply:
            for table, sql in SNAPSHOT_SQL.items():
                if before[table] > SNAPSHOT_ROW_LIMIT:
                    print(f"{table} holds {before[table]} rows (> {SNAPSHOT_ROW_LIMIT}): stop and ask Juniper",
                          file=sys.stderr)
                    return 2
                rows = await (await conn.execute(sql)).fetchall()
                (out / f"snapshot_{table}.json").write_text(json.dumps(rows, default=str, indent=1))
        with (out / "progress.log").open("a", buffering=1) as log:
            log.write(f"referent-backfill start mode={mode} at={datetime.now(timezone.utc).isoformat()}\n")
            report = await backfill_all(conn, now=datetime.now(timezone.utc), policy=policy, dry_run=not apply,
                                        on_progress=lambda p: log.write(progress_line(p) + "\n"))
        after = await _counts(conn)
    would = {t: sum(int(r.get(k, 0) or 0) for r in report) for t, k in (("aliases", "aliases"),
             ("questions", "questions"), ("proposals", "proposals"), ("decisions", "decisions"))}
    errors = [r for r in report if not r.get("ok")]
    with (out / "before_after.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["table", "before", "after", "mode"])
        for table in SNAPSHOT_SQL:
            writer.writerow([table, before[table], after[table], mode])
    (out / "report.md").write_text(
        f"# Referent backfill ({mode})\n\n"
        f"Verdict: {len(report) - len(errors)} of {len(report)} episodes {'applied' if apply else 'would apply'}; "
        f"{len(errors)} errors. Needs another pass: {'yes' if errors else 'no'}.\n\n"
        + ("Nothing was written (dry run: computed, then rolled back).\n\n" if not apply else "")
        + "Would add / added: " + ", ".join(f"{k} {v}" for k, v in would.items()) + "\n\n"
        + "".join(f"- {t}: {before[t]} -> {after[t]}\n" for t in SNAPSHOT_SQL)
        + "\nPer episode:\n\n" + "".join(f"- {json.dumps(r, default=str)}\n" for r in report)
        + "\nFiles: " + ", ".join(sorted(p.name for p in out.iterdir())) + "\n")
    print(f"{mode}: {len(report)} episodes, {len(errors)} errors; see {out}/report.md")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--apply", action="store_true", help="write (default: dry run, rolled back)")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    return asyncio.run(run(args.dsn, args.apply, Path(args.out)))


if __name__ == "__main__":
    raise SystemExit(main())
