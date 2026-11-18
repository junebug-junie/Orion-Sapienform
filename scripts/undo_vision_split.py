#!/usr/bin/env python3
"""Undo one walkway individual split (services/orion-sql-writer/app/vision_split.py).

Dry run by default: it runs the whole undo inside a transaction, prints what
would change, and ROLLS BACK. ``--apply`` commits it.

    python3 scripts/undo_vision_split.py --list
    python3 scripts/undo_vision_split.py split-<id>            # dry run
    python3 scripts/undo_vision_split.py split-<id> --apply    # undo it

What undo does (vision_split.undo_split):
  1. Moves the crops back to the individual they were in before (NULL for a
     crop older than stored membership).
  2. Restores the source individual and its rewritten sightings to the
     recorded ``before`` exactly when nothing touched them since; otherwise
     adds the moved crops back to their centroid and counts, so crops that
     joined later are not lost.
  3. Deletes the new individual (and its sightings, and its label) and the
     pair verdicts the split wrote.
  4. Sets ``undone_at``. The split is never re-applied for that answer.

Refuses, naming the crops, when crops joined the new individual since the
split: Juniper decides then. The new individual never got a substrate entity
node (the split labels it directly), so there is no node to retire.

DSN: --dsn, else POSTGRES_URI, else DATABASE_URL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "services" / "orion-sql-writer"
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split_id", nargs="?")
    ap.add_argument("--apply", action="store_true", help="commit the undo (default: dry run, rolled back)")
    ap.add_argument("--list", action="store_true", help="list recorded splits and exit")
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL"))
    args = ap.parse_args(argv)
    if not args.dsn:
        print("no DSN: pass --dsn or set POSTGRES_URI", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from sqlalchemy import create_engine, text

    from app.vision_split import undo_split

    engine = create_engine(args.dsn)
    try:
        if args.list or not args.split_id:
            with engine.connect() as conn:
                for r in conn.execute(text(
                        "SELECT split_id, status, source_individual_id, new_individual_id, name, "
                        "jsonb_array_length(moved), created_at, undone_at, reason FROM vision_individual_split "
                        "ORDER BY created_at")).fetchall():
                    print(f"{r[0]} {r[1]:8s} source={r[2][:8]} new={(r[3] or '-')[:8]} name={r[4]!r} "
                          f"moved={r[5]} at={r[6]:%Y-%m-%d %H:%M} undone={r[7] or '-'} {r[8] or ''}")
            return 0 if args.list else 2
        conn = engine.connect()
        trans = conn.begin()
        try:
            report = undo_split(conn, args.split_id)
            print(json.dumps(report, indent=2, default=str))
            if not report.get("ok"):
                trans.rollback()
                print("REFUSED: nothing changed", file=sys.stderr)
                return 1
            if args.apply:
                trans.commit()
                print("APPLIED")
            else:
                trans.rollback()
                print("DRY RUN: rolled back. Re-run with --apply to undo.")
            return 0
        except Exception:
            trans.rollback()
            raise
        finally:
            conn.close()
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
