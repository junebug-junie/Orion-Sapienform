#!/usr/bin/env python3
"""Undo one "same person" link Juniper made on a walkway story card (services/orion-sql-writer/app/vision_link_undo.py).

Dry run by default: it runs the whole undo inside a transaction, prints what would change, and ROLLS
BACK. ``--apply`` commits it.

    python3 scripts/undo_vision_link.py --list
    python3 scripts/undo_vision_link.py ask-<id>            # dry run
    python3 scripts/undo_vision_link.py ask-<id> --apply    # undo it

What undo does (vision_link_undo.undo_link): deletes the 'same' row the answer made, re-derives every
copied name in both groups (a typed name is never touched), and marks the ask ``undone_at``. It refuses,
naming the reason, when: the link is not one the story card made, it is older than 7 days, a split
recorded it, a name was typed on the linked person after it, or the two stay joined through other
'same' links (then undoing this one would change nothing). The Hub's "Undo" button asks the
individuals cycle to run the same function.

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
    ap.add_argument("ask_id", nargs="?")
    ap.add_argument("--apply", action="store_true", help="commit the undo (default: dry run, rolled back)")
    ap.add_argument("--list", action="store_true", help="list story-card links and exit")
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL"))
    args = ap.parse_args(argv)
    if not args.dsn:
        print("no DSN: pass --dsn or set POSTGRES_URI", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from sqlalchemy import create_engine, text

    from app.vision_link_undo import undo_link

    engine = create_engine(args.dsn)
    try:
        if args.list or not args.ask_id:
            with engine.connect() as conn:
                for r in conn.execute(text(
                        "SELECT a.ask_id, a.source_ref, a.answer, a.answered_at, a.answer_detail->>'undone_at', "
                        "p.verdict FROM orion_ask a LEFT JOIN vision_individual_pair_verdict p "
                        "ON p.individual_a || '|' || p.individual_b = a.source_ref AND p.ask_id = a.ask_id "
                        "WHERE a.source_kind = 'vision_same_individual' AND a.answer_detail->>'via' = 'motion_story' "
                        "ORDER BY a.answered_at")).fetchall():
                    print(f"{r[0]} pair={r[1]} answer={r[2]!r} at={r[3]:%Y-%m-%d %H:%M} row={r[5] or '-'} "
                          f"undone={r[4] or '-'}")
            return 0 if args.list else 2
        conn = engine.connect()
        trans = conn.begin()
        try:
            report = undo_link(conn, args.ask_id)
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
