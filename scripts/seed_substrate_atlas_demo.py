#!/usr/bin/env python3
"""Seed Substrate Atlas vision_observation demo trace."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQL_WRITER_ROOT = _REPO_ROOT / "services" / "orion-sql-writer"
# Drop the scripts/ entry Python adds when run as scripts/*.py — it shadows stdlib `platform`.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path[:0] = [str(_REPO_ROOT), str(_SQL_WRITER_ROOT)]

if not os.environ.get("DATABASE_URL"):
    print("DATABASE_URL is required", file=sys.stderr)
    sys.exit(1)

os.environ.setdefault("POSTGRES_URI", os.environ["DATABASE_URL"])

from app.db import get_session, remove_session  # noqa: E402
from orion.grammar.ledger import apply_grammar_event  # noqa: E402
from orion.grammar.seed_demo import TRACE_ID, build_vision_demo_events  # noqa: E402


def main() -> None:
    events = build_vision_demo_events()
    atom_count = 0
    edge_count = 0
    sess = get_session()
    try:
        for event in events:
            if apply_grammar_event(sess, event):
                if event.event_kind == "atom_emitted" and event.atom is not None:
                    atom_count += 1
                elif event.event_kind == "edge_emitted" and event.edge is not None:
                    edge_count += 1
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        try:
            sess.close()
        finally:
            remove_session()
    print(
        f"seeded trace_id={TRACE_ID} events={len(events)} atoms={atom_count} edges={edge_count}"
    )


if __name__ == "__main__":
    main()
