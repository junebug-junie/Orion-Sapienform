"""Replay: does world:chat keep one open-loop id through a conversation?

Live bug (found 2026-11-12): the world:chat label carried the message count
("Juniper is talking (2 messages in the last 15 min)"). The open-loop id is a
hash of the label (orion.substrate.attention.scoring.open_loop_id), so each new
message -- and each message ageing out of the 15-min window -- gave the same
conversation a new loop id. Reverie chains, the refractory table and loop
verdicts are all keyed by that id.

This reads every stored broadcast tick that world:chat won
(substrate_attention_broadcast_log), takes the chat loop's stored id and
description, and recomputes the id with today's label
(orion.attention.world_first.CHAT_TALKING_LABEL). Checks:

- ``formula_reproduces``: open_loop_id(stored description) == stored id on every
  tick (the replay uses the real id formula, not a copy).
- ``new_single_id``: today's label gives one id for every chat win.
- ``new_no_switches``: zero id switches between chat wins < 15 min apart.

Old-label distinct ids and switches are reported for the before/after.
Read-only. Exit status 1 if a check fails or there are no chat wins.

Run:
  python services/orion-substrate-runtime/evals/run_world_chat_loop_id_replay.py --days 7
  (POSTGRES_URI, or the default local conjourney DSN)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import CHAT_RATE_WINDOW, CHAT_TALKING_LABEL  # noqa: E402
from orion.schemas.attention_candidate import WORLD_CHAT_SOURCE_ID  # noqa: E402
from orion.substrate.attention.scoring import open_loop_id  # noqa: E402

SQL = """
SELECT generated_at, loop->>'id', loop->>'description'
FROM substrate_attention_broadcast_log,
     jsonb_array_elements(projection_json->'frame'->'open_loops') loop
WHERE generated_at >= now() - make_interval(days => :days)
  AND projection_json->'frame'->'debug'->'world_first'->>'winner' = :chat
  AND loop->'source_refs' ? :chat
ORDER BY generated_at
"""


def _switches(rows: list[tuple[datetime, str]], gap: timedelta) -> int:
    n = 0
    for (t0, a), (t1, b) in zip(rows, rows[1:]):
        if t1 - t0 < gap and a != b:
            n += 1
    return n


def evaluate(rows: list[tuple[datetime, str, str]], *, gap: timedelta = CHAT_RATE_WINDOW) -> dict:
    """rows: (generated_at, stored loop id, stored description), time-ordered."""
    new_id = open_loop_id(CHAT_TALKING_LABEL)
    old = [(t, lid) for t, lid, _ in rows]
    new = [(t, new_id) for t, _, _ in rows]
    reproduced = sum(1 for _, lid, desc in rows if open_loop_id(desc) == lid)
    report = {
        "chat_won_ticks": len(rows),
        "old_distinct_ids": len({lid for _, lid in old}),
        "old_switches_within_window": _switches(old, gap),
        "old_labels": sorted({desc for _, _, desc in rows}),
        "new_id": new_id,
        "new_distinct_ids": len({lid for _, lid in new}),
        "new_switches_within_window": _switches(new, gap),
        "formula_reproduced": reproduced,
    }
    report["checks"] = {
        "has_rows": len(rows) > 0,
        "formula_reproduces": reproduced == len(rows),
        "new_single_id": report["new_distinct_ids"] <= 1,
        "new_no_switches": report["new_switches_within_window"] == 0,
    }
    report["pass"] = all(report["checks"].values())
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    from sqlalchemy import create_engine, text

    dsn = os.getenv("POSTGRES_URI") or "postgresql://postgres:postgres@localhost:55432/conjourney"
    with create_engine(dsn).connect() as conn:
        rows = [
            (r[0], str(r[1]), str(r[2]))
            for r in conn.execute(text(SQL), {"days": args.days, "chat": WORLD_CHAT_SOURCE_ID})
        ]
    report = evaluate(rows)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
