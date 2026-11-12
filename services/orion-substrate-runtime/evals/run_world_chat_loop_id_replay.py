"""Replay: does world:chat keep one open-loop id through a conversation?

Live bug (found 2026-11-12): the world:chat label carried the message count
("Juniper is talking (2 messages in the last 15 min)"). The open-loop id is a
hash of the label (orion.substrate.attention.scoring.open_loop_id), so each new
message -- and each message ageing out of the 15-min window -- gave the same
conversation a new loop id. Reverie chains, the refractory table and loop
verdicts are all keyed by that id.

This reads every stored broadcast tick that world:chat won
(substrate_attention_broadcast_log), takes the chat loop's stored id and
description, and rebuilds the label at that tick with today's producer:
``chat_candidate(<Juniper's turns up to the tick>, now=<tick>).label`` from
chat_history_log (the real builder, so a count sneaking back into the label
fails the check). Checks:

- ``formula_reproduces``: open_loop_id(stored description) == stored id on every
  tick (the replay uses the real id formula, not a copy).
- ``new_single_id``: today's producer gives one id for every chat win.
- ``new_no_switches``: zero id switches between chat wins < 15 min apart.

``--stored-since <ISO time>`` (after deploy): also checks the STORED ids of chat
wins at or after that time directly -- the live proof (``stored_single_id``).

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import (  # noqa: E402
    CHAT_RATE_WINDOW,
    CHAT_TURN_TIMES_SQL,
    chat_candidate,
)
from orion.substrate.prediction_error_magnitude import WINDOW_7D  # noqa: E402
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


def evaluate(
    rows: list[tuple[datetime, str, str]],
    turn_times: list[datetime],
    *,
    gap: timedelta = CHAT_RATE_WINDOW,
    stored_since: datetime | None = None,
) -> dict:
    """rows: (generated_at, stored loop id, stored description), time-ordered.
    turn_times: Juniper's chat turns (aware UTC), covering 7 days before the first row."""
    times = sorted(turn_times)
    old = [(t, lid) for t, lid, _ in rows]
    new = []
    labels = set()
    for t, _, _ in rows:
        label = chat_candidate([x for x in times if x <= t], now=t).label
        labels.add(label)
        new.append((t, open_loop_id(label)))
    reproduced = sum(1 for _, lid, desc in rows if open_loop_id(desc) == lid)
    report = {
        "chat_won_ticks": len(rows),
        "old_distinct_ids": len({lid for _, lid in old}),
        "old_switches_within_window": _switches(old, gap),
        "old_labels": sorted({desc for _, _, desc in rows}),
        "new_labels": sorted(labels),
        "new_distinct_ids": len({lid for _, lid in new}),
        "new_switches_within_window": _switches(new, gap),
        "formula_reproduced": reproduced,
    }
    checks = {
        "has_rows": len(rows) > 0,
        "formula_reproduces": reproduced == len(rows),
        "new_single_id": report["new_distinct_ids"] <= 1,
        "new_no_switches": report["new_switches_within_window"] == 0,
    }
    if stored_since is not None:
        after = [(t, lid) for t, lid in old if t >= stored_since]
        report["stored_since"] = stored_since.isoformat()
        report["stored_ticks"] = len(after)
        report["stored_distinct_ids"] = len({lid for _, lid in after})
        checks["stored_has_rows"] = len(after) > 0
        checks["stored_single_id"] = report["stored_distinct_ids"] == 1
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--stored-since", default=None, help="ISO time; check stored ids of chat wins after it")
    args = ap.parse_args()
    from sqlalchemy import create_engine, text

    stored_since = None
    if args.stored_since:
        stored_since = datetime.fromisoformat(args.stored_since)
        if stored_since.tzinfo is None:
            stored_since = stored_since.replace(tzinfo=timezone.utc)
    dsn = os.getenv("POSTGRES_URI") or "postgresql://postgres:postgres@localhost:55432/conjourney"
    with create_engine(dsn).connect() as conn:
        rows = [
            (r[0], str(r[1]), str(r[2]))
            for r in conn.execute(text(SQL), {"days": args.days, "chat": WORLD_CHAT_SOURCE_ID})
        ]
        since = (rows[0][0] if rows else datetime.now(timezone.utc)) - WINDOW_7D - CHAT_RATE_WINDOW
        turns = [
            r[0] if r[0].tzinfo is not None else r[0].replace(tzinfo=timezone.utc)
            for r in conn.execute(
                text(CHAT_TURN_TIMES_SQL), {"since": since.astimezone(timezone.utc).replace(tzinfo=None)}
            )
        ]
    report = evaluate(rows, turns, stored_since=stored_since)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
