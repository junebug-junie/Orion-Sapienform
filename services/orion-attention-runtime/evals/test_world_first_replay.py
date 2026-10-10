"""Synthetic check of the world-first replay eval's own arithmetic (the live
run needs the production database; see replay_world_first.py)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_world_first import replay, spike_check, storm_check  # noqa: E402

END = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _calm_rows(node: str, value: float, *, days: float = 6.0, every_min: int = 5):
    t = END - timedelta(days=days)
    rows = []
    i = 0
    while t <= END:
        rows.append((node, t, value if i % 10 else value * 3))
        t += timedelta(minutes=every_min)
        i += 1
    return rows


def test_calm_body_and_quiet_chat_is_mostly_no_winner() -> None:
    rows = _calm_rows("node:substrate.biometrics", 0.02)
    out = replay(rows, [], None, start=END - timedelta(hours=6), end=END, step=timedelta(minutes=30))
    assert out["internal_winner_raw_lt_0_05_share"] < 0.2
    assert out["no_winner_share"] > 0.5


def test_real_spike_wins_and_a_sustained_storm_keeps_winning() -> None:
    rows = _calm_rows("node:substrate.execution", 0.0)
    rows.append(("node:substrate.execution", END - timedelta(hours=2), 1.0))
    out = spike_check(rows, [], None, start=END - timedelta(days=1), end=END,
                      node_id="node:substrate.execution", min_value=0.9)
    assert out["readings"] == 1 and out["won"] == 1
    storm = storm_check(rows, [], None, node_id="node:substrate.execution",
                        storm_start=END - timedelta(hours=5), hours=5.0)
    assert storm["won_share"] > 0.9


def test_one_carried_event_holds_at_most_the_orienting_window_with_decay() -> None:
    """A single codebase-style event, carried forward for 18 minutes until the
    next write: with decay it wins at most EVENT_ORIENTING_WINDOW_SEC; without,
    it holds until the next write (the 2026-10-10 05:51 live case)."""
    from orion.attention.world_first import EVENT_ORIENTING_WINDOW_SEC

    node = "node:substrate.codebase"
    rows = _calm_rows(node, 0.0, every_min=15)
    rows = [r for r in rows if r[1] < END - timedelta(minutes=40)]
    rows.append((node, END - timedelta(minutes=40), 0.988))
    rows.append((node, END - timedelta(minutes=22), 0.0))
    kw = dict(start=END - timedelta(minutes=40), end=END, step=timedelta(seconds=30))
    on = replay(rows, [], None, **kw)["longest_hold"][node]["seconds"]
    off = replay(rows, [], None, event_decay=False, **kw)["longest_hold"][node]["seconds"]
    assert on <= EVENT_ORIENTING_WINDOW_SEC
    assert off >= 17 * 60
