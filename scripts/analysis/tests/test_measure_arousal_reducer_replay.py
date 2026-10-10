"""The shipped-reducer replay eval (Temporal Self rev 4, order 3) on a synthetic day."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_arousal_reducer_replay import export_sql, replay  # noqa: E402

T0 = datetime(2026, 10, 10, 0, 0, tzinfo=timezone.utc)


def day_rows(*, hot_window=None, queue_window=None, depth=3):
    rows = [{"kind": "turn", "at": (T0 + timedelta(hours=9)).isoformat()}]
    t = T0 - timedelta(hours=1)
    while t < T0 + timedelta(hours=24):
        temp = 35.0 if hot_window and hot_window[0] <= t < hot_window[1] else 30.0
        rows.append({"kind": "heat", "at": t.isoformat(), "temp_c": temp})
        t += timedelta(seconds=30)
    t = T0 - timedelta(hours=1)
    while t < T0 + timedelta(hours=24):
        q = {"agent": depth} if queue_window and queue_window[0] <= t < queue_window[1] else {}
        rows.append({"kind": "gpu_state", "at": t.isoformat(), "host": "circe", "queue_depth": q})
        t += timedelta(seconds=5)
    return rows


def test_quiet_day_is_idle_with_an_engaged_burst():
    r = replay(day_rows(), T0, T0 + timedelta(hours=24))
    m = r["total_minutes"]
    assert m["unknown"] == 0 and m["strained"] == 0
    assert 44 <= m["engaged"] <= 46 and m["idle"] > 1300


def test_queue_strain_and_heat_strain_reach_strained_and_clear():
    rows = day_rows(queue_window=(T0 + timedelta(hours=2), T0 + timedelta(hours=3)),
                    hot_window=(T0 + timedelta(hours=14), T0 + timedelta(hours=14, minutes=10)))
    r = replay(rows, T0, T0 + timedelta(hours=24))
    spans = r["strained_spans"]
    assert len(spans) == 2
    # Queue: strained ~5 min after it starts, until ~10 min after it clears.
    assert 60 <= spans[0]["minutes"] <= 70 and spans[0]["reasons"]
    assert r["acceptance_bar_every_level_between_0_and_100pct"]
    assert r["strained_exits_per_day"] == {"2026-10-10": 2}


def test_missing_gpu_history_reads_unknown_not_idle():
    rows = [x for x in day_rows() if x["kind"] != "gpu_state"]
    r = replay(rows, T0, T0 + timedelta(hours=2))
    assert r["total_minutes"]["unknown"] == 120.0
    assert "stale:gpu_state" in r["unknown_step_reasons"]


def test_export_sql_uses_the_juniper_rule_and_is_read_only():
    sql = export_sql(T0, T0 + timedelta(hours=1))
    assert "READ ONLY" in sql and "unsolicited" in sql and "queue_depth" in sql
