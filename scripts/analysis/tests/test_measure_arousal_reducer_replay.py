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


# --- deferrable self-work (2026-10-23) --------------------------------------------------------

def _queue_day(deferrable=None):
    rows = day_rows(queue_window=(T0 + timedelta(hours=2), T0 + timedelta(hours=3)), depth=3)
    for r in rows:
        if r["kind"] == "gpu_state" and r["queue_depth"] and deferrable is not None:
            r["deferrable_depth"] = deferrable
    return rows


def test_saved_deferrable_depth_leaves_self_work_out_and_flag_restores_the_old_reading():
    end = T0 + timedelta(hours=24)
    assert replay(_queue_day({"agent": 3}), T0, end)["total_minutes"]["strained"] == 0
    assert replay(_queue_day({"agent": 1}), T0, end)["total_minutes"]["strained"] > 60   # 2 left: load
    assert replay(_queue_day({"agent": 3}), T0, end, gpu_deferrable=False)["total_minutes"]["strained"] > 60


def test_reconstructed_deferrable_from_hold_events():
    """Older rows (no column): the pool's own admitted/queued -> granted events for background holds."""
    rows = _queue_day()
    for lid in ("a", "b", "c"):
        rows.append({"kind": "deferrable_event", "at": (T0 + timedelta(hours=1)).isoformat(),
                     "event": "queued", "lease_id": lid, "work_class": "agent"})
    rows.append({"kind": "deferrable_event", "at": (T0 + timedelta(hours=2, minutes=30)).isoformat(),
                 "event": "granted", "lease_id": "a", "work_class": "agent"})
    end = T0 + timedelta(hours=24)
    assert replay(rows, T0, end)["total_minutes"]["strained"] > 60          # flag off: old reading
    r = replay(rows, T0, end, reconstruct_deferrable=True)
    # 2:00-2:30 all three deferrable (load 0); from 2:30 one is granted, load 1 < floor: never strained.
    assert r["total_minutes"]["strained"] == 0


def test_export_sql_reads_the_deferrable_column_and_hold_events():
    sql = export_sql(T0, T0 + timedelta(hours=1))
    assert "to_jsonb(h)->'deferrable_depth'" in sql
    assert "l.priority = 'background' AND l.kind = 'hold' AND l.hold_lease_id IS NULL" in sql
