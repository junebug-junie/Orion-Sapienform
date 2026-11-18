from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("repair_wp_read_timestamps", ROOT / "scripts" / "repair_wp_read_timestamps.py")
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod  # dataclasses resolve their module by name
_spec.loader.exec_module(mod)

NOW = datetime(2026, 10, 11, 1, 0, tzinfo=timezone.utc)
SERVER = datetime(2026, 10, 10, 14, 33, 48, 430353, tzinfo=timezone.utc)


def _node(node_id, trace, observed, stamp=None):
    return {"node_id": node_id, "trace_id": trace, "observed_at": observed,
            "activation_decayed_at": stamp, "_now": NOW}


def test_future_node_gets_server_time_and_frozen_stamp_is_moved():
    nodes = [_node("sub-concept-wp-read-a", "t1", "2026-10-11T02:30:00+00:00", "2026-10-11T02:30:00+00:00")]
    changes, unmatched = mod.plan_node_changes(nodes, {"t1": SERVER})
    assert unmatched == []
    assert [(c.store, c.new) for c in changes] == [
        ("falkor.observed_at", SERVER.isoformat()),
        ("falkor.activation_decayed_at", SERVER.isoformat()),
    ]
    assert changes[0].old == "2026-10-11T02:30:00+00:00"


def test_real_decay_stamp_is_left_alone():
    # Stamp written by a real decay tick (after the server time, not the bad
    # observed_at, not in the future) carries real decay state -- keep it.
    nodes = [_node("n", "t1", "2026-10-10T12:05:00+00:00", "2026-10-11T00:41:52.878546+00:00")]
    changes, _ = mod.plan_node_changes(nodes, {"t1": SERVER})
    assert [c.store for c in changes] == ["falkor.observed_at"]


def test_early_model_time_is_rewritten_but_near_server_stamp_is_not():
    early = _node("early", "t1", "2026-10-10T12:05:00+00:00")
    near = _node("near", "t2", "2026-10-10T14:33:40+00:00")  # 8s before the DB write
    changes, _ = mod.plan_node_changes([early, near], {"t1": SERVER, "t2": SERVER})
    assert [c.key for c in changes] == ["early"]


def test_any_value_later_than_server_time_is_rewritten():
    late = _node("late", "t1", "2026-10-10T14:33:49+00:00")  # 0.6s after
    changes, _ = mod.plan_node_changes([late], {"t1": SERVER})
    assert [c.key for c in changes] == ["late"]


def test_unmatched_node_is_untouched_and_listed():
    changes, unmatched = mod.plan_node_changes([_node("orphan", "gone", "2026-07-27T08:15:00+00:00")], {"t1": SERVER})
    assert changes == []
    assert [u["node_id"] for u in unmatched] == ["orphan"]


def test_journal_stage1_and_stage2_use_their_own_server_column():
    s2 = datetime(2026, 10, 10, 14, 38, 55, tzinfo=timezone.utc)
    rows = [
        {"entry_id": "e1", "source_ref": "world_pulse_read:t1",
         "created_at": datetime(2026, 10, 10, 12, 5, tzinfo=timezone.utc)},
        {"entry_id": "e2", "source_ref": "world_pulse_read_stage2:u1",
         "created_at": datetime(2026, 10, 10, 12, 10, tzinfo=timezone.utc)},
        {"entry_id": "e3", "source_ref": "world_pulse_read_stage2:missing",
         "created_at": datetime(2026, 10, 10, 12, 10, tzinfo=timezone.utc)},
    ]
    changes, unmatched = mod.plan_journal_changes(
        rows, table="journal_entries", handoff_at_by_trace={"t1": SERVER}, stage2_at_by_trace={"u1": s2})
    assert {c.key: c.new for c in changes} == {"e1": SERVER.isoformat(), "e2": s2.isoformat()}
    assert [u["entry_id"] for u in unmatched] == ["e3"]


def test_stage1_trace_does_not_satisfy_stage2_row():
    rows = [{"entry_id": "e", "source_ref": "world_pulse_read_stage2:t1",
             "created_at": datetime(2026, 10, 10, 12, 10, tzinfo=timezone.utc)}]
    changes, unmatched = mod.plan_journal_changes(
        rows, table="journal_entries", handoff_at_by_trace={"t1": SERVER}, stage2_at_by_trace={})
    assert changes == [] and len(unmatched) == 1


def test_parse_ts_pins_utc_whatever_the_session_timezone():
    from datetime import timedelta
    mdt = datetime(2026, 10, 10, 8, 33, 48, tzinfo=timezone(timedelta(hours=-6)))
    assert mod.parse_ts(mdt).isoformat() == "2026-10-10T14:33:48+00:00"


def test_settle_reapplies_until_two_clean_rounds(monkeypatch):
    applied = iter([238, 0, 0])
    verified = iter([1, 0, 0, 0])
    monkeypatch.setattr(mod, "_run", lambda args, reverse, quiet=False: next(applied))
    monkeypatch.setattr(mod, "cmd_verify", lambda args, quiet=False: next(verified))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    class A:
        max_rounds, wait_s = 5, 0
    assert mod.cmd_settle(A()) == 0
