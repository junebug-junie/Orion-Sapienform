"""The leisure replay's live mode after substrate_dispatch_results was dropped (2026-11-10).

Pins:
- against a database without the table, live mode stops with exit code 2 and a message that names
  the backup and the restore, and it never replays (no fake "PASS" on an empty world);
- with ``--dispatch-db``, the verdict queries go to that database and the session-length query
  still goes to the live database.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "evals" / "retro_leisure_self_start_replay.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("retro_leisure_self_start_replay_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def test_live_mode_without_the_table_exits_2_and_names_the_restore(ev, monkeypatch, capsys):
    calls = []

    def fake_psql(sql, db=ev.LIVE_DB):
        calls.append((db, sql))
        assert "to_regclass('substrate_dispatch_results')" in sql, "only the presence check may run"
        return [["f"]]

    monkeypatch.setattr(ev, "_psql", fake_psql)
    monkeypatch.setattr(ev, "replay", lambda *a, **k: pytest.fail("replayed on a missing table"))
    monkeypatch.setattr(sys, "argv", ["replay", "--live"])
    assert ev.main() == 2
    err = capsys.readouterr().err
    assert "substrate_dispatch_results is not in database 'conjourney'" in err
    assert ev.DISPATCH_BACKUP in err and "--dispatch-db" in err and "pg_restore" in err
    assert calls == [("conjourney", calls[0][1])]


def test_restored_db_feeds_verdicts_and_live_db_feeds_sessions(ev, monkeypatch):
    end = 1_762_000_000.0
    calls = []

    def fake_psql(sql, db=ev.LIVE_DB):
        calls.append((db, sql))
        if "to_regclass" in sql:
            return [["t"]]
        if "max(created_at)" in sql:
            return [[str(end)]]
        if "from substrate_dispatch_results" in sql:
            return [[str(end - 600), "started"], [str(end - 300), "cabinet_hot"]]
        if "from retro_play_sessions" in sql:
            return [[str(end - 590), "1200"]]
        raise AssertionError(sql)

    monkeypatch.setattr(ev, "_psql", fake_psql)
    world, start, got_end, real = ev.live_world(1.0, "dispatch_restore")
    assert got_end == end and start == end - 86400
    assert real["sends"] == 2 and real["open_verdicts"] == 1 and real["durations"] == [1200.0]
    assert world.session_sec == 1200.0
    for db, sql in calls:
        if "substrate_dispatch_results" in sql:
            assert db == "dispatch_restore", sql
        else:
            assert db == "conjourney", sql


def test_restored_db_without_leisure_verdicts_is_named_not_crashed(ev, monkeypatch):
    def fake_psql(sql, db=ev.LIVE_DB):
        return [["t"]] if "to_regclass" in sql else [[""]]

    monkeypatch.setattr(ev, "_psql", fake_psql)
    with pytest.raises(ev.DispatchHistoryMissing, match="no retro_leisure verdicts"):
        ev.live_world(1.0, "dispatch_restore")
