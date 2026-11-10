"""The disk-rule replay's ``--export`` after substrate_dispatch_results was dropped (2026-11-10).

Pins:
- against a database without the table, ``--export`` stops with exit code 2, names the backup and
  the restore, and leaves the committed fixture untouched;
- with ``--dispatch-db``, the prune-run query goes to that database and the disk query still goes to
  the live database.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "evals" / "run_image_prune_replay_eval.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("run_image_prune_replay_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def test_export_without_the_table_exits_2_and_keeps_the_fixture(ev, monkeypatch, tmp_path, capsys):
    fixture = tmp_path / "image_prune_replay.json.gz"
    fixture.write_bytes(b"kept")
    calls = []

    def fake_psql(sql, db=ev.LIVE_DB):
        calls.append((db, sql))
        assert "to_regclass('substrate_dispatch_results')" in sql, "only the presence check may run"
        return [["f"]]

    monkeypatch.setattr(ev, "_psql", fake_psql)
    monkeypatch.setattr(ev, "FIXTURE", fixture)
    monkeypatch.setattr(sys, "argv", ["eval", "--export"])
    assert ev.main() == 2
    err = capsys.readouterr().err
    assert "substrate_dispatch_results is not in database 'conjourney'" in err
    assert ev.DISPATCH_BACKUP in err and "--dispatch-db" in err and "pg_restore" in err
    assert fixture.read_bytes() == b"kept"
    assert len(calls) == 1


def test_export_reads_runs_from_the_restored_db_and_disk_from_live(ev, monkeypatch, tmp_path):
    fixture = tmp_path / "image_prune_replay.json.gz"
    calls = []

    def fake_psql(sql, db=ev.LIVE_DB):
        calls.append((db, sql))
        if "to_regclass" in sql:
            return [["t"]]
        if "from orion_biometrics_summary" in sql:
            return [["2026-10-22 00:00:10+00", "0.71"], ["2026-10-22 00:00:40+00", "0.72"]]
        if "from substrate_dispatch_results" in sql:
            return [["2026-10-22 01:00:00+00", "prune", "150", "0", "71.5", "150"],
                    ["2026-10-22 02:00:00+00", "declined_nothing_to_reclaim", "", "", "", ""]]
        raise AssertionError(sql)

    monkeypatch.setattr(ev, "_psql", fake_psql)
    monkeypatch.setattr(ev, "FIXTURE", fixture)
    monkeypatch.setattr(sys, "argv", ["eval", "--export", "--dispatch-db", "dispatch_restore"])
    assert ev.main() == 0
    with gzip.open(fixture, "rt") as f:
        data = json.load(f)
    assert data["disk"] == [[pytest.approx(ev._ts("2026-10-22 00:00:40+00").timestamp(), abs=0.1), 72.0]]
    assert [r["before"] for r in data["runs"]] == [150]   # a run with no dangling count is skipped
    for db, sql in calls:
        if "substrate_dispatch_results" in sql:
            assert db == "dispatch_restore", sql
        else:
            assert db == "conjourney", sql
