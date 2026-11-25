"""The one-time door backfill (scripts/self_question_door_backfill.py): a dry run never writes the database or the
apply's snapshot, an apply refuses to run twice, rollback is one transaction, and progress never counts a vanished
row as done. psql is faked; nothing touches a real database."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sqd_backfill", ROOT / "scripts/self_question_door_backfill.py")
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

ROWS = [{"question_id": "lived.orion.a", "text": "Who do I miss?", "door": None, "door_attempts": 0},
        {"question_id": "lived.self_monitor.x.broken.20261103T1200Z", "text": "Why?", "door": None, "door_attempts": 0}]


@pytest.fixture()
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "JOB", tmp_path)
    sent: list[str] = []

    def psql(sql, **vars_):
        sent.append(sql)
        if sql == bf.HAS_DOOR_SQL:
            return "1\n"
        if sql == bf.TARGET_SQL:
            return "".join(json.dumps(r) + "\n" for r in ROWS)
        if sql.startswith("UPDATE curiosity_self_questions SET door = 'held'"):
            assert vars_["ids"] == "{" + ",".join(r["question_id"] for r in ROWS) + "}"
            return "lived.orion.a\nlived.self_monitor.x.broken.20261103T1200Z\nUPDATE 2\n"
        if sql == bf.STATE_SQL:
            return "".join(json.dumps({"question_id": r["question_id"], "door": "routed", "door_attempts": 0}) + "\n"
                           for r in ROWS)
        raise AssertionError(sql)

    monkeypatch.setattr(bf, "psql", psql)
    monkeypatch.setattr(bf.time, "sleep", lambda s: None)
    return tmp_path, sent


def test_a_dry_run_writes_nothing_and_leaves_the_apply_snapshot_alone(job):
    path, sent = job
    assert bf.main([]) == 0
    assert not any(s.startswith("UPDATE") for s in sent)
    assert (path / "snapshot.dryrun.jsonl").exists() and not (path / "snapshot.jsonl").exists()
    # The documented flow: dry run, then apply.
    assert bf.main(["--apply", "--poll-sec", "0"]) == 0
    snap = [json.loads(x) for x in (path / "snapshot.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in snap] == [r["question_id"] for r in ROWS]
    assert "100% done" in (path / "progress.log").read_text() and (path / "report.md").exists()
    # A later dry run (nothing left to target) never overwrites or removes the apply's snapshot.
    assert bf.main([]) == 0 and (path / "snapshot.jsonl").exists()


def test_an_apply_refuses_to_run_over_an_earlier_applied_pass(job):
    path, _ = job
    (path / "snapshot.jsonl").write_text(json.dumps(ROWS[0]) + "\n")
    assert bf.main(["--apply"]) == 2


def test_rollback_is_one_transaction_over_the_snapshotted_ids(job, monkeypatch):
    path, _ = job
    (path / "snapshot.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    calls = []

    class _Done:
        stdout = "UPDATE 2"

    monkeypatch.setattr(bf.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw["input"])) or _Done())
    assert bf.main(["--rollback"]) == 0
    (cmd, sql), = calls
    assert "-1" in cmd and "ON_ERROR_STOP=1" in cmd and sql.count("UPDATE") == 1 and "minted_by = 'orion'" in sql


def test_progress_never_counts_a_vanished_row_as_done():
    line, held = bf.progress_line([{"question_id": "a", "door": "admitted"}], 2, 0.0, 60.0)
    assert "rows 1/2" in line and "1 target rows no longer found" in line and held == 0


def test_ids_outside_the_question_id_shape_are_refused():
    with pytest.raises(ValueError):
        bf.pg_array(["lived.orion.a", "x'; DROP TABLE y; --"])
