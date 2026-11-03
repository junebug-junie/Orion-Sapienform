"""Self-monitor E1 at orion-actions' shared journal seam (`_dispatch_journal`).

Every in-process journal compose (walkway_*, daily/notify summaries, town_episode, manual, collapse)
passes `_dispatch_journal`, so the gate sits there. Driven through the real lifespan closure with the
bus and cortex faked (test_act_emit._Harness). A refused journal opens no act and is audited
`skipped` / `track_record_failing`; a probe runs and is marked in its audit row; unknown state
changes nothing.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app import main as actions_main
from orion.self_monitor.view import self_monitor_view
from test_act_emit import _Harness, _journal_env

MANUAL = "act.actions.journal.make.manual"


def _state(status: str, *, faculty: str = MANUAL, age: timedelta = timedelta(0)) -> str:
    at = datetime.now(timezone.utc) - age
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": at.isoformat(), "window_days": 7.0,
                       "faculties": [{
                           "faculty": faculty, "gradeability": "ungradeable:too_few",
                           "window_start": (at - timedelta(days=7)).isoformat(), "window_end": at.isoformat(),
                           "n_bets": 28, "n_graded": 6, "n_failed_to_execute": 22, "n_expired": 0,
                           "weight_graded": 2.9, "candidate_status": status, "status": status,
                           "status_since": (at - timedelta(days=1)).isoformat()}]})


def _audits(hunter) -> list[dict]:
    out = []
    for c in hunter.bus.publish.call_args_list:
        payload = getattr(c.args[1], "payload", None)
        if isinstance(payload, dict) and "status" in payload and "action_name" in payload:
            out.append(payload)
    return out


def _run(tmp_path, *, raw, ledger, enabled=True):
    h = _Harness(orch_payload={"ok": True})

    async def body(app):
        await app.state.bus_handler(_journal_env())

    with (
        patch.object(actions_main.settings, "self_monitor_act_gate_enabled", enabled),
        patch.object(actions_main.settings, "postgres_uri", "postgresql://x/y"),
        patch.object(actions_main, "read_view_sync", lambda url, now: self_monitor_view(raw, now=now)),
        patch.object(actions_main, "_fetch_ledger_rows_blocking", lambda dsn, sql, params: ledger),
    ):
        events = asyncio.run(h.run(tmp_path, body))
    return events, _audits(h.hunters[0])


def test_failing_journal_is_refused_and_opens_no_act(tmp_path):
    recent = [{"object_ref": "journal_trigger:manual", "opened_at": datetime.now(timezone.utc) - timedelta(hours=1)}]
    events, audits = _run(tmp_path, raw=_state("broken"), ledger=recent)
    assert events == []
    skipped = [a for a in audits if a["status"] == "skipped"]
    assert skipped and skipped[0]["reason"] == "track_record_failing"
    assert "self_monitor:track_record_failing:act.actions.journal.make.manual:broken" in json.dumps(skipped[0])


def test_probe_after_six_hours_runs_and_is_marked(tmp_path):
    old = [{"object_ref": "journal_trigger:manual", "opened_at": datetime.now(timezone.utc) - timedelta(hours=7)}]
    events, audits = _run(tmp_path, raw=_state("unreliable"), ledger=old)
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    done = [a for a in audits if a["status"] == "completed"]
    assert done and "self_monitor_probe" in json.dumps(done[0])


@pytest.mark.parametrize("raw", [None, "{bad", "STALE", "RELIABLE", "OTHER_FACULTY"])
def test_unknown_or_healthy_view_changes_nothing(tmp_path, raw):
    raw = {"STALE": _state("broken", age=timedelta(hours=4)), "RELIABLE": _state("reliable"),
           "OTHER_FACULTY": _state("broken", faculty="act.actions.journal.make.walkway_grade")}.get(raw, raw)
    recent = [{"object_ref": "journal_trigger:manual", "opened_at": datetime.now(timezone.utc)}]
    gated, gated_audits = _run(tmp_path, raw=raw, ledger=recent)
    off, off_audits = _run(tmp_path, raw=raw, ledger=recent, enabled=False)
    assert [e["event_kind"] for e in gated] == [e["event_kind"] for e in off] == ["act_opened", "act_closed"]
    assert [a["status"] for a in gated_audits] == [a["status"] for a in off_audits]
    assert "self_monitor" not in json.dumps(gated_audits)
