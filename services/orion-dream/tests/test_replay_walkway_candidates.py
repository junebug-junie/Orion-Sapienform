"""Graded walkway predictions as dream replay material (2026-10-15).

Pinned: met/missed -> one candidate per subject with the 14-day hit rate;
open/unscorable dropped; window filter; a miss outweighs all-met; at most a
sixth of the replay; never sleep pressure; never starts a sleep; read failure
-> the sleep exactly as before.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 10, 22, 13, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=20)


def _g(eid, subject, status, hours_ago, *, now=NOW, peak=460, stream="walkway"):
    scored = now - timedelta(hours=hours_ago)
    return {"expectation_id": eid, "stream_id": stream, "subject_key": f"individual:{subject}",
            "subject_label": subject, "status": status, "peak_minute": peak,
            "window_start": scored - timedelta(minutes=30), "scored_at": scored}


def test_one_candidate_per_subject_with_hit_rate_and_declared_weights():
    from app.replay import WALKWAY_WEIGHT_MET, WALKWAY_WEIGHT_MISSED, walkway_candidates

    rows = [
        _g("e1", "the black dog", "met", 2),
        _g("e2", "the black dog", "missed", 10),
        _g("e3", "the mail van", "met", 3, peak=600),
        _g("e4", "a cyclist", "unscorable", 1),
        _g("e5", "a runner", "open", 1),
        _g("e6", "the black dog", "met", 30),  # before the window
    ]
    cal = [{"stream_id": "walkway", "subject_key": "individual:the black dog", "met": 6, "missed": 2}]
    items = walkway_candidates(rows, cal, SINCE, NOW, tz=TZ)
    assert [i.ref_id for i in items] == ["walkway:e1", "walkway:e3"]
    dog, van = items
    assert dog.source_kind == "walkway" and dog.weight == WALKWAY_WEIGHT_MISSED
    assert van.weight == WALKWAY_WEIGHT_MET
    assert dog.text == (
        "On the walkway I expected the black dog around 07:40: came on Thu 22 Oct; "
        "did not come on Wed 21 Oct. Over the last 14 days it came 6 of 8 times."
    )
    assert "Over the last" not in van.text  # no tally -> no claim
    assert "walkway" in dog.tags


def test_nothing_graded_is_nothing():
    from app.replay import walkway_candidates

    assert walkway_candidates([], [], SINCE, NOW, tz=TZ) == []
    assert walkway_candidates([_g("e", "x", "open", 1)], None, SINCE, NOW, tz=TZ) == []


def test_slots_are_a_sixth_never_zero():
    from app.replay import walkway_replay_slots

    assert walkway_replay_slots(12) == 2
    assert walkway_replay_slots(3) == 1
    assert walkway_replay_slots(0) == 0


def test_flag_ships_on():
    from app.settings import Settings

    assert Settings.model_fields["DREAM_REPLAY_WALKWAY_ENABLED"].default is True
    assert Settings.model_fields["DREAM_WALKWAY_STREAM_IDS"].default == "walkway"


def test_store_sql_reads_met_missed_only_and_failure_is_empty(monkeypatch):
    from app import cycle_store
    from orion.situational.street_material import GRADED_EXPECTATIONS_SQL

    assert "status IN ('met', 'missed')" in GRADED_EXPECTATIONS_SQL
    assert "INSERT" not in GRADED_EXPECTATIONS_SQL.upper() and "UPDATE" not in GRADED_EXPECTATIONS_SQL.upper()

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(cycle_store, "_get_engine", boom)
    assert cycle_store.load_walkway_rows(SINCE, NOW) == {"graded": [], "calibration": []}
    assert cycle_store.load_walkway_rows(SINCE, NOW, stream_ids=[]) == {"graded": [], "calibration": []}


# --- cycle ---------------------------------------------------------------------


def _rows():
    return {
        "metacog": [{"id": "m1", "summary": "Recall returned empty for three turns", "severity": "critical",
                     "trigger_kind": "recall_empty", "tags": ["recall"]}],
        "compaction_request": [{"request_id": "r1", "theme": "sleep schedule", "reason": "recurs"}],
        "resonance": [{"alert_id": "a1", "theme_key": "being watched", "violation_count": 4}],
    }


class _Fakes:
    def __init__(self, rows, walkway=None):
        self.rows, self.walkway = rows, walkway
        self.persisted, self.reads = [], []

    def deps(self):
        from app.cycle import CycleDeps

        def load(since, limit, until=None):
            return {} if until is not None else self.rows

        def load_walkway(since, until):
            self.reads.append((since, until))
            return self.walkway(until) if callable(self.walkway) else self.walkway

        async def complete(_prompt):
            return json.dumps({"link": True, "claim": "These two recur together more often than chance",
                               "why": "w"})

        return CycleDeps(
            load_source_rows=load, load_idle_minutes=lambda: 120.0,
            load_last_window_start=lambda: None, load_last_attempt_end=lambda: None,
            persist_cycle=lambda c: self.persisted.append(c) or True, complete=complete,
            load_walkway_rows=load_walkway if self.walkway is not None else None,
        )


def _many(until):
    return {"graded": [_g(f"e{i}", f"subject {i}", "missed", 0.1 * (i + 1), now=until) for i in range(8)],
            "calibration": []}


def test_walkway_reaches_replay_capped_and_never_pressure():
    from app.cycle import run_cycle_once

    off, on = _Fakes(_rows()), _Fakes(_rows(), walkway=_many)
    c_off = asyncio.run(run_cycle_once(off.deps()))
    c_on = asyncio.run(run_cycle_once(on.deps()))
    assert c_on.status == c_off.status == "completed"
    assert c_on.pressure.pressure == c_off.pressure.pressure
    assert c_on.pressure.counts == c_off.pressure.counts and "walkway" not in c_on.pressure.counts
    walk = [r for r in c_on.replay if r.source_kind == "walkway"]
    assert len(walk) == 2  # a sixth of DREAM_REPLAY_MAX=12
    assert c_on.replay[0].ref_id == "metacog:m1"
    (since, until), = on.reads
    assert since == c_on.pressure.since and until == c_on.started_at
    assert "walkway candidates=8" in c_on.note


def test_walkway_alone_never_starts_a_sleep():
    from app.cycle import run_cycle_once

    f = _Fakes({k: [] for k in _rows()}, walkway=_many)
    c = asyncio.run(run_cycle_once(f.deps(), trigger="manual", force=True))
    assert c.status == "empty" and c.replay == [] and f.reads == []


@pytest.mark.parametrize("bad", ["raise", "garbage"])
def test_walkway_read_failure_leaves_the_sleep_as_before(bad):
    from app.cycle import run_cycle_once

    def loader(until):
        if bad == "raise":
            raise RuntimeError("db down")
        return {"graded": [{"status": "met"}, {"nonsense": 1}], "calibration": "x"}

    off, on = _Fakes(_rows()), _Fakes(_rows(), walkway=loader)
    c_off = asyncio.run(run_cycle_once(off.deps()))
    c_on = asyncio.run(run_cycle_once(on.deps()))
    assert c_on.status == "completed"
    assert [r.ref_id for r in c_on.replay] == [r.ref_id for r in c_off.replay]
