"""E1 act gate (orion/self_monitor/gate.py) and E2 self-question (orion/self_monitor/questions.py).

Live shapes (2026-11-03): ``act.actions.journal.make.walkway_grade`` is held broken (22 of 28 never
ran), ``act.prime.predict`` unreliable (works 2% over 36), ``act.thought.painting.make`` reliable.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.self_monitor.adapters import act_faculty
from orion.self_monitor.gate import ActGate, act_gate, last_attempt_sql, latest_attempt_at
from orion.self_monitor.questions import question_id_for, self_questions_for
from orion.self_monitor.view import TRACK_RECORD_FAILING_REASON, SelfMonitorView, self_monitor_view

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
WALKWAY = "act.actions.journal.make.walkway_grade"
TOWN = "act.actions.journal.make.town_episode"
PAINT = "act.thought.painting.make"
PRIME = "act.prime.predict"
SIX_H = 6 * 3600.0


def _record(faculty: str, status: str, *, since: datetime | None = None, **kw) -> dict:
    base = {
        "faculty": faculty, "sources": ["grammar_acts"], "claim_kind": "success", "gradeability": "graded",
        "window_start": (NOW - timedelta(days=7)).isoformat(), "window_end": NOW.isoformat(),
        "n_bets": 28, "n_graded": 6, "n_failed_to_execute": 22, "n_expired": 0, "weight_graded": 2.9,
        "candidate_status": status, "status": status,
        "status_since": (since or NOW - timedelta(days=1)).isoformat(),
    }
    base.update(kw)
    return base


def _state(*records: dict, computed_at: datetime = NOW) -> str:
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": computed_at.isoformat(),
                       "window_days": 7.0, "faculties": list(records)})


LIVE = _state(
    _record(WALKWAY, "broken"),
    _record(PRIME, "unreliable", n_bets=36, n_graded=36, n_failed_to_execute=0, reliability=0.0235),
    _record(PAINT, "reliable", n_bets=84, n_graded=84, n_failed_to_execute=0, reliability=0.9),
    _record(TOWN, "reliable", n_bets=395, n_graded=395, n_failed_to_execute=0, reliability=1.0),
)


def _view(raw=LIVE, now=NOW) -> SelfMonitorView:
    return self_monitor_view(raw, now=now)


# --- faculty names agree with the organ ----------------------------------------------------------

@pytest.mark.parametrize("actor,verb,ref,expected", [
    ("actions.journal", "make", "journal_trigger:walkway_grade", WALKWAY),
    ("actions.journal", "make", "journal_trigger:town_episode", TOWN),
    # Allocator routes: every live object ref is an instance id, so the faculty is the base key,
    # which is what the dispatch worker asks for with an empty object_ref.
    ("thought.painting", "make", "durable_run:reverie-visual-014ea320631b5ca4aab5c6e221088688", PAINT),
    ("hub.world_pulse_read", "look", "https://academic.oup.com/nsr/article/doi/10.1093/nsr/nwag539/8770398",
     "act.hub.world_pulse_read.look"),
    ("hub.retro_leisure", "play", "retro_session:0b1d8ce8-f000-4ba0-9101-ba7d6d2f8724", "act.hub.retro_leisure.play"),
    ("aitown.session", "play", "aitown_session:13343ca1-b5f9-5143-a08e-de6c83fc2f6d", "act.aitown.session.play"),
])
def test_gate_faculty_is_the_organ_faculty(actor, verb, ref, expected) -> None:
    assert act_faculty(actor, verb, ref) == expected
    if not ref.startswith("journal_trigger:"):
        assert act_faculty(actor, verb, "") == expected


# --- the pure decision ---------------------------------------------------------------------------

def test_walkway_grade_broken_is_refused_track_record_failing() -> None:
    d = act_gate(_view(), WALKWAY, last_attempt_at=NOW - timedelta(hours=1), now=NOW, probe_interval_sec=SIX_H)
    assert (d.admit, d.probe, d.reason, d.status) == (False, False, TRACK_RECORD_FAILING_REASON, "broken")
    assert d.label().startswith(f"self_monitor:track_record_failing:{WALKWAY}")


def test_unreliable_is_refused_too() -> None:
    d = act_gate(_view(), PRIME, last_attempt_at=NOW - timedelta(minutes=5), now=NOW, probe_interval_sec=SIX_H)
    assert not d.admit and d.reason == TRACK_RECORD_FAILING_REASON


def test_one_probe_after_six_hours_none_before() -> None:
    before = act_gate(_view(), WALKWAY, last_attempt_at=NOW - timedelta(hours=5, minutes=59), now=NOW,
                      probe_interval_sec=SIX_H)
    after = act_gate(_view(), WALKWAY, last_attempt_at=NOW - timedelta(hours=6), now=NOW, probe_interval_sec=SIX_H)
    never = act_gate(_view(), WALKWAY, last_attempt_at=None, now=NOW, probe_interval_sec=SIX_H)
    assert not before.admit
    assert after.admit and after.probe and after.label().startswith("self_monitor:probe:")
    assert never.admit and never.probe


def test_healed_faculty_is_admitted_normally() -> None:
    healed = _state(_record(WALKWAY, "reliable", n_failed_to_execute=0, reliability=1.0))
    d = act_gate(_view(healed), WALKWAY, last_attempt_at=NOW, now=NOW, probe_interval_sec=SIX_H)
    assert (d.admit, d.probe, d.reason) == (True, False, "not_failing")


@pytest.mark.parametrize("raw,now,reason", [
    (None, NOW, "absent"),
    ("{not json", NOW, "unparseable"),
    (LIVE, NOW + timedelta(hours=3, seconds=1), "stale"),
    (_state(_record(WALKWAY, "broken"), computed_at=NOW + timedelta(hours=1)), NOW, "future_stamped"),
])
def test_unknown_view_changes_nothing(raw, now, reason) -> None:
    d = act_gate(self_monitor_view(raw, now=now), WALKWAY, last_attempt_at=now, now=now, probe_interval_sec=SIX_H)
    assert (d.admit, d.probe, d.reason) == (True, False, f"unknown:{reason}")


@pytest.mark.parametrize("status", ["miscalibrated", "unknown", "reliable"])
def test_only_broken_and_unreliable_refuse(status) -> None:
    raw = _state(_record(WALKWAY, status))
    assert act_gate(_view(raw), WALKWAY, last_attempt_at=NOW, now=NOW).admit


def test_untracked_faculty_and_disabled_change_nothing() -> None:
    assert act_gate(_view(), "act.nobody.make", last_attempt_at=NOW, now=NOW).admit
    d = act_gate(_view(), WALKWAY, last_attempt_at=NOW, now=NOW, enabled=False)
    assert (d.admit, d.reason) == (True, "disabled")


# --- ledger-derived probe bookkeeping ------------------------------------------------------------

def test_latest_attempt_only_counts_the_same_faculty() -> None:
    rows = [{"object_ref": "journal_trigger:town_episode", "opened_at": NOW - timedelta(minutes=1)},
            {"object_ref": "journal_trigger:walkway_grade", "opened_at": NOW - timedelta(hours=2)},
            {"object_ref": "journal_trigger:walkway_grade", "opened_at": NOW - timedelta(hours=4)}]
    assert latest_attempt_at(rows, actor="actions.journal", verb="make", faculty=WALKWAY) == NOW - timedelta(hours=2)
    assert latest_attempt_at([], actor="actions.journal", verb="make", faculty=WALKWAY) is None


def test_sql_in_both_placeholder_styles() -> None:
    assert ":actor" in last_attempt_sql() and "%(actor)s" in last_attempt_sql("pyformat")
    assert "grammar_acts" in last_attempt_sql()


class _Ledger:
    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.calls = list(rows or []), fail, 0

    def __call__(self, actor, verb, secs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("db down")
        return self.rows


def _fresh_live(now: datetime) -> str:
    """The live state as if just re-evaluated at ``now`` (the host writes hourly)."""
    data = json.loads(LIVE)
    data["computed_at"] = now.isoformat()
    return json.dumps(data)


def _gate(ledger, enabled=True) -> ActGate:
    return ActGate(enabled=enabled, probe_interval_sec=SIX_H, read_view=lambda now: _view(_fresh_live(now), now),
                   ledger_rows=ledger)


def test_act_gate_probe_then_refuse_until_window_passes() -> None:
    ledger = _Ledger()
    gate = _gate(ledger)
    gate.begin(NOW)
    first = gate.decide("actions.journal", "make", "journal_trigger:walkway_grade", now=NOW)
    assert first.probe
    # The probe's act row has not landed yet (bus lag): the memo still spends the slot.
    second = gate.decide("actions.journal", "make", "journal_trigger:walkway_grade", now=NOW + timedelta(minutes=1))
    assert not second.admit and second.reason == TRACK_RECORD_FAILING_REASON
    # Restart: a new gate sees only the ledger, where the probe's act now sits.
    ledger.rows = [{"object_ref": "journal_trigger:walkway_grade", "opened_at": NOW}]
    restarted = _gate(ledger)
    restarted.begin(NOW + timedelta(hours=1))
    assert not restarted.decide("actions.journal", "make", "journal_trigger:walkway_grade",
                                now=NOW + timedelta(hours=1)).admit
    later = NOW + timedelta(hours=6, minutes=1)
    restarted.begin(later)
    assert restarted.decide("actions.journal", "make", "journal_trigger:walkway_grade", now=later).probe


def test_act_gate_reads_no_ledger_for_a_healthy_faculty() -> None:
    ledger = _Ledger()
    gate = _gate(ledger)
    gate.begin(NOW)
    assert gate.decide("actions.journal", "make", "journal_trigger:town_episode", now=NOW).reason == "not_failing"
    assert ledger.calls == 0


def test_unreadable_ledger_fails_toward_admit_once_per_window() -> None:
    gate = _gate(_Ledger(fail=True))
    gate.begin(NOW)
    assert gate.decide("actions.journal", "make", "journal_trigger:walkway_grade", now=NOW).probe
    assert not gate.decide("actions.journal", "make", "journal_trigger:walkway_grade",
                           now=NOW + timedelta(hours=1)).admit


def test_view_read_failure_is_unknown() -> None:
    def boom(now):
        raise RuntimeError("redis down")

    gate = ActGate(enabled=True, probe_interval_sec=SIX_H, read_view=boom, ledger_rows=_Ledger())
    assert gate.begin(NOW).reason == "read_failed"
    assert gate.decide("actions.journal", "make", "journal_trigger:walkway_grade", now=NOW).reason == "unknown:read_failed"


def test_disabled_gate_never_reads_redis() -> None:
    def boom(now):
        raise AssertionError("read while disabled")

    gate = ActGate(enabled=False, probe_interval_sec=SIX_H, read_view=boom, ledger_rows=_Ledger())
    gate.begin(NOW)
    assert gate.decide("actions.journal", "make", "journal_trigger:walkway_grade", now=NOW).reason == "disabled"


# --- E2: one question per episode ----------------------------------------------------------------

def test_e2_mints_one_question_for_each_newly_failing_faculty() -> None:
    mints, seen = self_questions_for({}, _view(), now=NOW)
    assert sorted(m.faculty for m in mints) == [WALKWAY, PRIME]
    walkway = next(m for m in mints if m.faculty == WALKWAY)
    assert walkway.text == (f"My faculty {WALKWAY} didn't run 22 of its last 28 tries in the last 7 days. Why?")
    assert walkway.family == "lived" and walkway.question_id.startswith("lived.self_monitor.")
    prime = next(m for m in mints if m.faculty == PRIME)
    assert "about 2% of the time over 36 graded tries" in prime.text
    assert seen[PAINT].startswith("reliable@")


def test_e2_none_on_repeat_none_for_reliable() -> None:
    _, seen = self_questions_for({}, _view(), now=NOW)
    again, seen2 = self_questions_for(seen, _view(), now=NOW + timedelta(hours=1))
    assert again == [] and seen2 == seen
    reliable_only = _state(_record(PAINT, "reliable"), _record(TOWN, "miscalibrated"))
    assert self_questions_for({}, _view(reliable_only), now=NOW)[0] == []


def test_e2_new_episode_asks_again_with_a_new_id() -> None:
    first_since = NOW - timedelta(days=1)
    mints, seen = self_questions_for({}, _view(_state(_record(WALKWAY, "broken", since=first_since))), now=NOW)
    healed, seen = self_questions_for(seen, _view(_state(_record(WALKWAY, "reliable", since=NOW))), now=NOW)
    assert healed == []
    later = NOW + timedelta(days=2)
    again, _ = self_questions_for(seen, _view(_state(_record(WALKWAY, "broken", since=later), computed_at=later),
                                              now=later), now=later)
    assert len(again) == 1 and again[0].question_id != mints[0].question_id


def test_e2_unknown_view_mints_nothing_and_keeps_memory() -> None:
    memory = {WALKWAY: "reliable@20261101T0000Z"}
    for raw, now in ((None, NOW), (LIVE, NOW + timedelta(hours=4))):
        mints, seen = self_questions_for(memory, self_monitor_view(raw, now=now), now=now)
        assert mints == [] and seen == memory


def test_e2_episode_without_status_since_is_one_episode() -> None:
    raw = _state(_record(WALKWAY, "broken") | {"status_since": None})
    mints, seen = self_questions_for({}, _view(raw), now=NOW)
    assert len(mints) == 1
    assert self_questions_for(seen, _view(raw, NOW + timedelta(hours=2)), now=NOW + timedelta(hours=2))[0] == []


def test_question_id_fits_the_curiosity_id_rule() -> None:
    import re

    rule = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
    assert rule.match(question_id_for(WALKWAY, "broken", "20261103T1200Z"))
    long_id = question_id_for("act." + "x" * 200 + ".make", "unreliable", "20261103T1200Z")
    assert rule.match(long_id) and len(long_id) == 120
    assert rule.match(question_id_for("act.durable.journal_compose.make.world pulse/é", "broken", "20261103T1200Z"))


# --- replay (orion/self_monitor/replay.py) -------------------------------------------------------

def test_replay_spends_probes_only_on_admitted_attempts() -> None:
    from orion.self_monitor.replay import reach, replay_acts

    # The live 10-30 walkway night: first attempt 04:32, retries every ~3-7 min, success at 05:23.
    start = datetime(2026, 10, 30, 4, 32, tzinfo=timezone.utc)
    rows = [{"actor": "actions.journal", "verb": "make", "object_ref": "journal_trigger:walkway_grade",
             "opened_at": start + timedelta(minutes=4 * i), "closed_at": start + timedelta(minutes=4 * i + 1),
             "outcome_value": 0.0, "outcome_note": "raised:RuntimeError:journal_compose_failed"} for i in range(12)]
    rows[-1].update(outcome_value=1.0, outcome_note="entry_id=x mode=digest")
    rows.append({"actor": "actions.journal", "verb": "make", "object_ref": "journal_trigger:walkway_grade",
                 "opened_at": start + timedelta(days=1), "closed_at": start + timedelta(days=1, minutes=1),
                 "outcome_value": 1.0, "outcome_note": "entry_id=y mode=digest"})
    out = replay_acts(_view(_fresh_live(start), start), rows)[WALKWAY]
    assert (out["attempts"], out["would_probe"], out["would_refuse"]) == (13, 2, 11)
    # The night's eventual success is among the refused: the gate would have cost that entry.
    assert out["refused_outcomes"] == {"failed_to_execute": 10, "worked": 1}
    assert out["reach"] == "actions_journal" and reach("prime", "predict") == "not_gated"
    assert reach("thought.painting", "make") == "allocator"
