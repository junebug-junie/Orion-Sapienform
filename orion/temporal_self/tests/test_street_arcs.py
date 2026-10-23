"""Episodes from the street (walkway follow-ups spec PR #2763, item 1): notable outside events
become one closed ``street`` arc each, labelled as the outside world, capped per day.

T0 is 09:00 local (America/Denver, MDT, UTC-6) on 2026-10-09. Rows are shaped as
services/orion-durable-runs/app/temporal_self_sources.py selects them (``origin`` names the read).
"""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

from orion.temporal_self import sources as S
from orion.temporal_self.arcs import ReducerConfig, advance_clock, fold, initial_state
from orion.temporal_self.frame import build_frame
from orion.temporal_self.recent_arcs import project_recent_arcs
from orion.temporal_self.tests.fixtures import at, ev

GATE_OPEN = {"right": 50, "wrong_other": 2}  # 52 checked, 96% right, 0 wrong names: live 10-23
GATE_CLOSED = {"right": 10}


def story(story_id: str, start, **kw) -> dict:
    base = {"origin": "motion_story", "story_id": story_id, "stream_id": "walkway", "kind": "person",
            "started_at": start, "ended_at": start + timedelta(seconds=15), "speed_class": "walking",
            "dir_lateral": 1, "dir_depth": -1, "name_label": None, "individual_id": "ind-1",
            "companion_kinds": [], "together_story_ids": [], "published": True, "clean": True,
            "rules_version": 1, "gate_rules_version": 1, "gate_counts": GATE_OPEN, "hour_rollup": None}
    base.update(kw)
    return base


def ring(event_id: str, t) -> dict:
    return {"origin": "doorbell_ring", "event_id": event_id, "stream_id": "doorbell",
            "event_type": "doorbell_ring", "created_at": t}


def handoff(event_id: str, t, kind: str = "handoff_missed", p=0.01, **kw) -> dict:
    event = {"schema_version": "vision.street.event.v1", "event_id": event_id, "kind": kind,
             "stream_ids": ["walkway", "doorbell"], "occurred_at": t.isoformat(), "subject_kind": "person",
             "salience": 0.6, "p_value": p, "evidence_refs": ["story-a"], "correlation_id": "c"}
    event.update(kw)
    return {"origin": "street_event", "event_id": event_id, "handoff_id": f"h-{event_id}", "event": event,
            "created_at": t + timedelta(seconds=60)}


def _state(rows, now, cfg: ReducerConfig = ReducerConfig(), extra=()):
    return advance_clock(fold(initial_state(), events=S.adapt("street_event", rows) + list(extra), cfg=cfg), now, cfg)


def _street(state):
    return sorted((a for a in state.arcs.values() if a.kind == "street"), key=lambda a: a.began_at)


# --------------------------------------------------------------------------- notable -> episode


def test_a_gated_story_is_one_closed_street_arc_labelled_outside() -> None:
    state = _state([story("s1", at(0))], at(5))
    (arc,) = _street(state)
    assert (arc.status, arc.closed_reason, arc.began_at, arc.ended_at) == (
        "closed", "process_ended", at(0), at(0) + timedelta(seconds=15))
    assert arc.subject_ref == "street:story:s1"
    assert arc.subject_label == "Outside (walkway camera): a person walked left to right, away from the house"
    assert arc.evidence_refs == ["vision_motion_story:s1"]  # the stored row that proves the street origin
    e = S.street_event(story("s1", at(0)))
    assert e.payload["outside_world"] is True and e.payload["origin"] == "motion_story"
    assert e.payload["stream_ids"] == ["walkway"]
    # Replay identity: folding the same row twice changes nothing.
    again = advance_clock(fold(state, events=S.adapt("street_event", [story("s1", at(0))])), at(5))
    assert [a.arc_id for a in _street(again)] == [arc.arc_id]


def test_ring_and_low_p_handoff_are_episodes() -> None:
    state = _state([ring("r1", at(1)), handoff("e1", at(2), person_guess="Teddy")], at(10))
    labels = [a.subject_label for a in _street(state)]
    assert labels == ["Outside (doorbell camera): someone rang the doorbell",
                      "Outside (walkway camera and doorbell camera): Teddy (a guess) crossing between cameras "
                      "never arrived (p 0.01)"]


def test_named_story_uses_the_name_and_doorbell_camera() -> None:
    e = S.street_event(story("s2", at(0), stream_id="doorbell", name_label="Teddy", dir_lateral=0, dir_depth=1))
    assert e.label == "Outside (doorbell camera): Teddy walked toward the house"
    assert e.payload["named"] is True and e.payload["individual_id"] == "ind-1"
    assert S.street_event(story("s3", at(0)), ).payload["individual_id"] is None  # unnamed: no id carried


# --------------------------------------------------------------------------- ordinary -> nothing


def test_ordinary_parked_unpublished_and_ungated_events_are_not_episodes() -> None:
    dropped = [
        story("p1", at(0), published=False),               # glimpse / parked / prop: not published
        story("p2", at(0), clean=False),
        story("p3", at(0), kind="vehicle"),                # traffic, even when published
        story("p4", at(0), gate_counts=GATE_CLOSED),       # spot-check gate not passed
        story("p5", at(0), rules_version=2),               # newer rules nobody has checked
        story("p6", at(0), gate_rules_version=None),
        story("p7", at(0), stream_id="cam0"),              # the room is never the street
        story("p8", at(0), stream_id=None),
        handoff("h1", at(0), p=0.2),                       # graded but not unusual
        handoff("h2", at(0), p=None),
        handoff("h3", at(0), kind="approach_door"),         # ungraded street event (live: all 5 in 72 h)
        handoff("h4", at(0), stream_ids=["cam0"]),
        {"origin": "doorbell_ring", "event_id": "m1", "stream_id": "doorbell", "event_type": "doorbell_motion",
         "created_at": at(0)},                             # Ring motion alone (110 in 30 days)
        {"origin": "something_else", "event_id": "x"},
    ]
    assert [S.street_event(r) for r in dropped] == [None] * len(dropped)
    assert _street(_state(dropped, at(5))) == []


def test_gate_constants_match_the_street_attention_gate() -> None:
    """sources.py restates the gate constants (importing street_attention pulls orion.substrate); this
    reads that module's source so the two can never drift."""
    tree = ast.parse((Path(__file__).resolve().parents[2] / "substrate" / "street_attention.py").read_text())
    values = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("GATE_P_VALUE", "RING_KIND")}
    handoff_kinds = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "HANDOFF_KINDS")
    assert values == {"GATE_P_VALUE": S.STREET_GATE_P_VALUE, "RING_KIND": S.STREET_RING_KIND}
    assert set(ast.literal_eval(handoff_kinds.value.args[0])) == set(S.STREET_HANDOFF_KINDS)


# --------------------------------------------------------------------------- people as a range


def test_people_count_is_the_honest_range_never_a_bare_count() -> None:
    together = story("s1", at(0), companion_kinds=["person", "stroller"], together_story_ids=["s0"])
    e = S.street_event(together)
    assert e.label == ("Outside (walkway camera): a person walked left to right, away from the house, "
                       "with a stroller, with other people")
    assert not any(ch.isdigit() for ch in e.label)  # no track count ("together with 1 other person")
    rollup = {"rules_version": 2, "kinds": {"person": {"passings": 7, "count_range": {
        "low": 2, "high": 7, "method_low": "most_in_one_window", "method_high": "appearance_clusters"}}}}
    e = S.street_event(dict(together, hour_rollup=rollup))
    assert e.label.endswith("; that hour, people: at least 2, at most 7 (the camera splits one person into several)")
    assert e.payload["people_range"]["low"] == 2 and e.payload["people_range"]["high"] == 7
    # A pre-v2 rollup row has no stored range: derived from its fields, still a range.
    old = {"rules_version": 1, "peak_at_once": {"person": 1}, "kinds": {"person": {"passings": 4}}}
    assert "people: at least 1, at most 4" in S.street_event(dict(together, hour_rollup=old)).label
    # A dog story says nothing about people.
    assert "people" not in S.street_event(story("d1", at(0), kind="dog", hour_rollup=rollup)).label


# --------------------------------------------------------------------------- cap


def test_daily_cap_holds_first_come_first_kept_and_resets_next_day() -> None:
    cfg = ReducerConfig(street_arcs_per_day=3)
    rows = [story(f"s{i:02d}", at(i)) for i in range(6)]
    state = _state(rows, at(30), cfg)
    assert [a.subject_ref for a in _street(state)] == ["street:story:s00", "street:story:s01", "street:story:s02"]
    # The next local day starts with a fresh allowance.
    tomorrow = at(24 * 60)
    state2 = advance_clock(fold(state, events=S.adapt("street_event", [story("t1", tomorrow)]), cfg=cfg),
                           tomorrow + timedelta(minutes=5), cfg)
    assert [a.subject_ref for a in _street(state2)] == ["street:story:t1"]
    # 0 admits none.
    assert _street(_state(rows, at(30), ReducerConfig(street_arcs_per_day=0))) == []


# --------------------------------------------------------------------------- room unchanged


def _presence(ref: str, t, to_state: str) -> dict:
    return {"transition_id": ref, "presence_id": "cam0", "from_state": None, "to_state": to_state,
            "occurred_at": t, "identity_confirmed": False, "label": None}


def test_room_company_is_unchanged_and_street_binds_no_room_context() -> None:
    room = (S.adapt("presence_transition", [_presence("p1", at(0), "present"), _presence("p2", at(20), "absent")])
            + S.adapt("vision_percept", [{"event_id": "v1", "stream_id": "cam0", "entities": ["person"],
                                          "created_at": at(5)}]))
    without = advance_clock(fold(initial_state(), events=room), at(30))
    # The story spans the room percept at at(5); the ring lands on it exactly.
    with_street = _state([story("s1", at(5) - timedelta(seconds=5)), ring("r1", at(5))], at(30), extra=room)
    strip = lambda st: {k: v for k, v in st.arcs.items() if v.kind != "street"}  # noqa: E731
    assert strip(with_street) == strip(without)  # room arcs identical, field for field
    (company,) = [a for a in with_street.arcs.values() if a.kind == "company"]
    assert company.subject_ref == "cam0"
    assert len(_street(with_street)) == 2
    for arc in _street(with_street):
        assert arc.percept_entities == [] and arc.context_event_ids == []  # the room percept is not the street's
    # A walkway presence row or walkway percept still never reaches the room lanes.
    assert S.presence_transition(dict(_presence("p3", at(0), "present"), presence_id="walkway")) is None
    assert S.vision_percept({"event_id": "v2", "stream_id": "walkway", "entities": ["person"],
                             "created_at": at(0)}) is None


def test_recent_arcs_shows_only_the_latest_street_line() -> None:
    other = ev("curiosity_run", "c1", at(0), ended=at(1), subject="curiosity:1", label="probe")
    now = at(20)
    state = _state([story(f"s{i}", at(2 + i)) for i in range(4)], now, extra=[other])
    block = project_recent_arcs(build_frame(state, now), now, limit=3)
    lines = block.text.splitlines()[1:]
    assert sum("saw Outside (walkway camera)" in ln for ln in lines) == 1
    assert any("a curiosity run" in ln for ln in lines)
