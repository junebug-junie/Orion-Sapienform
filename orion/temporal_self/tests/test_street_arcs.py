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
    # Replay identity: a fresh re-fold of the same rows (more than the cap, in another order) gives
    # the same state, so a backfill or reset keeps the same street arcs.
    cfg = ReducerConfig(street_arcs_per_day=3)
    rows = [story(f"s{i:02d}", at(i)) for i in range(6)] + [ring("r1", at(2.5))]
    assert _state(rows, at(30), cfg) == _state(list(reversed(rows)), at(30), cfg)


def test_ring_and_low_p_handoff_are_episodes() -> None:
    state = _state([ring("r1", at(1)), handoff("e1", at(2), person_guess="Robin")], at(10))
    labels = [a.subject_label for a in _street(state)]
    assert labels == ["Outside (doorbell camera): someone rang the doorbell",
                      "Outside (walkway camera and doorbell camera): Robin (a guess) crossing between cameras "
                      "never arrived (p 0.01)"]


def test_named_story_says_where_the_name_came_from() -> None:
    """The written name is the recognizer matching crops to a cluster Juniper named: a guess, said the
    same way a handoff's person guess is. Only Juniper vouching for THIS story makes it hers."""
    row = story("s2", at(0), stream_id="doorbell", name_label="Robin", dir_lateral=0, dir_depth=1)
    e = S.street_event(row)
    assert e.label == "Outside (doorbell camera): Robin (a guess) walked toward the house"
    assert (e.payload["named"], e.payload["name_source"], e.payload["individual_id"]) == (True, "recognizer", "ind-1")
    # Her spot-check verdict "right" on this story vouches for the written name.
    right = S.street_event(dict(row, story_verdict="right"))
    assert right.label == "Outside (doorbell camera): Robin (named by Juniper) walked toward the house"
    assert right.payload["name_source"] == "juniper"
    assert "(a guess)" in S.street_event(dict(row, story_verdict="wrong_name")).label
    # Her latest answer on the story wins: a name, a correction, or "someone else" with no name.
    answers = [{"action": "name", "name": "Robin"}, {"action": "unsure"}, {"action": "name", "name": "Sam"}]
    assert S.street_event(dict(row, story_answers=answers)).label.startswith(
        "Outside (doorbell camera): Sam (named by Juniper) walked")
    gone = S.street_event(dict(row, story_verdict="right", story_answers=[{"action": "someone_else", "name": None}]))
    assert gone.label.startswith("Outside (doorbell camera): a person walked")
    assert (gone.payload["named"], gone.payload["name_source"], gone.payload["individual_id"]) == (False, None, None)
    # She can name an unnamed story too.
    assert S.street_event(story("s4", at(0), story_answers=[{"action": "name", "name": "Ana"}])).label.startswith(
        "Outside (walkway camera): Ana (named by Juniper) walked")
    assert S.street_event(story("s3", at(0))).payload["individual_id"] is None  # unnamed: no id carried


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


# --------------------------------------------------------------------------- no bare people count


def test_label_never_carries_a_people_count() -> None:
    """The story sentence says "together with 1 other person" (a track count). The label is built from
    fields and says "with other people", no number; the honest range (#2781) stays with the rollups,
    because it exists only hours after the chronicle reads the story."""
    together = story("s1", at(0), companion_kinds=["person", "stroller"], together_story_ids=["s0"],
                     sentence="A person with a stroller walked left to right, together with 1 other person.")
    e = S.street_event(together)
    assert e.label == ("Outside (walkway camera): a person walked left to right, away from the house, "
                       "with a stroller, with other people")
    assert not any(ch.isdigit() for ch in e.label)
    assert "people_range" not in e.payload


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
    (visit,) = _street(with_street)  # the ring and the story are one visit
    assert sorted(visit.evidence_refs) == ["vision_events:r1", "vision_motion_story:s1"]
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


def test_a_busy_street_never_pushes_orions_own_arcs_out_of_the_frame() -> None:
    """build_frame keeps ARCS_IN_FRAME_CAP arcs. Street arcs fill leftover slots (ranked with reverie),
    so a day full of Orion's own arcs keeps all of them; only today's latest street arc is pinned."""
    from orion.schemas.temporal_self import ARCS_IN_FRAME_CAP

    own = [ev("curiosity_run", f"r{i}", at(i), ended=at(i) + timedelta(seconds=30), subject=f"curiosity:{i}",
              label="probe") for i in range(ARCS_IN_FRAME_CAP)]
    now = at(ARCS_IN_FRAME_CAP + 30)
    cfg = ReducerConfig(street_arcs_per_day=12)
    streets = [story(f"s{i:02d}", at(ARCS_IN_FRAME_CAP + i)) for i in range(12)]  # all newer than own arcs
    f = build_frame(_state(streets, now, cfg, extra=own), now, cfg)
    kinds = [a.kind for a in f.arcs_today]
    assert len(kinds) == ARCS_IN_FRAME_CAP and kinds.count("street") == 1
    assert [a.subject_ref for a in f.arcs_today if a.kind == "street"] == ["street:story:s11"]
    assert kinds.count("curiosity") == ARCS_IN_FRAME_CAP - 1  # only the oldest one gave its slot to the pin


# --------------------------------------------------------------------------- one visit (follow-up to #2876)


def test_visit_gap_is_the_handoff_loops_walk_to_the_door() -> None:
    """84 s = approach_door's prior window end plus the handoff loop's late margin (2 x the window)."""
    from orion.vision.topology import load_topology

    lo, hi = load_topology().by_id("approach_door").prior_sec
    assert ReducerConfig().street_visit_gap_sec == hi + 2 * (hi - lo)


def test_walkway_story_doorbell_story_and_ring_of_one_visit_are_one_arc() -> None:
    """Live 10-24 20:47:57-20:48:07: a doorbell story, a ring inside it, and the walkway story the
    handoff loop paired with it. Before: up to 3 slots. Now one arc, one slot, all three rows as evidence."""
    t = at(0)
    rows = [
        story("door", t, stream_id="doorbell", ended_at=t + timedelta(seconds=10), dir_lateral=0, dir_depth=1,
              created_at=t + timedelta(seconds=60), visit_story_ids=["walk"]),
        story("walk", t + timedelta(seconds=2), ended_at=t + timedelta(seconds=8),
              created_at=t + timedelta(seconds=55), visit_story_ids=["door"]),
        ring("r1", t + timedelta(seconds=9)),
    ]
    cfg = ReducerConfig(street_arcs_per_day=2)
    state = _state(rows + [story("later", at(30))], at(60), cfg)
    visit, later = _street(state)
    assert sorted(visit.evidence_refs) == ["vision_events:r1", "vision_motion_story:door", "vision_motion_story:walk"]
    assert (visit.began_at, visit.ended_at, visit.status) == (t, t + timedelta(seconds=10), "closed")
    assert visit.subject_label == (
        "Outside (doorbell camera and walkway camera): a person walked toward the house; a person walked "
        "left to right, away from the house; someone rang the doorbell")
    assert later.subject_ref == "street:story:later"  # the visit spent 1 of the 2 slots, not 3
    # Replay identity: any input order gives the same state.
    assert _state(list(reversed(rows)) + [story("later", at(30))], at(60), cfg) == state


def test_a_ring_joins_a_story_only_within_the_gap_and_two_unlinked_stories_never_merge() -> None:
    near = _state([story("s1", at(0)), ring("r1", at(0) + timedelta(seconds=15 + 80))], at(10))
    assert len(_street(near)) == 1  # ring 80 s after the story ended: the walk to the door
    far = _state([story("s1", at(0)), ring("r1", at(0) + timedelta(seconds=15 + 90))], at(10))
    assert len(_street(far)) == 2
    before = _state([ring("r1", at(0)), story("s1", at(0) + timedelta(seconds=60))], at(10))
    assert len(_street(before)) == 1  # rang, then walked away on the walkway
    # Two passers-by 10 s apart with no handoff link are two arcs, and two rings are two visits.
    assert len(_street(_state([story("s1", at(0)), story("s2", at(0) + timedelta(seconds=10))], at(10)))) == 2
    assert len(_street(_state([ring("r1", at(0)), ring("r2", at(0.5))], at(10)))) == 2
    # Measured from the ring itself, never the visit's growing span: a busy walkway cannot chain story
    # after story onto one ring. B starts 80 s after A ended but 150 s after the ring.
    t = at(0)
    chain = _state([ring("r1", t), story("a", t + timedelta(seconds=60), ended_at=t + timedelta(seconds=70)),
                    story("b", t + timedelta(seconds=150), ended_at=t + timedelta(seconds=160))], at(10))
    assert [sorted(a.evidence_refs) for a in _street(chain)] == [
        ["vision_events:r1", "vision_motion_story:a"], ["vision_motion_story:b"]]
    (visit, _) = _street(chain)
    assert len(visit.segments) == len(visit.evidence_refs) and visit.cumulative_dwell_sec == 10.0
    # Review finding: a passer-by P (written first) and a doorbell visitor V (written after the ring)
    # are both within 84 s of the ring. The ring may join one of them; P and V never share an arc.
    pv = _state([story("P", t, ended_at=t + timedelta(seconds=10), created_at=t + timedelta(seconds=30)),
                 ring("r1", t + timedelta(seconds=40)),
                 story("V", t + timedelta(seconds=30), ended_at=t + timedelta(seconds=45), stream_id="doorbell",
                       created_at=t + timedelta(seconds=100))], at(10))
    stories = [sorted(r for r in a.evidence_refs if r.startswith("vision_motion_story:")) for a in _street(pv)]
    assert sorted(stories) == [["vision_motion_story:P"], ["vision_motion_story:V"]]


def test_linked_stories_merge_at_any_distance_the_same_day_and_a_graded_handoff_joins_its_story() -> None:
    # walkway_to_door "met" with a 372 s transit (live 10-25 21:50): beyond the time gap, linked anyway.
    rows = [story("a", at(0), visit_story_ids=["b"]), story("b", at(6.2), stream_id="doorbell"),
            handoff("e1", at(7), kind="handoff_met", p=0.01, evidence_refs=["handoff:h", "story:a", "story:b"])]
    (visit,) = _street(_state(rows, at(30)))
    assert sorted(visit.evidence_refs) == [
        "vision_handoff_street_outbox:e1", "vision_motion_story:a", "vision_motion_story:b"]
    # A visit lives in one local day: a link across midnight leaves two arcs, one per day.
    night = at(14 * 60 + 59)  # 23:59 local
    s = _state([story("a", night, visit_story_ids=["b"]), story("b", night + timedelta(minutes=3))],
               night + timedelta(minutes=30))
    assert [a.subject_ref for a in _street(s)] == ["street:story:b"]
    (closed,) = s.pending_closed_days
    assert [a.subject_ref for a in closed.arcs if a.kind == "street"] == ["street:story:a"]


def test_a_merged_event_never_spends_a_slot_even_when_the_day_is_full() -> None:
    cfg = ReducerConfig(street_arcs_per_day=1)
    rows = [story("s1", at(0)), ring("r1", at(0) + timedelta(seconds=30)), story("s2", at(10))]
    (arc,) = _street(_state(rows, at(30), cfg))
    assert sorted(arc.evidence_refs) == ["vision_events:r1", "vision_motion_story:s1"]  # s2 capped out


# --------------------------------------------------------------------------- late rows fold


def test_a_story_written_after_the_read_lag_still_folds_once() -> None:
    """A story is available when it is WRITTEN (created_at), not when it ended: 1 in 54 used to land
    after the chronicle's 300 s read lag and was stored late, never folded."""
    from orion.temporal_self.arcs import event_available_at

    t = at(0)
    e = S.street_event(story("late", t, created_at=t + timedelta(minutes=20)))
    assert event_available_at(e) == t + timedelta(minutes=20)
    # Never before its end (a write-clock skew cannot pull it earlier).
    assert event_available_at(S.street_event(story("x", t, created_at=t))) == t + timedelta(seconds=15)
    # The chronicle's clock passed t + 10 min before the row existed; it still folds when read.
    cfg = ReducerConfig()
    s = advance_clock(initial_state(), t + timedelta(minutes=10), cfg)
    s = advance_clock(fold(s, events=[e], cfg=cfg), t + timedelta(minutes=25), cfg)
    (arc,) = _street(s)
    assert (arc.began_at, arc.ended_at) == (t, t + timedelta(seconds=15))  # the real times, not the write time
    # The same row read again (a re-read window) is skipped: never a second arc.
    again = advance_clock(fold(s, events=[e], cfg=cfg), t + timedelta(minutes=30), cfg)
    assert len(_street(again)) == 1 and again.skipped_today == s.skipped_today + 1
    # Other kinds ignore a payload key of that name.
    other = ev("curiosity_run", "c1", t, ended=t + timedelta(seconds=5), subject="curiosity:1",
               payload={"available_at": (t + timedelta(hours=1)).isoformat()})
    assert event_available_at(other) == t + timedelta(seconds=5)
    # A handoff's outbox row is written after its occurred_at: available when written too.
    h = S.street_event(handoff("e9", t))
    assert event_available_at(h) == t + timedelta(seconds=60)
