"""Game company patch 3 (spec 2026-11-16 rev 4, sections 5 and 8): looks in the day, and who typed.

A note Orion stored while a person played becomes a ``retro_look`` event only once a line turn checked it
and it held. The looks open and refresh one play arc per sitting, which stays small however long the
sitting is; the sitting's own row closes it. A guest's chat-box message is never read as Juniper's turn.

T0 is 09:00 local (America/Denver, MDT, UTC-6) on 2026-10-09 (tests/fixtures.at).
"""

from __future__ import annotations

import re
from datetime import timedelta

from orion.schemas.temporal_self import LABEL_MAX

from orion.temporal_self import sources as S
from orion.temporal_self.arcs import LANE_RULES, advance_clock, fold, initial_state
from orion.temporal_self.tests.fixtures import at
from orion.temporal_self.tests.test_play_arcs import sitting_row


def look_row(look_id: str, settled_min: float, sitting: str = "s1", **kw) -> dict:
    """As services/orion-durable-runs/app/temporal_self_sources.py selects retro_watch_look."""
    base = {"look_id": look_id, "sitting_id": sitting, "game_id": "a-link-to-the-past",
            "looked_at": at(settled_min) - timedelta(seconds=40), "note": f"Link walks into the castle ({look_id})",
            "verdict": "checked_holds", "outcome": "stored", "settled_at": at(settled_min),
            "settled_by_line": "line-1"}
    base.update(kw)
    return base


def _fold(events, now):
    return advance_clock(fold(initial_state(), events=events), now)


def _play(state):
    return [a for a in state.arcs.values() if a.kind == "play"]


# --- the reader -----------------------------------------------------------------------------------

def test_a_checked_look_is_an_event_keyed_by_its_settle_time_and_labelled_as_a_look() -> None:
    e = S.retro_look(look_row("L1", 30))
    assert e is not None and e.source_kind == "retro_look" and e.event_id == "retro_look:L1"
    assert e.source_table == "retro_watch_look" and e.source_ref == "L1"
    assert e.occurred_at == at(30)  # settled_at, not looked_at
    assert e.subject_ref == "retro_sitting:s1"
    assert e.label == "Saw on the game screen: Link walks into the castle (L1)"
    assert e.payload["settled_by_line"] == "line-1" and e.payload["game_id"] == "a-link-to-the-past"
    assert e.payload["looked_at"] == (at(30) - timedelta(seconds=40)).isoformat()
    # no salience magnitude is carried: the payload has no duration or count to score
    assert "duration_sec" not in e.payload
    # the label cap holds for a long note, and the prefix survives the cut
    long = S.retro_look(look_row("L2", 30, note="word " * 400))
    assert len(long.label) <= LABEL_MAX and long.label.startswith("Saw on the game screen: ")


def test_an_unchecked_or_contradicted_look_never_becomes_an_event() -> None:
    assert S.retro_look(look_row("L1", 30, verdict="contradicted")) is None
    assert S.retro_look(look_row("L1", 30, verdict="unchecked")) is None
    assert S.retro_look(look_row("L1", 30, settled_at=None)) is None            # never settled
    assert S.retro_look(look_row("L1", 30, outcome="empty")) is None            # not a stored note
    assert S.retro_look(look_row("L1", 30, note="  ")) is None
    assert S.retro_look(look_row("L1", 30, sitting_id=None)) is None
    assert S.retro_look({k: v for k, v in look_row("L1", 30).items() if k != "verdict"}) is None
    assert [e.source_ref for e in S.adapt("retro_look", [
        look_row("ok", 30), look_row("no", 31, verdict="contradicted"), look_row("un", 32, verdict="unchecked")])] == ["ok"]


# --- the play arc ---------------------------------------------------------------------------------

def test_the_first_look_opens_the_sittings_play_arc_and_later_looks_refresh_it() -> None:
    state = _fold(S.adapt("retro_look", [look_row("L1", 10), look_row("L2", 20), look_row("L3", 30)]), at(31))
    (arc,) = _play(state)
    assert arc.status == "open" and arc.subject_ref == "retro_sitting:s1"
    assert arc.began_at == at(10) and arc.last_seen_at == at(30)
    assert arc.subject_label == "Saw on the game screen: Link walks into the castle (L3)"
    # first and newest look kept; the one between is counted
    assert arc.evidence_refs == ["retro_watch_look:L1", "retro_watch_look:L3"] and arc.evidence_overflow == 1
    assert len(arc.segments) == 1


def test_the_sitting_row_closes_the_arc_its_looks_opened() -> None:
    events = S.adapt("retro_look", [look_row("L1", 10), look_row("L2", 20)]) \
        + S.adapt("retro_play", [sitting_row("s1", at(0), 40, looks=2)])
    state = _fold(events, at(60))
    (arc,) = _play(state)  # one arc, not a second born-closed one
    assert arc.status == "closed" and arc.closed_reason == "process_ended"
    assert arc.subject_label == "Someone played A Link to the Past for 40 min. Orion watched: 2 notes"
    assert arc.began_at == at(0) and arc.ended_at == at(40)  # the sitting's whole interval
    assert arc.cumulative_dwell_sec == 40 * 60.0
    assert arc.evidence_refs == ["retro_play_sessions:sitting:s1", "retro_watch_look:L1", "retro_watch_look:L2"]
    # a look settled after the sitting closed adds to the counts and never reopens it
    late = fold(state, events=S.adapt("retro_look", [look_row("L9", 70)]))
    (arc,) = _play(late)
    assert arc.status == "closed" and arc.ended_at == at(40) and arc.evidence_overflow == 1
    assert arc.subject_label.startswith("Someone played")


def test_a_sitting_with_no_settled_look_and_orions_own_play_keep_todays_behaviour() -> None:
    state = _fold(S.adapt("retro_play", [sitting_row("s1", at(0), 40)]), at(60))
    (arc,) = _play(state)
    assert arc.status == "closed" and arc.evidence_refs == ["retro_play_sessions:sitting:s1"]
    # another sitting's look never touches it
    state = _fold(S.adapt("retro_look", [look_row("L1", 10, sitting="s2")])
                  + S.adapt("retro_play", [sitting_row("s1", at(0), 40)]), at(60))
    assert sorted((a.subject_ref, a.status) for a in _play(state)) == [
        ("retro_sitting:s1", "closed"), ("retro_sitting:s2", "open")]


def test_300_looks_fold_into_an_arc_of_fixed_size() -> None:
    def size(n: int) -> tuple[int, int]:
        rows = [look_row(f"L{i:04d}", i * 0.5) for i in range(n)]
        (arc,) = _play(_fold(S.adapt("retro_look", rows), at(n * 0.5 + 1)))
        # Every number written as one digit: the counters and times grow in digits (28 -> 298), nothing
        # else may grow. Labels differ only by the fixed-width look id.
        return len(re.sub(r"\d+", "0", arc.model_dump_json())), arc.evidence_overflow

    small, big = size(30), size(300)
    assert small[0] == big[0] < 2000
    assert (small[1], big[1]) == (28, 298)  # the count of looks is kept: 2 refs + the overflow


def test_lane_rule_names_the_looks() -> None:
    assert "retro_look" in LANE_RULES["play"] and "human_play" in LANE_RULES["play"]


# --- who typed (section 5) --------------------------------------------------------------------------

def chat_row(i: int, source: str, *, prompt: bool = True, minute: float = 0) -> dict:
    return {"id": f"turn-{i}", "correlation_id": f"c-{i}", "session_id": "sess-juniper", "source": source,
            "created_at": at(minute).replace(tzinfo=None), "has_prompt": prompt, "unsolicited": False}


def test_a_guests_turn_is_never_juniper_and_opens_no_conversation() -> None:
    e = S.chat_turn(chat_row(1, "retro_guest"))
    assert e.payload["speaker"] == "guest" and e.subject_ref is None
    assert e.related_refs == ["sess-juniper"]
    assert S.chat_turn(chat_row(2, "hub_orion")).payload["speaker"] == "juniper"
    # the prompt rule still reads the text when the driver gave no flag
    assert S._speaker({"source": "retro_guest", "prompt": "hi"}) == "guest"
    assert S._speaker({"source": "hub_orion", "prompt": "hi"}) == "juniper"


def test_a_flip_mid_sitting_keeps_each_rows_own_source() -> None:
    # A guest typed for 20 minutes, then tapped "Juniper": the later rows are hers, nothing is rewritten.
    rows = [chat_row(i, "retro_guest", minute=i) for i in range(3)] + \
        [chat_row(10 + i, "hub_orion", minute=20 + i) for i in range(3)]
    events = S.adapt("chat_turn", rows)
    assert [e.payload["speaker"] for e in events] == ["guest"] * 3 + ["juniper"] * 3
    state = _fold(events, at(25))
    convs = [a for a in state.arcs.values() if a.kind == "conversation"]
    (conv,) = convs
    assert conv.began_at == at(20)  # the conversation starts with her first own turn, not the guest's
    assert all(r.startswith("chat_history_log:turn-1") for r in conv.evidence_refs)
