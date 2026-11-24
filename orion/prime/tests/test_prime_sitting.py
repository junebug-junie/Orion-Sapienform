"""Game company patch 4: a game sitting wakes Prime twice at most (orion/prime/sitting.py)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.prime import sitting as sw
from orion.prime.state import PrimeStateV1

T0 = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)


def sit(minutes: float, *, sid="S1", player=None, title="A Link to the Past", now=T0):
    start = now.timestamp() - minutes * 60.0
    return sw.PrimeSittingV1(sitting_id=sid, started_at=start, last_input_at=now.timestamp() - 5,
                             game_title=title, game_id="a-link-to-the-past", player_stated=player)


def named(state, sid="S1"):
    return [c for c in state.pending_changes if sw.names_sitting(c, sid)]


def test_a_sitting_under_three_minutes_wakes_nothing_and_the_open_wake_fires_once_past_it():
    s = sw.observe_sitting(PrimeStateV1(), sit(2.0), read_ok=True, now=T0)
    assert named(s) == [] and not s.sittings["S1"].opened_woke
    s = sw.observe_sitting(s, sit(3.5), read_ok=True, now=T0 + timedelta(seconds=90))
    [c] = named(s)
    assert c.startswith("a game sitting is open: Someone has played A Link to the Past for 3 min")
    assert c.endswith("(retro_sitting:S1)")
    for k in range(5):        # later ticks of the same sitting add nothing
        s = sw.observe_sitting(s, sit(5.0 + k), read_ok=True, now=T0 + timedelta(minutes=3 + k))
    assert len(named(s)) == 1


def test_the_closed_wake_fires_once_when_the_sitting_is_gone_and_only_for_one_prime_saw_open():
    s = sw.observe_sitting(PrimeStateV1(), sit(10.0, player="juniper"), read_ok=True, now=T0)
    s = s.model_copy(update={"pending_changes": []})          # the open wake was thought about
    s = sw.observe_sitting(s, None, read_ok=True, now=T0 + timedelta(minutes=6))
    [c] = named(s)
    assert c.startswith("a game sitting ended: Juniper played A Link to the Past for about 10 min")
    s = s.model_copy(update={"pending_changes": []})
    for k in range(3):
        s = sw.observe_sitting(s, None, read_ok=True, now=T0 + timedelta(minutes=7 + k))
    assert named(s) == []
    # A sitting Prime never saw past 3 minutes ends with no wake at all.
    s2 = sw.observe_sitting(PrimeStateV1(), sit(1.0, sid="S2"), read_ok=True, now=T0)
    s2 = sw.observe_sitting(s2, None, read_ok=True, now=T0 + timedelta(minutes=6))
    assert named(s2, "S2") == []


def test_another_sitting_opening_closes_the_first():
    s = sw.observe_sitting(PrimeStateV1(), sit(10.0), read_ok=True, now=T0)
    s = sw.observe_sitting(s, sit(4.0, sid="S2", title="Crystalis"), read_ok=True, now=T0 + timedelta(minutes=8))
    assert len(named(s, "S1")) == 2 and len(named(s, "S2")) == 1
    assert any(c.startswith("a game sitting ended:") for c in named(s, "S1"))


def test_an_unread_sitting_is_never_a_closed_one():
    s = sw.observe_sitting(PrimeStateV1(), sit(10.0), read_ok=True, now=T0)
    s = s.model_copy(update={"pending_changes": []})
    for k in range(4):
        s = sw.observe_sitting(s, None, read_ok=False, now=T0 + timedelta(minutes=k + 1))
    assert named(s) == [] and not s.sittings["S1"].closed_woke


def test_at_most_two_thoughts_name_one_sitting():
    s = sw.observe_sitting(PrimeStateV1(), sit(10.0), read_ok=True, now=T0)
    # A thought opens, never reaches the model, keeps its pending changes; the next try opens another thought.
    s = sw.note_sitting_thought(s)
    s = sw.observe_sitting(s, sit(11.0), read_ok=True, now=T0 + timedelta(minutes=1))
    s = sw.note_sitting_thought(s)
    assert s.sittings["S1"].thoughts == 2
    # A third try: the change naming the sitting is gone, so no third thought can name it...
    s = sw.observe_sitting(s, sit(12.0), read_ok=True, now=T0 + timedelta(minutes=2))
    assert named(s) == []
    # ...and the closed wake adds nothing once the cap is reached.
    s = sw.observe_sitting(s, None, read_ok=True, now=T0 + timedelta(minutes=9))
    assert named(s) == [] and s.sittings["S1"].closed_woke
    # A thought woken by something else does not count against the sitting.
    t = sw.observe_sitting(PrimeStateV1(), sit(1.0, sid="S3"), read_ok=True, now=T0)
    t = t.model_copy(update={"pending_changes": ["first wake: I have not thought yet"]})
    assert sw.note_sitting_thought(t).sittings["S3"].thoughts == 0


def test_the_state_saves_byte_for_byte_without_sittings_and_round_trips_with_them():
    assert "sittings" not in PrimeStateV1().model_dump_json()
    s = sw.observe_sitting(PrimeStateV1(), sit(10.0), read_ok=True, now=T0)
    back = PrimeStateV1.model_validate_json(s.model_dump_json())
    assert back.sittings["S1"].opened_woke and back.sittings["S1"].title == "A Link to the Past"


def test_the_tracked_sittings_are_bounded():
    s = PrimeStateV1()
    for i in range(sw.MAX_TRACKED + 7):
        s = sw.observe_sitting(s, sit(10.0, sid=f"S{i}"), read_ok=True, now=T0 + timedelta(minutes=i))
    assert len(s.sittings) == sw.MAX_TRACKED and f"S{sw.MAX_TRACKED + 6}" in s.sittings


def test_parse_sitting_reads_the_gateway_copy_and_refuses_junk_or_a_closed_one():
    raw = {"sitting_id": "S1", "retro_session_id": "r", "started_at": 1.0, "last_input_at": 200.0,
           "game_title": "Crystalis", "mask_changes": 9, "duration_sec": 199.0, "backfilled": False}
    assert sw.parse_sitting(raw).game_title == "Crystalis"
    assert sw.parse_sitting({**raw, "ended_at": 300.0}) is None
    assert sw.parse_sitting({"sitting_id": "S1"}) is None and sw.parse_sitting("x") is None


def test_the_frame_line_says_who_what_how_long_the_watch_and_the_newest_note():
    s = sit(34.0)
    w = {"sitting_id": "S1", "mode": "off", "set_by": "orion", "why": "I want to just listen tonight",
         "counts": {"notes_stored": 12, "model_calls": 14, "failed_looks": 1},
         "last_note": "Link walks into a cave by the well.", "last_note_at": T0.timestamp() - 120}
    line = sw.sitting_frame_line(s, w, now=T0)
    assert line.startswith("Someone is playing A Link to the Past on your Retro service, 33 min so far")
    assert "nobody said who holds the controller" in line
    assert "Your watch: off, set by you: I want to just listen tonight." in line
    assert "Looks: 12 notes stored of 14 model calls, 1 failed." in line
    assert "Newest note (2 min ago; a look, not a fact): Link walks into a cave by the well." in line
    # The watch state of another sitting is not this one's.
    assert "Your watch: unknown" in sw.sitting_frame_line(s, {**w, "sitting_id": "S9"}, now=T0)
    assert sw.sitting_frame_line(None, None, now=T0) is None
    assert "could not be read" in sw.sitting_frame_line(None, None, now=T0, read_error="redis down")
