"""The sitting tracker (orion/retro/sitting.py): what counts as a person playing, and when a stretch ends."""
import json

from orion.retro.sitting import (
    END_IDLE, END_SESSION_CHANGED, END_SESSION_STOPPED, SittingTracker, parse_input_message, sittings_from_journal,
    worth_recording,
)

SID, OTHER = "a" * 36, "b" * 36


def ids():
    n = iter(range(1, 100))
    return lambda: f"s{next(n)}"


def test_a_neutral_first_message_is_not_a_hand():
    t = SittingTracker(new_id=ids())
    assert t.on_input(SID, 0, 0, 100.0) == []
    assert t.current is None


def test_repeats_of_a_held_state_do_not_count_and_a_release_does():
    t = SittingTracker(new_id=ids())
    t.on_input(SID, 0, 128, 100.0)                      # press
    for i in range(1, 40):
        t.on_input(SID, 0, 128, 100.0 + i * 0.05)       # the page repeats the held state every 50 ms
    assert t.current.mask_changes == 1 and t.current.last_input_at == 100.0
    t.on_input(SID, 0, 0, 103.0)                        # release
    assert t.current.mask_changes == 2 and t.current.last_input_at == 103.0


def test_a_focused_page_with_no_press_never_extends_a_sitting():
    """The 2026-11-15 journal: the page kept sending the same mask for 15 minutes while the player was away."""
    t = SittingTracker(gap_sec=300, new_id=ids())
    t.on_input(SID, 0, 128, 0.0)
    t.on_input(SID, 0, 0, 1.0)
    closed = []
    for sec in range(2, 900):
        closed += t.on_input(SID, 0, 0, float(sec))     # repeats only
        closed += t.tick(float(sec))
    assert [s.end_reason for s in closed] == [END_IDLE]
    assert closed[0].ended_at == 1.0 and closed[0].duration_sec == 1.0
    assert t.current is None


def test_a_resting_stick_is_not_a_hand_and_a_pushed_stick_is():
    t = SittingTracker(new_id=ids())
    for i in range(50):                                  # noise around the centre, no button
        t.on_input(SID, 0, 0, float(i), axes=[300 - i * 7, -150 + i, 12, 0])
    assert t.current is None
    t.on_input(SID, 0, 0, 60.0, axes=[30000, 0, 0, 0])   # stick pushed right
    assert t.current is not None and t.current.mask_changes == 1
    t.on_input(SID, 0, 0, 61.0, axes=[31000, 100, 0, 0])  # still pushed right: no new change
    assert t.current.mask_changes == 1
    t.on_input(SID, 0, 0, 62.0, axes=[0, 0, 0, 0])       # let go
    assert t.current.mask_changes == 2


def test_the_whole_life_of_two_sittings():
    t = SittingTracker(gap_sec=300, new_id=ids())
    assert t.on_input(SID, 0, 16, 1000.0) == []
    t.on_input(SID, 0, 0, 1200.0)
    assert t.tick(1499.0) == []                         # 299 s after the last change: still open
    first = t.tick(1501.0)
    assert [s.sitting_id for s in first] == ["s1"] and first[0].ended_at == 1200.0
    assert not first[0].open and first[0].duration_sec == 200.0
    assert t.on_input(SID, 0, 16, 2000.0) == []         # a new stretch opens a new sitting
    assert t.current.sitting_id == "s2" and t.current.started_at == 2000.0


def test_an_input_after_the_gap_closes_the_old_sitting_and_opens_a_new_one():
    t = SittingTracker(gap_sec=300, new_id=ids())
    t.on_input(SID, 0, 16, 0.0)
    closed = t.on_input(SID, 0, 0, 400.0)               # nobody ticked in between
    assert [s.sitting_id for s in closed] == ["s1"] and closed[0].ended_at == 0.0
    assert t.current.sitting_id == "s2"


def test_another_session_ends_the_sitting():
    t = SittingTracker(new_id=ids())
    t.on_input(SID, 0, 16, 0.0)
    closed = t.on_input(OTHER, 0, 16, 10.0)
    assert closed[0].end_reason == END_SESSION_CHANGED and closed[0].retro_session_id == SID
    assert t.current.retro_session_id == OTHER


def test_a_stopped_session_ends_the_sitting_at_its_last_input():
    t = SittingTracker(new_id=ids())
    t.on_input(SID, 0, 16, 0.0)
    t.on_input(SID, 0, 0, 50.0)
    assert t.end_session(OTHER) == []
    closed = t.end_session(SID)
    assert closed[0].end_reason == END_SESSION_STOPPED and closed[0].ended_at == 50.0
    # the port's mask is forgotten with the session: a neutral message after a restart opens nothing
    assert t.on_input(SID, 0, 0, 60.0) == [] and t.current is None


def test_actions_count_only_on_the_open_sitting_of_that_session():
    t = SittingTracker(new_id=ids())
    t.on_action(SID, "save")                            # no sitting: nothing to count on
    t.on_input(SID, 0, 16, 0.0)
    for kind in ("save", "save", "load", "pause", "stop", "plan"):
        t.on_action(SID, kind)
    t.on_action(OTHER, "save")
    assert (t.current.saves, t.current.loads, t.current.pauses) == (2, 1, 1)


def test_the_player_is_only_what_the_page_was_told():
    t = SittingTracker(new_id=ids())
    assert t.state_player("juniper") is False           # no sitting
    t.on_input(SID, 0, 16, 0.0)
    assert t.current.player_stated is None
    assert t.state_player("someone from the camera") is False and t.current.player_stated is None
    assert t.state_player("guest") is True and t.current.player_stated == "guest"


def test_short_sittings_are_not_worth_a_record():
    t = SittingTracker(new_id=ids())
    t.on_input(SID, 0, 16, 0.0)
    t.on_input(SID, 0, 0, 119.0)
    short = t.end_session(SID)[0]
    assert not worth_recording(short, 120) and worth_recording(short, 119)


def test_parse_input_message():
    ok = json.dumps({"version": 1, "session_id": SID, "generation": "g", "lease": "l", "epoch": 1, "seq": 2, "port": 0, "mask": 128})
    assert parse_input_message(ok) == (0, 128, None)
    n64 = json.dumps({"version": 1, "port": 1, "mask": 0, "axes": [-32767, 9000, 120, -7999]})
    assert parse_input_message(n64) == (1, 0, [-32767, 9000, 120, -7999])    # raw: the tracker makes them coarse
    assert parse_input_message(json.dumps({"port": 0, "mask": 0, "axes": [1, 2]})) == (0, 0, [1, 2])
    for bad in ("", "{", "[]", json.dumps({"port": 0}), json.dumps({"port": 2, "mask": 1}),
                json.dumps({"port": 0, "mask": -1}), json.dumps({"port": 0, "mask": True}),
                json.dumps({"port": "0", "mask": 1}), json.dumps({"ack": 3})):
        assert parse_input_message(bad) is None


def _live(tracker, now, mask=0, axes=None, port=0):
    """One browser message through the live path: the parser, then the tracker."""
    body = {"version": 1, "session_id": SID, "port": port, "mask": mask}
    if axes is not None:
        body["axes"] = axes
    return tracker.on_input(SID, *parse_input_message(json.dumps(body))[:2], now, parse_input_message(json.dumps(body))[2])


def test_a_stick_push_counts_through_the_live_path():
    """Review finding: the parser and the tracker both made the axes coarse, so a full push (32767)
    became 1, then 0. A stick alone never opened a sitting, and a sitting played on the stick closed
    as idle after 5 minutes of play."""
    t = SittingTracker(gap_sec=300)
    _live(t, 0.0, axes=[120, -300, 0, 0])                       # a resting stick's noise is not a hand
    assert t.current is None
    _live(t, 1.0, axes=[32767, 0, 0, 0])                        # a full push right, no button
    assert t.current is not None and t.current.mask_changes == 1
    for i in range(1, 61):                                      # ten minutes on the stick, A held throughout
        _live(t, 1.0 + i * 10, mask=256, axes=[32767 if i % 2 else -32767, 0, 0, 0])
    assert t.current.mask_changes == 61 and t.tick(601.0 + 299) == []
    assert t.current.duration_sec == 600.0
    _live(t, 620.0, mask=256, axes=[-20000, 0, 0, 0])           # the same side, a little less: no change
    assert t.current.mask_changes == 61


def test_a_dropped_socket_leaves_no_button_state_behind():
    """Review finding: a socket that dropped while a button was held left that mask in the tracker,
    so the neutral message of a reconnect hours later read as a release and opened a sitting."""
    t = SittingTracker(gap_sec=300)
    _live(t, 0.0, mask=128)
    t.forget_masks(SID)                                         # the input socket closed
    assert [s.end_reason for s in t.tick(301.0)] == [END_IDLE]
    _live(t, 7200.0, mask=0)                                    # the page reconnects and announces itself
    assert t.current is None
    _live(t, 7201.0, mask=128)                                  # a real press still opens a sitting
    assert t.current is not None and t.current.started_at == 7201.0
    other = SittingTracker(gap_sec=300)                         # the session of another socket is untouched
    other.on_input("b" * 36, 0, 128, 0.0); other.forget_masks(SID)
    assert other.on_input("b" * 36, 0, 128, 1.0) == [] and other.current.mask_changes == 1


def _row(kind, sec, **extra):
    return json.dumps({"kind": kind, "session_id": SID, "utc": f"2026-11-15T20:{sec // 60:02d}:{sec % 60:02d}+00:00", **extra})


def test_replay_of_a_journal_counts_mask_changes_and_only_rows_the_journal_calls_human():
    lines = [json.dumps({"kind": "start", "session_id": SID, "game_id": "a-link-to-the-past", "utc": "2026-11-15T20:00:00+00:00"})]
    lines += [_row("applied", 1, port=0, mask=16)]                           # before attribution: no controller field
    lines += [_row("applied", 10, port=0, mask=128, controller="human")]     # press
    lines += [_row("applied", 10 + i, port=0, mask=128, controller="human") for i in range(1, 30)]   # repeats
    lines += [_row("applied", 40, port=0, mask=0, controller="human")]       # release
    lines += [_row("save", 45), _row("load", 46), _row("transition", 47, state="paused", reason="pause"),
              _row("transition", 48, state="paused", reason="disconnect")]
    lines += [_row("applied", 200, port=0, mask=1, controller="script")]     # Orion's own input is not a person
    lines += [_row("applied", 300, port=0, mask=16, controller="human"), _row("applied", 330, port=0, mask=0, controller="human")]
    lines += [_row("applied", 800, port=0, mask=16, controller="human"), "not json", _row("applied", 801, port=0, mask=0, controller="human")]
    got = list(sittings_from_journal(lines, gap_sec=300, new_id=ids()))
    assert [(s.sitting_id, s.mask_changes, round(s.duration_sec), s.end_reason) for s in got] == [
        ("s1", 4, 320, END_IDLE), ("s2", 2, 1, END_IDLE)]
    assert (got[0].saves, got[0].loads, got[0].pauses) == (1, 1, 1)
    assert all(s.backfilled and s.game_id == "a-link-to-the-past" and not s.open for s in got)


def test_replay_ends_a_sitting_when_the_session_stops():
    lines = [_row("applied", 10, port=0, mask=16, controller="human"), _row("applied", 20, port=0, mask=0, controller="human"),
             _row("transition", 25, state="stopped", reason="operator_or_service_stop")]
    got = list(sittings_from_journal(lines, new_id=ids()))
    assert [(s.end_reason, round(s.duration_sec)) for s in got] == [(END_SESSION_STOPPED, 10)]
