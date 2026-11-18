"""The look's pure rules (orion/retro/watch.py): calibration, gates, counters, notes, the switch, the page."""
from datetime import datetime, timedelta, timezone

import pytest

from orion.retro import watch as W


def hist(values):
    h = [0] * W.N_BINS
    for v in values:
        h[W.bin_of(v)] += 1
    return h


# --- calibration -------------------------------------------------------------------------------------

def test_bins_separate_the_zelda_nes_walking_values_from_each_other():
    # step 0: Zelda NES walking read 0.0032-0.0047, still up to 0.0054; the bins must be fine enough to tell
    assert W.bin_of(0.0) == 0 and W.bin_of(5e-5) == 0 and W.bin_of(1.0) == W.N_BINS - 1
    assert W.bin_of(0.0032) < W.bin_of(0.0047) < W.bin_of(0.01) < W.bin_of(0.1)
    assert W.EDGES[0] == pytest.approx(1e-4) and W.EDGES[-1] == pytest.approx(1.0)
    for k in range(1, W.N_EDGES):                                # bin k holds [EDGES[k-1], EDGES[k])
        assert W.bin_of(W.EDGES[k - 1] * 1.0001) == k


def test_a_game_with_too_little_play_gets_no_floor():
    cal = W.calibrate(hist([0.1] * 400), hist([0.0003] * 400))       # 800 checks x 2 s = 26.7 min < 30
    assert cal.floor is None and cal.status == "too_little_play" and "26 of 30 minutes" in cal.words()


def test_a_separable_game_gets_the_widest_gap_floor_and_its_counts():
    # A Link to the Past in step 0: still 0.0002-0.0004, walking 0.05-0.11
    played = [0.05 + 0.06 * i / 499 for i in range(500)]
    still = [0.0002 + 0.0002 * i / 499 for i in range(500)]
    cal = W.calibrate(hist(played), hist(still))
    assert cal.status == "set" and 0.0004 < cal.floor <= 0.05
    assert cal.noinput_above <= 0.10 and cal.input_above >= 0.90
    assert (cal.checks_input, cal.checks_no_input, cal.seconds) == (500, 500, 2000.0)
    # the widest gap is 1.0 - 0.0; ties go to the lowest edge, which looks most often
    assert cal.noinput_above == 0.0 and cal.input_above == 1.0
    assert cal.floor == min(e for e in W.EDGES if e > 0.0004 * 1.0001)


def test_zelda_nes_like_play_where_still_and_walking_overlap_gets_no_floor():
    # step 0: walking 0.0032-0.0047 inside the still range 0.0000-0.0054
    played = [0.0032 + 0.0015 * i / 499 for i in range(500)]
    still = [0.0054 * i / 499 for i in range(500)]
    cal = W.calibrate(hist(played), hist(still))
    assert cal.floor is None and cal.status == "no_separation"
    assert "no screen-change value separates" in cal.words()


def test_the_ten_and_ninety_percent_lines_are_inclusive():
    # 10% of still checks at 0.2 and 90% of played checks at 0.2: exactly on both lines still passes
    still = [0.0003] * 900 + [0.2] * 100
    played = [0.2] * 900 + [0.0003] * 100
    cal = W.calibrate(hist(played), hist(still))
    assert cal.status == "set" and cal.noinput_above == pytest.approx(0.10) and cal.input_above == pytest.approx(0.90)
    still[0] = 0.2                                               # 10.1%: no value passes any more
    assert W.calibrate(hist(played), hist(still)).floor is None


def test_one_sided_play_gets_no_floor():
    assert W.calibrate(hist([0.1] * 1000), hist([0.0003] * 10)).status == "one_sided"


# --- gates and counters --------------------------------------------------------------------------------

def test_the_gates_in_order():
    now = 1000.0
    assert W.gate(0.5, None, now=now, last_input_at=now, last_look_at=None) == W.GATE_NO_FLOOR
    assert W.gate(0.001, 0.01, now=now, last_input_at=now, last_look_at=None) == W.GATE_STATIC
    assert W.gate(0.5, 0.01, now=now, last_input_at=now - 60, last_look_at=None) == W.GATE_INPUT_STALE
    assert W.gate(0.5, 0.01, now=now, last_input_at=now - 59.9, last_look_at=now - 30) == W.GATE_TOO_SOON
    assert W.gate(0.5, 0.01, now=now, last_input_at=now - 59.9, last_look_at=now - 60) == W.GATE_READY
    assert W.gate(0.5, 0.01, now=now, last_input_at=now, last_look_at=now - 90, look_gap_sec=120) == W.GATE_TOO_SOON
    assert W.gate(0.01, 0.01, now=now, last_input_at=now, last_look_at=None) == W.GATE_READY   # at the floor = over


def test_a_paused_game_counts_static_skips_and_a_still_screen_with_presses_counts_a_likely_miss():
    c = {k: 0 for k in W.COUNTERS}
    for _ in range(300):                       # 10 minutes paused, nobody pressing
        W.count(c, W.GATE_STATIC, input_in_window=False)
    W.count(c, W.GATE_STATIC, input_in_window=True)
    W.count(c, W.GATE_INPUT_STALE, input_in_window=False)
    W.count(c, W.GATE_TOO_SOON, input_in_window=True)
    W.count(c, W.GATE_READY, input_in_window=True)
    W.count(c, W.GATE_NO_FLOOR, input_in_window=True)
    assert c["static_skips"] == 301 and c["under_floor_with_input"] == 1
    assert c["checks_over_floor"] == 3 and c["input_stale_skips"] == 1


def test_frames_spread_over_the_last_thirty_seconds_oldest_first():
    times = [float(t) for t in range(0, 62, 2)]                    # a full ring, one every 2 s
    idx = W.pick_frames(times, now=60.0)
    assert len(idx) == 4 and idx[-1] == len(times) - 1 and times[idx[0]] >= 30.0
    assert idx == sorted(idx)
    assert W.pick_frames([0.0, 2.0], now=60.0) == []               # nothing in the window
    assert W.pick_frames([40.0, 50.0], now=60.0) == [0, 1]


# --- the note ------------------------------------------------------------------------------------------

def test_outcomes_of_a_note():
    assert W.classify("", None) == "empty" and W.classify("   ", "x") == "empty" and W.classify(None, None) == "empty"
    assert W.classify("unclear", None) == "unclear" and W.classify(" Unclear. ", None) == "unclear"
    assert W.classify("Link walks north; unclear what the item is.", None) == "stored"
    assert W.classify("Link walks north.", "Link walks north.") == "copied"
    assert W.classify("Link walks north.", "Link walks  north.") == "copied"     # whitespace only
    assert W.classify("Link walks north!", "Link walks north.") == "stored"
    assert len(W.clean_note("x" * 900)) == W.NOTE_MAX_CHARS


def test_similarity_is_word_overlap():
    assert W.similar("Link walks north in a cave", "Link walks south in a cave") == pytest.approx(5 / 7, abs=1e-4)
    assert W.similar("a b", None) is None and W.similar("same words", "Same words.") == 1.0


def test_the_prompt_is_stateless_and_says_unclear_and_no_status_bar_numbers():
    text = W.prompt("The Legend of Zelda: A Link to the Past", 47.4, 4)
    assert "A Link to the Past" in text and "47 minutes" in text and "4 frames" in text
    assert "write exactly: unclear" in text and "Do not quote numbers from the status bar" in text
    assert "previous" not in text.lower() and "last note" not in text.lower()
    assert "2 frames" in W.prompt(None, 1, 2) and "a game" in W.prompt("", 1, 2)


# --- the switch -----------------------------------------------------------------------------------------

def test_the_lower_setting_wins_and_juniper_cannot_turn_orions_off_back_on():
    assert W.effective(None, None) == W.Mode("on", None)
    assert W.effective("off", "on", orion_why="I want to just keep company") == W.Mode("off", "orion", "I want to just keep company")
    assert W.effective("on", "off").set_by == "juniper"
    assert W.effective("quiet", "on").mode == "quiet"
    assert W.effective("off", "off", orion_why="mine").set_by == "orion"     # a tie names Orion's own choice
    with pytest.raises(ValueError):
        W.valid_mode("loud")


def test_a_choice_for_this_sitting_wins_over_the_games_default():
    t0 = datetime(2026, 11, 18, tzinfo=timezone.utc)
    rows = [
        {"scope": "game", "scope_id": "zelda", "mode": "off", "why": "standing", "set_at": t0},
        {"scope": "sitting", "scope_id": "s1", "mode": "on", "why": None, "set_at": t0 - timedelta(days=1)},
        {"scope": "game", "scope_id": "zelda", "mode": "quiet", "why": "newer", "set_at": t0 + timedelta(hours=1)},
    ]
    assert W.orion_choice(rows, "s1", "zelda")["mode"] == "on"
    assert W.orion_choice(rows, "s2", "zelda")["why"] == "newer"
    assert W.orion_choice(rows, "s2", "tetris") is None and W.orion_choice(rows, None, None) is None


# --- the page tells the truth ------------------------------------------------------------------------

def status(**kw):
    base = dict(enabled=True, sitting_open=True, mode=W.Mode("on", None),
                calibration=W.Calibration(0.01, "set", 500, 500, 2000.0, 0.0, 1.0), now=10_000.0,
                last_stored_at=None, last_refusal=None, last_over_floor_at=None, last_input_at=10_000.0,
                sitting_started_at=9_990.0)
    return W.status(**(base | kw))


def test_watching_only_with_a_fresh_stored_note():
    assert status(last_stored_at=10_000.0 - 179)["text"] == "Orion is watching."
    assert status(last_stored_at=10_000.0 - 181)["code"] != "watching"
    assert status()["code"] == "waiting"


def test_why_orion_is_not_watching():
    assert status(mode=W.Mode("off", "orion", "I would rather just keep company"))["text"] == \
        "Orion chose not to watch: I would rather just keep company"
    assert status(mode=W.Mode("off", "juniper"))["code"] == "off_juniper"
    assert status(calibration=None)["text"].startswith("No looks for this game yet")
    assert "no screen-change value" in status(calibration=W.Calibration(None, "no_separation", 500, 500, 2000.0))["text"]
    assert status(last_refusal=("deferred:price", 9_999.0))["text"] == "Orion is not looking: bid refused (deferred:price)."
    assert status(last_refusal=(W.HEAT_PREFIX + "pool:shed:cabinet_hot", 9_999.0))["code"] == "refused_heat"
    assert status(last_over_floor_at=10_000.0 - 360)["text"] == "Orion is looking; the screen has not changed for 6 min."
    assert status(last_over_floor_at=10_000.0, last_input_at=10_000.0 - 130)["code"] == "input_stale"
    assert status(enabled=False)["code"] == "disabled" and status(sitting_open=False)["code"] == "no_sitting"
    # an old refusal does not hide a newer note
    assert status(last_refusal=("deferred:price", 9_000.0), last_stored_at=9_950.0)["code"] == "watching"
