"""world:play (game company patch 4): someone plays on Orion, as an attention candidate.

Acceptance (spec 2026-11-16, "Acceptance checks: Patch 4"): world:play appears in the attention broadcast during a
watched sitting, is not eligible when Orion is not watching, is absent with a reason when the table cannot be read,
and reads zero when nobody plays. It yields to Juniper's chat; nothing makes reading yield to it.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.attention import world_first as wf
from orion.attention.frame_rollup import classify
from orion.schemas.attention_candidate import WORLD_PLAY_SOURCE_ID, is_world_source_id

NOW = datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]


def week():
    """A week of play like the live one: one hour-long sitting a day (backfilled history)."""
    return [wf.PlaySitting(f"b{d}", NOW - timedelta(days=d, hours=3), NOW - timedelta(days=d, hours=2), backfilled=True,
                           ended_at=NOW - timedelta(days=d, hours=2) + timedelta(minutes=5)) for d in range(1, 7)]


def live(minutes=25.0, *, idle_sec=20.0, player=None, ended=False):
    return wf.PlaySitting("S1", NOW - timedelta(minutes=minutes), NOW - timedelta(seconds=idle_sec),
                          ended_at=NOW if ended else None, game_title="A Link to the Past",
                          game_id="a-link-to-the-past", player_stated=player)


WATCHING = wf.PlayWatch(mode="on", newest_note_at=NOW - timedelta(seconds=40),
                        newest_note="Link walks into a cave by the well.")


def test_reads_zero_when_nobody_plays_and_that_rest_is_quiet_not_absent():
    c = wf.play_candidate(week(), now=NOW)
    v = wf.judge_candidate(c)
    assert c.unusualness.value == 0.0 and not c.absent and c.not_eligible_reason is None
    assert c.label == wf.PLAY_QUIET_LABEL and v.band == "quiet" and not v.eligible
    assert c.evidence is None and c.value_kind == "count" and is_world_source_id(c.source_id)


def test_absent_with_a_reason_when_the_table_cannot_be_read():
    c = wf.play_candidate(None, now=NOW, absent_reason="retro_sitting unreadable (OperationalError)")
    v = wf.judge_candidate(c)
    assert c.absent and not v.eligible and v.reason == "absent: retro_sitting unreadable (OperationalError)"
    assert wf.judge_candidate(wf.play_candidate(None, now=NOW)).reason == "absent: retro_sitting unreadable"


def test_eligible_while_orion_watches_a_busy_sitting():
    c = wf.play_candidate(week() + [live()], now=NOW, watch=WATCHING)
    v = wf.judge_candidate(c)
    assert v.eligible and v.band in wf.EXTERNAL_ELIGIBLE_BANDS, v.reason
    assert c.label == "Someone is playing A Link to the Past"
    assert c.evidence["minutes_in_window"] == int(c.unusualness.value) == 25
    assert c.evidence["newest_note"] == "Link walks into a cave by the well."
    assert "retro_sitting:S1" in c.evidence_refs


@pytest.mark.parametrize("watch,reason", [
    (None, "not watching: the watch mode could not be read"),
    (wf.PlayWatch(mode="off", set_by="orion", why="I want to listen tonight",
                  newest_note_at=NOW - timedelta(seconds=40)),
     "not watching: the watch is off (set by Orion: I want to listen tonight)"),
    (wf.PlayWatch(mode="off", set_by="juniper", newest_note_at=NOW - timedelta(seconds=40)),
     "not watching: the watch is off (set by Juniper)"),
    (wf.PlayWatch(mode="on", newest_note_at=NOW - timedelta(seconds=200)),
     "not watching: no look note stored in the last 3 min"),
    (wf.PlayWatch(mode="on"), "not watching: no look note stored in the last 3 min"),
])
def test_not_eligible_when_orion_is_not_watching_but_the_reading_still_shows(watch, reason):
    c = wf.play_candidate(week() + [live()], now=NOW, watch=watch)
    v = wf.judge_candidate(c)
    assert not v.eligible and v.reason == reason
    # The reading, its percentile and band are still traced: held out, not hidden.
    assert v.score is not None and v.band in wf.EXTERNAL_ELIGIBLE_BANDS and c.unusualness.value > 0
    assert classify(f"world-first: not attended -- {v.reason}; {c.label}") == "not watching"


def test_quiet_mode_still_watches():
    quiet = wf.PlayWatch(mode="quiet", newest_note_at=NOW - timedelta(seconds=30))
    assert wf.judge_candidate(wf.play_candidate(week() + [live()], now=NOW, watch=quiet)).eligible


def test_a_closed_or_dead_sitting_is_not_watched_even_while_its_minutes_still_count():
    for s in (live(ended=True), live(idle_sec=400.0)):
        c = wf.play_candidate(week() + [s], now=NOW, watch=WATCHING)
        assert c.unusualness.value > 0
        assert wf.judge_candidate(c).reason == "not watching: no sitting is open"
    # A backfilled row is history, never open.
    hist = wf.PlaySitting("b0", NOW - timedelta(minutes=20), NOW - timedelta(seconds=5), backfilled=True)
    assert wf.open_play_sitting([hist], now=NOW) is None


def test_the_label_names_who_and_what_never_a_count():
    """The open loop's id is a hash of its label (the world:chat lesson): one sitting must keep one label."""
    early = wf.play_candidate(week() + [live(6.0)], now=NOW, watch=WATCHING)
    late = wf.play_candidate(week() + [live(29.0)], now=NOW, watch=WATCHING)
    assert early.unusualness.value != late.unusualness.value and early.label == late.label
    assert wf.play_candidate(week() + [live(player="juniper")], now=NOW, watch=WATCHING).label.startswith("Juniper")


def test_first_week_reads_insufficient_history_but_hunger_needs_no_week():
    # The old band rule needs a week of play to stand on.
    c = wf.play_candidate([live(3.0)], now=NOW, watch=WATCHING, hunger=False)
    v = wf.judge_candidate(c)
    assert c.unusualness.band == "insufficient_history" and v.reason == "insufficient_history"
    # Hunger only needs Orion to be watching: the very first sitting can win.
    h = wf.judge_candidate(wf.play_candidate([live(3.0)], now=NOW, watch=WATCHING))
    assert h.eligible and h.band == "insufficient_history" and h.reason.startswith("watching; hunger")


def test_overlapping_sittings_count_a_minute_once():
    a = wf.PlaySitting("a", NOW - timedelta(minutes=10), NOW - timedelta(minutes=2))
    b = wf.PlaySitting("b", NOW - timedelta(minutes=6), NOW)
    assert len(wf.play_minute_times([a, b], now=NOW)) == 11


def _chat(eligible: bool):
    turns = [NOW - timedelta(days=d, minutes=m) for d in range(1, 7) for m in range(0, 60, 20)]
    if eligible:
        turns += [NOW - timedelta(minutes=m) for m in range(1, 13)]
    return wf.chat_candidate(turns, now=NOW)


def _reading(eligible: bool):
    links = [(NOW - timedelta(days=d, hours=h), "node:a", "node:b") for d in range(1, 7) for h in range(3)]
    if eligible:
        links += [(NOW - timedelta(minutes=m), f"node:x{m}", f"node:y{m}") for m in range(1, 14)]
    return wf.reading_candidate(links, now=NOW)


def test_play_yields_to_juniper_and_nothing_makes_reading_yield_to_play():
    play = wf.play_candidate(week() + [live()], now=NOW, watch=WATCHING)
    r = wf.yield_reading_to_chat(wf.rank_candidates([play, _chat(True), _reading(True)]))
    ids = [v.candidate.source_id for v in r.eligible]
    assert ids == ["world:chat"]
    assert {v.candidate.source_id: v.reason for v in r.ineligible
            if v.candidate.source_id in ("world:play", "world:reading")} == {
        "world:play": "yields to Juniper (world:chat eligible)",
        "world:reading": "yields to Juniper (world:chat eligible)"}
    r2 = wf.yield_reading_to_chat(wf.rank_candidates([play, _chat(False), _reading(True)]))
    assert {v.candidate.source_id for v in r2.eligible} == {"world:play", "world:reading"}


def test_the_broadcast_carries_play_as_a_world_signal_with_its_sitting():
    from orion.substrate import attention_broadcast as ab

    play = wf.play_candidate(week() + [live()], now=NOW, watch=WATCHING)
    signals, trace, scores = ab.world_first_signals([], magnitude_by_node_id={}, external_candidates=[play], now=NOW)
    [sig] = signals
    assert sig.evidence_refs[:2] == [WORLD_PLAY_SOURCE_ID, "retro_sitting:S1"]
    assert sig.signal_kind == "world_activity" and sig.provenance["world_evidence"]["kind"] == "retro_play"
    assert trace["winner"] == WORLD_PLAY_SOURCE_ID and WORLD_PLAY_SOURCE_ID in scores
    # Not watching: in the trace with its reason, no signal.
    held = wf.play_candidate(week() + [live()], now=NOW, watch=None)
    signals, trace, _ = ab.world_first_signals([], magnitude_by_node_id={}, external_candidates=[held], now=NOW)
    assert signals == [] and trace["candidates"][0]["reason"].startswith("not watching")


def test_the_field_contest_treats_play_as_a_channel_not_a_node():
    from orion.attention.field_attention.selectors import world_first_targets
    from orion.attention.field_attention.policy import load_attention_policy
    from orion.schemas.field_state import FieldStateV1

    policy = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
    field = FieldStateV1(generated_at=NOW, tick_id="t")
    play = wf.play_candidate(week() + [live()], now=NOW, watch=WATCHING)
    attended, _ = world_first_targets(field, policy, [play])
    [t] = attended
    assert t.target_id == WORLD_PLAY_SOURCE_ID and t.target_kind == "channel"
    assert t.pressure_score == 0.0 and t.dominant_channels == {}


def test_watch_from_preference_rows_uses_the_game_pages_lower_wins_rule():
    s = live()
    rows = [{"scope": "all", "scope_id": "", "set_by": "juniper", "mode": "on", "why": None, "set_at": NOW},
            {"scope": "game", "scope_id": "a-link-to-the-past", "set_by": "orion", "mode": "off", "why": "not tonight",
             "set_at": NOW - timedelta(hours=1)}]
    w = wf.play_watch_from_rows(s, rows, (NOW - timedelta(seconds=10), "note"))
    assert (w.mode, w.set_by, w.why, w.newest_note) == ("off", "orion", "not tonight", "note")
    assert wf.play_watch_from_rows(s, [], None).mode == "on"
    assert wf.play_watch_from_rows(None, rows, None) is None and wf.play_watch_from_rows(s, None, None) is None


def test_the_rule_in_force_is_the_one_the_replay_chose():
    assert wf.PLAY_RULE_DEFAULT == wf.PLAY_RULE_NOTE
    assert wf.PLAY_WATCHING_FRESH_SEC == __import__("orion.retro.watch", fromlist=["x"]).WATCHING_FRESH_SEC


# --- hunger (2026-11-25): play competes on time since attention last went to it ---------------------------------


def _rival(rank: float):
    """An eligible world source with a fixed rank (a reading at its own top tenth, say)."""
    from orion.schemas.attention_candidate import AttentionCandidateV1
    from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

    mag = PredictionErrorMagnitudeV1(value=1.0, percentile_now=rank, band="high", n_readings_7d=500, age_sec=1.0)
    return AttentionCandidateV1(candidate_id="c:r", source_id="world:reading", source_kind="external", label="r",
                                unusualness=mag, rank_percentile=rank, value_kind="count")


def _short_sitting(start):
    return wf.PlaySitting("S6", start, start + timedelta(minutes=6), game_title="SimEarth")


def _watch_at(t):
    return wf.PlayWatch(mode="on", newest_note_at=t - timedelta(seconds=30), newest_note="A planet cools.")


def _run_sitting(rivals_rank, ticks_sec, *, clock=None):
    """Broadcast-like ticks through a 6-minute watched sitting; returns (winner per tick, clock)."""
    clock = clock or wf.PlayHungerClock()
    start = NOW - timedelta(minutes=10)
    s = _short_sitting(start)
    winners = []
    for sec in ticks_sec:
        t = start + timedelta(seconds=sec)
        live_s = wf.PlaySitting(s.sitting_id, s.started_at, min(s.last_input_at, t), game_title=s.game_title)
        play = wf.play_candidate(week() + [live_s], now=t, watch=_watch_at(t),
                                 last_won_at=clock.last_won_at(s.sitting_id))
        cands = [play] + ([_rival(rivals_rank)] if rivals_rank is not None else [])
        r = wf.yield_reading_to_chat(wf.rank_candidates(cands))
        win = r.winner.candidate.source_id if r.winner else None
        clock.note_winner(win, wf.play_sitting_id(play), now=t)
        winners.append(win)
    return winners, clock


def test_a_six_minute_watched_sitting_wins_when_nothing_stronger_is_eligible():
    # Tonight's 8e19 sitting: broadcast ticks ~2 and ~6 minutes in (the live median gap is 218 s).
    winners, _ = _run_sitting(None, [127, 366])
    assert winners[0] == WORLD_PLAY_SOURCE_ID
    # Against a reading at its own top tenth, the pull passes it before the sitting ends.
    winners, _ = _run_sitting(0.90, [127, 366])
    assert winners == ["world:reading", WORLD_PLAY_SOURCE_ID]


def test_the_old_band_rule_never_lets_a_short_sitting_win():
    """The live failure (2026-11-25): 9 play minutes against a week holding long sittings reads 'usual'."""
    start = NOW - timedelta(minutes=10)
    long_week = [wf.PlaySitting(f"b{d}", NOW - timedelta(days=d, hours=6), NOW - timedelta(days=d, hours=1),
                                backfilled=True) for d in range(1, 7)]
    t = start + timedelta(seconds=366)
    s = wf.PlaySitting("S6", start, t, game_title="SimEarth")
    old = wf.judge_candidate(wf.play_candidate(long_week + [s], now=t, watch=_watch_at(t), hunger=False))
    new = wf.judge_candidate(wf.play_candidate(long_week + [s], now=t, watch=_watch_at(t)))
    assert not old.eligible and "not busy" in old.reason
    assert new.eligible and new.rank_score == pytest.approx(1 - math.exp(-366 / wf.PLAY_HUNGER_TAU_SEC))


def test_the_pull_resets_after_a_win_and_grows_back():
    s = live(30.0)
    t0 = NOW
    clock = wf.PlayHungerClock()
    before = wf.play_candidate(week() + [s], now=t0, watch=WATCHING, last_won_at=clock.last_won_at("S1"))
    assert before.hunger > 0.99 and before.hunger_unfed_sec == pytest.approx(1800.0)
    assert clock.note_winner(WORLD_PLAY_SOURCE_ID, wf.play_sitting_id(before), now=t0)
    after = wf.play_candidate(week() + [s], now=t0, watch=WATCHING, last_won_at=clock.last_won_at("S1"))
    assert after.hunger == 0.0 and after.hunger_unfed_sec == 0.0
    # Just after the win it loses to any eligible rival; the pull then grows back past it.
    r = wf.rank_candidates([after, _rival(0.90)])
    assert r.winner.candidate.source_id == "world:reading"
    later = t0 + timedelta(seconds=300)
    grown = wf.play_candidate(week() + [s], now=later, watch=wf.PlayWatch(mode="on", newest_note_at=later),
                              last_won_at=clock.last_won_at("S1"))
    assert grown.hunger == pytest.approx(1 - math.exp(-300 / wf.PLAY_HUNGER_TAU_SEC))
    assert wf.rank_candidates([grown, _rival(0.90)]).winner.candidate.source_id == WORLD_PLAY_SOURCE_ID


def test_a_long_sitting_cannot_take_over_from_a_steady_rival():
    # 2 h of ticks every 218 s against a reading at its top tenth: play wins about every other tick, never all.
    ticks = list(range(0, 7200, 218))
    clock = wf.PlayHungerClock()
    start = NOW - timedelta(hours=3)
    wins = 0
    for sec in ticks:
        t = start + timedelta(seconds=sec)
        s = wf.PlaySitting("L", start, t, game_title="Zelda")
        play = wf.play_candidate(week() + [s], now=t, watch=_watch_at(t), last_won_at=clock.last_won_at("L"))
        win = wf.rank_candidates([play, _rival(0.95)]).winner.candidate.source_id
        clock.note_winner(win, "L", now=t)
        wins += win == WORLD_PLAY_SOURCE_ID
    assert 0 < wins <= len(ticks) // 2 + 1


def test_chat_still_wins_however_hungry_play_is():
    play = wf.play_candidate(week() + [live(30.0)], now=NOW, watch=WATCHING)
    assert play.hunger > 0.99
    r = wf.yield_reading_to_chat(wf.rank_candidates([play, _chat(True)]))
    assert [v.candidate.source_id for v in r.eligible] == ["world:chat"]
    [p] = [v for v in r.ineligible if v.candidate.source_id == WORLD_PLAY_SOURCE_ID]
    assert p.reason == "yields to Juniper (world:chat eligible)"


@pytest.mark.parametrize("watch", [
    None,
    wf.PlayWatch(mode="off", set_by="juniper", newest_note_at=NOW - timedelta(seconds=40)),
    wf.PlayWatch(mode="on", newest_note_at=NOW - timedelta(seconds=200)),
])
def test_play_has_no_pull_and_is_not_eligible_when_not_watching(watch):
    c = wf.play_candidate(week() + [live(30.0)], now=NOW, watch=watch)
    v = wf.judge_candidate(c)
    assert c.hunger is None and c.hunger_unfed_sec is None
    assert not v.eligible and v.reason.startswith("not watching")
    # No open sitting at all: quiet, no pull.
    q = wf.play_candidate(week(), now=NOW, watch=WATCHING)
    assert q.hunger is None and not wf.judge_candidate(q).eligible


def test_other_world_sources_are_judged_exactly_as_before():
    for cand in (_chat(True), _chat(False), _reading(True), _reading(False), _rival(0.95)):
        assert cand.hunger is None and cand.hunger_unfed_sec is None
        v = wf.judge_candidate(cand)
        assert "hunger" not in v.trace() and "hunger" not in v.reason
    assert wf.judge_candidate(_chat(True)).reason.startswith("world busy for itself")
    assert wf.judge_candidate(_reading(False)).reason.endswith("not busy")


def test_hunger_off_restores_the_band_rule_exactly():
    on = wf.judge_candidate(wf.play_candidate(week() + [live()], now=NOW, watch=WATCHING))
    off = wf.judge_candidate(wf.play_candidate(week() + [live()], now=NOW, watch=WATCHING, hunger=False))
    assert off.candidate.hunger is None and off.reason.startswith("world busy for itself")
    assert on.reason.startswith("watching; hunger") and on.trace()["hunger"] == pytest.approx(on.rank_score, abs=1e-6)


def test_the_hunger_clock_is_per_sitting_and_ignores_a_win_older_than_the_sitting():
    clock = wf.PlayHungerClock(max_sittings=2)
    assert not clock.note_winner("world:reading", "S1", now=NOW)
    assert not clock.note_winner(WORLD_PLAY_SOURCE_ID, None, now=NOW)
    assert clock.note_winner(WORLD_PLAY_SOURCE_ID, "A", now=NOW)
    clock.note_winner(WORLD_PLAY_SOURCE_ID, "B", now=NOW)
    clock.note_winner(WORLD_PLAY_SOURCE_ID, "C", now=NOW)
    assert clock.last_won_at("A") is None and clock.last_won_at("C") == NOW
    s = live(10.0)
    pull, unfed = wf.play_hunger(s, NOW - timedelta(hours=1), now=NOW)
    assert unfed == pytest.approx(600.0)
    assert wf.play_sitting_id(wf.play_candidate(week() + [s], now=NOW, watch=WATCHING)) == "S1"
    assert wf.play_sitting_id(_chat(True)) is None


def test_the_hunger_drive_is_the_appetite_drive():
    from orion.physiology.appetite import intensity

    pull, unfed = wf.play_hunger(live(5.0), None, now=NOW)
    assert pull == intensity(unfed, wf.PLAY_HUNGER_TAU_SEC)
