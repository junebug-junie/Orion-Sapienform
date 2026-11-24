"""world:play (game company patch 4): someone plays on Orion, as an attention candidate.

Acceptance (spec 2026-11-16, "Acceptance checks: Patch 4"): world:play appears in the attention broadcast during a
watched sitting, is not eligible when Orion is not watching, is absent with a reason when the table cannot be read,
and reads zero when nobody plays. It yields to Juniper's chat; nothing makes reading yield to it.
"""
from __future__ import annotations

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


def test_first_week_reads_insufficient_history():
    c = wf.play_candidate([live(3.0)], now=NOW, watch=WATCHING)
    v = wf.judge_candidate(c)
    assert c.unusualness.band == "insufficient_history" and v.reason == "insufficient_history"


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
