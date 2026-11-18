"""Orion plays Zelda, section F (same day): one `play` arc per finished Retro session.

T0 is 09:00 local (America/Denver, MDT, UTC-6) on 2026-10-09. Rows are shaped exactly as
services/orion-hub/scripts/retro_memory_export.py writes retro_play_sessions and as
services/orion-durable-runs/app/temporal_self_sources.py selects them.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from orion.schemas.temporal_self import STANCE_CUE_MAX_CHARS
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import advance_clock, drain_closed_days, fold, initial_state
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.frame import build_frame
from orion.temporal_self.recent_arcs import project_recent_arcs
from orion.temporal_self.stance_cue import project_stance_cue
from orion.temporal_self.tests.fixtures import at

SOLO = ("Played The Legend of Zelda solo for 22 min, 9 new rooms (3 caves), got the wooden sword, died twice, "
        "14 of 31 predictions met.")
TOGETHER = ("Played The Legend of Zelda together for 40 min (a human drove 64% of frames), 3 new rooms, "
            "5 of 9 predictions met.")


def row(session_id: str, start, minutes: float, summary: str, **kw) -> dict:
    base = {"session_id": session_id, "lineage_id": "lin-1", "game_id": "zelda-prg0",
            "game_title": "The Legend of Zelda", "kind": "leisure", "control_basis": "leisure_sponsor",
            "predictions_unknown": 0,
            "started_at": start, "ended_at": start + timedelta(minutes=minutes), "summary": summary,
            "played_together": False, "human_share": 0.0, "decisions": 31, "predictions_met": 14,
            "rooms_new": 9, "deaths": 2, "memory_card_id": UUID("00000000-0000-4000-8000-000000000001")}
    base.update(kw)
    return base


def _state(rows, now):
    return advance_clock(fold(initial_state(), events=S.adapt("retro_play", rows)), now)


def _frame(rows, now):
    return build_frame(_state(rows, now), now)


def test_adapter_shapes_one_session_event() -> None:
    e = S.retro_play(row("sess-1", at(0), 22, SOLO))
    assert (e.event_id, e.source_kind, e.source_table, e.source_ref) == (
        "retro_play:sess-1", "retro_play", "retro_play_sessions", "sess-1")
    assert e.subject_ref == "retro_session:sess-1" and e.label == "The Legend of Zelda solo, 14 of 31 predictions met"
    assert (e.occurred_at, e.ended_at) == (at(0), at(22))
    assert e.payload["memory_card_id"] == "00000000-0000-4000-8000-000000000001"
    assert e.payload["played_together"] is False
    assert S.retro_play(row("sess-2", at(0), 5, "")) is None   # nothing to say -> no event
    assert S.retro_play(row("", at(0), 5, SOLO)) is None


def test_one_session_is_one_closed_play_arc_labelled_with_its_summary() -> None:
    state = _state([row("sess-1", at(0), 22, SOLO)], at(30))
    (arc,) = [a for a in state.arcs.values() if a.kind == "play"]
    assert (arc.status, arc.closed_reason, arc.began_at, arc.ended_at) == ("closed", "process_ended", at(0), at(22))
    assert arc.subject_label == "The Legend of Zelda solo, 14 of 31 predictions met"
    assert arc.subject_ref == "retro_session:sess-1"
    assert arc.evidence_refs == ["retro_play_sessions:sess-1"]
    # Folding the same row twice changes nothing (replay identity).
    again = advance_clock(fold(state, events=S.adapt("retro_play", [row("sess-1", at(0), 22, SOLO)])), at(30))
    assert [a.arc_id for a in again.arcs.values() if a.kind == "play"] == [arc.arc_id]


def test_label_says_who_played_from_the_control_share() -> None:
    assert S.retro_play_label(row("a", at(0), 1, TOGETHER, kind="co_play", played_together=True, decisions=9,
                                  predictions_met=5, control_basis="frames")) == "The Legend of Zelda together, 5 of 9 predictions met"
    assert S.retro_play_label(row("b", at(0), 1, SOLO, kind="co_play", control_basis="unknown")) == (
        "The Legend of Zelda with a human, 14 of 31 predictions met")
    # A leisure session a human may have taken over is never "solo".
    assert S.retro_play_label(row("d", at(0), 1, SOLO, control_basis="unknown", predictions_unknown=5)) == (
        "The Legend of Zelda with a human, 14 of 31 predictions met, 5 unknown")
    assert S.retro_play_label(row("e", at(0), 1, SOLO, decisions=None)) == "The Legend of Zelda solo"
    assert S.retro_play_label(row("f", at(0), 1, SOLO, human_may_be_in_control=True, decisions=None)) == (
        "The Legend of Zelda with a human")
    # A co-play handoff's arc names the Retro session, the same ref its memory card anchors.
    e = S.retro_play(row("sid-9:1792699320.000", at(0), 5, SOLO, retro_session_id="sid-9", kind="co_play"))
    assert e.subject_ref == "retro_session:sid-9"
    assert S.retro_play_label(row("c", at(0), 1, SOLO, decisions=0)) == "The Legend of Zelda solo, no planner decisions"


def test_stance_cue_names_todays_latest_session() -> None:
    now = at(90)
    cue = project_stance_cue(_frame([row("sess-1", at(0), 22, SOLO),
                                     row("sess-2", at(30), 40, TOGETHER, kind="co_play", played_together=True,
                                         control_basis="frames", decisions=9, predictions_met=5)], now), now)
    (arc_id,) = [i for i in cue.arc_ids]
    label = "The Legend of Zelda together, 5 of 9 predictions met"
    assert f"Played {label} (09:30 to 10:10) [arc:{arc_id}]." in cue.text
    assert "solo" not in cue.text  # only the latest
    assert len(cue.text) <= STANCE_CUE_MAX_CHARS
    assert cue.labels == {label: [arc_id]}  # shown verbatim: a match label for citation truth
    cue1 = project_stance_cue(_frame([row("sess-1", at(0), 22, SOLO)], now), now)
    assert f"Played The Legend of Zelda solo, 14 of 31 predictions met (09:00 to 09:22) [arc:{cue1.arc_ids[-1]}]." in cue1.text


def test_a_busy_day_keeps_its_latest_session_in_the_frame() -> None:
    """build_frame caps arcs_today at the most recent ARCS_IN_FRAME_CAP; a morning session must still
    be there in the evening, or the cue could not say Orion played."""
    from orion.schemas.temporal_self import ARCS_IN_FRAME_CAP
    from orion.temporal_self.tests.fixtures import ev

    later = [ev("curiosity_run", f"r{i}", at(60 + i), ended=at(60 + i, ) + timedelta(seconds=30),
                subject=f"curiosity:{i}", label="probe") for i in range(ARCS_IN_FRAME_CAP + 5)]
    now = at(60 + ARCS_IN_FRAME_CAP + 10)
    state = advance_clock(fold(initial_state(), events=S.adapt("retro_play", [row("sess-1", at(0), 22, SOLO)]) + later), now)
    f = build_frame(state, now)
    assert len(f.arcs_today) == ARCS_IN_FRAME_CAP and f.arcs_today_total == ARCS_IN_FRAME_CAP + 6
    assert [a.kind for a in f.arcs_today].count("play") == 1


def test_no_session_today_no_play_clause() -> None:
    now = at(90)
    assert "Played" not in project_stance_cue(_frame([], now), now).text


def test_recent_arcs_and_day_letter_carry_the_session() -> None:
    now = at(90)
    block = project_recent_arcs(_frame([row("sess-1", at(0), 22, SOLO)], now), now)
    assert "09:00-09:22 (22 min) played The Legend of Zelda solo, 14 of 31 predictions met [arc:" in block.text
    state = _state([row("sess-1", at(0), 22, SOLO)], at(60 * 24))  # past local midnight: the day closes
    _, days = drain_closed_days(state)
    (day,) = [d for d in days if d.day_id == "2026-10-09"]
    rendered = render_closed_day_block(day)
    assert rendered is not None and "1 play" in rendered.text


# --- a person's own sitting (game company design, patch 1): kind human_play --------------------------

def sitting_row(sitting_id: str, start, minutes: float, **kw) -> dict:
    """As services/orion-hub/scripts/retro_sitting.py play_row writes it."""
    base = {"session_id": f"sitting:{sitting_id}", "retro_session_id": "retro-1", "lineage_id": None,
            "game_id": "a-link-to-the-past", "game_title": "A Link to the Past", "kind": "human_play",
            "control_basis": "human_input", "human_may_be_in_control": True, "played_together": False,
            "human_share": None, "decisions": None, "predictions_met": 0, "predictions_unknown": None,
            "rooms_new": None, "deaths": None, "memory_card_id": None, "player_stated": None,
            "started_at": start, "ended_at": start + timedelta(minutes=minutes), "duration_sec": minutes * 60.0,
            "summary": "Someone played A Link to the Past."}
    base.update(kw)
    return base


def test_a_persons_sitting_is_a_play_arc_that_never_credits_orion() -> None:
    e = S.retro_play(sitting_row("s1", at(0), 142))
    assert e.subject_ref == "retro_sitting:s1" and e.source_ref == "sitting:s1"
    assert e.label == "Someone played A Link to the Past for 2 h 22 min"
    assert e.payload["duration_sec"] == 142 * 60.0 and e.payload["kind"] == "human_play"
    told = S.retro_play(sitting_row("s2", at(0), 40, player_stated="juniper"))
    assert told.label == "Juniper played A Link to the Past for 40 min"
    # The label does not say "Orion did not play": a short handoff to Orion can sit inside a sitting
    # (1 of 32 long sittings in the journals), and the label must never state a thing that can be false.
    assert "Orion" not in e.label and "Orion" not in told.label
    # Only the page's own word makes it Juniper: anything else stays "Someone".
    assert S.retro_play_label(sitting_row("s3", at(0), 5, player_stated="guest")).startswith("Someone played")
    assert S.retro_play_label(sitting_row("s4", at(0), 5, player_stated="a name from a camera")).startswith("Someone played")
    # duration falls back to the row's own times; an Orion session keeps its old subject and label
    assert S.retro_play_seconds(sitting_row("s5", at(0), 7, duration_sec=None)) == 420.0
    assert S.retro_play(row("sess-1", at(0), 22, SOLO)).subject_ref == "retro_session:sess-1"
    assert S.retro_play(row("sess-1", at(0), 22, SOLO)).payload["duration_sec"] == 22 * 60.0


def test_a_watched_sitting_label_counts_orions_notes_from_the_table() -> None:
    # Game company patch 2: the count of stored notes (retro_play_sessions.looks), never a model's words.
    e = S.retro_play(sitting_row("s1", at(0), 142, looks=3))
    assert e.label == "Someone played A Link to the Past for 2 h 22 min. Orion watched: 3 notes"
    assert e.payload["looks"] == 3
    assert S.retro_play_label(sitting_row("s2", at(0), 40, looks=1)).endswith("Orion watched: 1 note")
    for none in (0, None, True, "3"):                      # nothing stored, unknown, or not a count: not said
        assert "watched" not in S.retro_play_label(sitting_row("s3", at(0), 40, looks=none))


def test_no_reader_words_a_persons_sitting_as_orion_playing() -> None:
    now = at(200)
    rows = [row("sess-1", at(0), 22, SOLO), sitting_row("s1", at(30), 142, player_stated="juniper")]
    frame = _frame(rows, now)
    label = "Juniper played A Link to the Past for 2 h 22 min"
    cue = project_stance_cue(frame, now)
    assert f"On your Retro service: {label} (09:30 to 11:52) [arc:" in cue.text
    # Review finding: the later sitting must not push Orion's own play of the day out of the cue.
    assert "Played The Legend of Zelda solo, 14 of 31 predictions met (09:00 to 09:22) [arc:" in cue.text
    assert "Played Juniper" not in cue.text and "Played Someone" not in cue.text
    assert len(cue.arc_ids) == 2 and len(cue.text) <= STANCE_CUE_MAX_CHARS
    block = project_recent_arcs(frame, now)
    assert f"(2 h 22 min) {label} [arc:" in block.text and f"played {label}" not in block.text
    assert "played The Legend of Zelda solo" in block.text    # Orion's own session is still Orion's
    state = _state(rows, at(60 * 24))
    _, days = drain_closed_days(state)
    (day,) = [d for d in days if d.day_id == "2026-10-09"]
    text = render_closed_day_block(day).text
    assert "2 play" in text
    assert f"Play on the Retro service: 09:00 The Legend of Zelda solo, 14 of 31 predictions met [arc:" in text
    assert f"; 09:30 {label} [arc:" in text


def test_sitting_minutes_is_the_salience_magnitude_of_play() -> None:
    from orion.temporal_self.now import MAGNITUDE_RULES

    rule = MAGNITUDE_RULES["sitting_minutes"]
    assert rule({"payload": {"duration_sec": 8520.0}}) == 142.0
    assert rule({"payload": {"duration_sec": 0.0}}) == 0.0          # rest is a true zero
    for nothing in ({}, {"payload": None}, {"payload": {}}, {"payload": {"duration_sec": None}},
                    {"payload": {"duration_sec": "long"}}):
        assert rule(nothing) is None                                 # not recorded is not zero


def test_a_busy_day_keeps_orions_own_play_and_a_persons_sitting_both_in_the_frame() -> None:
    """Review finding: the frame pinned only the single latest play arc, so a person's sitting took
    the pin from Orion's own session (and the reverse)."""
    from orion.schemas.temporal_self import ARCS_IN_FRAME_CAP
    from orion.temporal_self.tests.fixtures import ev

    later = [ev("curiosity_run", f"r{i}", at(200 + i), ended=at(200 + i) + timedelta(seconds=30),
                subject=f"curiosity:{i}", label="probe") for i in range(ARCS_IN_FRAME_CAP + 5)]
    now = at(200 + ARCS_IN_FRAME_CAP + 10)
    plays = S.adapt("retro_play", [row("sess-1", at(0), 22, SOLO), sitting_row("s1", at(30), 142)])
    f = build_frame(advance_clock(fold(initial_state(), events=plays + later), now), now)
    assert len(f.arcs_today) == ARCS_IN_FRAME_CAP
    assert sorted(a.subject_ref for a in f.arcs_today if a.kind == "play") == ["retro_session:sess-1", "retro_sitting:s1"]


def test_a_play_row_written_after_its_end_is_ordered_by_when_it_was_written() -> None:
    """Review finding: a sitting ends at its last button press and its row is written 5 minutes later.
    Read by ended_at alone, a row that landed after the chronicle's read lag was stored late and never
    folded. The adapter now carries the write time and the reducer orders by it."""
    from orion.temporal_self.arcs import AVAILABLE_AT_KINDS, event_available_at

    assert "retro_play" in AVAILABLE_AT_KINDS
    ended = at(142)
    late = S.retro_play(sitting_row("s1", at(0), 142, created_at=ended + timedelta(seconds=312)))
    assert late.ended_at == ended and event_available_at(late) == ended + timedelta(seconds=312)
    assert late.occurred_at == at(0)                                   # the arc still spans the play itself
    plain = S.retro_play(row("sess-1", at(0), 22, SOLO))               # no write time on the row: its end
    assert event_available_at(plain) == at(22)
    # a watermark that passed the sitting's end before the row existed no longer hides it
    state = fold(initial_state(), events=[S.retro_play(row("sess-9", ended + timedelta(seconds=60), 1, SOLO))])
    state = fold(state, events=[late])                                  # read one step later, as the chronicle would
    assert state.skipped_today == 0
    now = ended + timedelta(seconds=600)
    frame = build_frame(advance_clock(state, now), now)
    assert {"retro_sitting:s1", "retro_session:sess-9"} <= {a.subject_ref for a in frame.arcs_today if a.kind == "play"}
    # the same row with no write time is behind the watermark: counted, never folded (the bug, kept as a contrast)
    blind = S.retro_play(sitting_row("s2", at(0), 142))
    assert fold(fold(initial_state(), events=[S.retro_play(row("sess-9", ended + timedelta(seconds=60), 1, SOLO))]),
                events=[blind]).skipped_today == 1


def test_the_day_block_play_line_has_a_cap() -> None:
    """Review finding: the play line listed every play arc of the day, with no item cap. On the real
    closed day 2026-11-15 its four sittings made a 403-character line that pushed the conversation
    threads line out of the block. Now: the longest two, in time order, after the conversation lines."""
    from orion.temporal_self.day_block import PLAY_LINE_ITEMS

    minutes = [10, 40, 142, 15]
    rows = [sitting_row(f"s{i}", at(i * 150), m) for i, m in enumerate(minutes)]
    _, days = drain_closed_days(_state(rows, at(60 * 24)))
    (day,) = [d for d in days if d.day_id == "2026-10-09"]
    block = render_closed_day_block(day)
    (line,) = [l for l in block.text.split("\n") if l.startswith("Play on the Retro service")]
    assert PLAY_LINE_ITEMS == 2 and line.startswith("Play on the Retro service (4 today, the 2 longest): ")
    assert line.count("Someone played") == 2 and line.index("for 40 min") < line.index("for 2 h 22 min")
    assert "for 10 min" not in line and "for 15 min" not in line and len(line) < 220
    # among lines of its own priority it is the one the cap drops first, so it cannot evict a conversation line
    tight = render_closed_day_block(day, max_chars=len(block.text) - 1).text
    assert "Play on the Retro service" not in tight and "left out to fit" in tight
