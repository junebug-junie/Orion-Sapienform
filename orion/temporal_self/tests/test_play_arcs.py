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
