"""The line's pure rules (orion/retro/commentary.py): grouping near-copies, checkable notes, the game block, reading
the CHECKS block, settling the three buckets, the error rate, and the gate."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from orion.retro import commentary as C
from orion.retro.watch import similar

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "services/orion-hub/evals/fixtures/retro_look_step0.json"


def note(i: int, at: float, text: str = "Link walks", sim: float | None = None) -> C.Note:
    return C.Note(look_id=f"look-{i}", looked_at=at, note=text, similar_to_previous=sim)


# --- near-copies -------------------------------------------------------------------------------------

def test_near_copies_group_into_one_line_with_a_count():
    notes = [note(1, 0, "a cave", None), note(2, 60, "a cave again", 0.8), note(3, 120, "a cave still", 0.5),
             note(4, 180, "the overworld", 0.1)]
    groups = C.group_notes(notes)
    assert [len(g.notes) for g in groups] == [3, 1]
    assert groups[0].shown.look_id == "look-3"          # the newest note's words are shown
    lines = C.note_lines(groups)
    assert "3 notes over 2 min" in lines[0] and "a cave still" in lines[0]


def test_a_note_under_the_cutoff_starts_its_own_group():
    groups = C.group_notes([note(1, 0, sim=None), note(2, 60, sim=C.NEAR_COPY_MIN - 0.01)])
    assert len(groups) == 2


def test_at_most_twelve_groups_newest_kept():
    groups = C.group_notes([note(i, i * 60.0, sim=0.0) for i in range(20)])
    assert len(groups) == C.MAX_NOTES == 12
    assert groups[-1].shown.look_id == "look-19" and groups[0].label == "N1"


def test_cutoff_sits_above_every_different_scene_pair_of_step0():
    """The cut-off is set from the step 0 notes (module docstring): two notes about different sequences never
    reach it, apart from the one word-for-word copy step 0 recorded."""
    if not FIXTURE.exists():
        pytest.skip("step 0 fixture missing")
    data = json.loads(FIXTURE.read_text())
    seqs = [[look["note"] for look in s["looks"] if look["model"].startswith("qwen3.8") and look["note"]
             and look["note"] != "nothing new"] for s in data["sequences"]]
    diff = sorted(similar(a, b) for s1, s2 in itertools.combinations(seqs, 2) for a in s1 for b in s2)
    assert diff[-1] == 1.0 and diff[-2] < C.NEAR_COPY_MIN       # the copy, then every real pair below the cut-off
    same = [similar(a, b) for s in seqs for a, b in itertools.combinations(s, 2)]
    assert any(v >= C.NEAR_COPY_MIN for v in same)              # it does catch some same-scene pairs


# --- checkable ----------------------------------------------------------------------------------------

def test_checkable_when_the_note_window_overlaps_the_turn_frames():
    frames = [1000.0, 1010.0, 1020.0, 1030.0]
    assert C.is_checkable(1015.0, frames)           # inside
    assert C.is_checkable(1040.0, frames)           # its window [1010, 1040] overlaps
    assert not C.is_checkable(990.0, frames)        # ended before the turn's first frame
    assert not C.is_checkable(1070.0, frames)       # began after the turn's last frame
    assert not C.is_checkable(1015.0, [])


def test_only_the_newest_note_is_usually_checkable():
    frames = [1000.0 + 10 * i for i in range(4)]      # the last 30 s
    notes = [note(i, 1030.0 - 60.0 * (11 - i), sim=0.0) for i in range(12)]
    groups = C.group_notes(notes, frames)
    assert [g.checkable for g in groups] == [False] * 11 + [True]
    assert "CHECKABLE" in C.note_lines(groups)[-1] and "CHECKABLE" not in C.note_lines(groups)[0]


# --- the prompt -----------------------------------------------------------------------------------------

def test_nobody_stated_means_no_name_and_maybe_a_child():
    text = C.build_prompt(game_title="Zelda", minutes_played=12, player_stated=None, groups=[], n_frames=4)
    assert "You do not know who holds the controller" in text and "child" in text
    assert "Juniper, as stated" not in text


def test_guest_stated_and_juniper_stated():
    guest = C.build_prompt(game_title="Zelda", minutes_played=1, player_stated="guest", groups=[], n_frames=1)
    assert "not Juniper" in guest and "Use no name" in guest
    jun = C.build_prompt(game_title="Zelda", minutes_played=1, player_stated="juniper", groups=[], n_frames=2)
    assert "Juniper, as stated on the game page" in jun


def test_the_prompt_names_the_real_frame_count_and_never_zero():
    assert "Attached: 3 frames" in C.build_prompt(game_title="Z", minutes_played=5, player_stated=None, groups=[],
                                                  n_frames=3)
    assert "Attached: 1 frame of" in C.build_prompt(game_title="Z", minutes_played=5, player_stated=None, groups=[],
                                                    n_frames=1)
    with pytest.raises(ValueError):
        C.build_prompt(game_title="Z", minutes_played=5, player_stated=None, groups=[], n_frames=0)


def test_the_game_block_holds_notes_lines_chat_and_at_most_three_now_items():
    groups = C.group_notes([note(1, 0, "a dungeon room")], [0.0])
    text = C.build_prompt(game_title="ALttP", minutes_played=30, saves=2, player_stated=None, groups=groups,
                          n_frames=4, previous_lines=["Nice jump."], recent_chat=[("Player", "hi")],
                          now_lines=["[experience:a] x", "[experience:b] y", "[experience:c] z", "[experience:d] w"])
    assert "Game: ALttP. Played for 30 minutes; 2 saves." in text
    assert "N1 [" in text and "a dungeon room" in text and "Nice jump." in text and "Player: hi" in text
    assert "[experience:c]" in text and "[experience:d]" not in text
    assert "PASS" in text and "CHECKS:" in text


# --- reading the reply ----------------------------------------------------------------------------------

def test_the_checks_block_is_cut_from_the_line_and_parsed():
    r = C.parse_reply("You found the cave behind the well.\n\nCHECKS:\nN2: holds\nN3: contradicted\nN9: maybe")
    assert r.line == "You found the cave behind the well."
    assert r.judgments == {"N2": "checked_holds", "N3": "contradicted"} and r.had_checks_block


def test_a_pass_with_checks_is_still_a_pass_and_still_judges():
    r = C.parse_reply("PASS\n**CHECKS:**\n- N1: held")
    assert r.line == "PASS" and r.judgments == {"N1": "checked_holds"}


def test_no_checks_block_leaves_everything_unchecked():
    r = C.parse_reply("Just a line.")
    assert r.line == "Just a line." and r.judgments == {} and not r.had_checks_block


def test_settle_counts_only_a_checkable_shown_note():
    frames = [1000.0, 1030.0]
    groups = C.group_notes([note(1, 900.0, sim=0.0), note(2, 1020.0, "a", sim=0.0)], frames)
    s = {x.look_id: x for x in C.settle(groups, {"N1": "contradicted", "N2": "contradicted"}, frames)}
    assert s["look-1"].verdict == "unchecked" and not s["look-1"].checkable   # judged, but not checkable: ignored
    assert s["look-2"].verdict == "contradicted" and s["look-2"].checkable
    rate = C.error_rate(s.values())
    assert rate == {"shown": 2, "checkable": 1, "contradicted": 1, "checkable_share": 0.5, "error_rate": 1.0}


def test_a_grouped_member_that_is_not_shown_stays_unchecked():
    frames = [1000.0, 1030.0]
    groups = C.group_notes([note(1, 1001.0, "x y z"), note(2, 1029.0, "x y z w", sim=0.9)], frames)
    assert len(groups) == 1
    s = {x.look_id: x for x in C.settle(groups, {"N1": "checked_holds"}, frames)}
    assert s["look-2"].verdict == "checked_holds"
    assert s["look-1"].verdict == "unchecked" and s["look-1"].checkable


def test_error_rate_with_nothing_checkable_is_none():
    assert C.error_rate([C.Settlement("a", False, "unchecked")])["error_rate"] is None


# --- the gate --------------------------------------------------------------------------------------------

BASE = dict(enabled=True, sitting_open=True, mode="on", juniper_turn_running=False, line_running=False,
            now=1000.0, last_spoke_at=800.0, newest_note_at=900.0)


@pytest.mark.parametrize("change,expected", [
    ({}, C.GATE_READY),
    ({"enabled": False}, C.GATE_DISABLED),
    ({"sitting_open": False}, C.GATE_NO_SITTING),
    ({"mode": "quiet"}, C.GATE_MODE),
    ({"mode": "off"}, C.GATE_MODE),
    ({"juniper_turn_running": True}, C.GATE_TURN),
    ({"line_running": True}, C.GATE_LINE),
    ({"hot": True}, C.GATE_HEAT),
    ({"last_spoke_at": 950.0}, C.GATE_GAP),
    ({"newest_note_at": 790.0}, C.GATE_NO_NOTE),
    ({"newest_note_at": None}, C.GATE_NO_NOTE),
])
def test_gate(change, expected):
    assert C.gate(**{**BASE, **change}) == expected


def test_the_line_never_asks_for_interactive():
    assert C.POOL_PRIORITY == "system" and C.POOL_CLASS == "chat"
