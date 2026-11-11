from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.consult.core import (
    budget_refusal,
    claude_turn_prompt,
    consult_id_from_room,
    consult_outcomes_section,
    consult_room_id,
    gate_ending,
    orion_session_id,
    orion_turn_prompt,
    parse_claude_turn,
    render_brief,
    self_monitor_sources,
    stuck_prior_source,
)

NOW = datetime(2026, 11, 11, 18, 0, tzinfo=timezone.utc)


def good_brief(**over) -> dict:
    brief = {
        "why_it_matters": "Journal runs stopped, so Orion's day has no record.",
        "checked": [
            {"claim": "The journal tick skips when the lease is held.", "proof": "orion/curiosity/journal.py:88"},
            {"claim": "It happened 5 times this week.", "not_checked": True},
        ],
        "disagreement": "none",
        "impact": {"what_changes": "Journal runs again.", "number_to_watch": "journal rows per day",
                   "how_we_see_it": "python -P -m orion.dig journal count"},
        "cost": {"doing_the_work": {"value": "about 2 GPU-minutes a day", "source": "orion.dig energy acts"},
                 "doing_nothing": {"unknown_reason": "no dig measures a missing journal"}},
        "recommendation": "Release the lease after each run.",
        "question_for_juniper": "Should a dev session make this change?",
    }
    brief.update(over)
    return brief


def block(payload: dict) -> str:
    return "Here is my view.\n\n```json\n" + json.dumps(payload) + "\n```"


# --- room ids --------------------------------------------------------------------------------

def test_room_id_round_trip_and_social_rooms_are_not_consults():
    assert consult_id_from_room(consult_room_id("abc")) == "abc"
    assert consult_id_from_room("hub-direct") is None
    assert consult_id_from_room("consult:") is None
    assert consult_id_from_room(None) is None
    assert orion_session_id("abc") == "consult-abc"


# --- parsing Claude's reply ------------------------------------------------------------------

def test_no_block_means_continue():
    turn = parse_claude_turn("What does the lease log say?")
    assert turn.ending == "continue" and turn.payload is None and turn.error is None
    assert turn.room_text == "What does the lease log say?"


def test_last_block_wins_and_is_stripped_from_room_text():
    text = "a\n```json\n{\"ending\": \"continue\"}\n```\nb" + "\n" + block({"ending": "answered", "answer": "x"})
    turn = parse_claude_turn(text)
    assert turn.ending == "answered" and "```" not in turn.room_text.split("b")[-1]


@pytest.mark.parametrize("raw,err", [
    ("x\n```json\n{not json}\n```", "ending_block_unparseable"),
    ("x\n```json\n{\"ending\": \"maybe\"}\n```", "unknown_ending:maybe"),
])
def test_bad_block_is_continue_with_error(raw, err):
    turn = parse_claude_turn(raw)
    assert turn.ending == "continue" and turn.error == err


# --- the brief gate --------------------------------------------------------------------------

def test_good_brief_passes():
    result = gate_ending("agreed_proposal", {"brief": good_brief()})
    assert result.ok and result.brief is not None


@pytest.mark.parametrize("mutate,part", [
    (lambda b: b.pop("why_it_matters"), "why_it_matters"),
    (lambda b: b.update(checked=[]), "checked"),
    (lambda b: b.update(checked=[{"claim": "x"}]), "checked.0"),
    (lambda b: b.update(checked=[{"claim": "x", "proof": "f:1", "not_checked": True}]), "checked.0"),
    (lambda b: b.pop("impact"), "impact"),
    (lambda b: b["cost"].update(doing_nothing={"value": "$40 a month"}), "cost.doing_nothing"),
    (lambda b: b["cost"].update(doing_the_work={"value": "unknown"}), "cost.doing_the_work"),
    (lambda b: b.update(recommendation="  "), "recommendation"),
    (lambda b: b.update(question_for_juniper="Make the change."), "question_for_juniper"),
])
def test_a_missing_part_is_refused_by_name(mutate, part):
    brief = good_brief()
    mutate(brief)
    result = gate_ending("agreed_proposal", {"brief": brief})
    assert not result.ok and result.refused_part == part


def test_an_unsourced_cost_number_is_refused():
    brief = good_brief()
    brief["cost"]["doing_the_work"] = {"value": "3 kWh a day"}
    assert gate_ending("agreed_proposal", {"brief": brief}).refused_part == "cost.doing_the_work"


def test_disagree_needs_both_positions():
    assert gate_ending("disagree", {"brief": good_brief()}).refused_part == "orion_position"
    ok = gate_ending("disagree", {"brief": good_brief(orion_position="keep it", claude_position="change it")})
    assert ok.ok


def test_answered_needs_an_answer_and_continue_is_not_final():
    assert gate_ending("answered", {"answer": "It is by design: see x.py:3"}).ok
    assert gate_ending("answered", {"answer": " "}).refused_part == "answer"
    assert gate_ending("continue", {}).refused_part == "ending"
    assert gate_ending("agreed_proposal", {}).refused_part == "brief"


# --- the budget gate -------------------------------------------------------------------------

def obs(state="clear", observed=True, age=10.0):
    return SimpleNamespace(state=state, observed=observed, observed_at=NOW - timedelta(seconds=age))


@pytest.mark.parametrize("session,weekly,reason", [
    (obs(), None, "weekly_meter_missing"),
    (obs(), obs(age=4 * 3600), "weekly_meter_stale"),
    (obs(), obs(observed=False, state="unknown"), "weekly_unobserved"),
    (obs(), obs(state="limited"), "weekly_limited"),
    (None, obs(), "session_meter_missing"),
    (obs(age=3600), obs(), "session_meter_stale"),
    (obs(state="limited"), obs(), "session_limited"),
])
def test_budget_fails_closed(session, weekly, reason):
    assert budget_refusal(session, weekly, now=NOW) == reason


def test_budget_allows_clear_and_an_idle_session_pool():
    assert budget_refusal(obs(), obs(), now=NOW) is None
    # Nobody used Claude Code for 5 hours: the session pool is fullest, and the weekly reading
    # proves the meter can see.
    assert budget_refusal(obs(state="unknown", observed=False), obs(), now=NOW) is None


# --- prompts, rendering, sources -------------------------------------------------------------

def test_last_claude_turn_must_end():
    assert "last turn" in claude_turn_prompt(orion_text="x", turn_no=6, max_turns=6, first=False, source="s", question="q")
    assert "last turn" not in claude_turn_prompt(orion_text="x", turn_no=2, max_turns=6, first=True, source="s", question="q")


def test_orion_prompt_points_at_digs_and_keeps_chats_out():
    text = orion_turn_prompt(source="s", question="q", transcript=[], turn_no=1, max_turns=6)
    assert "python -P -m orion.dig" in text and "private chats" in text
    later = orion_turn_prompt(source="s", question="q", transcript=[("Claude", "why?")], turn_no=5, max_turns=6)
    assert "Claude: why?" in later and "last turn" in later


def test_render_brief_has_every_part_in_order():
    brief = gate_ending("disagree", {"brief": good_brief(orion_position="o", claude_position="c")}).brief
    text = render_brief(brief, ending="disagree", source="From x.", consult_cost="$0.40", transcript_hint="(see chat)")
    order = ["@Juniper", "Where it came from", "Why it matters", "What they checked", "Where they disagree",
             "- Orion: o", "- Claude: c", "Impact", "Cost", "This consult: $0.40", "Recommendation", "Question for you"]
    positions = [text.index(word) for word in order]
    assert positions == sorted(positions)
    assert "Not checked." in text and "unknown -- no dig measures" in text


def test_self_monitor_sources_one_per_broken_episode():
    since = datetime(2026, 11, 10, 3, 0, tzinfo=timezone.utc)
    view = SimpleNamespace(known=True, faculties={
        "journal": SimpleNamespace(status="broken", status_since=since, n_graded=2, n_failed_to_execute=5),
        "fetch": SimpleNamespace(status="reliable", status_since=since, n_graded=9, n_failed_to_execute=0),
        "odd": SimpleNamespace(status="broken", status_since=None, n_graded=0, n_failed_to_execute=1),
    })
    out = self_monitor_sources(view)
    assert [ref for ref, _ in out] == [f"self_finding:journal:broken@{since.isoformat()}"]
    assert "5 of 7 tries never ran" in out[0][1]
    assert self_monitor_sources(SimpleNamespace(known=False, faculties={})) == []


def test_stuck_prior_source():
    assert stuck_prior_source(SimpleNamespace(subject_prior_id=None, subject_claim="x")) is None
    ref, q = stuck_prior_source(SimpleNamespace(subject_prior_id="p1", subject_claim="Claim."))
    assert ref == "prior:p1" and "Claim." in q


def test_outcomes_section_separates_juniper_answer_from_truth():
    lines = consult_outcomes_section([
        {"consult_id": "c1", "ending": "answered", "answer": "by design", "question": "why?"},
        {"consult_id": "c2", "ending": "agreed_proposal", "decision": "no_because", "decision_reason": "too costly",
         "question": "fix?"},
    ])
    text = "\n".join(lines)
    assert "not proof the claim is true" in text
    assert "c1 was answered: by design" in text and "no, because: too costly" in text
    assert consult_outcomes_section([]) == []
