"""Stage 0A: greetings and Hub skill commands never become memories.

Every prompt here is a real one from the live intake (memory_crystallizations
auto-activated rows, 2026-09-01..09-29), not an invented example. Real memories
that must keep passing come from the redesign spec's Austin day.
"""

from __future__ import annotations

import pytest

from orion.memory.consolidation_gate import consolidation_memory_gate
from orion.memory.intake_junk import hub_command_workflow, prompt_junk_reason

# Real junk rows that were auto-saved as memories.
GREETINGS = [
    "sup",
    "hi",
    "ty!",
    "yo",
    "sup yo",
    "hey, which queue?",
    "howdy, how goes it",
    "Hey hey how are things going?",
    "what else is on your mind?",
]

COMMANDS = [
    ("Do a journal pass.", "journal_pass"),
    ("Run github compactor.", "github_compactor_pass"),
    ("Compact the last 24 hours of chat into a memory digest.", "chat_history_compactor_pass"),
    ("Run your dream cycle.", "dream_cycle"),
    ("hey orion, run github compactor please", "github_compactor_pass"),
]

# Real memories (spec, Austin day 2026-09-28) plus the shapes the spec says
# must survive: health and family updates. The health/family lines are
# written for this test, not copied from chat -- the repo is public.
REAL = [
    "Thanks. Headed to Austin and will fly back on Wednesday.",
    "It's a team offsite for AI/ML. I'll be meeting my peers for the first time. "
    "It's nice to meet them, but super draining for me--I'm an introvert :)",
    "yup I'll be away from home :(",
    "My labs came back and the doctor wants a follow-up next month.",
    "My sister is visiting this weekend with the kids.",
    "I've got the blues.",
    "no, that literal title is incorrect.",
    # Contains a workflow alias but is a sentence, not a command.
    "I keep wondering what have we been building, and whether the curiosity work is worth it.",
]


def _turn(prompt, response="A long, thoughtful reply from Orion about many things.", **kw):
    return {
        "prompt": prompt,
        "response": response,
        "spark_meta": {
            "turn_change_appraisal": {
                "novelty_score": kw.get("novelty", 0.1),
                "shift_kind": kw.get("shift", "NONE"),
            },
            "memory_significance_score": kw.get("significance", 0.1),
        },
    }


def _gate(turns, repair=False):
    return consolidation_memory_gate(
        turns=turns, grammar_repair_signal=repair, min_novelty=0.35, min_significance=0.40
    )


@pytest.mark.parametrize("prompt", GREETINGS)
def test_greeting_is_junk(prompt):
    assert prompt_junk_reason(prompt) == "low_info_social"


@pytest.mark.parametrize("prompt,workflow", COMMANDS)
def test_command_is_junk_and_names_the_real_workflow(prompt, workflow):
    assert hub_command_workflow(prompt) == workflow
    assert prompt_junk_reason(prompt) == "hub_command"


@pytest.mark.parametrize("prompt", REAL)
def test_real_memory_is_not_junk(prompt):
    assert prompt_junk_reason(prompt) is None


@pytest.mark.parametrize("prompt", GREETINGS + [c for c, _ in COMMANDS])
@pytest.mark.parametrize("repair", [False, True])
def test_greeting_or_command_window_produces_no_row(prompt, repair):
    # High novelty, a TOPIC shift and a repair signal: every shortcut that used
    # to admit these windows is on, and the window is still skipped.
    result = _gate([_turn(prompt, novelty=0.9, shift="TOPIC", significance=0.9)], repair=repair)
    assert result.action == "skip"


def test_mixed_junk_window_is_skipped():
    result = _gate([_turn("hi"), _turn("Run github compactor."), _turn("ty!")], repair=True)
    assert result.action == "skip"
    assert result.reasons == ["hub_command", "low_info_social"]


@pytest.mark.parametrize("prompt", REAL)
def test_real_memory_window_still_proposes(prompt):
    assert _gate([_turn(prompt)]).action == "propose"


def test_real_memory_next_to_a_greeting_still_proposes():
    turns = [_turn("hey, which queue?"), _turn(REAL[0])]
    assert _gate(turns).action == "propose"


def test_long_orion_reply_does_not_rescue_a_greeting():
    # The old rule judged prompt AND response; Orion's reply is never small
    # talk, so every greeting passed.
    long_reply = "Here is a long and genuinely substantive reply " * 10
    assert _gate([_turn("sup", response=long_reply)]).action == "skip"
