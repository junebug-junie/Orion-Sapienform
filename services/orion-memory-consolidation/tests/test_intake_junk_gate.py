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
    turns = [_turn("sup yo"), _turn(REAL[0])]
    assert _gate(turns).action == "propose"


def test_long_orion_reply_does_not_rescue_a_greeting():
    # The old rule judged prompt AND response; Orion's reply is never small
    # talk, so every greeting passed.
    long_reply = "Here is a long and genuinely substantive reply " * 10
    assert _gate([_turn("sup", response=long_reply)]).action == "skip"


@pytest.mark.parametrize("prompt", ["thanks for those updates", "what's crackalacking"])
def test_short_ack_or_wh_question_is_junk(prompt):
    assert prompt_junk_reason(prompt) == "low_info_social"


@pytest.mark.parametrize(
    "prompt",
    ["sleepy", "hey, I'm pregnant", "ok, mom died", "thanks, my labs came back fine", "Do you believe in god(s)?"],
)
def test_short_real_statement_survives_the_ack_and_question_rules(prompt):
    assert prompt_junk_reason(prompt) is None


def test_row_summary_skips_a_trailing_greeting():
    # The gate keeps ["Headed to Austin...", "sup yo"] for the Austin line; the
    # row used to be summarised by the last prompt, i.e. saved as "sup yo".
    from orion.memory.crystallization.intake_consolidation_window import _window_summary

    turns = [_turn(REAL[0]), _turn("sup yo"), _turn("Run github compactor.")]
    assert _window_summary(turns) == REAL[0]
    assert _window_summary([_turn("hi"), _turn("")]) == "hi"  # all junk: unchanged fallback


# --- Review of PR #2457: over-dropping. "Over-index on remembering." ---------

# Finding 1: non-Latin scripts and accented Latin were read as "no words".
NON_LATIN = ["мама умерла сегодня", "母が亡くなった", "אמא שלי חולה", "Mamá está enferma", "café?"]
# Finding 2: negations were stopwords, so a short bad day read as filler.
SHORT_FEELINGS = ["I'm not ok", "not good", "not great", "I'm sad", "rough day", "I can't sleep", "no"]
# Finding 3: a command followed by real content is also a memory.
COMMAND_PLUS_CONTENT = ["Do a journal pass about my labs", "run a self review on my divorce"]
# Finding 4: short questions about real things are kept.
REAL_QUESTIONS = ["when is the surgery?", "where is mom?", "who is Sarah?", "hey, which queue?"]
# Still junk after the fixes.
PURE_SOCIAL_QUESTIONS = ["you back?", "what's up?", "how are you?", "what's new?", "what else is on your mind?"]


@pytest.mark.parametrize(
    "prompt",
    NON_LATIN + SHORT_FEELINGS + COMMAND_PLUS_CONTENT + REAL_QUESTIONS
    + ["hi, my son was diagnosed today", "please, my mom is sick"],
)
def test_review_overdrop_cases_are_kept(prompt):
    assert prompt_junk_reason(prompt) is None
    # And the window proposes, even with nothing else going for it.
    assert _gate([_turn(prompt)]).action == "propose"


@pytest.mark.parametrize("prompt", NON_LATIN)
def test_non_latin_text_survives_even_alone_in_a_window_with_a_greeting(prompt):
    assert _gate([_turn("hi"), _turn(prompt)]).action == "propose"


@pytest.mark.parametrize("prompt", PURE_SOCIAL_QUESTIONS)
def test_pure_social_question_is_still_junk(prompt):
    assert prompt_junk_reason(prompt) == "low_info_social"


@pytest.mark.parametrize(
    "prompt", ["please run github compactor now", "hey orion, run github compactor please"]
)
def test_command_with_only_politeness_is_still_a_command(prompt):
    assert prompt_junk_reason(prompt) == "hub_command"


def test_command_with_content_is_not_a_command():
    for prompt in COMMAND_PLUS_CONTENT:
        assert hub_command_workflow(prompt) is None, prompt
