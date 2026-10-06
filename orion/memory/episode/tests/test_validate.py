"""Deterministic validation of distilled episode memories (spec Stage 1 acceptance 2-7).

Turns here are synthetic. The two Austin lines reused below are already quoted verbatim in the
public spec; nothing private is in this file.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.memory.episode.validate import (
    EpisodeTurn,
    coverage,
    normalize_referent_key,
    quote_in_text,
    validate_distillation,
)
from orion.schemas.memory_episode import EpisodeDistillationV1

AUSTIN_0906 = (
    "It's a team offsite for AI/ML, so I'll be meeting my peers in person for the first time. "
    "They are nice people, but meeting new people all day is super draining for me--I'm an introvert :)"
)
assert len(AUSTIN_0906) > 160 and AUSTIN_0906.index("introvert") > 160  # the revision-1 trap

TURNS = [
    EpisodeTurn("t1", "c-1", "I'll be pretty busy the next few days with work travel.", "Got it, safe travels.",
                datetime(2026, 9, 28, 6, 59, tzinfo=timezone.utc)),
    EpisodeTurn("t2", "c-2", "Headed to Austin and will fly back on Wednesday.",
                "Austin! I hope the trip goes smoothly, and I'll be here when you're back.",
                datetime(2026, 9, 28, 8, 57, tzinfo=timezone.utc)),
    EpisodeTurn("t3", "c-3", AUSTIN_0906, "That sounds like a lot. I hope you get some quiet time between sessions.",
                datetime(2026, 9, 28, 9, 6, tzinfo=timezone.utc)),
    EpisodeTurn("t4", "c-4", "Run github compactor.", "Workflow: GitHub Compactor\nStatus: ok",
                datetime(2026, 9, 28, 9, 44, tzinfo=timezone.utc), is_command=True),
]


def _mem(**kw):
    base = {"purpose": "happened", "voice": "juniper_said", "channel": "chat", "stakes": "low", "stakes_reason": "none",
            "statement": "Juniper flew to Austin for a work offsite this week.",
            "referents": [{"key": "event:austin-ai-ml-offsite-2026-09", "aliases": ["austin", "offsite"]}],
            "evidence": [{"turn": "t2", "field": "prompt", "quote": "Headed to Austin"}]}
    base.update(kw)
    return base


def _run(*memories, questions=()):
    d = EpisodeDistillationV1.model_validate({"memories": list(memories), "questions": list(questions)})
    return validate_distillation(d, TURNS, episode_id="ep-test")


def test_quote_beyond_char_160_still_verifies():
    """Revision 1 judged fidelity against left(prompt, 160) and withdrew a false claim. The
    validator must check the FULL text."""
    r = _run(_mem(purpose="about_juniper", statement="Juniper told me she is an introvert and meeting people drains her.",
                  referents=[{"key": "person:juniper"}],
                  evidence=[{"turn": "t3", "field": "prompt", "quote": "super draining for me--I'm an introvert :)"}]))
    assert len(r.memories) == 1 and not r.rejections
    m = r.memories[0]
    assert m.voice == "juniper_said" and m.evidence[0].verified
    assert m.evidence[0].quote.endswith("introvert :)")


def test_truncated_text_would_have_failed():
    assert not quote_in_text("I'm an introvert", AUSTIN_0906[:160])
    assert quote_in_text("I'm an introvert", AUSTIN_0906)


def test_whitespace_and_wrapping_quote_marks_are_tolerated_but_words_are_not():
    assert quote_in_text('"Headed  to\nAustin"', "Headed to Austin and will fly back")
    assert not quote_in_text("Heading to Austin", "Headed to Austin and will fly back")


def test_no_verified_quote_is_rejected_and_not_stored():
    r = _run(_mem(evidence=[{"turn": "t2", "field": "prompt", "quote": "flying to Austin on Monday"}]))
    assert r.memories == []
    assert [x.reason for x in r.rejections] == ["no_verified_quote"]


def test_quote_from_wrong_field_does_not_verify():
    # The words are in Orion's response, not Juniper's prompt.
    r = _run(_mem(evidence=[{"turn": "t2", "field": "prompt", "quote": "I'll be here when you're back"}]))
    assert [x.reason for x in r.rejections] == ["no_verified_quote"]


def test_juniper_said_backed_only_by_orions_words_is_downgraded():
    r = _run(_mem(statement="Juniper hopes the trip to Austin goes smoothly for her.",
                  evidence=[{"turn": "t2", "field": "response", "quote": "I hope the trip goes smoothly"}]))
    m = r.memories[0]
    assert m.voice == "orion_thought"
    assert [e.op for e in m.events] == ["downgraded_voice"]
    assert r.downgrades == 1


def test_worked_out_together_needs_both_sides():
    both = _run(_mem(voice="worked_out_together", statement="Juniper and I agreed the Austin trip will be busy for her.",
                     evidence=[{"turn": "t2", "field": "prompt", "quote": "Headed to Austin"},
                               {"turn": "t2", "field": "response", "quote": "I hope the trip goes smoothly"}]))
    assert both.memories[0].voice == "worked_out_together"
    prompt_only = _run(_mem(voice="worked_out_together"))
    assert prompt_only.memories[0].voice == "juniper_said"


def test_internal_reverie_never_becomes_something_juniper_said():
    """Source monitoring: something Orion turned over on its own is never 'we discussed'."""
    r = _run(_mem(voice="worked_out_together", channel="reverie",
                  statement="Juniper and I discussed how travel changes what feels like home.",
                  evidence=[{"turn": "t2", "field": "prompt", "quote": "Headed to Austin"},
                            {"turn": "t2", "field": "response", "quote": "I'll be here when you're back"}]))
    assert r.memories == []
    assert [x.reason for x in r.rejections] == ["internal_channel_labelled_as_juniper"]
    r2 = _run(_mem(voice="juniper_said", channel="curiosity"))
    assert [x.reason for x in r2.rejections] == ["internal_channel_labelled_as_juniper"]


def test_internal_thought_is_kept_as_orions_own():
    r = _run(_mem(voice="orion_thought", channel="reverie",
                  statement="I kept turning over what being away from home means for Juniper.",
                  evidence=[{"turn": "t2", "field": "response", "quote": "I'll be here when you're back"}]))
    m = r.memories[0]
    assert (m.voice, m.channel) == ("orion_thought", "reverie")


def test_commands_produce_no_memories():
    r = _run(_mem(statement="Juniper asked me to run the github compactor this morning.",
                  evidence=[{"turn": "t4", "field": "prompt", "quote": "Run github compactor."}]))
    assert [x.reason for x in r.rejections] == ["command_turn_only"]


def test_short_and_duplicate_statements_are_rejected():
    r = _run(_mem(statement="Austin trip."), _mem(), _mem())
    assert [x.reason for x in r.rejections] == ["statement_too_short", "duplicate_statement"]
    assert len(r.memories) == 1


def test_high_stakes_from_the_distiller_means_pending_confirmation():
    r = _run(_mem(stakes="high", stakes_reason="family_relationships"))
    assert (r.memories[0].stakes, r.memories[0].confirmation_state) == ("high", "pending_confirmation")


@pytest.mark.parametrize("raw, key", [
    ("event: Austin AI/ML offsite 2026-09", "event:austin-ai-ml-offsite-2026-09"),
    ("Service:orion_durable_runs", "service:orion-durable-runs"),
    ("austin", None),
    ("mood:tired", None),
])
def test_referent_keys_are_kind_slug(raw, key):
    assert normalize_referent_key(raw) == key


def test_follow_up_dates_and_strength():
    r = _run(_mem(purpose="follow_up", voice="orion_thought",
                  statement="Ask Juniper how the Austin offsite went and how she is recovering.",
                  due_after="2026-09-30T18:00:00-06:00", expires_at="2026-10-07T00:00:00Z",
                  evidence=[{"turn": "t2", "field": "prompt", "quote": "will fly back on Wednesday"}]))
    m = r.memories[0]
    assert m.due_after >= datetime(2026, 9, 30, tzinfo=timezone.utc)
    assert (m.strength, m.half_life_days) == (1.0, None)


def test_memory_ids_are_deterministic():
    a, b = _run(_mem()), _run(_mem())
    assert a.memories[0].memory_id == b.memories[0].memory_id


def test_questions_need_a_verified_quote():
    r = _run(questions=[
        {"text": "Is being away from home hard for Juniper, or was it this trip?",
         "evidence": [{"turn": "t1", "field": "prompt", "quote": "days with work travel"}]},
        {"text": "Does Juniper like Austin?", "evidence": [{"turn": "t1", "field": "prompt", "quote": "loves Austin"}]},
    ])
    assert len(r.questions) == 1 and [x.reason for x in r.rejections] == ["no_verified_quote"]


def test_coverage_counts_non_command_turns_cited():
    r = _run(_mem(), _mem(statement="Juniper will be busy with work travel for a few days.",
                          evidence=[{"turn": "t1", "field": "prompt", "quote": "busy the next few days"}]))
    assert coverage(r, TURNS) == {"content_turns": 3, "cited_turns": 2, "coverage": 0.667}


# --- stakes: presence and consistency only (Juniper's rubric, 2026-10-06) -------------------------

from typing import get_args  # noqa: E402

from orion.memory.episode.validate import resolve_stakes  # noqa: E402
from orion.schemas.memory_episode import HIGH_STAKES_REASONS, StakesReason  # noqa: E402


@pytest.mark.parametrize("reason", sorted(HIGH_STAKES_REASONS))
def test_high_with_a_category_is_kept_as_is(reason):
    assert resolve_stakes("high", reason) == ("high", reason, None)


def test_low_with_none_is_auto():
    r = _run(_mem(stakes="low", stakes_reason="none"))
    m = r.memories[0]
    assert (m.stakes, m.stakes_reason, m.confirmation_state, m.events) == ("low", "none", "auto", [])


@pytest.mark.parametrize("stakes, reason, asks, expect, op", [
    ("low", "juniper_feelings", False, ("high", "juniper_feelings"), "stakes_raised"),   # category is the judgment
    ("low", None, False, ("high", None), "stakes_raised"),                               # not judged
    ("low", "safety_location", False, ("high", None), "stakes_raised"),                  # retired / unknown
    ("low", "none", True, ("high", "orion_asks_direction"), "stakes_raised"),            # asks for direction
    ("high", "none", False, ("high", None), "stakes_reason_missing"),
    ("high", None, False, ("high", None), "stakes_reason_missing"),
    ("high", None, True, ("high", "orion_asks_direction"), "stakes_reason_set"),
])
def test_inconsistent_pairs_resolve_toward_high_and_are_logged(stakes, reason, asks, expect, op):
    got_stakes, got_reason, event = resolve_stakes(stakes, reason, asks)
    assert (got_stakes, got_reason) == expect
    assert event is not None and event.op == op


def test_stakes_never_lowered_and_statement_never_read():
    """The same statement gets whatever the distiller's pair says; vocabulary changes nothing."""
    scary = "Juniper told me she was terrified her sister was in the hospital."
    low = _run(_mem(statement=scary, stakes="low", stakes_reason="none")).memories[0]
    assert (low.stakes, low.confirmation_state) == ("low", "auto")
    calm = "Juniper flew to Austin for a work offsite this week."
    high = _run(_mem(statement=calm, stakes="high", stakes_reason="health")).memories[0]
    assert (high.stakes, high.confirmation_state) == ("high", "pending_confirmation")


def test_unknown_reason_does_not_drop_the_memory_at_parse():
    from orion.memory.episode.distill import parse_distillation

    d = parse_distillation('{"memories": [{"purpose": "happened", "voice": "juniper_said", '
                           '"statement": "s", "stakes": "low", "stakes_reason": "made_up"}]}')
    assert len(d.memories) == 1


def test_none_is_the_only_low_reason():
    assert set(get_args(StakesReason)) - HIGH_STAKES_REASONS == {"none"}
