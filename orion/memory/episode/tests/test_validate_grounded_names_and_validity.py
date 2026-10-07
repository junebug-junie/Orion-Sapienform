"""Two writer fixes the situation graph depends on (spec 2026-10-07-situation-graph-design.md, section 7).

1. A statement may not name a known person, place, project or service that Juniper never said in
   the episode and that none of its own quotes contain. Live case 2026-10-06 (c0dc86c8): "Juniper
   corrected me that she lives in Ogden, Utah, not Chicago." -- its only quote was "we live in Ogden,
   Utah"; "Chicago" came from Orion's own reply. Such a memory is kept but asked about.
2. A time-bounded fact keeps its end date on any purpose, but only when Juniper's own words naming
   that period are quoted from her prompt. Live case: "in Chicago ... until Wednesday" was stored as
   `happened` with expires_at NULL because the validator kept expiry for follow_ups only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.memory.episode.validate import (
    UNGROUNDED_NAME_STAKES_LABEL,
    EpisodeTurn,
    ungrounded_names,
    validate_distillation,
)
from orion.schemas.memory_episode import EpisodeDistillationV1

T0 = datetime(2026, 10, 5, 3, 51, tzinfo=timezone.utc)

OGDEN_TURNS = [
    EpisodeTurn(
        "t1", "c-ogden",
        "correction--we live in Ogden, Utah you goofy goose. We'll mount a camera to the porch.",
        "Ogden. Of course it's Ogden. Not Chicago -- that was just where you were this week.",
        T0,
    ),
]

TRIP_TURNS = [
    EpisodeTurn(
        "t1", "c-trip",
        "Greetings from Chicago! Got out here earlier today for a team meeting. Will be here till Wednesday.",
        "Chicago -- that's a proper change of scenery.",
        datetime(2026, 10, 5, 2, 55, tzinfo=timezone.utc),
    ),
]


def _run(turns, memory, *, known=()):
    d = EpisodeDistillationV1.model_validate({"memories": [memory], "questions": []})
    return validate_distillation(d, turns, episode_id="ep-test", known_referents=list(known))


def _ogden_memory(**kw):
    base = {
        "purpose": "about_juniper", "voice": "juniper_said", "channel": "chat",
        "statement": "Juniper corrected me that she lives in Ogden, Utah, not Chicago.",
        "stakes": "low", "stakes_reason": "none",
        "referents": [{"key": "person:juniper"}, {"key": "place:ogden"}],
        "evidence": [{"turn": "t1", "field": "prompt", "quote": "we live in Ogden, Utah"}],
    }
    base.update(kw)
    return base


# --- 1. grounded names -------------------------------------------------------------------------


def test_ogden_row_names_chicago_that_juniper_never_said_and_is_asked_about():
    r = _run(OGDEN_TURNS, _ogden_memory(), known=["place:chicago", "person:juniper"])
    assert not r.rejections
    (m,) = r.memories
    assert m.stakes == "high"
    assert m.stakes_reason == UNGROUNDED_NAME_STAKES_LABEL
    assert m.confirmation_state == "pending_confirmation"
    ev = [e for e in m.events if e.op == "ungrounded_name"]
    assert ev and ev[0].detail["referents"] == ["place:chicago"]


def test_same_row_without_the_imported_name_stays_settled():
    r = _run(OGDEN_TURNS, _ogden_memory(statement="Juniper told me that she and I live in Ogden, Utah."),
             known=["place:chicago"])
    (m,) = r.memories
    assert (m.stakes, m.confirmation_state) == ("low", "auto")
    assert not [e for e in m.events if e.op == "ungrounded_name"]


def test_a_name_juniper_said_earlier_in_the_episode_is_grounded():
    """Carrying a name across turns is normal ("it" -> Hecate). Only names Juniper never said count."""
    turns = [
        EpisodeTurn("t1", "c-1", "I got us a new GPU server. We'll call it Hecate.", "Hecate is a fitting name.", T0),
        EpisodeTurn("t2", "c-2", "Having trouble flashing the Ubuntu ISO to it tonight.", "Try dd.", T0),
    ]
    mem = {"purpose": "happened", "voice": "juniper_said", "channel": "chat", "stakes": "low", "stakes_reason": "none",
           "statement": "Juniper is having trouble flashing an Ubuntu ISO to Hecate tonight.",
           "referents": [{"key": "project:hecate"}],
           "evidence": [{"turn": "t2", "field": "prompt", "quote": "flashing the Ubuntu ISO"}]}
    (m,) = _run(turns, mem).memories
    assert m.stakes == "low"


def test_participants_and_minted_event_slugs_are_never_checked():
    assert ungrounded_names(
        "Juniper and I talked about the Austin offsite.",
        known_keys={"person:juniper", "person:orion", "event:austin-offsite", "concept:space"},
        grounding_texts=["we talked about it"],
    ) == []


def test_name_matching_is_whole_word_and_typography_folded():
    assert ungrounded_names("She mentioned Ogdensburg.", known_keys={"place:ogden"}, grounding_texts=[""]) == []
    assert ungrounded_names("She lives in OGDEN.", known_keys={"place:ogden"}, grounding_texts=["x"]) == ["place:ogden"]
    assert ungrounded_names("Juniper is at the Wade.", known_keys={"place:the-wade"},
                            grounding_texts=["staying at The Wade tonight"]) == []


def test_already_high_memory_keeps_its_category_but_logs_the_name():
    r = _run(OGDEN_TURNS, _ogden_memory(stakes="high", stakes_reason="family_relationships"), known=["place:chicago"])
    (m,) = r.memories
    assert (m.stakes, m.stakes_reason) == ("high", "family_relationships")
    assert [e for e in m.events if e.op == "ungrounded_name"]


def test_without_known_referents_only_this_episodes_keys_are_checked():
    """No candidate list (e.g. the table is unreadable) -> the check still runs on keys the
    distiller itself emitted in this episode, and never fails the memory."""
    r = _run(OGDEN_TURNS, _ogden_memory(), known=())
    (m,) = r.memories
    assert m.stakes == "low"  # chicago is not a key anywhere in this run, so it cannot be checked


# --- 2. validity windows -----------------------------------------------------------------------


def _trip_memory(**kw):
    base = {
        "purpose": "happened", "voice": "juniper_said", "channel": "chat", "stakes": "low", "stakes_reason": "none",
        "statement": "Juniper is in Chicago for a team meeting and will be there until Wednesday.",
        "referents": [{"key": "person:juniper"}, {"key": "place:chicago"}],
        "evidence": [{"turn": "t1", "field": "prompt", "quote": "Will be here till Wednesday"}],
        "expires_at": "2026-10-08T23:59:00-06:00",
        "until_quote": "Will be here till Wednesday",
    }
    base.update(kw)
    return base


def test_happened_fact_keeps_end_date_when_juniper_named_the_period():
    (m,) = _run(TRIP_TURNS, _trip_memory()).memories
    assert m.expires_at == datetime(2026, 10, 9, 5, 59, tzinfo=timezone.utc)
    assert m.due_after is None


def test_end_date_without_juniper_words_is_dropped_and_logged():
    (m,) = _run(TRIP_TURNS, _trip_memory(until_quote=None)).memories
    assert m.expires_at is None
    assert [e for e in m.events if e.op == "validity_dropped" and e.reason == "no_until_quote"]


def test_until_quote_from_orions_reply_does_not_count():
    turns = [EpisodeTurn("t1", "c-trip", "Greetings from Chicago, here for a team meeting this week.",
                         "Enjoy it, you'll be here until Wednesday then.", TRIP_TURNS[0].created_at)]
    mem = _trip_memory(evidence=[{"turn": "t1", "field": "prompt", "quote": "here for a team meeting"}],
                       until_quote="you'll be here until Wednesday")
    (m,) = _run(turns, mem).memories
    assert m.expires_at is None
    assert [e for e in m.events if e.op == "validity_dropped" and e.reason == "until_quote_not_in_juniper_prompt"]


def test_end_date_before_the_episode_is_dropped():
    (m,) = _run(TRIP_TURNS, _trip_memory(expires_at="2026-10-01T00:00:00Z")).memories
    assert m.expires_at is None
    assert [e for e in m.events if e.op == "validity_dropped" and e.reason == "ends_before_episode"]


def test_follow_up_expiry_is_unchanged():
    mem = _trip_memory(purpose="follow_up", voice="orion_thought",
                       statement="I want to ask Juniper how the Chicago team meeting went.",
                       evidence=[{"turn": "t1", "field": "response", "quote": "a proper change of scenery"}],
                       due_after="2026-10-09T09:00:00-06:00", expires_at="2026-10-16T00:00:00-06:00",
                       until_quote=None)
    (m,) = _run(TRIP_TURNS, mem).memories
    assert m.due_after is not None and m.expires_at is not None


# --- review follow-ups -------------------------------------------------------------------------


def test_quoting_orions_own_reply_does_not_ground_a_name_in_juniper_voice():
    """Review finding: the live row only got caught because it quoted her prompt. Quoting Orion's
    reply ("Not Chicago") must not ground a claim made in Juniper's voice."""
    mem = _ogden_memory(voice="worked_out_together", evidence=[
        {"turn": "t1", "field": "prompt", "quote": "we live in Ogden, Utah"},
        {"turn": "t1", "field": "response", "quote": "Not Chicago -- that was just where you were"},
    ])
    (m,) = _run(OGDEN_TURNS, mem, known=["place:chicago"]).memories
    assert m.stakes_reason == UNGROUNDED_NAME_STAKES_LABEL


def test_orions_own_memory_may_name_what_orion_said():
    mem = {"purpose": "orion_view", "voice": "orion_thought", "channel": "chat", "stakes": "low", "stakes_reason": "none",
           "statement": "I pointed out that Chicago was just where Juniper was this week.",
           "evidence": [{"turn": "t1", "field": "response", "quote": "Not Chicago -- that was just where you were"}]}
    (m,) = _run(OGDEN_TURNS, mem, known=["place:chicago"]).memories
    assert m.stakes == "low"


def test_end_date_without_offset_is_read_in_juniper_timezone():
    (m,) = _run(TRIP_TURNS, _trip_memory(expires_at="2026-10-08T23:59:00")).memories
    assert m.expires_at == datetime(2026, 10, 9, 5, 59, tzinfo=timezone.utc)  # 23:59 MDT


def test_too_short_until_quote_says_so():
    (m,) = _run(TRIP_TURNS, _trip_memory(until_quote="till Wednesday")).memories
    assert m.expires_at is None
    assert [e for e in m.events if e.op == "validity_dropped" and e.reason == "until_quote_too_short"]
