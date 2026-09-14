"""Pure helpers: when Orion should walk out of a dead town conversation."""

from __future__ import annotations

from orion.embodiment.dead_chat import should_abandon_dead_chat


def test_abandon_when_last_line_is_stale():
    assert (
        should_abandon_dead_chat(
            status="participating",
            messages=[{"author_id": "p:29", "text": "hi", "created_ms": 1_000.0}],
            own_player_id="p:29",
            now_ms=1_000.0 + 180_000.0,
            abandon_after_ms=180_000.0,
            own_utterances_this_convo=1,
            participating_since_ms=1_000.0,
        )
        is True
    )


def test_no_abandon_when_last_line_is_fresh():
    assert (
        should_abandon_dead_chat(
            status="participating",
            messages=[{"author_id": "p:0", "text": "hey", "created_ms": 1_000.0}],
            own_player_id="p:29",
            now_ms=1_000.0 + 30_000.0,
            abandon_after_ms=180_000.0,
            own_utterances_this_convo=0,
            participating_since_ms=1_000.0,
        )
        is False
    )


def test_abandon_when_never_spoke_and_participating_too_long():
    assert (
        should_abandon_dead_chat(
            status="participating",
            messages=[{"author_id": "p:0", "text": "spam", "created_ms": 50_000.0}],
            own_player_id="p:29",
            now_ms=1_000.0 + 180_000.0,
            abandon_after_ms=180_000.0,
            own_utterances_this_convo=0,
            participating_since_ms=1_000.0,
        )
        is True
    )


def test_no_abandon_when_disabled():
    assert (
        should_abandon_dead_chat(
            status="participating",
            messages=[{"author_id": "p:29", "text": "hi", "created_ms": 1.0}],
            own_player_id="p:29",
            now_ms=999_999.0,
            abandon_after_ms=0.0,
            own_utterances_this_convo=1,
            participating_since_ms=1.0,
        )
        is False
    )


def test_no_abandon_when_transcript_shows_orion_spoke_but_counter_zero():
    """Worker restart wipes the in-memory counter; transcript must still count."""
    assert (
        should_abandon_dead_chat(
            status="participating",
            messages=[
                {"author_id": "p:29", "text": "earlier", "created_ms": 50_000.0},
                {"author_id": "p:0", "text": "fresh", "created_ms": 170_000.0},
            ],
            own_player_id="p:29",
            now_ms=181_000.0,
            abandon_after_ms=180_000.0,
            own_utterances_this_convo=0,
            participating_since_ms=1_000.0,
        )
        is False
    )


def test_abandon_when_never_spoke_despite_fresh_partner_spam():
    assert (
        should_abandon_dead_chat(
            status="participating",
            messages=[{"author_id": "p:0", "text": "spam", "created_ms": 170_000.0}],
            own_player_id="p:29",
            now_ms=181_000.0,
            abandon_after_ms=180_000.0,
            own_utterances_this_convo=0,
            participating_since_ms=1_000.0,
        )
        is True
    )
