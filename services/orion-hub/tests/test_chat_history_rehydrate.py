"""Conversation-history rehydration on websocket connect.

Regression cover for 2026-08-14: a Hub rebuild mid-conversation wiped the
in-memory per-connection ``history``, and because
``build_continuity_messages`` has no fallback, the next turns went out with
nothing but the current prompt. Orion answered a direct follow-up with "What's
on your mind?" and then surfaced an unrelated old conversation.
"""

from __future__ import annotations

import asyncio

import pytest

import inspect

import scripts.chat_history_rehydrate as rehydrate_mod
from scripts.chat_history_rehydrate import (
    fetch_recent_rows,
    history_is_empty,
    rehydrate_history,
    rows_to_history_messages,
)


def _system() -> list[dict]:
    return [{"role": "system", "content": "you are orion"}]


def _sql_where_clause(source: str) -> str:
    """The literal SQL text inside fetch_recent_rows's ``text(...)`` call --
    isolated from the surrounding docstring/comments so a check against it
    can't be satisfied by prose alone."""
    start = source.index('SELECT prompt, response')
    end = source.index('"""', start)
    return source[start:end]


def test_fetch_recent_rows_excludes_room_claude_turns() -> None:
    """Claude's room replies are persisted with client_meta.room_claude set
    (see room_claude_relay.py's _publish_history) and role="assistant", same
    shape as a genuine Orion turn. Without this exclusion they come back
    through rehydration labeled plain "assistant" and get fed into Orion's own
    continuity context as if Orion had said them.

    No live Postgres in this test env, so this asserts the actual SQL clause
    text -- not the surrounding docstring, which would pass even with the
    clause deleted -- same level the existing 'unsolicited' exclusion would
    need if it were tested at all.
    """
    where = _sql_where_clause(inspect.getsource(fetch_recent_rows))
    assert "client_meta->>'room_claude'" in where, "the SQL must exclude room_claude-tagged rows"
    assert "= ''" in where
    assert "client_meta->>'unsolicited'" in where, "the existing outreach exclusion must still be present"


# --------------------------------------------------------------------------
# Row -> message conversion
# --------------------------------------------------------------------------


def test_prompt_and_response_become_two_messages() -> None:
    rows = [{"prompt": "hi", "response": "hey there"}]
    assert rows_to_history_messages(rows, max_turns=10) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey there"},
    ]


def test_half_rows_contribute_one_message_each() -> None:
    """Hub writes prompt-only and response-only rows separately."""
    rows = [
        {"prompt": "well, that was Muse Glimmer.", "response": ""},
        {"prompt": "", "response": "It's good to hear from you."},
    ]
    assert rows_to_history_messages(rows, max_turns=10) == [
        {"role": "user", "content": "well, that was Muse Glimmer."},
        {"role": "assistant", "content": "It's good to hear from you."},
    ]


def test_order_is_preserved_oldest_first() -> None:
    rows = [
        {"prompt": "first", "response": "1"},
        {"prompt": "second", "response": "2"},
    ]
    got = [m["content"] for m in rows_to_history_messages(rows, max_turns=10)]
    assert got == ["first", "1", "second", "2"]


def test_cap_keeps_the_newest_messages() -> None:
    rows = [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(10)]
    # max_turns=2 -> 4 messages -> the last two pairs.
    got = [m["content"] for m in rows_to_history_messages(rows, max_turns=2)]
    assert got == ["q8", "a8", "q9", "a9"]


def test_persisted_llm_error_is_not_replayed_as_speech() -> None:
    """A llamacpp 400 reached chat_history_log as an assistant row (2026-08-14).

    Replaying it would teach Orion it once said an HTTP error out loud.
    """
    rows = [
        {"prompt": "you there?", "response": "[Error: llamacpp failed: Client error '400 Bad Request']"},
        {"prompt": "hello", "response": "I'm here."},
    ]
    got = rows_to_history_messages(rows, max_turns=10)
    assert got == [
        {"role": "user", "content": "you there?"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I'm here."},
    ]


def test_prose_mentioning_errors_survives() -> None:
    rows = [{"prompt": "how goes it", "response": "The codebase is throwing errors I can't map yet."}]
    got = rows_to_history_messages(rows, max_turns=10)
    assert got[-1]["content"] == "The codebase is throwing errors I can't map yet."


def test_blank_rows_contribute_nothing() -> None:
    assert rows_to_history_messages([{"prompt": "  ", "response": None}], max_turns=10) == []


# --------------------------------------------------------------------------
# Emptiness detection
# --------------------------------------------------------------------------


def test_system_only_history_counts_as_empty() -> None:
    assert history_is_empty(_system()) is True


def test_history_with_a_real_turn_is_not_empty() -> None:
    assert history_is_empty(_system() + [{"role": "user", "content": "hi"}]) is False


def test_completely_empty_list_is_empty() -> None:
    assert history_is_empty([]) is True


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _patch_rows(monkeypatch, rows) -> dict:
    seen: dict = {}

    def fake_fetch(session_id, *, max_turns, max_age_hours):
        seen.update(session_id=session_id, max_turns=max_turns, max_age_hours=max_age_hours)
        return rows

    # Patch the module OBJECT, not a dotted string. tests/conftest.py drops
    # every `scripts.*` entry from sys.modules before each test, so a string
    # target re-imports a fresh module and patches a copy that the already-bound
    # rehydrate_history never looks at.
    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", fake_fetch)
    return seen


def test_restores_conversation_after_a_restart(monkeypatch) -> None:
    """The actual failure: reconnect with a system-only history."""
    history = _system()
    _patch_rows(monkeypatch, [{"prompt": "tell me about Muse", "response": "It's a 30B distill."}])

    restored = asyncio.run(
        rehydrate_history(history, session_id="orion_journal", max_turns=12, max_age_hours=48.0)
    )

    assert restored == 2
    assert history[0]["role"] == "system"  # priming preserved at index 0
    assert history[1] == {"role": "user", "content": "tell me about Muse"}
    assert history[2] == {"role": "assistant", "content": "It's a 30B distill."}


def test_mutates_in_place_not_by_rebinding(monkeypatch) -> None:
    """The ws loop holds this exact list for the connection's lifetime."""
    history = _system()
    original_id = id(history)
    _patch_rows(monkeypatch, [{"prompt": "q", "response": "a"}])

    asyncio.run(rehydrate_history(history, session_id="s", max_turns=12, max_age_hours=48.0))

    assert id(history) == original_id
    assert len(history) == 3


def test_never_stomps_a_live_conversation(monkeypatch) -> None:
    history = _system() + [{"role": "user", "content": "already talking"}]
    _patch_rows(monkeypatch, [{"prompt": "old", "response": "older"}])

    restored = asyncio.run(
        rehydrate_history(history, session_id="s", max_turns=12, max_age_hours=48.0)
    )

    assert restored == 0
    assert history == _system() + [{"role": "user", "content": "already talking"}]


def test_disabled_flag_restores_nothing(monkeypatch) -> None:
    history = _system()
    _patch_rows(monkeypatch, [{"prompt": "q", "response": "a"}])

    restored = asyncio.run(
        rehydrate_history(
            history, session_id="s", max_turns=12, max_age_hours=48.0, enabled=False
        )
    )

    assert restored == 0
    assert history == _system()


@pytest.mark.parametrize("session_id", [None, "", "   "])
def test_missing_session_id_restores_nothing(monkeypatch, session_id) -> None:
    history = _system()
    _patch_rows(monkeypatch, [{"prompt": "q", "response": "a"}])

    restored = asyncio.run(
        rehydrate_history(history, session_id=session_id, max_turns=12, max_age_hours=48.0)
    )

    assert restored == 0
    assert history == _system()


def test_db_failure_degrades_to_cold_start(monkeypatch) -> None:
    """A rehydrate failure must never break the connection."""
    history = _system()

    def boom(session_id, *, max_turns, max_age_hours):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", boom)

    restored = asyncio.run(
        rehydrate_history(history, session_id="s", max_turns=12, max_age_hours=48.0)
    )

    assert restored == 0
    assert history == _system()


def test_no_rows_leaves_history_untouched(monkeypatch) -> None:
    history = _system()
    _patch_rows(monkeypatch, [])

    assert asyncio.run(
        rehydrate_history(history, session_id="s", max_turns=12, max_age_hours=48.0)
    ) == 0
    assert history == _system()


def test_settings_are_passed_through_to_the_query(monkeypatch) -> None:
    history = _system()
    seen = _patch_rows(monkeypatch, [{"prompt": "q", "response": "a"}])

    asyncio.run(
        rehydrate_history(history, session_id="orion_journal", max_turns=7, max_age_hours=6.0)
    )

    assert seen == {"session_id": "orion_journal", "max_turns": 7, "max_age_hours": 6.0}


def test_history_without_system_message_still_restores(monkeypatch) -> None:
    history: list[dict] = []
    _patch_rows(monkeypatch, [{"prompt": "q", "response": "a"}])

    asyncio.run(rehydrate_history(history, session_id="s", max_turns=12, max_age_hours=48.0))

    assert history == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_restored_history_feeds_continuity_messages(monkeypatch) -> None:
    """End-to-end with the real consumer -- the thing that was actually broken.

    Note what the cold path really does: a system-only history is NOT empty as
    far as build_continuity_messages is concerned (it accepts role="system"),
    so it never reaches its latest_user_prompt fallback and the turn goes out
    carrying the priming message and nothing else -- not even the user's own
    words. That is worse than the docstring-level reading of the code suggests,
    and it is why history_is_empty() ignores system rows.
    """
    from scripts.cortex_request_builder import build_continuity_messages

    history = _system()
    cold = build_continuity_messages(
        history=history, latest_user_prompt="just checking in after our deep convo", turns=12
    )
    assert [m["role"] for m in cold] == ["system"]
    assert not any("Muse" in m["content"] for m in cold)

    _patch_rows(
        monkeypatch,
        [{"prompt": "tell me about Muse Glimmer", "response": "A 30B distill of Muse Spark."}],
    )
    asyncio.run(rehydrate_history(history, session_id="s", max_turns=12, max_age_hours=48.0))

    warm = build_continuity_messages(
        history=history, latest_user_prompt="just checking in after our deep convo", turns=12
    )
    contents = [m["content"] for m in warm]
    assert "tell me about Muse Glimmer" in contents
    assert "A 30B distill of Muse Spark." in contents
