"""Patch 1a, Hub half: the whole conversation for Claude, and Q11's light view for Orion.

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md
(acceptance check 1, open question 11).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import scripts.chat_history_rehydrate as rehydrate_mod
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomTranscriptEntryV1
from scripts.room_claude_context import (
    HUB_INDEX_MAX_CHARS,
    clip_long_text,
    render_hub_index,
    row_entries,
    select_window,
    transcript_note,
)


def _row(i: int, *, prompt="", response="", meta=None, chars=None):
    if chars is not None:
        prompt = "p" * chars
    return {
        "id": f"id-{i}",
        "correlation_id": f"{i:08d}-corr-rest",
        "prompt": prompt,
        "response": response,
        "created_at": datetime(2026, 11, 12, 10, i % 60, tzinfo=timezone.utc),
        "client_meta": meta,
    }


# -- one row -> entries ---------------------------------------------------------------------


def test_normal_turn_gives_juniper_then_orion_with_real_time_and_turn_ref():
    entries = row_entries(_row(5, prompt="hi", response="hello"))
    assert [e["speaker_name"] for e in entries] == ["Juniper", "Oríon"]
    assert all(e["turn_ref"] == "00000005" for e in entries)
    assert entries[0]["created_at"] == "2026-11-12T10:05:00+00:00"
    assert entries[1]["unprompted"] is False
    for e in entries:
        RoomTranscriptEntryV1.model_validate(e)  # the shape the companion consumes


def test_unprompted_orion_row_is_labelled():
    (e,) = row_entries(_row(1, response="I noticed something", meta={"unsolicited": True}))
    assert e["speaker_id"] == "orion" and e["unprompted"] is True


def test_claude_side_exchange_carries_the_prompt_that_asked_for_it():
    meta = {"room_claude": {"request_id": "r", "prompt": "what do you think?", "invited_by": "Juniper"}}
    entries = row_entries(_row(2, response="I think so.", meta=meta))
    assert [(e["speaker_name"], e["text"]) for e in entries] == [
        ("Juniper (to Claude)", "what do you think?"),
        ("Claude", "I think so."),
    ]
    assert all(e["side_exchange"] for e in entries)


def test_old_claude_row_without_a_stored_prompt_still_shows_the_reply():
    (e,) = row_entries(_row(3, response="old reply", meta='{"room_claude": {"request_id": "r"}}'))
    assert e["speaker_id"] == "claude" and e["side_exchange"] is True


def test_error_shaped_reply_is_not_shown():
    assert [e["speaker_id"] for e in row_entries(_row(4, prompt="q", response="[Error: 400]"))] == ["juniper"]


def test_long_row_keeps_head_and_tail():
    text = "a" * 8000 + "b" * 2000 + "c" * 3000
    clipped = clip_long_text(text)
    assert clipped.startswith("a" * 8000) and clipped.endswith("c" * 3000)
    assert "[... 2000 characters cut ...]" in clipped
    assert clip_long_text("x" * 12000) == "x" * 12000


# -- the window ---------------------------------------------------------------------------------


def test_window_keeps_whole_rows_newest_first_and_never_half_a_row():
    rows_newest_first = [_row(9, chars=40), _row(8, chars=40), _row(7, chars=40)]
    entries, kept, stopped = select_window(rows_newest_first, max_chars=100)
    assert kept == 2 and stopped["id"] == "id-7"
    assert [e["turn_ref"] for e in entries] == ["00000008", "00000009"], "oldest kept first"
    assert sum(len(e["text"]) for e in entries) == 80


def test_a_row_exactly_at_the_cap_fits():
    entries, kept, stopped = select_window([_row(1, chars=50), _row(0, chars=50)], max_chars=100)
    assert kept == 2 and stopped is None


def test_empty_rows_cost_nothing():
    entries, kept, stopped = select_window([_row(1), _row(0, chars=10)], max_chars=10)
    assert kept == 1 and stopped is None


def test_earlier_note_says_how_many_and_when():
    note = transcript_note(323, datetime(2026, 9, 18, tzinfo=timezone.utc), datetime(2026, 11, 5, tzinfo=timezone.utc))
    assert note == "[Earlier: 323 turns from 2026-09-18 to 2026-11-05 not shown.]"
    assert transcript_note(0, None, None) is None
    assert transcript_note(1, "2026-11-05T00:00:00", "2026-11-05T01:00:00") == "[Earlier: 1 turn from 2026-11-05 not shown.]"


def test_hub_index_is_capped_at_800_characters():
    lines = [f"- line {i} " + "x" * 90 for i in range(20)]
    out = render_hub_index(lines)
    assert len(out) <= HUB_INDEX_MAX_CHARS and out.startswith("- line 0")
    assert render_hub_index([]) is None


def test_request_with_full_context_is_valid_and_large_payloads_survive():
    """160,000 characters of transcript in one request validates (extra=forbid)."""
    rows = [_row(i % 60, prompt="p" * 4000, response="r" * 4000) for i in range(40)]
    entries, kept, _ = select_window(rows, max_chars=160_000)
    req = RoomClaudeRequestV1(
        room_id="hub-direct", invited_by="Juniper", prompt="q", transcript=entries,
        hub_index="- Open asks waiting on Juniper: 5", transcript_note="[Earlier: 20 turns not shown.]",
    )
    assert kept == 20 and sum(len(e.text) for e in req.transcript) == 160_000


# -- the invite route ------------------------------------------------------------------------------


class _Relay:
    enabled = True

    def __init__(self):
        self.kw = None

    async def invite(self, **kw):
        self.kw = kw
        return RoomClaudeRequestV1(
            room_id=kw["room_id"], invited_by=kw["invited_by"], prompt=kw["prompt"],
            transcript=kw["transcript"], hub_index=kw.get("hub_index"),
            transcript_note=kw.get("transcript_note"),
        )


def _invite(monkeypatch, mode: str):
    import scripts.api_routes as routes
    import scripts.main as hub_main
    import scripts.room_claude_context as ctx
    from scripts.settings import settings

    relay = _Relay()
    monkeypatch.setattr(hub_main, "room_claude_relay", relay, raising=False)
    monkeypatch.setattr(settings, "HUB_ROOM_CLAUDE_CONTEXT_MODE", mode, raising=False)
    seen = {}

    async def fake_build(session_id, *, max_chars, bus=None, claude_name="Claude"):
        seen["max_chars"] = max_chars
        return [row_entries(_row(1, prompt="hi", response="hello"))[0]], "[Earlier: 1 turn not shown.]", "- Open asks waiting on Juniper: 2"

    async def fake_recent(session_id, limit):
        seen["recent"] = limit
        return []

    monkeypatch.setattr(ctx, "build_room_claude_context", fake_build)
    monkeypatch.setattr(routes, "_recent_room_transcript", fake_recent)
    payload = routes.RoomClaudeInviteRequest(prompt="hey", session_id="sess-1")
    out = asyncio.run(routes.api_room_claude_invite(payload))
    return out, relay, seen


def test_invite_sends_the_full_window_by_default(monkeypatch):
    out, relay, seen = _invite(monkeypatch, "full")
    assert seen["max_chars"] == 160000
    assert relay.kw["hub_index"] == "- Open asks waiting on Juniper: 2"
    assert relay.kw["transcript_note"] == "[Earlier: 1 turn not shown.]"
    assert out["transcript_note"] == "[Earlier: 1 turn not shown.]"
    assert "recent" not in seen


def test_recent_mode_is_the_rollback(monkeypatch):
    _, relay, seen = _invite(monkeypatch, "recent")
    assert seen["recent"] == 12 and relay.kw["hub_index"] is None


# -- relay stores what asked (Q11) ----------------------------------------------------------------


def test_relay_stores_the_prompt_in_client_meta_not_the_prompt_column():
    from orion.schemas.room_claude import ExternalRoomResponderV1, RoomClaudeUtteranceV1
    from scripts.room_claude_relay import RoomClaudeRelay

    class _Bus:
        enabled = True

        async def publish(self, channel, envelope):
            pass

    relay = RoomClaudeRelay(request_channel="a", utterance_channel="b", enabled=True)
    relay._bus = _Bus()
    published = []

    async def _fake_publish(bus, envelopes):
        published.extend(envelopes)

    import scripts.chat_history as chat_history

    original = chat_history.publish_chat_history
    chat_history.publish_chat_history = _fake_publish
    try:
        request = asyncio.run(relay.invite(prompt="@claude thoughts?", invited_by="Juniper",
                                           session_id="s1", room_id="hub-direct"))
        utt = RoomClaudeUtteranceV1(
            request_id=request.request_id, room_id="hub-direct", session_id="s1",
            responder=ExternalRoomResponderV1(participant_id="claude", participant_name="Claude"),
            text="Here is a thought.",
        )
        asyncio.run(relay._handle_utterance({"payload": utt.model_dump(mode="json")}))
    finally:
        chat_history.publish_chat_history = original

    (env,) = published
    assert env.payload.role == "assistant", "never a user message: that would make it a Juniper turn"
    rc = env.payload.client_meta["room_claude"]
    assert rc["prompt"] == "@claude thoughts?" and rc["invited_by"] == "Juniper"


# -- rehydrate: Orion's light view (Q11) ------------------------------------------------------------


def _side(i, prompt, reply):
    return {
        "response": reply,
        "created_at": datetime(2026, 11, 12, 10, i, tzinfo=timezone.utc),
        "client_meta": {"room_claude": {"prompt": prompt, "invited_by": "Juniper"}} if prompt else {"room_claude": {}},
    }


def test_side_block_is_one_labelled_system_message_with_trimmed_text():
    block = rehydrate_mod.render_side_exchange_block(
        [_side(1, "short q", "a" * 500), _side(2, None, "old reply")], trim_chars=300
    )
    assert block["role"] == "system"
    lines = block["content"].split("\n")
    assert lines[0] == "Side conversation, Juniper ↔ Claude (you were not asked to reply):"
    assert lines[1] == "  Juniper → Claude: short q"
    assert lines[2] == "  Claude: " + "a" * 300 + " …"
    assert lines[3] == "  Claude: old reply"
    assert block["ts"] == "2026-11-12T10:02:00+00:00"


def test_side_block_is_none_when_there_is_nothing():
    assert rehydrate_mod.render_side_exchange_block([], trim_chars=300) is None


def test_block_goes_in_at_its_time_never_first():
    msgs = [
        {"role": "user", "content": "a", "ts": "2026-11-12T09:00:00+00:00"},
        {"role": "assistant", "content": "b", "ts": "2026-11-12T11:00:00+00:00"},
    ]
    rehydrate_mod.insert_by_time(msgs, {"role": "system", "content": "S", "ts": "2026-11-12T10:00:00+00:00"})
    assert [m["content"] for m in msgs] == ["a", "S", "b"]
    early = [{"role": "user", "content": "a", "ts": "2026-11-12T12:00:00+00:00"}]
    rehydrate_mod.insert_by_time(early, {"role": "system", "content": "S", "ts": "2026-11-12T10:00:00+00:00"})
    assert early[0]["content"] == "a", "a leading system message reads as persona priming to the trim"


def _patch(monkeypatch, rows, side_rows):
    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", lambda sid, **kw: rows)
    seen = {}

    def fake_side(sid, **kw):
        seen.update(kw)
        return side_rows

    monkeypatch.setattr(rehydrate_mod, "fetch_side_exchange_rows", fake_side)
    return seen


def test_rehydrate_adds_the_side_block_when_enabled(monkeypatch):
    rows = [{"prompt": "hi", "response": "hello", "created_at": datetime(2026, 11, 12, 9, tzinfo=timezone.utc)}]
    seen = _patch(monkeypatch, rows, [_side(30, "q", "r")])
    history = [{"role": "system", "content": "persona"}]
    n = asyncio.run(rehydrate_mod.rehydrate_history(
        history, session_id="s", max_turns=10, max_age_hours=48.0, side_exchange_max=3, side_exchange_trim_chars=300,
    ))
    assert n == 3 and seen["limit"] == 3
    assert [m["role"] for m in history] == ["system", "user", "assistant", "system"]
    assert history[-1]["content"].startswith("Side conversation, Juniper")
    # Nothing about it is a prompt Orion should answer.
    assert not any(m["role"] == "user" and "Claude" in m["content"] for m in history)


def test_rehydrate_without_the_flag_is_unchanged(monkeypatch):
    rows = [{"prompt": "hi", "response": "hello"}]
    _patch(monkeypatch, rows, [_side(30, "q", "r")])
    history = [{"role": "system", "content": "persona"}]
    asyncio.run(rehydrate_mod.rehydrate_history(history, session_id="s", max_turns=10, max_age_hours=48.0))
    assert len(history) == 3


def test_side_fetch_failure_never_costs_the_real_history(monkeypatch):
    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", lambda sid, **kw: [{"prompt": "hi", "response": "yo"}])

    def boom(*a, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(rehydrate_mod, "fetch_side_exchange_rows", boom)
    history = [{"role": "system", "content": "persona"}]
    n = asyncio.run(rehydrate_mod.rehydrate_history(
        history, session_id="s", max_turns=10, max_age_hours=48.0, side_exchange_max=3,
    ))
    assert n == 2


def test_orions_own_history_query_still_excludes_room_claude_rows():
    import inspect

    src = inspect.getsource(rehydrate_mod.fetch_recent_rows)
    assert "coalesce(client_meta->>'room_claude', '') = ''" in src


@pytest.mark.parametrize("value", [0, -1])
def test_zero_turns_the_block_off(monkeypatch, value):
    seen = _patch(monkeypatch, [{"prompt": "hi", "response": "yo"}], [_side(1, "q", "r")])
    history = [{"role": "system", "content": "persona"}]
    asyncio.run(rehydrate_mod.rehydrate_history(
        history, session_id="s", max_turns=10, max_age_hours=48.0, side_exchange_max=value,
    ))
    assert seen == {} and len(history) == 3


# -- review findings (2026-11-12) ----------------------------------------------------


def test_side_block_reaches_only_the_unified_turn():
    from scripts.cortex_request_builder import build_continuity_messages

    block = rehydrate_mod.render_side_exchange_block([_side(1, "q", "r")], trim_chars=300)
    history = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi", "ts": "2026-11-12T09:00:00+00:00"},
        block,
        {"role": "assistant", "content": "hello", "ts": "2026-11-12T11:00:00+00:00"},
    ]
    plain = build_continuity_messages(history=history, latest_user_prompt="x", turns=10)
    assert not any("Side conversation" in m["content"] for m in plain), "a mid-list system message breaks some chat templates"
    unified = build_continuity_messages(history=history, latest_user_prompt="x", turns=10, keep_timestamps=True)
    assert any(m["content"].startswith("Side conversation") and m["role"] == "system" for m in unified)


def test_no_side_block_without_a_restored_conversation(monkeypatch):
    """A history holding only the block still reads as empty, so a second
    session_hello would append it again."""
    _patch(monkeypatch, [], [_side(1, "q", "r")])
    history = [{"role": "system", "content": "persona"}]
    n = asyncio.run(rehydrate_mod.rehydrate_history(
        history, session_id="s", max_turns=10, max_age_hours=48.0, side_exchange_max=3,
    ))
    assert n == 0 and len(history) == 1


def test_trim_zero_means_no_trim():
    block = rehydrate_mod.render_side_exchange_block([_side(1, None, "a" * 900)], trim_chars=0)
    assert block["content"].endswith("a" * 900)


def test_earlier_count_excludes_the_shown_boundary_row_when_the_fetch_limit_stops():
    from scripts.room_claude_context import older_summary_sql

    assert "created_at <= :before" in older_summary_sql(cap_stopped=True)
    assert "created_at < :before" in older_summary_sql(cap_stopped=False)


def test_fetch_conversation_against_a_fake_connection(monkeypatch):
    """The real query path: cap stop -> the 'Earlier' note uses <= and the stopping row."""
    import scripts.room_claude_context as ctx

    rows = [_row(9, chars=60), _row(8, chars=60)]
    calls = []

    class _Result:
        def __init__(self, data):
            self._data = data

        def mappings(self):
            return self

        def all(self):
            return self._data

        def first(self):
            return self._data[0] if self._data else None

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt, params):
            calls.append((str(stmt), params))
            if "count(*)" in str(stmt):
                return _Result([{"n": 4, "first_at": "2026-11-01T00:00:00", "last_at": "2026-11-12T10:08:00"}])
            return _Result(rows)

    class _Engine:
        def connect(self):
            return _Conn()

    monkeypatch.setattr(ctx, "_engine", lambda: _Engine())
    entries, note = ctx.fetch_conversation("sess", max_chars=100)
    assert [e["turn_ref"] for e in entries] == ["00000009"]
    assert note == "[Earlier: 4 turns from 2026-11-01 to 2026-11-12 not shown.]"
    sql, params = calls[1]
    assert "created_at <= :before" in sql and params["before"] == rows[1]["created_at"]
    assert "NULLS LAST" in calls[0][0]


class _Redis:
    def __init__(self, values):
        self.values = values

    async def get(self, key):
        return self.values.get(key)


class _Bus:
    def __init__(self, values):
        self.redis = _Redis(values)


def _prime(as_of):
    from orion.prime.block import PRIME_STATE_KEY
    from orion.prime.state import PRIME_CONCERNS_KEY, PrimeConcernsViewV1, PrimeStateV1

    view = PrimeConcernsViewV1.model_validate({"as_of": as_of.isoformat()})
    return {PRIME_STATE_KEY: PrimeStateV1().model_dump_json(), PRIME_CONCERNS_KEY: view.model_dump_json()}


def test_prime_count_is_shown_fresh_and_hidden_stale():
    from datetime import timedelta

    from scripts.room_claude_context import prime_concern_count_line

    now = datetime.now(timezone.utc)
    fresh = asyncio.run(prime_concern_count_line(_Bus(_prime(now))))
    assert fresh == "- Prime holds 0 concern(s)"
    assert asyncio.run(prime_concern_count_line(_Bus(_prime(now - timedelta(hours=2))))) is None
    assert asyncio.run(prime_concern_count_line(_Bus({}))) is None
