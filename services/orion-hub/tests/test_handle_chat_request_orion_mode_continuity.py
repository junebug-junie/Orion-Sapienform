from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")


def test_handle_chat_request_orion_mode_builds_continuity_messages(monkeypatch) -> None:
    """Regression for the parity gap: the mode=="orion" HTTP branch of /api/chat
    previously called execute_unified_turn with no continuity_messages at all,
    unlike the sibling non-orion branch a few lines below (which calls
    build_continuity_messages(history=user_messages, ...)) and unlike the
    WebSocket path (the real Hub UI), which already built it correctly. This
    was inert while turn_orchestrator dropped continuity_messages on the
    floor, but now that HarnessRunRequestV1.recent_turns threads it into the
    harness prompt, an HTTP-driven orion-mode turn must build it the same way
    the sibling branch does or it silently gets no history at all.

    NOTE: scripts.api_routes/scripts.main must be imported fresh inside this
    test body -- conftest.py's autouse _hub_service_isolation fixture clears
    scripts.*/app.* from sys.modules before every test.
    """
    import scripts.api_routes as api_routes
    import scripts.main as hub_main
    from orion.hub import turn_orchestrator

    monkeypatch.setattr(api_routes.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "ORION_HARNESS_GOVERNOR_ENABLED", True, raising=False)

    class _Bus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _Bus(), raising=False)
    monkeypatch.setattr(hub_main, "harness_step_relay", None, raising=False)
    monkeypatch.setattr(hub_main, "rpc_bus", None, raising=False)

    captured_kwargs: dict = {}

    async def _fake_execute_unified_turn(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            {
                "type": "final",
                "correlation_id": "corr-continuity",
                "llm_response": "a fully reflected answer",
                "finalize_ran": True,
            }
        ]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    payload = {
        "mode": "orion",
        "messages": [
            {"role": "user", "content": "what's the weather like?"},
            {"role": "assistant", "content": "I don't have live weather access."},
            {"role": "user", "content": "ok what about tomorrow?"},
        ],
    }
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-orion-continuity", no_write=True))

    assert out["type"] == "final"
    continuity_messages = captured_kwargs.get("continuity_messages")
    assert continuity_messages, "orion-mode HTTP branch must build continuity_messages, not pass None"
    assert continuity_messages == [
        {"role": "user", "content": "what's the weather like?"},
        {"role": "assistant", "content": "I don't have live weather access."},
        {"role": "user", "content": "ok what about tomorrow?"},
    ]


def test_handle_chat_request_orion_mode_rehydrates_thin_history(monkeypatch) -> None:
    """Regression for the WebSocket-down HTTP fallback gap (confirmed live
    2026-08-22): app.js's fallback (fires when waitForWebSocketOpen times out)
    sends `messages: [{"role": "user", "content": value}]` -- just the current
    turn, no history, because the browser holds no client-side message array
    of its own. Unlike the WebSocket connect path (chat_history_rehydrate.
    rehydrate_history in websocket_handler.py), this HTTP handler had no
    fallback to persisted chat_history_log rows, so a turn landing in that
    reconnect window read as a first-ever message even mid-session, and Orion
    greeted fresh instead of continuing the thread.
    """
    import scripts.api_routes as api_routes
    import scripts.main as hub_main
    from orion.hub import turn_orchestrator

    monkeypatch.setattr(api_routes.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "ORION_HARNESS_GOVERNOR_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_MAX_AGE_HOURS", 48.0, raising=False)

    class _Bus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _Bus(), raising=False)
    monkeypatch.setattr(hub_main, "harness_step_relay", None, raising=False)
    monkeypatch.setattr(hub_main, "rpc_bus", None, raising=False)

    captured_kwargs: dict = {}

    async def _fake_execute_unified_turn(**kwargs):
        captured_kwargs.update(kwargs)
        return [{"type": "final", "correlation_id": "corr-rehydrate", "llm_response": "ok", "finalize_ran": True}]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    import scripts.chat_history_rehydrate as rehydrate_mod

    persisted_rows = [
        {"prompt": "hey Orion, quick check-in", "response": "Hi Juniper -- here and steady."},
    ]

    def _fake_fetch_recent_rows(session_id, *, max_turns, max_age_hours):
        assert session_id == "sid-orion-thin"
        return persisted_rows

    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", _fake_fetch_recent_rows)

    payload = {
        "mode": "orion",
        # Shape app.js's WebSocket-down HTTP fallback actually sends: just the
        # current turn, no prior thread.
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-orion-thin", no_write=True))

    assert out["type"] == "final"
    continuity_messages = captured_kwargs.get("continuity_messages")
    assert continuity_messages == [
        {"role": "user", "content": "hey Orion, quick check-in"},
        {"role": "assistant", "content": "Hi Juniper -- here and steady."},
        {"role": "user", "content": "hi"},
    ], "thin client-supplied history must be backfilled from persisted chat_history_log rows"


def test_handle_chat_request_orion_mode_rehydrate_disabled(monkeypatch) -> None:
    """HUB_HISTORY_REHYDRATE_ENABLED=false must skip the DB round-trip entirely,
    same opt-out the WebSocket connect path already respects."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main
    from orion.hub import turn_orchestrator

    monkeypatch.setattr(api_routes.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "ORION_HARNESS_GOVERNOR_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_ENABLED", False, raising=False)

    class _Bus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _Bus(), raising=False)
    monkeypatch.setattr(hub_main, "harness_step_relay", None, raising=False)
    monkeypatch.setattr(hub_main, "rpc_bus", None, raising=False)

    captured_kwargs: dict = {}

    async def _fake_execute_unified_turn(**kwargs):
        captured_kwargs.update(kwargs)
        return [{"type": "final", "correlation_id": "corr-no-rehydrate", "llm_response": "ok", "finalize_ran": True}]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    import scripts.chat_history_rehydrate as rehydrate_mod

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("fetch_recent_rows must not be called when rehydrate is disabled")

    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", _fail_if_called)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hi"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-orion-disabled", no_write=True))

    assert out["type"] == "final"
    assert captured_kwargs.get("continuity_messages") == [{"role": "user", "content": "hi"}]


# ─── Direct unit tests for _augment_thin_history_from_persisted ──────────────
#
# Exercised in isolation (not through the full handle_chat_request pipeline)
# because the helper is mode-agnostic -- it is called identically from both
# the mode=="orion" branch and the sibling default ("brain") branch, and the
# default branch's downstream substrate/pre-turn-appraisal wiring has
# pre-existing, unrelated environment flakiness in this sandbox (confirmed:
# the same failures reproduce on unmodified main) that would make a
# full-pipeline test of that branch noisy without adding real coverage of
# this helper's own logic.


def test_augment_thin_history_skips_when_real_thread_present(monkeypatch) -> None:
    """>1 real turn already present must not touch the DB at all."""
    import scripts.api_routes as api_routes

    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_ENABLED", True, raising=False)

    import scripts.chat_history_rehydrate as rehydrate_mod

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("fetch_recent_rows must not be called when real history is already present")

    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", _fail_if_called)

    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    out = asyncio.run(
        api_routes._augment_thin_history_from_persisted(
            session_id="sid-has-thread", user_messages=messages, max_turns=10
        )
    )
    assert out is messages


def test_augment_thin_history_fails_open_on_fetch_error(monkeypatch) -> None:
    """A DB error must never break the turn -- fall back to the caller's own messages."""
    import scripts.api_routes as api_routes

    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_ENABLED", True, raising=False)

    import scripts.chat_history_rehydrate as rehydrate_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", _boom)

    messages = [{"role": "user", "content": "hi"}]
    out = asyncio.run(
        api_routes._augment_thin_history_from_persisted(
            session_id="sid-db-down", user_messages=messages, max_turns=10
        )
    )
    assert out == messages


def test_augment_thin_history_no_op_when_nothing_persisted(monkeypatch) -> None:
    """A genuinely new session (no persisted rows) must leave messages untouched,
    not inject an empty prefix."""
    import scripts.api_routes as api_routes

    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_ENABLED", True, raising=False)

    import scripts.chat_history_rehydrate as rehydrate_mod

    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", lambda *a, **k: [])

    messages = [{"role": "user", "content": "hi"}]
    out = asyncio.run(
        api_routes._augment_thin_history_from_persisted(
            session_id="sid-brand-new", user_messages=messages, max_turns=10
        )
    )
    assert out == messages


def test_augment_thin_history_default_mode_branch_wiring(monkeypatch) -> None:
    """The sibling (non-orion / 'brain') branch of handle_chat_request must build
    continuity_messages the same augmented way as the orion-mode branch --
    confirms both call sites route through the helper, not just the orion one."""
    import scripts.api_routes as api_routes

    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "HUB_HISTORY_REHYDRATE_MAX_AGE_HOURS", 48.0, raising=False)

    import scripts.chat_history_rehydrate as rehydrate_mod

    persisted_rows = [{"prompt": "earlier today", "response": "still here"}]
    monkeypatch.setattr(rehydrate_mod, "fetch_recent_rows", lambda *a, **k: persisted_rows)

    captured: dict = {}
    real_build_chat_request = api_routes.build_chat_request

    def _spy_build_chat_request(*, messages, **kwargs):
        captured["messages"] = messages
        return real_build_chat_request(messages=messages, **kwargs)

    monkeypatch.setattr(api_routes, "build_chat_request", _spy_build_chat_request)

    class _FakeCortex:
        async def chat(self, req, correlation_id=None):
            from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult

            result = CortexClientResult(
                ok=True, mode="brain", verb="chat_general", status="success",
                final_text="ok", correlation_id=correlation_id or "corr-brain-thin",
            )
            return CortexChatResult(cortex_result=result, final_text="ok")

    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hi"}]}
    asyncio.run(api_routes.handle_chat_request(_FakeCortex(), payload, "sid-brain-thin", no_write=True))

    assert captured.get("messages") == [
        {"role": "user", "content": "earlier today"},
        {"role": "assistant", "content": "still here"},
        {"role": "user", "content": "hi"},
    ], "default-mode branch must also backfill thin history via _augment_thin_history_from_persisted"
