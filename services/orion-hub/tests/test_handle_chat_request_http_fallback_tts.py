"""2026-08-27: the HTTP fallback path for `mode=="orion"` now speaks too.

Real incident, corr=11215a1b-d3c8-438b-901b-0d6cadf3d637: a live chat turn
came through this endpoint -- not `websocket_handler.py`'s WebSocket loop --
and got neither text nor voice back. Root cause of the missing voice half:
this HTTP route has NEVER had any TTS wiring, in any mode, since it existed.
It gets reached whenever `app.js`'s own fallback (built 2026-08-22, for a WS
connection dying while a tab is open -- e.g. a Hub redeploy) sends a turn
via `POST /api/chat` because the live WebSocket is down.

Unlike the WS lane (PR #1905's `dispatch_tts_reply`, fire-and-forget via a
queue a `drain_task` relays), this is a single request/response HTTP cycle
with nothing to push audio into afterward -- so synthesis is SYNCHRONOUS,
awaited before the response returns, with the audio riding in the same
JSON body as the text.

NOTE: scripts.api_routes/scripts.main must be imported fresh inside each
test body, not at module top level -- conftest.py's autouse
_hub_service_isolation fixture clears scripts.*/app.* from sys.modules
before every test, so a module-level import here would bind to a stale
module/settings singleton the monkeypatches below never touch. Matches
test_handle_chat_request_orion_mode_degraded.py's own established
convention for this exact file.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")


def _wire_common(
    monkeypatch, api_routes, hub_main, *, tts_client, final_text="a real answer", tts_timeout_sec=5.0
):
    from orion.hub import turn_orchestrator

    monkeypatch.setattr(api_routes.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "ORION_HARNESS_GOVERNOR_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "HUB_TTS_TIMEOUT_SEC", tts_timeout_sec, raising=False)

    class _Bus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _Bus(), raising=False)
    monkeypatch.setattr(hub_main, "harness_step_relay", None, raising=False)
    monkeypatch.setattr(hub_main, "rpc_bus", None, raising=False)
    monkeypatch.setattr(hub_main, "tts_client", tts_client, raising=False)

    async def _fake_execute_unified_turn(**_kwargs):
        return [
            {
                "type": "final",
                "correlation_id": "corr-http-tts",
                "llm_response": final_text,
                "finalize_ran": True,
            }
        ]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)


class _FakeTTSClient:
    def __init__(self, *, result=None, exc=None, delay=0.0):
        self._result = result
        self._exc = exc
        self._delay = delay
        self.calls = []

    async def speak(self, request):
        self.calls.append(request)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._result


class _FakeTTSResult:
    def __init__(self, audio_b64="ZmFrZWF1ZGlv", content_type="audio/wav", duration_sec=1.5, metadata=None):
        self.audio_b64 = audio_b64
        self.content_type = content_type
        self.duration_sec = duration_sec
        self.metadata = metadata or {"backend": "coqui"}


def test_http_fallback_synthesizes_and_returns_audio(monkeypatch) -> None:
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult())
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True))

    assert out["llm_response"] == "a real answer"
    assert out["audio_response"] == "ZmFrZWF1ZGlv"
    assert out["tts_source_text"] == "a real answer"
    assert out["tts_meta"]["content_type"] == "audio/wav"
    assert "tts_error" not in out
    assert len(client.calls) == 1
    assert client.calls[0].text == "a real answer"


def test_http_fallback_respects_disable_tts(monkeypatch) -> None:
    """The one client-facing toggle that must still work identically to the
    WS lane -- a caller that explicitly opted out of voice must not get it
    anyway just because this path can now speak."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult())
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client)

    payload = {
        "mode": "orion",
        "messages": [{"role": "user", "content": "hello"}],
        "disable_tts": True,
    }
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True))

    assert out["llm_response"] == "a real answer"
    assert "audio_response" not in out
    assert client.calls == []


def test_http_fallback_omits_audio_when_no_tts_client_configured(monkeypatch) -> None:
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    _wire_common(monkeypatch, api_routes, hub_main, tts_client=None)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True))

    assert out["llm_response"] == "a real answer"
    assert "audio_response" not in out
    assert "tts_error" not in out


def test_http_fallback_surfaces_a_tts_timeout_without_losing_the_text_reply(monkeypatch) -> None:
    """A slow/hung synthesis must not take the whole turn down with it --
    the text half of the reply, which already succeeded, must still reach
    the caller."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult(), delay=999)
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client, tts_timeout_sec=0.05)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True))

    assert out["llm_response"] == "a real answer"
    assert "audio_response" not in out
    assert "timed out" in out["tts_error"]


def test_http_fallback_surfaces_a_tts_exception_without_losing_the_text_reply(monkeypatch) -> None:
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(exc=RuntimeError("synthesis backend unreachable"))
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True))

    assert out["llm_response"] == "a real answer"
    assert "audio_response" not in out
    assert out["tts_error"] == "synthesis backend unreachable"


def test_http_fallback_skips_synthesis_when_client_already_disconnected(monkeypatch) -> None:
    """Review finding, 2026-08-27: this fallback exists precisely for
    Hub-wide WS outages, when many clients can hit this route at once. A
    client that reloads/navigates away mid-wait must not hold a request
    slot + TTS-backend RPC slot for the full HUB_TTS_TIMEOUT_SEC for
    nobody -- checked via the real FastAPI Request.is_disconnected(), not
    just accepted as an unavoidable cost."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult())
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client)

    class _DisconnectedRequest:
        async def is_disconnected(self):
            return True

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(
        api_routes.handle_chat_request(
            object(), payload, "sid-http-tts", http_request=_DisconnectedRequest(), no_write=True
        )
    )

    assert out["llm_response"] == "a real answer"
    assert "audio_response" not in out
    assert client.calls == [], "synthesis must not run once the client is confirmed gone"


def test_http_fallback_still_synthesizes_when_request_object_is_unavailable(monkeypatch) -> None:
    """The disconnect check is best-effort, gated on http_request actually
    being supplied -- existing/non-HTTP callers of handle_chat_request
    (direct test invocation, or a future non-FastAPI caller) must not lose
    TTS just because they have no Request object to check."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult())
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(
        api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True)
    )

    assert out["audio_response"] == "ZmFrZWF1ZGlv"
    assert len(client.calls) == 1


def test_http_fallback_a_broken_disconnect_check_does_not_block_synthesis(monkeypatch) -> None:
    """The disconnect check itself failing must not be treated as proof
    the client is gone -- a client that is very likely still there must
    not lose its voice reply to a broken probe."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult())
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client)

    class _BrokenRequest:
        async def is_disconnected(self):
            raise RuntimeError("connection state unavailable")

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(
        api_routes.handle_chat_request(
            object(), payload, "sid-http-tts", http_request=_BrokenRequest(), no_write=True
        )
    )

    assert out["audio_response"] == "ZmFrZWF1ZGlv"
    assert len(client.calls) == 1


def test_http_fallback_omits_audio_when_final_text_is_empty(monkeypatch) -> None:
    """Nothing to speak -- must not call the client at all, matching the
    WS lane's own dispatch_tts_reply gate (a whitespace/empty reply is not
    real text)."""
    import scripts.api_routes as api_routes
    import scripts.main as hub_main

    client = _FakeTTSClient(result=_FakeTTSResult())
    _wire_common(monkeypatch, api_routes, hub_main, tts_client=client, final_text="   ")

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-http-tts", no_write=True))

    assert "audio_response" not in out
    assert client.calls == []
