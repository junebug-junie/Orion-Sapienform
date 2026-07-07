from __future__ import annotations

import asyncio
import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import EmbodimentWorker
from orion.schemas.embodiment import WorldPerceptionV1


def _worker(*, unified: bool = True) -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._speaking_conversations = set()
    w._opened_conversations = set()
    w._settings = SimpleNamespace(
        speech_enabled=True,
        speech_lane="quick",
        speech_verb="chat_quick",
        speech_timeout_sec=30.0,
        cortex_request_channel="orion:cortex:exec:request",
        cortex_result_prefix="orion:exec:result",
        speech_unified_enabled=unified,
        hub_chat_url="http://orion-athena-hub:8080/api/chat",
        unified_timeout_sec=120.0,
        unified_session_prefix="aitown",
    )
    return w


def _perception_in_convo() -> WorldPerceptionV1:
    return WorldPerceptionV1(
        player_id="orion",
        position={"x": 0.0, "y": 0.0},
        nearby_players=[{"player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}],
        active_conversation={
            "conversation_id": "conv1",
            "status": "participating",
            "participants": ["orion", "p9"],
            "messages": [{"author_id": "p9", "author": "Juniper", "text": "hey Orion"}],
        },
    )


class _FakeResp:
    """Minimal urlopen context-manager stand-in returning a JSON body."""

    def __init__(self, payload) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _final_frame(text: str) -> dict:
    return {"type": "final", "correlation_id": "c1", "mode": "orion", "llm_response": text, "finalize_ran": True}


def test_unified_success_injects_hub_response_and_skips_quick():
    w = _worker(unified=True)
    quick = AsyncMock(return_value="QUICK_SHOULD_NOT_BE_USED")
    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen", return_value=_FakeResp(_final_frame("Hello from unified"))) as uo, \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Hello from unified"
    quick.assert_not_awaited()
    # session_id is built from the active conversation id.
    posted = uo.call_args.args[0]
    body = json.loads(posted.data.decode("utf-8"))
    assert body["mode"] == "orion"
    assert body["session_id"] == "aitown:conv1"
    assert body["messages"][0]["role"] == "user"


def test_unified_timeout_falls_back_to_quick():
    w = _worker(unified=True)
    quick = AsyncMock(return_value="Quick fallback reply")
    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick fallback reply"
    quick.assert_awaited_once()


def test_unified_turn_error_frame_falls_back_to_quick():
    w = _worker(unified=True)
    quick = AsyncMock(return_value="Quick fallback reply")
    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen", return_value=_FakeResp({"type": "turn_error", "detail": "boom"})), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick fallback reply"
    quick.assert_awaited_once()


def test_unified_turn_deferred_frame_falls_back_to_quick():
    w = _worker(unified=True)
    quick = AsyncMock(return_value="Quick fallback reply")
    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen", return_value=_FakeResp({"type": "turn_deferred", "reason": "cooldown"})), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick fallback reply"
    quick.assert_awaited_once()


def test_unified_empty_llm_response_falls_back_to_quick():
    w = _worker(unified=True)
    quick = AsyncMock(return_value="Quick fallback reply")
    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen", return_value=_FakeResp(_final_frame("   "))), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick fallback reply"
    quick.assert_awaited_once()


def test_unified_bad_json_falls_back_to_quick():
    w = _worker(unified=True)
    quick = AsyncMock(return_value="Quick fallback reply")

    class _BadResp(_FakeResp):
        def __init__(self):
            self._body = b"not json"

    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen", return_value=_BadResp()), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick fallback reply"
    quick.assert_awaited_once()


def test_unified_disabled_uses_quick_directly():
    w = _worker(unified=False)
    quick = AsyncMock(return_value="Quick direct reply")
    with patch.object(w, "_request_utterance_quick", new=quick), \
         patch("urllib.request.urlopen") as uo, \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick direct reply"
    quick.assert_awaited_once()
    uo.assert_not_called()
