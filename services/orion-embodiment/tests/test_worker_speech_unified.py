from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import EmbodimentWorker
from orion.schemas.embodiment import WorldPerceptionV1


def _worker(*, grounded_enabled: bool = True) -> EmbodimentWorker:
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
        speech_unified_enabled=grounded_enabled,
        speech_hub_llm_route="chat",
        unified_timeout_sec=45.0,
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


def test_grounded_cortex_success_skips_quick():
    w = _worker(grounded_enabled=True)
    grounded = AsyncMock(return_value="Hello from grounded_small")
    quick = AsyncMock(return_value="QUICK_SHOULD_NOT_BE_USED")
    with patch.object(w, "_request_utterance_cortex", new=grounded), \
         patch.object(w, "_request_utterance_quick", new=quick), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Hello from grounded_small"
    quick.assert_not_awaited()
    grounded.assert_awaited_once()
    kwargs = grounded.await_args.kwargs
    assert kwargs["verb"] == "chat_general"
    assert kwargs["lane"] == "chat"
    assert kwargs["hub_chat_lane"] == "grounded_small"


def test_grounded_cortex_error_falls_back_to_quick():
    w = _worker(grounded_enabled=True)
    grounded = AsyncMock(side_effect=TimeoutError("slow"))
    quick = AsyncMock(return_value="Quick fallback reply")
    with patch.object(w, "_request_utterance_cortex", new=grounded), \
         patch.object(w, "_request_utterance_quick", new=quick), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick fallback reply"
    quick.assert_awaited_once()


def test_grounded_disabled_uses_quick_directly():
    w = _worker(grounded_enabled=False)
    grounded = AsyncMock(return_value="nope")
    quick = AsyncMock(return_value="Quick direct reply")
    with patch.object(w, "_request_utterance_cortex", new=grounded), \
         patch.object(w, "_request_utterance_quick", new=quick), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Quick direct reply"
    grounded.assert_not_awaited()
    quick.assert_awaited_once()
