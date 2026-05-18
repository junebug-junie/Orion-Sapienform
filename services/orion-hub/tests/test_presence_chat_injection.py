from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HUB_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
from scripts.api_routes import handle_chat_request


class _Store:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)


class _FakeCortexClient:
    def __init__(self) -> None:
        self.requests = []

    async def chat(self, req, correlation_id=None):
        self.requests.append((req, correlation_id))
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_quick",
            status="success",
            final_text="ok",
            memory_used=False,
            recall_debug={},
            steps=[],
            correlation_id=correlation_id or str(uuid4()),
            metadata={},
        )
        return CortexChatResult(cortex_result=result, final_text=result.final_text)


def test_handle_chat_request_injects_session_presence(monkeypatch) -> None:
    store = _Store()
    store.values["sid-presence-chat"] = {
        "audience_mode": "kid_present",
        "requestor": {"display_name": "Juniper"},
    }
    monkeypatch.setitem(
        sys.modules,
        "scripts.main",
        types.SimpleNamespace(presence_context_store=store, presence_state=None),
    )
    client = _FakeCortexClient()
    payload = {
        "mode": "brain",
        "verbs": ["chat_quick"],
        "messages": [{"role": "user", "content": "explain GPUs simply"}],
    }
    result = asyncio.run(handle_chat_request(client, payload, "sid-presence-chat", no_write=True))
    assert not result.get("error")
    req, _ = client.requests[0]
    assert req.metadata["presence_context"]["audience_mode"] == "kid_present"
    assert result["routing_debug"]["presence_context_present"] is True
    assert result["routing_debug"]["audience_mode"] == "kid_present"
