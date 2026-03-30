from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

from scripts.api_routes import api_debug_build, handle_chat_request
from scripts.websocket_handler import websocket_endpoint
from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
from fastapi import WebSocketDisconnect


class _FakeCortexClient:
    def __init__(self) -> None:
        self.requests = []

    async def chat(self, req, correlation_id=None):
        self.requests.append((req, correlation_id))
        workflow_request = (req.metadata or {}).get("workflow_request") or {}
        policy = workflow_request.get("execution_policy") or {}
        workflow_id = workflow_request.get("workflow_id")
        invocation_mode = policy.get("invocation_mode")
        schedule = policy.get("schedule") or {}
        summary = f"Workflow request accepted: {workflow_id} ({schedule.get('label') or 'scheduled'})."
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb=workflow_id or "chat_general",
            status="success",
            final_text=summary,
            memory_used=False,
            recall_debug={},
            steps=[],
            correlation_id=correlation_id or str(uuid4()),
            metadata={
                "workflow": {
                    "workflow_id": workflow_id,
                    "status": "scheduled" if invocation_mode == "scheduled" else "completed",
                    "scheduled": [f"req-1:{schedule.get('run_at_utc') or schedule.get('label') or 'scheduled'}"]
                    if invocation_mode == "scheduled"
                    else [],
                    "persisted": [],
                    "main_result": summary,
                    "execution_policy": policy,
                }
            },
        )
        return CortexChatResult(cortex_result=result, final_text=result.final_text)


class _FakeWebSocket:
    def __init__(self, payload: dict) -> None:
        self._inbound = [json.dumps(payload)]
        self.sent = []
        self.client_state = SimpleNamespace(name="CONNECTED")
        self.headers = {}
        self.client = SimpleNamespace(host="127.0.0.1", port=9000)

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        if self._inbound:
            return self._inbound.pop(0)
        self.client_state.name = "DISCONNECTED"
        raise WebSocketDisconnect()

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def test_http_chat_path_preserves_scheduled_workflow_policy(caplog) -> None:
    caplog.set_level("INFO")
    client = _FakeCortexClient()
    payload = {
        "mode": "auto",
        "session_id": "sid-http",
        "messages": [{"role": "user", "content": "Orion, would you schedule a self review for 2:46 PM?"}],
    }
    result = asyncio.run(handle_chat_request(client, payload, "sid-http", no_write=True))
    assert not result.get("error")
    req, _ = client.requests[0]
    policy = req.metadata["workflow_request"]["execution_policy"]
    assert req.metadata["workflow_request"]["workflow_id"] == "self_review"
    assert policy["invocation_mode"] == "scheduled"
    assert policy["schedule"]["kind"] == "one_shot"
    assert result["workflow"]["status"] == "scheduled"
    assert result["workflow"]["scheduled"]
    assert any("hub_workflow_request" in rec.message for rec in caplog.records)
    assert any("hub_workflow_response" in rec.message for rec in caplog.records)


def test_hub_debug_build_identity_surface() -> None:
    payload = api_debug_build()
    assert payload["hub"]["service"]
    assert payload["hub"]["version"]
    assert payload["hub"]["process_started_at"]
    assert "cortex_gateway_request_channel" in payload["downstream"]


def test_hub_ui_asset_version_helper_returns_non_empty_token(monkeypatch) -> None:
    import scripts.main as hub_main

    monkeypatch.setenv("HUB_UI_BUILD", "build-20260330")
    assert hub_main.build_hub_ui_asset_version() == "build-20260330"


def test_root_response_disables_html_caching() -> None:
    from scripts.api_routes import root

    response = asyncio.run(root())
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"


def test_websocket_chat_path_preserves_scheduled_workflow_policy(monkeypatch) -> None:
    import scripts.main as hub_main

    client = _FakeCortexClient()
    monkeypatch.setattr(hub_main, "bus", object())
    monkeypatch.setattr(hub_main, "cortex_client", client)
    monkeypatch.setattr(hub_main, "tts_client", None)
    monkeypatch.setattr(hub_main, "biometrics_cache", None)
    monkeypatch.setattr(hub_main, "notification_cache", None)
    monkeypatch.setattr(hub_main, "presence_state", None)

    ws = _FakeWebSocket(
        {
            "text_input": "Orion, would you schedule a self review for 2:46 PM?",
            "mode": "auto",
            "session_id": "sid-ws",
            "no_write": True,
        }
    )
    asyncio.run(websocket_endpoint(ws))

    req, _ = client.requests[0]
    policy = req.metadata["workflow_request"]["execution_policy"]
    assert req.metadata["workflow_request"]["workflow_id"] == "self_review"
    assert policy["invocation_mode"] == "scheduled"
    assert policy["schedule"]["kind"] == "one_shot"
    workflow_messages = [msg for msg in ws.sent if isinstance(msg, dict) and isinstance(msg.get("workflow"), dict)]
    assert workflow_messages
    assert workflow_messages[-1]["workflow"]["status"] == "scheduled"
