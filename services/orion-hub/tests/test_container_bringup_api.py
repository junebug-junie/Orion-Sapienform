"""Tests for POST /api/debug/container-bringup (skills.docker.compose_service_bringup.v1 direct dispatch).

This is a small dedicated endpoint, not the Skill Runner catalogue pipeline -- see
docs/operator_skill_prompt_catalogue.md's "Container bring-up" section for why.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

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

from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult  # noqa: E402
from scripts.api_routes import ContainerBringupRequest, api_debug_container_bringup  # noqa: E402


class _FakeCortexClient:
    def __init__(self, *, skill_result: dict | None = None, ok: bool = True) -> None:
        self.requests = []
        self._skill_result = skill_result or {}
        self._ok = ok

    async def chat(self, req, correlation_id=None, rpc_timeout_sec=None):
        self.requests.append((req, correlation_id, rpc_timeout_sec))
        result = CortexClientResult(
            ok=self._ok,
            mode="brain",
            verb=req.verb or "chat_general",
            status="success" if self._ok else "fail",
            final_text="ok",
            memory_used=False,
            recall_debug={},
            steps=[],
            correlation_id=correlation_id or str(uuid4()),
            metadata={"skill_name": req.verb, "skill_result": self._skill_result},
        )
        return CortexChatResult(cortex_result=result, final_text=result.final_text)


def test_dispatch_threads_service_into_skill_args(monkeypatch) -> None:
    import scripts.main as hub_main

    client = _FakeCortexClient(skill_result={"service": "orion-widget", "status": "healthy"})
    monkeypatch.setattr(hub_main, "cortex_client", client)

    result = asyncio.run(api_debug_container_bringup(ContainerBringupRequest(service="orion-widget")))

    assert result["ok"] is True
    assert result["service"] == "orion-widget"
    assert result["skill_result"]["status"] == "healthy"
    req, _corr, _timeout = client.requests[0]
    assert req.verb == "skills.docker.compose_service_bringup.v1"
    assert req.metadata["skill_args"] == {"service": "orion-widget"}


def test_dispatch_rejects_empty_service(monkeypatch) -> None:
    import scripts.main as hub_main

    monkeypatch.setattr(hub_main, "cortex_client", _FakeCortexClient())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_debug_container_bringup(ContainerBringupRequest(service="   ")))
    assert exc_info.value.status_code == 422


def test_dispatch_raises_503_when_cortex_client_unavailable(monkeypatch) -> None:
    import scripts.main as hub_main

    monkeypatch.setattr(hub_main, "cortex_client", None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_debug_container_bringup(ContainerBringupRequest(service="orion-widget")))
    assert exc_info.value.status_code == 503


def test_dispatch_surfaces_not_ok_result(monkeypatch) -> None:
    import scripts.main as hub_main

    client = _FakeCortexClient(skill_result={"status": "blocked", "policy_blocked": True}, ok=False)
    monkeypatch.setattr(hub_main, "cortex_client", client)

    result = asyncio.run(api_debug_container_bringup(ContainerBringupRequest(service="orion-widget")))
    assert result["ok"] is False
    assert result["skill_result"]["status"] == "blocked"
