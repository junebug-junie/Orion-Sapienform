from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request


HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

import importlib

hub_api_routes = importlib.import_module("scripts.api_routes")


def _request(method: str = "GET", body: bytes = b"") -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": "/api/knowledge/status",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope, receive=receive)


def test_knowledge_forge_alias_routes_forward_expected_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_proxy(path: str, request: Request):
        calls.append(path)
        return {"ok": True}

    monkeypatch.setattr(hub_api_routes, "_proxy_knowledge_forge_request", fake_proxy)

    asyncio.run(hub_api_routes.proxy_knowledge_forge_health(_request("GET")))
    asyncio.run(hub_api_routes.proxy_knowledge_forge_status(_request("GET")))
    asyncio.run(
        hub_api_routes.proxy_knowledge_forge_context_packs_compile(
            _request("POST", b'{"task":"review","scope":"claims"}')
        )
    )
    asyncio.run(
        hub_api_routes.proxy_knowledge_forge_ideation_run(
            _request("POST", b'{"task":"test"}')
        )
    )
    asyncio.run(
        hub_api_routes.proxy_knowledge_forge_sources_ingest(
            _request("POST", b'{"path":"/tmp/x.md","source_id":"source:x"}')
        )
    )
    asyncio.run(hub_api_routes.proxy_knowledge_forge("v1/custom-endpoint", _request("GET")))

    assert calls == [
        "health",
        "v1/status",
        "v1/context-packs/compile",
        "v1/ideation/run",
        "v1/sources/ingest",
        "v1/custom-endpoint",
    ]


def test_knowledge_forge_proxy_returns_controlled_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_proxy(path: str, request: Request):
        raise HTTPException(status_code=502, detail="Knowledge Forge proxy request failed")

    monkeypatch.setattr(hub_api_routes, "_proxy_knowledge_forge_request", fake_proxy)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(hub_api_routes.proxy_knowledge_forge_status(_request("GET")))

    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "Knowledge Forge proxy request failed"


def test_knowledge_forge_proxy_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_api_routes.settings, "KNOWLEDGE_FORGE_BASE_URL", "")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(hub_api_routes._proxy_knowledge_forge_request("v1/status", _request("GET")))

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Knowledge Forge base URL not configured"
