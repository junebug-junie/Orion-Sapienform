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


def _request(method: str = "GET", path: str = "/aitown/index.html") -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "query_string": b"",
    }
    return Request(scope, receive=receive)


@pytest.mark.asyncio
async def test_aitown_proxy_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_api_routes.settings, "HUB_AITOWN_ENABLED", False, raising=False)
    with pytest.raises(HTTPException) as exc:
        await hub_api_routes.aitown_proxy("index.html", _request())
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_aitown_proxy_forwards_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_proxy(path: str, request: Request):
        calls.append(path)
        from starlette.responses import Response

        return Response(content=b"ok", status_code=200)

    monkeypatch.setattr(hub_api_routes.settings, "HUB_AITOWN_ENABLED", True, raising=False)
    monkeypatch.setattr(hub_api_routes.settings, "HUB_AITOWN_UI_URL", "http://127.0.0.1:5173", raising=False)
    monkeypatch.setattr(hub_api_routes, "_proxy_aitown_request", fake_proxy)

    resp = await hub_api_routes.aitown_proxy("assets/app.js", _request())
    assert resp.status_code == 200
    assert calls == ["assets/app.js"]


def test_upstream_aitown_url_uses_vite_base() -> None:
    assert (
        hub_api_routes._upstream_aitown_url("http://127.0.0.1:5173", "")
        == "http://127.0.0.1:5173/ai-town/"
    )
    assert (
        hub_api_routes._upstream_aitown_url("http://127.0.0.1:5173", "@vite/client")
        == "http://127.0.0.1:5173/ai-town/@vite/client"
    )
