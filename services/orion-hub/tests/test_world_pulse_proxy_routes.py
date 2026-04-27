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
        "path": "/api/world-pulse/latest",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope, receive=receive)


def test_world_pulse_alias_routes_forward_expected_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_proxy(path: str, request: Request):
        calls.append(path)
        return {"ok": True}

    monkeypatch.setattr(hub_api_routes, "_proxy_world_pulse_request", fake_proxy)

    asyncio.run(hub_api_routes.proxy_world_pulse_healthz(_request("GET")))
    asyncio.run(hub_api_routes.proxy_world_pulse_latest(_request("GET")))
    asyncio.run(hub_api_routes.proxy_world_pulse_run(_request("POST", b'{"dry_run":true}')))
    asyncio.run(hub_api_routes.proxy_world_pulse("api/world-pulse/runs", _request("GET")))

    assert calls == [
        "healthz",
        "api/world-pulse/latest",
        "api/world-pulse/run",
        "api/world-pulse/runs",
    ]


def test_world_pulse_proxy_returns_controlled_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_proxy(path: str, request: Request):
        raise HTTPException(status_code=502, detail="World Pulse proxy request failed")

    monkeypatch.setattr(hub_api_routes, "_proxy_world_pulse_request", fake_proxy)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(hub_api_routes.proxy_world_pulse_latest(_request("GET")))

    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "World Pulse proxy request failed"


def test_world_pulse_ui_has_shared_fallback_metric_helper() -> None:
    app_js = (HUB_ROOT / "static/js/app.js").read_text(encoding="utf-8")
    assert "function computeEffectiveWorldPulseMetrics(model)" in app_js
    assert "const metrics = computeEffectiveWorldPulseMetrics(data);" in app_js
    assert "Evidence: ${metrics.acceptedArticleCount} accepted articles / ${metrics.articleClusterCount} clusters" in app_js
