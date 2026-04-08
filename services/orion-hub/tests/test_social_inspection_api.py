from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import aiohttp
import pytest
from fastapi import HTTPException


HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    'CHANNEL_VOICE_TRANSCRIPT': 'orion:voice:transcript',
    'CHANNEL_VOICE_LLM': 'orion:voice:llm',
    'CHANNEL_VOICE_TTS': 'orion:voice:tts',
    'CHANNEL_COLLAPSE_INTAKE': 'orion:collapse:intake',
    'CHANNEL_COLLAPSE_TRIAGE': 'orion:collapse:triage',
}.items():
    os.environ.setdefault(key, value)

import importlib
hub_api_routes = importlib.import_module('scripts.api_routes')


@pytest.fixture(autouse=True)
def _social_memory_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_api_routes.settings, "SOCIAL_MEMORY_BASE_URL", "http://orion-social-memory:8765")


def test_api_social_memory_inspection_proxies_existing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "platform": "callsyne",
        "room_id": "room-alpha",
        "participant_id": "peer-1",
        "summary": "6 sections · 3 context candidates selected · 1 softened · 1 excluded",
        "sections": [],
    }

    async def fake_fetch(path: str, params: dict[str, object]) -> dict[str, object]:
        assert path == "/inspection"
        assert params == {"platform": "callsyne", "room_id": "room-alpha", "participant_id": "peer-1"}
        return expected

    monkeypatch.setattr(hub_api_routes, "_fetch_social_memory", fake_fetch)

    result = asyncio.run(
        hub_api_routes.api_social_memory_inspection(
            platform="callsyne",
            room_id="room-alpha",
            participant_id="peer-1",
        )
    )

    assert result == expected


def test_api_social_memory_inspection_surfaces_upstream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    request_info = object()
    error = aiohttp.ClientResponseError(
        request_info=request_info,  # type: ignore[arg-type]
        history=(),
        status=404,
        message="inspection_not_found",
        headers=None,
    )

    async def fake_fetch(path: str, params: dict[str, object]) -> dict[str, object]:
        raise error

    monkeypatch.setattr(hub_api_routes, "_fetch_social_memory", fake_fetch)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            hub_api_routes.api_social_memory_inspection(
                platform="callsyne",
                room_id="room-alpha",
                participant_id=None,
            )
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "inspection_not_found"
