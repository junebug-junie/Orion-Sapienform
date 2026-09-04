"""Tests for scripts/field_digester_client.py (orion-field-digester /health client)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import field_digester_client  # noqa: E402


class _FakeResponse:
    def __init__(self, *, status=200, json_result=None, json_error=None):
        self.status = status
        self._json_result = json_result
        self._json_error = json_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_result


class _FakeSession:
    def __init__(self, *a, response=None, get_error=None, **k):
        self._response = response
        self._get_error = get_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, *a, **k):
        if self._get_error is not None:
            raise self._get_error
        return self._response


def _patch_session(monkeypatch, **kwargs):
    monkeypatch.setattr(
        field_digester_client.aiohttp, "ClientSession", lambda *a, **k: _FakeSession(**kwargs)
    )


@pytest.fixture(autouse=True)
def _configured_base_url(monkeypatch):
    monkeypatch.setattr(field_digester_client.settings, "FIELD_DIGESTER_BASE_URL", "http://x:8116")


@pytest.mark.asyncio
async def test_fetch_health_returns_the_parsed_body(monkeypatch):
    _patch_session(monkeypatch, response=_FakeResponse(json_result={"status": "ok"}))
    result = await field_digester_client.fetch_health()
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_fetch_health_wraps_connection_failure(monkeypatch):
    import aiohttp

    _patch_session(monkeypatch, get_error=aiohttp.ClientConnectionError("boom"))
    with pytest.raises(field_digester_client.FieldDigesterClientError):
        await field_digester_client.fetch_health()


@pytest.mark.asyncio
async def test_fetch_health_wraps_malformed_json(monkeypatch):
    """Review finding (2026-09-04): a truncated/malformed body with a
    correct content-type raises json.JSONDecodeError, not aiohttp.ClientError
    -- must not escape as an unhandled exception."""
    bad_json = json.JSONDecodeError("Expecting value", "not json", 0)
    _patch_session(monkeypatch, response=_FakeResponse(json_error=bad_json))
    with pytest.raises(field_digester_client.FieldDigesterClientError):
        await field_digester_client.fetch_health()


@pytest.mark.asyncio
async def test_fetch_health_raises_on_http_error_status(monkeypatch):
    _patch_session(monkeypatch, response=_FakeResponse(status=500, json_result={"detail": "boom"}))
    with pytest.raises(field_digester_client.FieldDigesterClientError):
        await field_digester_client.fetch_health()


@pytest.mark.asyncio
async def test_fetch_health_raises_on_non_object_payload(monkeypatch):
    _patch_session(monkeypatch, response=_FakeResponse(json_result=["not", "a", "dict"]))
    with pytest.raises(field_digester_client.FieldDigesterClientError):
        await field_digester_client.fetch_health()


@pytest.mark.asyncio
async def test_fetch_health_raises_when_base_url_unconfigured(monkeypatch):
    monkeypatch.setattr(field_digester_client.settings, "FIELD_DIGESTER_BASE_URL", "")
    with pytest.raises(field_digester_client.FieldDigesterClientError):
        await field_digester_client.fetch_health()
