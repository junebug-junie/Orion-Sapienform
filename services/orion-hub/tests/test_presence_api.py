from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

HUB_ROOT = Path(__file__).resolve().parents[1]
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

hub_api_routes = importlib.import_module("scripts.api_routes")


class _Store:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, payload):
        self.values[key] = dict(payload)
        return self.values[key]

    def clear(self, key):
        self.values.pop(key, None)


def test_presence_roundtrip(monkeypatch):
    store = _Store()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(presence_context_store=store, presence_state=None))
    payload = hub_api_routes.api_presence_set(
        {"audience_mode": "kid_present", "requestor": {"display_name": "Juniper"}},
        x_orion_session_id="sid-1",
    )
    assert payload["audience_mode"] == "kid_present"
    got = hub_api_routes.api_presence(x_orion_session_id="sid-1")
    assert got["audience_mode"] == "kid_present"
    cleared = hub_api_routes.api_presence_clear(x_orion_session_id="sid-1")
    assert cleared["audience_mode"] == "solo"


def test_situation_status_and_brief(monkeypatch):
    store = _Store()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(presence_context_store=store, presence_state=None))
    status = hub_api_routes.api_situation_status()
    assert status["enabled"] is True
    assert status["hub_enabled"] is True
    assert status["reason"] is None
    brief = hub_api_routes.api_situation_brief(x_orion_session_id="sid-2")
    assert brief["kind"] == "situation.brief.v1"
    assert isinstance(brief.get("time"), dict)
    assert isinstance(brief.get("conversation_phase"), dict)
    assert isinstance(brief.get("presence"), dict)


def test_situation_status_and_brief_disabled(monkeypatch):
    store = _Store()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(presence_context_store=store, presence_state=None))
    monkeypatch.setattr(hub_api_routes.settings, "ORION_SITUATION_ENABLED", False)
    status = hub_api_routes.api_situation_status()
    assert status["enabled"] is False
    assert status["hub_enabled"] is False
    assert status["reason"] == "disabled_by_config"
    brief = hub_api_routes.api_situation_brief(x_orion_session_id="sid-disabled")
    assert brief["enabled"] is False
    assert brief["reason"] == "disabled_by_config"


def test_presence_invalid_payload_is_422(monkeypatch):
    store = _Store()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(presence_context_store=store, presence_state=None))
    with pytest.raises(HTTPException) as exc:
        hub_api_routes.api_presence_set(None, x_orion_session_id="sid-3")  # type: ignore[arg-type]
    assert exc.value.status_code == 422
