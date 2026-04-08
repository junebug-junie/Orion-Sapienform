from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from orion.schemas.registry import _REGISTRY


ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = ROOT / "services" / "orion-dream"
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.main import app  # noqa: E402


def _channel_entry(name: str) -> dict:
    raw = yaml.safe_load(CHANNELS_YAML.read_text()) or {}
    channels = raw.get("channels", [])
    for entry in channels:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise AssertionError(f"Missing channel catalog entry: {name}")


def test_dream_trigger_channel_and_schema_are_registered() -> None:
    entry = _channel_entry("orion:dream:trigger")
    result_entry = _channel_entry("orion:dream:log")

    assert entry["schema_id"] == "DreamTriggerPayload"
    assert result_entry["schema_id"] == "DreamResultV1"
    assert "DreamTriggerPayload" in _REGISTRY
    assert "DreamInternalTriggerV1" in _REGISTRY
    assert "DreamResultV1" in _REGISTRY


def test_post_dreams_run_publishes_trigger(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeBus:
        def __init__(self, url: str):
            self.url = url

        async def connect(self) -> None:
            return None

        async def publish(self, channel: str, env: object) -> None:
            calls.append((channel, env))

        async def close(self) -> None:
            return None

    monkeypatch.setattr("app.main.OrionBusAsync", FakeBus)
    monkeypatch.setattr(
        "app.main.enforcer.entry_for",
        lambda channel: {"name": channel, "schema_id": "DreamTriggerPayload"},
    )

    with TestClient(app) as client:
        resp = client.post("/dreams/run", params={"mode": "lucid"})

    assert resp.status_code == 200
    assert resp.json() == {"status": "triggered", "mode": "lucid"}
    assert len(calls) == 1
    channel, env = calls[0]
    assert channel == "orion:dream:trigger"
    assert getattr(env, "kind") == "dream.trigger"
    assert getattr(env, "payload") == {"mode": "lucid"}


def test_post_dreams_run_returns_actionable_publish_error(monkeypatch) -> None:
    class FakeBus:
        def __init__(self, url: str):
            self.url = url

        async def connect(self) -> None:
            return None

        async def publish(self, channel: str, env: object) -> None:
            raise ValueError("Channel not found in catalog: orion:dream:trigger")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("app.main.OrionBusAsync", FakeBus)
    monkeypatch.setattr("app.main.enforcer.entry_for", lambda channel: None)

    with TestClient(app) as client:
        resp = client.post("/dreams/run")

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error"] == "dream_trigger_publish_failed"
    assert detail["channel"] == "orion:dream:trigger"
    assert detail["kind"] == "dream.trigger"
    assert "channels.yaml" in detail["hint"]
    assert "registry.py" in detail["hint"]
