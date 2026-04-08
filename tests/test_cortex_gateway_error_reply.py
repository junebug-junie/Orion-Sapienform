from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_bus_client_module():
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "services" / "orion-cortex-gateway" / "app"

    pkg = types.ModuleType("orion_cortex_gateway")
    pkg.__path__ = [str(app_dir.parent)]
    subpkg = types.ModuleType("orion_cortex_gateway.app")
    subpkg.__path__ = [str(app_dir)]
    sys.modules.setdefault("orion_cortex_gateway", pkg)
    sys.modules.setdefault("orion_cortex_gateway.app", subpkg)

    settings_spec = importlib.util.spec_from_file_location("orion_cortex_gateway.app.settings", app_dir / "settings.py")
    settings_mod = importlib.util.module_from_spec(settings_spec)
    assert settings_spec and settings_spec.loader
    settings_spec.loader.exec_module(settings_mod)
    sys.modules["orion_cortex_gateway.app.settings"] = settings_mod

    mod_spec = importlib.util.spec_from_file_location("orion_cortex_gateway.app.bus_client", app_dir / "bus_client.py")
    mod = importlib.util.module_from_spec(mod_spec)
    assert mod_spec and mod_spec.loader
    mod_spec.loader.exec_module(mod)
    return mod


bus_mod = _load_bus_client_module()


class _Decoded:
    def __init__(self, env):
        self.ok = True
        self.envelope = env
        self.error = None


class _Codec:
    def __init__(self, env):
        self._env = env

    def decode(self, _data):
        return _Decoded(self._env)


class _Bus:
    def __init__(self, env):
        self.codec = _Codec(env)


class _Env:
    def __init__(self):
        self.kind = "cortex.gateway.chat.request"
        self.correlation_id = "11111111-1111-1111-1111-111111111111"
        self.source = {"name": "hub"}
        self.reply_to = "orion:cortex:gateway:result:11111111-1111-1111-1111-111111111111"
        self.causality_chain = []
        self.created_at = "2026-01-01T00:00:00Z"
        self.payload = {"prompt": "hello", "mode": "agent", "verb": "chat_general"}


def test_gateway_replies_when_orch_rpc_raises(monkeypatch):
    client = bus_mod.BusClient()
    client.bus = _Bus(_Env())
    captured = {}

    async def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    async def _capture(*, reply_to, correlation_id, causality_chain, payload):
        captured["reply_to"] = reply_to
        captured["correlation_id"] = correlation_id
        captured["payload"] = payload

    monkeypatch.setattr(client, "rpc_call_cortex_orch", _raise)
    monkeypatch.setattr(client, "_publish_gateway_reply", _capture)

    asyncio.run(client.handle_gateway_request({"data": b"x"}))

    assert captured["reply_to"].endswith("11111111-1111-1111-1111-111111111111")
    assert captured["payload"].cortex_result.ok is False
    assert captured["payload"].cortex_result.error.get("message") == "boom"


def test_gateway_replies_when_result_validation_fails(monkeypatch):
    client = bus_mod.BusClient()
    client.bus = _Bus(_Env())
    captured = {}

    async def _bad_result(*_args, **_kwargs):
        return {"nonsense": True}

    async def _capture(*, reply_to, correlation_id, causality_chain, payload):
        captured["payload"] = payload

    monkeypatch.setattr(client, "rpc_call_cortex_orch", _bad_result)
    monkeypatch.setattr(client, "_publish_gateway_reply", _capture)

    asyncio.run(client.handle_gateway_request({"data": b"x"}))

    assert captured["payload"].cortex_result.ok is False
    assert captured["payload"].cortex_result.error.get("type") == "invalid_cortex_result"


def test_gateway_disabled_recall_forwards_supervised_request_and_replies_to_hub(monkeypatch):
    client = bus_mod.BusClient()
    client.bus = _Bus(_Env())
    captured = {}

    async def _fake_rpc(req, correlation_id=None, causality_chain=None):
        captured["orch_req"] = req
        captured["orch_corr"] = correlation_id
        captured["orch_chain"] = causality_chain
        return {
            "ok": True,
            "mode": "agent",
            "verb": "agent_runtime",
            "status": "success",
            "final_text": "done",
            "memory_used": False,
            "recall_debug": {"skipped": "disabled_by_client"},
            "steps": [],
            "error": None,
            "correlation_id": str(correlation_id),
            "metadata": {"trace_verb": "agent_runtime"},
        }

    async def _capture(*, reply_to, correlation_id, causality_chain, payload):
        captured["reply_to"] = reply_to
        captured["reply_corr"] = correlation_id
        captured["reply_payload"] = payload

    env = _Env()
    env.payload = {
        "prompt": "hello",
        "mode": "agent",
        "options": {"supervised": True},
        "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
    }
    client.bus = _Bus(env)

    monkeypatch.setattr(client, "rpc_call_cortex_orch", _fake_rpc)
    monkeypatch.setattr(client, "_publish_gateway_reply", _capture)

    asyncio.run(client.handle_gateway_request({"data": b"x"}))

    orch_req = captured["orch_req"]
    assert orch_req.mode == "agent"
    assert orch_req.options["supervised"] is True
    assert orch_req.recall.enabled is False
    assert orch_req.recall.required is False
    assert orch_req.recall.mode == "hybrid"
    assert orch_req.recall.profile is None
    assert captured["reply_to"].endswith(str(env.correlation_id))
    assert captured["reply_payload"].cortex_result.ok is True
    assert captured["reply_payload"].cortex_result.verb == "agent_runtime"
