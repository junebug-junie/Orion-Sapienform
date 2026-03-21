from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
APP_DIR = ROOT / "services" / "orion-agent-chain" / "app"


def _load_tool_executor_module():
    pkg = types.ModuleType("orion_agent_chain")
    pkg.__path__ = [str(APP_DIR.parent)]
    subpkg = types.ModuleType("orion_agent_chain.app")
    subpkg.__path__ = [str(APP_DIR)]
    sys.modules.setdefault("orion_agent_chain", pkg)
    sys.modules.setdefault("orion_agent_chain.app", subpkg)

    settings_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.settings", APP_DIR / "settings.py")
    settings_mod = importlib.util.module_from_spec(settings_spec)
    assert settings_spec and settings_spec.loader
    settings_spec.loader.exec_module(settings_mod)
    sys.modules["orion_agent_chain.app.settings"] = settings_mod

    exec_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.tool_executor", APP_DIR / "tool_executor.py")
    exec_mod = importlib.util.module_from_spec(exec_spec)
    assert exec_spec and exec_spec.loader
    exec_spec.loader.exec_module(exec_mod)
    return exec_mod


tool_executor = _load_tool_executor_module()


class _FakeCodec:
    def decode(self, _data):
        envelope = SimpleNamespace(payload={"content": "ok", "model_used": "stub-model"})
        return SimpleNamespace(ok=True, envelope=envelope, error=None)


class _FakeBus:
    def __init__(self) -> None:
        self.codec = _FakeCodec()
        self.captured = None

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.captured = {
            "channel": channel,
            "env": env,
            "reply_channel": reply_channel,
            "timeout_sec": timeout_sec,
        }
        return {"data": b"ok"}


def test_tool_executor_routes_llm_verbs_to_agent_lane():
    bus = _FakeBus()
    executor = tool_executor.ToolExecutor(bus, base_dir=str(ROOT / "orion" / "cognition"))

    result = asyncio.run(
        executor.execute_llm_verb(
            "finalize_response",
            {"original_request": "Summarize the implementation plan.", "text": "Summarize the implementation plan."},
            parent_correlation_id="parent-1",
        )
    )

    assert result["llm_output"] == "ok"
    assert bus.captured is not None
    assert bus.captured["env"].payload["route"] == "agent"
