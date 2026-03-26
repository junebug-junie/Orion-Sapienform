from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.tool_executor import ToolExecutor


class _FakeCodec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=SimpleNamespace(payload=data), error=None)


class _FakeBus:
    def __init__(self):
        self.codec = _FakeCodec()
        self.calls = []

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.calls.append((channel, env.kind, env.payload.get("verb")))
        return {
            "data": {
                "ok": True,
                "mode": "brain",
                "verb": "skills.docker.ps_status.v1",
                "status": "success",
                "final_text": "docker healthy",
                "memory_used": False,
                "steps": [],
                "recall_debug": {},
                "metadata": {},
            }
        }


def test_capability_backed_tool_exec_normalizes_observation():
    bus = _FakeBus()
    executor = ToolExecutor(bus, base_dir="/workspace/Orion-Sapienform/orion/cognition")
    result = asyncio.run(
        executor.execute_llm_verb(
            "assess_runtime_state",
            {"text": "check runtime"},
            parent_correlation_id="corr-1",
        )
    )
    assert result["selected_verb"] == "assess_runtime_state"
    assert result["selected_skill_family"] == "system_inspection"
    assert result["selected_skill"] is not None
    assert "execution_summary" in result
    assert bus.calls and bus.calls[0][0] == "orion:cortex:request"
