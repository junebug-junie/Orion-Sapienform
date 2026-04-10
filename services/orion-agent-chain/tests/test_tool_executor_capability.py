from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import yaml

from app.tool_executor import ToolExecutor
from orion.core.bus.enforce import ChannelCatalogEnforcer


class _FakeCodec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=SimpleNamespace(payload=data), error=None)


class _FakeBus:
    def __init__(self):
        self.codec = _FakeCodec()
        self.calls = []

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.calls.append(
            (
                channel,
                env.kind,
                env.payload.get("verb"),
                env.payload.get("context", {}).get("metadata", {}),
                reply_channel,
                env.reply_to,
            )
        )
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


class _FakeBusEmptyFinalText(_FakeBus):
    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.calls.append(
            (
                channel,
                env.kind,
                env.payload.get("verb"),
                env.payload.get("context", {}).get("metadata", {}),
                reply_channel,
                env.reply_to,
            )
        )
        return {
            "data": {
                "ok": True,
                "mode": "brain",
                "verb": "skills.runtime.docker_prune_stopped_containers.v1",
                "status": "success",
                "final_text": "",
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
    metadata = bus.calls[0][3]
    assert metadata["requested_verb"] == "assess_runtime_state"
    assert "risk_class" in metadata
    assert bus.calls[0][4].startswith("orion:cortex:result:")
    assert bus.calls[0][5] == bus.calls[0][4]


def test_capability_backed_tool_exec_fail_closes_on_empty_terminal_text():
    bus = _FakeBusEmptyFinalText()
    executor = ToolExecutor(bus, base_dir="/workspace/Orion-Sapienform/orion/cognition")
    result = asyncio.run(
        executor.execute_llm_verb(
            "housekeep_runtime",
            {"text": "dry-run cleanup of stopped containers"},
            parent_correlation_id="corr-empty",
        )
    )
    assert result["selected_verb"] == "housekeep_runtime"
    assert result["selected_skill"] is None
    assert result["raw_payload_ref"]["status"] == "empty_terminal_output"
    assert "failed closed" in result["execution_summary"].lower()


def test_capability_backed_tool_reply_channel_is_catalog_valid():
    bus = _FakeBus()
    executor = ToolExecutor(bus, base_dir="/workspace/Orion-Sapienform/orion/cognition")
    asyncio.run(
        executor.execute_llm_verb(
            "housekeep_runtime",
            {"text": "dry-run cleanup of stopped containers"},
            parent_correlation_id="corr-2",
        )
    )
    reply_channel = bus.calls[0][4]
    assert reply_channel.startswith("orion:cortex:result:")

    channels_yaml = Path("/workspace/Orion-Sapienform/orion/bus/channels.yaml")
    raw = yaml.safe_load(channels_yaml.read_text(encoding="utf-8")) or {}
    channels = raw.get("channels", [])
    catalog = {entry["name"]: entry for entry in channels if isinstance(entry, dict) and entry.get("name")}
    enforcer = ChannelCatalogEnforcer(enforce=True, catalog=catalog)

    entry = enforcer.entry_for(reply_channel)
    assert entry is not None
    assert entry["schema_id"] == "CortexClientResult"
