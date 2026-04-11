from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import yaml

from app.capability_bridge import skill_domain_failure_reason_from_final_text

from app.tool_executor import ToolExecutor
from orion.core.bus.enforce import ChannelCatalogEnforcer

# Repo root: services/orion-agent-chain/tests -> parents[3] = Orion-Sapienform
_REPO_ROOT = Path(__file__).resolve().parents[3]
_COGNITION_DIR = _REPO_ROOT / "orion" / "cognition"


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


class _FakeBusDomainNegativeOkTrue(_FakeBus):
    """Nested cortex reports HTTP/RPC success but skill JSON is domain-negative (false-green)."""

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
                "verb": "skills.mesh.mesh_ops_round.v1",
                "status": "success",
                "final_text": json.dumps(
                    {
                        "available": False,
                        "reason": "tailscale_not_installed",
                        "unsupported": True,
                        "nodes": [],
                    }
                ),
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


def test_skill_domain_failure_reason_from_mesh_json():
    ft = json.dumps(
        {"available": False, "reason": "tailscale_not_installed", "unsupported": True, "nodes": []}
    )
    assert skill_domain_failure_reason_from_final_text(ft) == "tailscale_not_installed"


def test_capability_backed_tool_exec_domain_negative_overrides_false_green():
    bus = _FakeBusDomainNegativeOkTrue()
    executor = ToolExecutor(bus, base_dir=str(_COGNITION_DIR))
    result = asyncio.run(
        executor.execute_llm_verb(
            "assess_mesh_presence",
            {"text": "which nodes are up"},
            parent_correlation_id="corr-domain-neg",
        )
    )
    assert result["raw_payload_ref"]["ok"] is False
    assert result["raw_payload_ref"].get("domain_negative") is True
    summary = result["execution_summary"]
    assert "Executed" not in summary
    assert "tailscale_not_installed" in summary or "mesh status" in summary.lower()


def test_capability_backed_tool_exec_normalizes_observation():
    bus = _FakeBus()
    executor = ToolExecutor(bus, base_dir=str(_COGNITION_DIR))
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
    executor = ToolExecutor(bus, base_dir=str(_COGNITION_DIR))
    result = asyncio.run(
        executor.execute_llm_verb(
            "housekeep_runtime",
            {"text": "dry-run cleanup of stopped containers"},
            parent_correlation_id="corr-empty",
        )
    )
    assert result["selected_verb"] == "housekeep_runtime"
    assert result["selected_skill"] == "skills.runtime.docker_prune_stopped_containers.v1"
    assert result["raw_payload_ref"]["status"] == "empty_terminal_output"
    assert "empty terminal output" in result["execution_summary"].lower()


def test_capability_backed_tool_reply_channel_is_catalog_valid():
    bus = _FakeBus()
    executor = ToolExecutor(bus, base_dir=str(_COGNITION_DIR))
    asyncio.run(
        executor.execute_llm_verb(
            "housekeep_runtime",
            {"text": "dry-run cleanup of stopped containers"},
            parent_correlation_id="corr-2",
        )
    )
    reply_channel = bus.calls[0][4]
    assert reply_channel.startswith("orion:cortex:result:")

    channels_yaml = _REPO_ROOT / "orion" / "bus" / "channels.yaml"
    raw = yaml.safe_load(channels_yaml.read_text(encoding="utf-8")) or {}
    channels = raw.get("channels", [])
    catalog = {entry["name"]: entry for entry in channels if isinstance(entry, dict) and entry.get("name")}
    enforcer = ChannelCatalogEnforcer(enforce=True, catalog=catalog)

    entry = enforcer.entry_for(reply_channel)
    assert entry is not None
    assert entry["schema_id"] == "CortexClientResult"
