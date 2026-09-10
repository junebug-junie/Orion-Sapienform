"""Configured tool scope and actual MCP protocol validation (no model/network)."""
import asyncio
import json

import pytest

from orion.fcc import mcp_config
from orion.schemas.reading import ReadingToolBindingV1


@pytest.mark.parametrize("context", ["unified_chat", "curiosity"])
def test_render_reading_binding(context, monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    binding = ReadingToolBindingV1(invocation_context=context, parent_run_id="run-7", parent_trace_id="trace-8")
    path = mcp_config.render_mcp_config(
        correlation_id="test", fcc_env={"GITHUB_PAT": "test", "FIRECRAWL_API_KEY": "test"},
        reading_binding=binding, reading_bus_url="redis://100.92.216.81:6379/0", tmp_dir=tmp_path,
    )
    server = json.loads(path.read_text())["mcpServers"]["orion-reading"]
    assert server["type"] == "stdio"
    assert server["args"] == ["-P", "-m", "orion.world_pulse_read.mcp_server"]
    assert ReadingToolBindingV1.model_validate_json(server["env"]["ORION_READING_BINDING"]) == binding
    assert set(server["env"]) == {"ORION_READING_BINDING", "ORION_BUS_URL", "PYTHONPATH"}


def test_reader_config_excludes_mutating_tools_even_when_other_flags_are_on(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    path = mcp_config.render_mcp_config(
        correlation_id="test", fcc_env={"GITHUB_PAT": "test", "FIRECRAWL_API_KEY": "test"},
        reading_only=True, include_aitown=True, include_gitnexus=True, include_context_mode=True,
        tmp_dir=tmp_path,
    )
    assert json.loads(path.read_text())["mcpServers"] == {}


def test_mcp_protocol_rejects_spoofed_arguments_and_returns_receipt():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from orion.world_pulse_read.mcp_server import build_server
    from orion.world_pulse_read.tools import ReadingTools
    calls = []
    class Tools:
        async def invoke(self, name, arguments):
            calls.append((name, arguments))
            return {"status": "queued", "request_id": "saved-id"}
    async def run():
        async with create_connected_server_and_client_session(build_server(Tools())) as client:
            listing = await client.list_tools()
            assert {t.name for t in listing.tools} == {"recommend_reading", "reading_status"}
            tool = next(t for t in listing.tools if t.name == "recommend_reading")
            assert set(tool.inputSchema["properties"]) == {"url", "why_now"}
            assert tool.inputSchema["additionalProperties"] is False
            bad = await client.call_tool("recommend_reading", {"url": "https://example.org/a", "why_now": "Read", "requested_by": "juniper"})
            assert bad.isError
            assert calls == []
            good = await client.call_tool("recommend_reading", {"url": "https://example.org/a", "why_now": "Read"})
            assert not good.isError
            assert json.loads(good.content[0].text)["status"] == "queued"
    asyncio.run(run())
