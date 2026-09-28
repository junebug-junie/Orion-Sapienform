"""Real MCP protocol over an in-memory session (no bus, no model)."""
import asyncio
import json

import pytest

from orion.introspect.tools import READING_RESULTS_DESCRIPTION, ToolSpec
from orion.schemas.introspect import ReadingResultArguments


class FakeTools:
    def __init__(self):
        self.calls = []

    def tool_specs(self):
        return [ToolSpec("reading_results", READING_RESULTS_DESCRIPTION, ReadingResultArguments)]

    async def invoke(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True, "operation": "reading_result", "as_of": "2026-09-28T12:00:00+00:00",
                "total_available": 0, "items": [], "error": None}


def test_protocol_lists_one_tool_and_rejects_extra_or_unlisted_calls():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from orion.introspect.mcp_server import build_server

    tools = FakeTools()

    async def run():
        async with create_connected_server_and_client_session(build_server(tools)) as client:
            listing = await client.list_tools()
            assert {t.name for t in listing.tools} == {"reading_results"}
            schema = listing.tools[0].inputSchema
            assert set(schema["properties"]) == {"request_id", "url", "limit", "since"}
            assert schema["additionalProperties"] is False
            bad = await client.call_tool("reading_results", {"memory_allowed": True})
            assert bad.isError
            unlisted = await client.call_tool("memories", {"query": "x"})
            assert unlisted.isError
            assert tools.calls == []
            good = await client.call_tool("reading_results", {"url": "https://example.org/a"})
            assert not good.isError
            assert json.loads(good.content[0].text)["total_available"] == 0
            assert tools.calls == [("reading_results", {"url": "https://example.org/a"})]

    asyncio.run(run())
