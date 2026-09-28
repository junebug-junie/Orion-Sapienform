"""Real MCP protocol over an in-memory session (no bus, no model)."""
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from orion.introspect.tools import READING_RESULTS_DESCRIPTION, ToolSpec
from orion.schemas.introspect import (
    DEFAULT_TEXT_CAP,
    MAX_ITEMS,
    SHORT_FIELD_CAP,
    URL_CAP,
    IntrospectItemV1,
    IntrospectResultV1,
    ReadingResultArguments,
)

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
MCP_TOOL_RESULT_MAX_CHARS = 12000


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
            assert set(schema["properties"]) == {"request_id", "url", "limit", "since", "query"}
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


def _unicode_worst_case_result_dict():
    accent = "é"
    item = IntrospectItemV1(
        id="reading:0",
        occurred_at=NOW,
        kind="reading_result",
        epistemic_status="unsettled",
        text=accent * DEFAULT_TEXT_CAP,
        truncated=True,
        extra={
            "url": "https://example.org/" + accent * (URL_CAP - 20),
            "title": accent * SHORT_FIELD_CAP,
            "why_now": accent * SHORT_FIELD_CAP,
            "reading_status": "landing_pending",
            "source_read": True,
            "learned": True,
            "request_id": str(uuid4()),
        },
    )
    return IntrospectResultV1(
        ok=True,
        operation="reading_result",
        as_of=NOW,
        total_available=500,
        items=[item.model_copy(update={"id": f"reading:{i}"}) for i in range(MAX_ITEMS)],
    ).model_dump(mode="json")


def test_unicode_worst_case_tool_text_stays_under_mcp_cap():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from orion.introspect.mcp_server import build_server

    class WorstCaseTools:
        def tool_specs(self):
            return [ToolSpec("reading_results", READING_RESULTS_DESCRIPTION, ReadingResultArguments)]

        async def invoke(self, name, arguments):
            return _unicode_worst_case_result_dict()

    async def run():
        async with create_connected_server_and_client_session(build_server(WorstCaseTools())) as client:
            response = await client.call_tool("reading_results", {})
            assert not response.isError
            text = response.content[0].text
            assert len(text) < MCP_TOOL_RESULT_MAX_CHARS
            payload = json.loads(text)
            assert payload["total_available"] == 500
            assert len(payload["items"]) == MAX_ITEMS

    asyncio.run(run())
