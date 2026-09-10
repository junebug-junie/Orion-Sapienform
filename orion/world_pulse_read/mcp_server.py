"""Stdio MCP adapter, following the FCC per-turn tool transport (no HTTP ingress)."""
from __future__ import annotations

import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.reading import ReadingStatusArguments, ReadingToolBindingV1, RecommendReadingArguments
from orion.world_pulse_read.tools import ReadingTools, RECOMMEND_DESCRIPTION, STATUS_DESCRIPTION


def build_server(tools: ReadingTools) -> Server:
    server = Server("orion-reading")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(name="recommend_reading", description=RECOMMEND_DESCRIPTION,
                 inputSchema=RecommendReadingArguments.model_json_schema()),
            Tool(name="reading_status", description=STATUS_DESCRIPTION,
                 inputSchema=ReadingStatusArguments.model_json_schema()),
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        result = await tools.invoke(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


async def run():
    binding = ReadingToolBindingV1.model_validate_json(os.environ["ORION_READING_BINDING"])
    bus = OrionBusAsync(os.environ["ORION_BUS_URL"])
    try:
        await bus.connect()
        server = build_server(ReadingTools(bus, binding))
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(run())
