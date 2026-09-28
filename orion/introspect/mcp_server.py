"""Stdio MCP adapter for orion-introspect, following the FCC per-turn tool transport."""
from __future__ import annotations

import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from orion.core.bus.async_service import OrionBusAsync
from orion.introspect.tools import IntrospectTools
from orion.schemas.introspect import IntrospectToolBindingV1


def build_server(tools) -> Server:
    server = Server("orion-introspect")
    specs = {spec.name: spec for spec in tools.tool_specs()}

    @server.list_tools()
    async def list_tools():
        return [
            Tool(name=s.name, description=s.description, inputSchema=s.arguments.model_json_schema())
            for s in specs.values()
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name not in specs:
            raise ValueError(f"tool not available this turn: {name}")
        result = await tools.invoke(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


async def run():
    binding = IntrospectToolBindingV1.model_validate_json(os.environ["ORION_INTROSPECT_BINDING"])
    bus = OrionBusAsync(os.environ["ORION_BUS_URL"])
    try:
        await bus.connect()
        server = build_server(IntrospectTools(bus, binding))
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(run())
