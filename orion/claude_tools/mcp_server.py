"""Stdio MCP server for Claude in Hub chat: Orion's read-only tools, each gated by Orion's consent.

Runs inside orion-room-companion, started by Claude Code from the per-turn ``--mcp-config`` the
companion writes (server name ``orion_room``). Follows orion/world_pulse_read/mcp_server.py.

Its ONLY ability is one bus request per tool call, to the broker in orion-harness-governor
(orion:claude:tool:request). It reads nothing itself, holds no grant, no reader and no credential.
The turn's context (room, Claude session, the invite text) comes from the companion in
``ORION_CLAUDE_TOOLS_BINDING`` (JSON, ``ClaudeToolBindingV1``), never from Claude's arguments.

After each call it appends one line to the binding's ``call_log_path``; the companion turns those
lines into the footer under Claude's reply ("Looked at: ...").
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from orion.claude_tools.registry import TOOLS
from orion.claude_tools.render import unavailable_text
from orion.claude_tools.transport import (
    TOOL_REQUEST_CHANNEL,
    TOOL_REQUEST_KIND,
    TOOL_RESULT_KIND,
    TOOL_RESULT_PREFIX,
)
from orion.schemas.claude_tools import ClaudeToolBindingV1, ClaudeToolRequestV1, ClaudeToolResultV1

SERVER_NAME = "orion_room"
BINDING_ENV = "ORION_CLAUDE_TOOLS_BINDING"
# Consent (40 s) + the slowest reader (15 s) + bus slack, under the CLI's MCP tool timeout
# (the companion sets MCP_TOOL_TIMEOUT above this).
RPC_TIMEOUT_SEC = 58.0
NO_BROKER_TEXT = unavailable_text("the broker did not answer, so nothing was shared")


def tool_list() -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "inputSchema": t.arguments.model_json_schema()}
            for t in TOOLS.values()]


def split_reason(arguments: Optional[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    args = dict(arguments or {})
    reason = args.pop("reason", "")
    return (reason if isinstance(reason, str) else str(reason))[:300], args


def log_call(path: Optional[str], result: ClaudeToolResultV1) -> None:
    if not path:
        return
    line = {k: getattr(result, k) for k in ("call_id", "tool", "category", "decision_path", "shared",
                                            "orion_answer", "orion_reason", "error")}
    try:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError:
        pass  # the footer is best-effort; the audit row is the record


class ToolClient:
    def __init__(self, bus: Any, binding: ClaudeToolBindingV1, *, timeout_sec: float = RPC_TIMEOUT_SEC) -> None:
        self.bus = bus
        self.binding = binding
        self.timeout_sec = timeout_sec

    async def call(self, name: str, arguments: Optional[dict[str, Any]]) -> ClaudeToolResultV1:
        reason, args = split_reason(arguments)
        call_id = uuid4().hex
        try:
            req = ClaudeToolRequestV1(call_id=call_id, binding=self.binding, tool=name[:64] or "unknown",
                                      arguments=args, reason=reason)
        except ValueError:
            result = ClaudeToolResultV1(call_id=call_id, tool=name[:64] or "unknown", shared=False,
                                        decision_path="invalid", text=unavailable_text("bad arguments"))
            log_call(self.binding.call_log_path, result)
            return result
        result = await self._rpc(req)
        log_call(self.binding.call_log_path, result)
        return result

    async def _rpc(self, req: ClaudeToolRequestV1) -> ClaudeToolResultV1:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        def down(why: str) -> ClaudeToolResultV1:
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path="error",
                                      text=NO_BROKER_TEXT, error=why)

        correlation_id = uuid4()
        reply = f"{TOOL_RESULT_PREFIX}{correlation_id}"
        envelope = BaseEnvelope(kind=TOOL_REQUEST_KIND, correlation_id=correlation_id, reply_to=reply,
                                source=ServiceRef(name="orion-room-companion"), payload=req.model_dump(mode="json"))
        try:
            raw = await self.bus.rpc_request(TOOL_REQUEST_CHANNEL, envelope, reply_channel=reply,
                                             timeout_sec=self.timeout_sec, require_listener=True)
        except Exception as exc:  # noqa: BLE001 -- no listener, timeout, bus down: nothing was shared
            return down(type(exc).__name__)
        decoded = self.bus.codec.decode(raw.get("data"))
        if (not decoded.ok or str(decoded.envelope.correlation_id) != str(correlation_id)
                or decoded.envelope.kind != TOOL_RESULT_KIND):
            return down("malformed reply")
        try:
            result = ClaudeToolResultV1.model_validate(decoded.envelope.payload)
        except ValueError:
            return down("malformed result")
        if result.call_id != req.call_id:
            return down("reply for another call")
        return result


def build_server(client: ToolClient):
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools():
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in tool_list()]

    @server.call_tool()
    async def call_tool(name, arguments):
        result = await client.call(str(name), arguments)
        return [TextContent(type="text", text=result.text)]

    return server


async def run() -> None:
    from mcp.server.stdio import stdio_server

    from orion.core.bus.async_service import OrionBusAsync

    binding = ClaudeToolBindingV1.model_validate_json(os.environ[BINDING_ENV])
    bus = OrionBusAsync(os.environ["ORION_BUS_URL"])
    try:
        await bus.connect()
        server = build_server(ToolClient(bus, binding))
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(run())
