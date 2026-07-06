"""MCP stdio server exposing AI Town gameplay tools."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from orion_aitown_mcp.client import (
    AitownClientError,
    _default_agent_id,
    _default_player_id,
    _world_id,
    convex_query,
    fetch_version,
    send_input,
)

mcp = FastMCP("orion-aitown")


def _json(data: Any) -> str:
  return json.dumps(data, indent=2, default=str)


@mcp.tool()
async def aitown_world_status() -> str:
  """Return Convex version and configured world id."""
  version = fetch_version()
  return _json({"version": version, "world_id": _world_id()})


@mcp.tool()
async def aitown_list_players() -> str:
  """List players in the configured world."""
  try:
    data = convex_query("aiTown/world:players", {"worldId": _world_id()})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(data)


@mcp.tool()
async def aitown_list_agents() -> str:
  """List agents in the configured world."""
  try:
    data = convex_query("aiTown/world:agents", {"worldId": _world_id()})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(data)


@mcp.tool()
async def aitown_list_characters() -> str:
  """List available character descriptions."""
  try:
    data = convex_query("aiTown/characters:list", {})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(data)


@mcp.tool()
async def aitown_move_player(
    destination_x: float,
    destination_y: float,
    player_id: Optional[str] = None,
) -> str:
  """Move a player to map coordinates (defaults to AITOWN_ORION_PLAYER_ID)."""
  pid = str(player_id or _default_player_id()).strip()
  if not pid:
    return _json({"error": "player_id required (set AITOWN_ORION_PLAYER_ID or pass player_id)"})
  try:
    result = send_input(
        name="moveTo",
        args={"playerId": pid, "destination": {"x": destination_x, "y": destination_y}},
    )
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(result)


@mcp.tool()
async def aitown_create_agent(description_index: int) -> str:
  """Create an agent from a character description index."""
  try:
    result = send_input(name="createAgent", args={"descriptionIndex": description_index})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(result)


@mcp.tool()
async def aitown_stop_world() -> str:
  """Stop the simulation engine."""
  try:
    result = send_input(name="stop", args={})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(result)


@mcp.tool()
async def aitown_resume_world() -> str:
  """Resume the simulation engine."""
  try:
    result = send_input(name="resume", args={})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(result)


@mcp.tool()
async def aitown_kick_engine() -> str:
  """Kick the simulation engine loop."""
  try:
    result = send_input(name="kick", args={})
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(result)


@mcp.tool()
async def aitown_send_input(name: str, args_json: str, world_id: Optional[str] = None) -> str:
  """God-mode: arbitrary sendInput(name, args) against a world."""
  try:
    args: Dict[str, Any] = json.loads(args_json) if args_json.strip() else {}
  except json.JSONDecodeError as exc:
    return _json({"error": f"invalid args_json: {exc}"})
  try:
    result = send_input(name=name, args=args, world_id=world_id)
  except AitownClientError as exc:
    return _json({"error": str(exc)})
  return _json(result)


def main() -> None:
  asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
  main()
