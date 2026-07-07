"""Thin env-driven Convex HTTP client for the AI Town backend (shared lib).

Synchronous urllib on purpose (mirrors the existing MCP client). Async callers
should wrap these in ``asyncio.to_thread``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class AitownClientError(Exception):
    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def _base_url() -> str:
    return str(os.environ.get("AITOWN_CONVEX_URL") or "").rstrip("/")


def _admin_key() -> str:
    return str(os.environ.get("AITOWN_ADMIN_KEY") or "").strip()


def _world_id() -> str:
    return str(os.environ.get("AITOWN_WORLD_ID") or "").strip()


def convex_request(endpoint: str, *, path: str, args: Optional[Dict[str, Any]] = None) -> Any:
    base = _base_url()
    key = _admin_key()
    if not base:
        raise AitownClientError("AITOWN_CONVEX_URL is not set")
    if not key:
        raise AitownClientError("AITOWN_ADMIN_KEY is not set")
    url = f"{base}/api/{endpoint.lstrip('/')}"
    payload = json.dumps({"path": path, "args": args or {}, "format": "json"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Convex {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise AitownClientError(f"Convex HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AitownClientError(f"Convex unreachable: {exc.reason}") from exc
    if not body.strip():
        return None
    parsed = json.loads(body)
    if isinstance(parsed, dict) and parsed.get("status") == "error":
        raise AitownClientError(str(parsed.get("errorMessage") or parsed))
    if isinstance(parsed, dict) and "value" in parsed:
        return parsed["value"]
    return parsed


def convex_query(path: str, args: Optional[Dict[str, Any]] = None) -> Any:
    return convex_request("query", path=path, args=args)


def convex_mutation(path: str, args: Optional[Dict[str, Any]] = None) -> Any:
    return convex_request("mutation", path=path, args=args)


def send_input(*, name: str, args: Dict[str, Any], world_id: Optional[str] = None) -> Any:
    wid = str(world_id or _world_id()).strip()
    if not wid:
        raise AitownClientError("AITOWN_WORLD_ID is not set")
    return convex_mutation("aiTown/main:sendInput", {"worldId": wid, "name": name, "args": args})


def heartbeat_world(world_id: Optional[str] = None) -> Any:
    """Wake/keep-alive the town engine. An `inactive` world drops all queued inputs
    (join/moveTo) until restarted, so callers that actuate must heartbeat first."""
    wid = str(world_id or _world_id()).strip()
    if not wid:
        raise AitownClientError("AITOWN_WORLD_ID is not set")
    return convex_mutation("world:heartbeatWorld", {"worldId": wid})


def join_player(*, name: str, character: str, description: str, world_id: Optional[str] = None) -> Any:
    """Spawn a named, externally-driven player (no town-AI agent) via the ``join`` input."""
    return send_input(
        name="join",
        args={"name": name, "character": character, "description": description},
        world_id=world_id,
    )


def _world_snapshot(world_id: Optional[str] = None) -> Dict[str, Any]:
    wid = str(world_id or _world_id()).strip()
    if not wid:
        raise AitownClientError("AITOWN_WORLD_ID is not set")
    ws = convex_query("world:worldState", {"worldId": wid})
    return ws if isinstance(ws, dict) else {}


def _game_descriptions(world_id: Optional[str] = None) -> Dict[str, Any]:
    wid = str(world_id or _world_id()).strip()
    gd = convex_query("world:gameDescriptions", {"worldId": wid})
    return gd if isinstance(gd, dict) else {}


def list_players(world_id: Optional[str] = None) -> Any:
    """Players from ``world:worldState`` enriched with names from ``world:gameDescriptions``.

    This deployment exposes the monolithic ``world:*`` layout, not ``aiTown/world:*``.
    Returned dicts keep the raw player fields (id/position/human/...) plus ``name``.
    """
    world = _world_snapshot(world_id).get("world") or {}
    players = world.get("players") or []
    names = {
        str(d.get("playerId")): d.get("name")
        for d in (_game_descriptions(world_id).get("playerDescriptions") or [])
    }
    return [{**p, "id": str(p.get("id")), "name": names.get(str(p.get("id")))} for p in players]


def get_world_map(world_id: Optional[str] = None) -> Dict[str, Any]:
    """Static map descriptor (dimensions + object/collision layers)."""
    wm = _game_descriptions(world_id).get("worldMap")
    return wm if isinstance(wm, dict) else {}


def list_agents(world_id: Optional[str] = None) -> Any:
    world = _world_snapshot(world_id).get("world") or {}
    agents = world.get("agents") or []
    desc = {
        str(d.get("agentId")): d
        for d in (_game_descriptions(world_id).get("agentDescriptions") or [])
    }
    out = []
    for a in agents:
        aid = str(a.get("id"))
        d = desc.get(aid, {})
        out.append({**a, "id": aid, "identity": d.get("identity"), "plan": d.get("plan")})
    return out


def move_to(*, player_id: str, x: float, y: float, world_id: Optional[str] = None) -> Any:
    kwargs: Dict[str, Any] = {
        "name": "moveTo",
        "args": {"playerId": player_id, "destination": {"x": float(x), "y": float(y)}},
    }
    if world_id is not None:
        kwargs["world_id"] = world_id
    return send_input(**kwargs)
