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


def list_players(world_id: Optional[str] = None) -> Any:
    return convex_query("aiTown/world:players", {"worldId": str(world_id or _world_id()).strip()})


def list_agents(world_id: Optional[str] = None) -> Any:
    return convex_query("aiTown/world:agents", {"worldId": str(world_id or _world_id()).strip()})


def move_to(*, player_id: str, x: float, y: float, world_id: Optional[str] = None) -> Any:
    kwargs: Dict[str, Any] = {
        "name": "moveTo",
        "args": {"playerId": player_id, "destination": {"x": float(x), "y": float(y)}},
    }
    if world_id is not None:
        kwargs["world_id"] = world_id
    return send_input(**kwargs)
