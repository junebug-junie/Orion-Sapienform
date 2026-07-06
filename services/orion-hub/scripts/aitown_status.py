"""AI Town status probe for Hub API."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _convex_base_from_settings(settings: Any) -> str:
  from scripts.fcc_env_catalog import expand_env_path, load_fcc_env

  fcc_path = expand_env_path(getattr(settings, "HUB_FCC_ENV_PATH", ""))
  env = load_fcc_env(fcc_path)
  return str(env.get("AITOWN_CONVEX_URL") or "").strip()


def fetch_aitown_status(settings: Any) -> Dict[str, Any]:
  if not bool(getattr(settings, "HUB_AITOWN_ENABLED", False)):
    return {
        "ok": False,
        "convex_reachable": False,
        "engine_running": False,
        "player_count": 0,
        "agent_count": 0,
        "generation": None,
        "error": "aitown_disabled",
    }

  base = _convex_base_from_settings(settings)
  if not base:
    return {
        "ok": False,
        "convex_reachable": False,
        "engine_running": False,
        "player_count": 0,
        "agent_count": 0,
        "generation": None,
        "error": "aitown_convex_url_missing",
    }

  out: Dict[str, Any] = {
      "ok": False,
      "convex_reachable": False,
      "engine_running": False,
      "player_count": 0,
      "agent_count": 0,
      "generation": None,
  }

  try:
    req = urllib.request.Request(f"{base.rstrip('/')}/version", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
      json.loads(resp.read().decode("utf-8"))
    out["convex_reachable"] = True
  except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
    out["error"] = f"convex_unreachable: {exc}"
    return out

  world_id = ""
  try:
    from scripts.fcc_env_catalog import expand_env_path, load_fcc_env

    env = load_fcc_env(expand_env_path(getattr(settings, "HUB_FCC_ENV_PATH", "")))
    world_id = str(env.get("AITOWN_WORLD_ID") or "").strip()
    admin_key = str(env.get("AITOWN_ADMIN_KEY") or "").strip()
  except Exception:
    admin_key = ""

  if world_id and admin_key:
    try:
      payload = json.dumps(
          {"path": "aiTown/world:status", "args": {"worldId": world_id}, "format": "json"}
      ).encode("utf-8")
      req = urllib.request.Request(
          f"{base.rstrip('/')}/api/query",
          data=payload,
          headers={
              "Content-Type": "application/json",
              "Authorization": f"Convex {admin_key}",
          },
          method="POST",
      )
      with urllib.request.urlopen(req, timeout=8) as resp:
        body = json.loads(resp.read().decode("utf-8"))
      value = body.get("value") if isinstance(body, dict) else body
      if isinstance(value, dict):
        out["engine_running"] = bool(value.get("running") or value.get("engineRunning"))
        out["generation"] = value.get("generation")
        out["player_count"] = int(value.get("playerCount") or value.get("players") or 0)
        out["agent_count"] = int(value.get("agentCount") or value.get("agents") or 0)
    except Exception:
      pass

  out["ok"] = bool(out["convex_reachable"])
  return out
