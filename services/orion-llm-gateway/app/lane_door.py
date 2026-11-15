"""The gateway's half of the lane door (leased-evals design, Phase 2).

docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md. Every call the gateway sends
to a model server under a pool grant carries two headers; the door in front of the server
(services/orion-lane-door) checks the first, logs the second, and strips both:

* ``X-Orion-Lane-Key``: LLM_GATEWAY_LANE_DOOR_KEY, the secret only the gateway and the doors hold;
* ``X-Orion-Lease-Id``: the granted lease, so a door log line joins to ``gpu_pool_events``.

Both paths send them: the HTTP passthrough (passthrough_proxy.py) and the bus path (llm_backend's
HTTP client, which reads the lease id the grant's worker thread was started with, upstream_cancel.py).

Key drift is the dangerous failure (Phase 3 would refuse every chat call), so the gateway checks it:
the first time it dispatches to a server URL (and every LLM_GATEWAY_LANE_DOOR_CHECK_SEC after), it
reads that server's ``/door/stats`` in the background -- never on the call's path -- and compares the
door's key hash with its own. It logs ``lane_door_key_ok``, ``lane_door_key_mismatch`` (WARNING) or
``lane_door_absent`` (no door in front of that server yet), and ``status()`` reports the last result
per server for /health.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from .settings import settings

logger = logging.getLogger("orion-llm-gateway.lane-door")

KEY_HEADER = "X-Orion-Lane-Key"
LEASE_HEADER = "X-Orion-Lease-Id"
STATS_PATH = "/door/stats"

_lock = threading.Lock()
_checked: dict[str, float] = {}           # server base URL -> monotonic time of the last check start
_status: dict[str, dict[str, Any]] = {}   # server base URL -> last result


def key_hash(key: str) -> Optional[str]:
    """Same as services/orion-lane-door/app/settings.py key_hash (tests/test_lane_door_exempt_parity.py)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] if key else None


def _key() -> str:
    return str(getattr(settings, "llm_gateway_lane_door_key", "") or "").strip()


def headers(lease_id: Optional[str]) -> dict[str, str]:
    """The door headers for one dispatch. No key configured: only the lease id (the door then counts the
    call as ``missing``, which is what the shadow report should show until the key is set)."""
    out: dict[str, str] = {}
    key = _key()
    if key:
        out[KEY_HEADER] = key
    if lease_id:
        out[LEASE_HEADER] = str(lease_id)
    return out


def _base(url: str) -> Optional[str]:
    parts = urlsplit(str(url or ""))
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


def check_now(base: str, timeout_sec: float = 3.0) -> dict[str, Any]:
    """Read one server's /door/stats and compare key hashes. Blocking; call off the request path."""
    mine = key_hash(_key())
    result: dict[str, Any] = {"checked_at": time.time(), "gateway_key_hash": mine}
    try:
        r = httpx.get(base + STATS_PATH, timeout=timeout_sec)
        body = r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        body = None
    if not isinstance(body, dict) or "key_hash" not in body or "lane" not in body:
        result.update(state="absent")
        logger.info("lane_door_absent server=%s", base)
    else:
        result.update(lane=body.get("lane"), mode=body.get("mode"), door_key_hash=body.get("key_hash"))
        if not mine and not body.get("key_hash"):
            result["state"] = "unkeyed"   # neither side has a key yet: deploy step 1 not done; not a drift
            logger.info("lane_door_unkeyed server=%s lane=%s mode=%s", base, body.get("lane"), body.get("mode"))
        elif mine and body.get("key_hash") == mine:
            result["state"] = "ok"
            logger.info("lane_door_key_ok server=%s lane=%s mode=%s", base, body.get("lane"), body.get("mode"))
        else:
            result["state"] = "mismatch"
            logger.warning("lane_door_key_mismatch server=%s lane=%s mode=%s door_key_hash=%s gateway_key_hash=%s",
                           base, body.get("lane"), body.get("mode"), body.get("key_hash"), mine)
    with _lock:
        _status[base] = result
    return result


def observe_grant(url: str) -> None:
    """Called on every dispatch; starts a background door check for this server when one is due."""
    if not getattr(settings, "llm_gateway_lane_door_check_enabled", True):
        return
    base = _base(url)
    if base is None:
        return
    every = float(getattr(settings, "llm_gateway_lane_door_check_sec", 600.0) or 600.0)
    now = time.monotonic()
    with _lock:
        last = _checked.get(base)
        if last is not None and now - last < every:
            return
        _checked[base] = now
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(target=check_now, args=(base,), daemon=True, name="lane-door-check").start()
        return
    loop.run_in_executor(None, check_now, base)


def status() -> dict[str, Any]:
    with _lock:
        return {"gateway_key_hash": key_hash(_key()), "servers": {k: dict(v) for k, v in _status.items()}}


def _reset_for_tests() -> None:
    with _lock:
        _checked.clear()
        _status.clear()
