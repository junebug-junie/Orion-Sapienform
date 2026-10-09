"""Who still reads the retiring ``GET /routes`` view, so PR 6.5 can prove 24 h of zero reads.

GPU pool stage 6.3 moved every known reader to pool state; stage 6.5 deletes the endpoint only
after a full day with no reads (spec 2026-09-30-gpu-pool-stage6-telemetry-reducers-lockdown.md,
acceptance check 4). Two records, because each covers the other's blind spot:

- one WARNING log line per read (``routes_compat_read``), naming the caller's address and
  User-Agent. It survives a container restart in ``docker logs``; the in-process counter does not.
- an in-process counter since boot (``GET /debug/routes-compat-reads``). A restart resets it, but
  it is exact and needs no log scraping: ``reads_total == 0`` with ``uptime_sec >= 86400`` is the
  zero-read window on its own.

The User-Agent is what tells readers apart: every athena-host caller arrives from the same bridge
gateway address. aiohttp (Hub route picker), Python-urllib (situational runtime line) and
python-httpx (fcc_motor, context-exec) were the pre-6.3 readers.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("orion-llm-gateway.routes_compat_reads")

_MAX_CALLERS = 64  # a scanner cycling User-Agents must not grow this without bound

_lock = threading.Lock()
_state: Dict[str, Any] = {}


def reset() -> None:
    with _lock:
        _state.clear()
        _state.update(started_monotonic=time.monotonic(),
                      started_at=datetime.now(timezone.utc).isoformat(),
                      reads_total=0, last_read_at=None, by_caller={})


reset()


def record(client_host: str | None, user_agent: str | None) -> None:
    host = str(client_host or "unknown")
    agent = str(user_agent or "unknown")[:120]
    key = f"{host} {agent}"
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _state["reads_total"] += 1
        _state["last_read_at"] = now
        callers: Dict[str, Any] = _state["by_caller"]
        if key in callers or len(callers) < _MAX_CALLERS:
            row = callers.setdefault(key, {"reads": 0, "last_read_at": None})
        else:
            row = callers.setdefault("other", {"reads": 0, "last_read_at": None})
        row["reads"] += 1
        row["last_read_at"] = now
        total = _state["reads_total"]
    logger.warning("routes_compat_read client=%s user_agent=%r total_since_boot=%d "
                   "(GET /routes is retiring: GPU pool stage 6.5 deletes it after 24 h of zero reads)",
                   host, agent, total)


def snapshot() -> Dict[str, Any]:
    with _lock:
        return {
            "endpoint": "GET /routes",
            "counting_since": _state["started_at"],
            "uptime_sec": round(time.monotonic() - _state["started_monotonic"], 1),
            "reads_total": _state["reads_total"],
            "last_read_at": _state["last_read_at"],
            "by_caller": {k: dict(v) for k, v in _state["by_caller"].items()},
        }
