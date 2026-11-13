"""GET /health carries the town session timer's block (ladder retirement PR 3, 2026-11-05)."""
from __future__ import annotations

import asyncio
import os

# The worker reads EMBODIMENT_FCC_ENV_PATH at construction (main.py builds it at import); a sandboxed
# runner cannot read /root/.fcc/.env (same guard as test_heartbeat_chassis.py).
os.environ.setdefault("EMBODIMENT_FCC_ENV_PATH", "/tmp/nonexistent-embodiment-fcc-test.env")


def test_health_shows_the_town_self_start_block() -> None:
    from app import main

    out = asyncio.run(main.health())
    assert out["status"] == "ok"
    block = out["town_self_start"]
    assert block["enabled"] is True and block["tick_sec"] == 300.0
    assert block["ticks"] == 0 and block["running"] is False and block["session_open"] is False
