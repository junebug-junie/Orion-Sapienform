"""Hub Mind run read APIs (session-gated; mock PG pool)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_HUB = Path(__file__).resolve().parents[1]
_REPO = _HUB.parents[1]
for p in (str(_REPO), str(_HUB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import mind_routes


def test_get_mind_run_returns_row() -> None:
    row = {
        "mind_run_id": "mid-1",
        "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "session_id": "s1",
        "trigger": "user_turn",
        "ok": True,
        "error_code": None,
        "snapshot_hash": "abc",
        "router_profile_id": "default",
        "result_jsonb": {},
        "request_summary_jsonb": {},
        "redaction_profile_id": None,
        "created_at_utc": None,
    }

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)

    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(memory_pg_pool=pool)))

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess"):
        out = asyncio.run(mind_routes.get_mind_run("mid-1", req, None))
        assert out["mind_run_id"] == "mid-1"
