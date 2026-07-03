from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService


@pytest.mark.asyncio
async def test_load_state_decodes_bytes_redis_keys_to_str():
    """Bytes hash keys from redis must not duplicate str keys from heartbeats."""
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.redis = MagicMock()
    now = datetime.now(timezone.utc).isoformat()
    svc.bus.redis.hgetall = AsyncMock(
        return_value={
            b"sql-writer@athena": json.dumps(
                {
                    "service": "sql-writer",
                    "node": "athena",
                    "status": "ok",
                    "last_seen_ts": now,
                    "heartbeat_interval_sec": 10.0,
                }
            ).encode(),
        }
    )

    await svc._load_state()

    assert "sql-writer@athena" in svc._state
    assert b"sql-writer@athena" not in svc._state
