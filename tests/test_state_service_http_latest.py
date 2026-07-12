from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-state-service"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app import main as state_service_main  # noqa: E402
from orion.schemas.state.contracts import StateLatestReply  # noqa: E402


def test_http_get_latest_passes_biometrics_stale_after_sec(monkeypatch) -> None:
    # Regression: http_get_latest previously called STORE.get_latest(req)
    # without the required keyword-only biometrics_stale_after_sec argument
    # that StateStore.get_latest() needs (the bus-RPC path, _handle_get_latest,
    # already passed it correctly) -- confirmed live via a TypeError on every
    # GET /state/latest call (2026-07-12 research finding). The HTTP route
    # must pass it the same way the RPC path does.
    fake_store = AsyncMock()
    fake_store.get_latest.return_value = StateLatestReply(status="missing")
    monkeypatch.setattr(state_service_main, "STORE", fake_store)

    asyncio.run(state_service_main.http_get_latest(scope="global", node=None))

    fake_store.get_latest.assert_awaited_once()
    _, kwargs = fake_store.get_latest.call_args
    assert "biometrics_stale_after_sec" in kwargs
    assert isinstance(kwargs["biometrics_stale_after_sec"], float)
