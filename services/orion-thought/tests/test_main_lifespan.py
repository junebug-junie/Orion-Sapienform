"""Tests for main.py's lifespan wiring of the shared pool warm-up task.

Targeted coverage for the specific addition (pool_warmup_task creation at
startup, cancellation at shutdown) -- not an attempt to close the pre-existing
gap that the other four lifespan workers (bus/reverie/reverie_chain/reasoning)
have no direct test coverage either; that's a broader, separate concern.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_lifespan_creates_and_cancels_pool_warmup_task(monkeypatch) -> None:
    import app.main as main_module
    from app.settings import settings

    # Keep the other four workers as instant no-ops so this test only
    # exercises the pool-warmup wiring, not real bus/redis connections.
    monkeypatch.setattr(settings, "orion_bus_enabled", False)
    monkeypatch.setattr(settings, "reverie_enabled", False)
    monkeypatch.setattr(settings, "reverie_chain_enabled", False)

    warmed: list[bool] = []

    async def _fake_warm_pool() -> None:
        warmed.append(True)

    monkeypatch.setattr(main_module, "warm_pool", _fake_warm_pool)

    app = SimpleNamespace(state=SimpleNamespace())
    async with main_module.lifespan(app):
        assert app.state.pool_warmup_task is not None
        # Give the fire-and-forget task a tick to actually run.
        await app.state.pool_warmup_task

    assert warmed == [True]
    assert app.state.pool_warmup_task.done()
