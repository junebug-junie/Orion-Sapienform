"""/health carries the boot schema self-heal state, so a degraded pool is visible, not silent."""
from __future__ import annotations

import asyncio


def test_health_reports_a_degraded_schema():
    import app.main as main
    from app.store import PostgresStore

    store = PostgresStore(pool=None)
    store._missing = {("gpu_pool_cards", "actuation_paused_at")}
    store._db_missing = [("gpu_pool_cards", "actuation_paused_at")]
    store._schema.update(state="degraded", last_error="gpu_pool_cards.actuation_paused_at: LockNotAvailable")
    saved = main._store, main.runtime
    main._store, main.runtime = store, None
    try:
        body = asyncio.run(main.health())
    finally:
        main._store, main.runtime = saved
    assert body["schema"]["state"] == "degraded"
    assert body["schema"]["missing"] == ["gpu_pool_cards.actuation_paused_at"]
    assert body["schema"]["in_memory"] == ["gpu_pool_cards.actuation_paused_at"]
    assert "LockNotAvailable" in body["schema"]["last_error"]
