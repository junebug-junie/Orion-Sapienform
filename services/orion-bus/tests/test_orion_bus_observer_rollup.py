from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.bus_observer import build_rollup_from_redis_snapshot, run_observer_tick
from app.settings import Settings


@pytest.mark.asyncio
async def test_rollup_records_depth_and_backpressure() -> None:
    # model_copy field names — .env may override Settings() kwargs via env_file
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_stream_depth_warning": 100,
            "bus_stream_depth_critical": 1000,
            "bus_observer_streams": "orion:evt:gateway",
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {"orion:evt:gateway": 150},
        "catalog_names": {"orion:grammar:event"},
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.ping_ok is True
    assert rollup.stream_lengths["orion:evt:gateway"] == 150
    collector = rollup.to_collector(code_version="0.1.0")
    roles = {a.semantic_role for a in collector._atoms.values()}
    assert "bus_stream_depth_observed" in roles
    assert "bus_backpressure_observed" in roles
    assert "bus_configured_stream_uncataloged" in roles


@pytest.mark.asyncio
async def test_run_tick_publishes_when_enabled() -> None:
    with patch("app.bus_observer._fetch_redis_snapshot", new_callable=AsyncMock) as snap:
        snap.return_value = {
            "ping_ok": True,
            "stream_lengths": {"orion:evt:gateway": 1},
            "catalog_names": {"orion:evt:gateway"},
        }
        bus = AsyncMock()
        from app.settings import settings as default_settings

        s = default_settings.model_copy(
            update={
                "publish_orion_bus_grammar": True,
                "bus_observer_streams": "orion:evt:gateway",
            }
        )
        await run_observer_tick(bus=bus, settings=s)
        assert bus.publish.await_count >= 1
