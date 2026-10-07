from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.bus_observer import (
    ActivityEwmaTracker,
    _fetch_redis_snapshot,
    build_rollup_from_redis_snapshot,
    run_observer_tick,
)
from app.settings import Settings


def test_rollup_emits_no_depth_or_backpressure_even_for_a_huge_stream() -> None:
    """2026-09-25 (fix/bus-observer-scope): XLEN depth and backpressure were
    retired. Even a snapshot carrying a pre-retirement-shaped stream_lengths
    entry far past the old 100k critical threshold must produce neither atom;
    uncataloged detection still works."""
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:evt:gateway",
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {"orion:evt:gateway": 5_000_000},
        "catalog_names": {"orion:grammar:event"},
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.ping_ok is True
    assert rollup.streams_observed == 1
    collector = rollup.to_collector(code_version="0.1.0")
    roles = {a.semantic_role for a in collector._atoms.values()}
    assert "bus_stream_depth_observed" not in roles
    assert "bus_backpressure_observed" not in roles
    assert "bus_configured_stream_uncataloged" in roles
    done = next(a for a in collector._atoms.values() if a.semantic_role == "bus_observer_tick_completed")
    assert "streams_observed=1" in done.summary


@pytest.mark.asyncio
async def test_run_tick_publishes_when_enabled() -> None:
    with patch("app.bus_observer._fetch_redis_snapshot", new_callable=AsyncMock) as snap:
        snap.return_value = {
            "ping_ok": True,
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


# ── schema-sample retirement (2026-10-07) ────────────────────────
#
# The bounded XREVRANGE schema sample behind contract_pressure was retired
# (fix/transport-lattice-names-and-contract): it covered two world_pulse
# streams, read 0 on every live tick, and a mesh-wide version reads 0 by
# construction because OrionBusAsync.publish() validates before sending.


def test_schema_sample_machinery_is_gone() -> None:
    import app.bus_observer as bo

    for name in ("count_schema_mismatches", "load_channel_catalog_schema_ids"):
        assert not hasattr(bo, name), name
    assert not hasattr(Settings(), "bus_observer_schema_sample_count")


def test_rollup_has_no_schema_mismatch_atom_even_with_legacy_snapshot_keys() -> None:
    """A snapshot still carrying the old sample keys produces no
    bus_schema_validation_failed atom."""
    settings = Settings().model_copy(
        update={"bus_observer_streams": "orion:stream:world_pulse:run:result"}
    )
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot={
            "ping_ok": True,
            "catalog_names": {"orion:stream:world_pulse:run:result"},
            "catalog_schema_ids": {"orion:stream:world_pulse:run:result": "WorldPulseRunResultV1"},
            "stream_samples": {"orion:stream:world_pulse:run:result": [("1-0", {"envelope": "{}"})]},
        },
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert not hasattr(rollup, "schema_mismatches")
    roles = {a.semantic_role for a in rollup.to_collector(code_version=None)._atoms.values()}
    assert "bus_schema_validation_failed" not in roles


@pytest.mark.asyncio
async def test_fetch_snapshot_does_no_stream_reads() -> None:
    settings = Settings().model_copy(update={"bus_observer_census_enabled": False})
    client = AsyncMock()
    client.ping.return_value = True
    with patch("app.bus_observer.aioredis.from_url", return_value=client):
        snapshot = await _fetch_redis_snapshot(settings)
    client.xrevrange.assert_not_called()
    assert "stream_samples" not in snapshot
    assert "catalog_schema_ids" not in snapshot


# ── mesh-wide census (catalog_drift_pressure fix, 2026-07-25) ──────────


def test_rollup_threads_undeclared_active_count_into_census_atom() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:evt:gateway",
        }
    )
    snapshot = {
        "ping_ok": True,
        "catalog_names": {"orion:evt:gateway", "orion:other:channel"},
        "undeclared_active_count": 3,
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.undeclared_active_count == 3
    assert rollup.catalog_size == 2

    collector = rollup.to_collector(code_version="0.1.0")
    census_atoms = [a for a in collector._atoms.values() if a.semantic_role == "bus_census_computed"]
    assert len(census_atoms) == 1
    assert "undeclared_active_count=3" in census_atoms[0].summary
    assert "catalog_size=2" in census_atoms[0].summary


def test_rollup_omits_census_atom_when_not_measured() -> None:
    # snapshot has no "undeclared_active_count" key at all -- same
    # backward-compatibility shape as the pre-existing schema-snapshot test
    # above, confirming a rollup without census data never emits the atom
    # (None must stay None, not silently become 0).
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:evt:gateway",
        }
    )
    snapshot = {
        "ping_ok": True,
        "catalog_names": {"orion:evt:gateway"},
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.undeclared_active_count is None

    collector = rollup.to_collector(code_version="0.1.0")
    roles = {a.semantic_role for a in collector._atoms.values()}
    assert "bus_census_computed" not in roles


@pytest.mark.asyncio
async def test_fetch_snapshot_skips_census_when_disabled() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:evt:gateway",
            "bus_observer_census_enabled": False,
        }
    )
    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=0)
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch("app.bus_observer.load_channel_catalog_names", return_value={"orion:evt:gateway"}),
        patch("app.bus_observer.scan_active_channels", new_callable=AsyncMock) as scan_mock,
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    scan_mock.assert_not_awaited()
    assert snapshot["undeclared_active_count"] is None


@pytest.mark.asyncio
async def test_fetch_snapshot_computes_census_when_enabled() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:evt:gateway",
            "bus_observer_census_enabled": True,
        }
    )
    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=0)
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch("app.bus_observer.load_channel_catalog_names", return_value={"orion:declared:one"}),
        patch(
            "app.bus_observer.scan_active_channels",
            new_callable=AsyncMock,
            return_value={"orion:declared:one": 1.0, "orion:undeclared:two": 1.0},
        ),
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    # orion:undeclared:two is active but not in the declared catalog -- the
    # one real undeclared_active entry compute_census() should find.
    assert snapshot["undeclared_active_count"] == 1


@pytest.mark.asyncio
async def test_fetch_snapshot_census_scan_failure_fails_open_to_none() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:evt:gateway",
            "bus_observer_census_enabled": True,
        }
    )
    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=0)
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch("app.bus_observer.load_channel_catalog_names", return_value={"orion:declared:one"}),
        patch("app.bus_observer.scan_active_channels", side_effect=RuntimeError("redis unreachable")),
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    # Must not raise (fail-open), and must report None (not measured), never
    # a silent 0 that would misrepresent "scan failed" as "confirmed clean."
    assert snapshot["undeclared_active_count"] is None


# ── bus_activity_zscore (Idea 6, 2026-07-25) ──────────────────────────


@pytest.mark.asyncio
async def test_fetch_snapshot_computes_total_mesh_publish_rate_when_census_enabled() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:evt:gateway",
            "bus_observer_census_enabled": True,
        }
    )
    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=0)
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch("app.bus_observer.load_channel_catalog_names", return_value={"orion:declared:one"}),
        patch(
            "app.bus_observer.scan_active_channels",
            new_callable=AsyncMock,
            return_value={"orion:declared:one": 1.5, "orion:undeclared:two": 2.5},
        ),
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    # Reuses the same scan_active_channels() result the census diff already
    # paid for -- sum of every active channel's rate, no extra Redis call.
    assert snapshot["total_mesh_publish_rate"] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_fetch_snapshot_total_mesh_publish_rate_none_when_census_disabled() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:evt:gateway",
            "bus_observer_census_enabled": False,
        }
    )
    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=0)
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch("app.bus_observer.load_channel_catalog_names", return_value={"orion:declared:one"}),
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    assert snapshot["total_mesh_publish_rate"] is None


def test_rollup_threads_total_mesh_publish_rate_through() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:evt:gateway",
        }
    )
    snapshot = {
        "ping_ok": True,
        "catalog_names": {"orion:evt:gateway"},
        "total_mesh_publish_rate": 7.25,
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.total_mesh_publish_rate == 7.25


class TestActivityEwmaTracker:
    def test_first_observation_has_no_zscore(self) -> None:
        tracker = ActivityEwmaTracker(alpha=0.2)
        update = tracker.observe(10.0)
        assert update.zscore is None
        assert update.ewma == 10.0

    def test_second_observation_zscores_against_prior_baseline(self) -> None:
        tracker = ActivityEwmaTracker(alpha=0.2)
        tracker.observe(10.0)
        update = tracker.observe(10.0)
        assert update.zscore == pytest.approx(0.0)

    def test_tracker_state_persists_across_observe_calls(self) -> None:
        # The whole point: a fresh tracker per tick would never build a
        # baseline. Same tracker instance must accumulate state.
        tracker = ActivityEwmaTracker(alpha=0.5)
        tracker.observe(10.0)
        tracker.observe(10.0)
        update = tracker.observe(100.0)
        assert update.zscore is not None
        assert update.zscore > 0


@pytest.mark.asyncio
async def test_run_observer_tick_emits_activity_zscore_atom_when_tracker_given() -> None:
    with patch("app.bus_observer._fetch_redis_snapshot", new_callable=AsyncMock) as snap:
        snap.return_value = {
            "ping_ok": True,
            "catalog_names": {"orion:evt:gateway"},
            "total_mesh_publish_rate": 12.0,
        }
        bus = AsyncMock()
        s = Settings().model_copy(
            update={"publish_orion_bus_grammar": True, "bus_observer_streams": "orion:evt:gateway"}
        )
        tracker = ActivityEwmaTracker(alpha=0.2)

        captured_events = []

        async def _capture(bus_arg, events, **kwargs):
            captured_events.extend(events)

        with patch("app.bus_observer.publish_bus_transport_grammar_trace", side_effect=_capture):
            await run_observer_tick(bus=bus, settings=s, activity_tracker=tracker)

    roles = {e.atom.semantic_role for e in captured_events if e.atom}
    assert "bus_activity_zscore_computed" in roles


@pytest.mark.asyncio
async def test_run_observer_tick_without_tracker_omits_activity_zscore_atom() -> None:
    # Backward-compatible default (activity_tracker=None) -- same as every
    # pre-existing call site that doesn't know about Idea 6 yet.
    with patch("app.bus_observer._fetch_redis_snapshot", new_callable=AsyncMock) as snap:
        snap.return_value = {
            "ping_ok": True,
            "catalog_names": {"orion:evt:gateway"},
            "total_mesh_publish_rate": 12.0,
        }
        bus = AsyncMock()
        s = Settings().model_copy(
            update={"publish_orion_bus_grammar": True, "bus_observer_streams": "orion:evt:gateway"}
        )

        captured_events = []

        async def _capture(bus_arg, events, **kwargs):
            captured_events.extend(events)

        with patch("app.bus_observer.publish_bus_transport_grammar_trace", side_effect=_capture):
            await run_observer_tick(bus=bus, settings=s)

    roles = {e.atom.semantic_role for e in captured_events if e.atom}
    assert "bus_activity_zscore_computed" not in roles
