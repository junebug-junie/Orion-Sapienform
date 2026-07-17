from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.bus_observer import (
    _fetch_redis_snapshot,
    build_rollup_from_redis_snapshot,
    count_schema_mismatches,
    load_channel_catalog_names,
    load_channel_catalog_schema_ids,
    run_observer_tick,
)
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


# ── schema-mismatch (contract_pressure) ──────────────────────────


def test_load_channel_catalog_schema_ids(tmp_path) -> None:
    catalog = tmp_path / "channels.yaml"
    catalog.write_text(
        """
channels:
  - name: "orion:core:events"
    schema_id: "CoreEventV1"
  - name: "orion:no:schema"
  - name: "orion:effect:*"
    schema_id: "GenericPayloadV1"
""",
        encoding="utf-8",
    )
    schema_ids = load_channel_catalog_schema_ids(str(catalog))
    assert schema_ids == {
        "orion:core:events": "CoreEventV1",
        "orion:effect:*": "GenericPayloadV1",
    }


def _entry(field_name: str, raw: dict) -> tuple[str, dict]:
    return ("1-0", {field_name: json.dumps(raw)})


def test_count_schema_mismatches_valid_payload_no_mismatch() -> None:
    entries = [_entry("data", {"event": "something_happened", "payload": {}})]
    mismatches, sampled = count_schema_mismatches(entries, schema_id="CoreEventV1")
    assert (mismatches, sampled) == (0, 1)


def test_count_schema_mismatches_missing_required_field_counts_as_mismatch() -> None:
    # CoreEventV1.event is required -- omitting it is a real contract violation,
    # distinct in kind from "stream not in the catalog" (catalog_drift_pressure).
    entries = [_entry("data", {"payload": {}})]
    mismatches, sampled = count_schema_mismatches(entries, schema_id="CoreEventV1")
    assert (mismatches, sampled) == (1, 1)


def test_count_schema_mismatches_unparseable_entry_counts_as_mismatch() -> None:
    entries = [("1-0", {"data": "{not valid json"})]
    mismatches, sampled = count_schema_mismatches(entries, schema_id="CoreEventV1")
    assert (mismatches, sampled) == (1, 1)


def test_count_schema_mismatches_no_known_envelope_field_counts_as_mismatch() -> None:
    entries = [("1-0", {"some_unknown_field": "x"})]
    mismatches, sampled = count_schema_mismatches(entries, schema_id="CoreEventV1")
    assert (mismatches, sampled) == (1, 1)


def test_count_schema_mismatches_unknown_schema_id_is_noop() -> None:
    entries = [_entry("data", {"event": "x"})]
    mismatches, sampled = count_schema_mismatches(entries, schema_id="NotARealModelV1")
    assert (mismatches, sampled) == (0, 0)


def test_count_schema_mismatches_mixed_sample() -> None:
    entries = [
        _entry("data", {"event": "ok"}),
        _entry("data", {"payload": {}}),  # missing required event
        _entry("envelope", {"event": "also_ok"}),
    ]
    mismatches, sampled = count_schema_mismatches(entries, schema_id="CoreEventV1")
    assert (mismatches, sampled) == (1, 3)


def test_rollup_records_schema_mismatch_only_for_cataloged_streams_with_bad_samples() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:core:events,orion:no:schema",
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {"orion:core:events": 10, "orion:no:schema": 5},
        "catalog_names": {"orion:core:events", "orion:no:schema"},
        "catalog_schema_ids": {"orion:core:events": "CoreEventV1"},
        "stream_samples": {
            "orion:core:events": [_entry("data", {"payload": {}})],
            "orion:no:schema": [_entry("data", {"anything": "goes"})],
        },
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    # orion:no:schema has no schema_id in catalog_schema_ids -> never checked,
    # even though it has a sample -- there's nothing to validate it against.
    assert rollup.schema_mismatches == [("orion:core:events", 1, 1)]
    collector = rollup.to_collector(code_version="0.1.0")
    roles = {a.semantic_role for a in collector._atoms.values()}
    assert "bus_schema_validation_failed" in roles


def test_rollup_omits_schema_mismatch_atom_when_all_samples_valid() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:core:events",
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {"orion:core:events": 10},
        "catalog_names": {"orion:core:events"},
        "catalog_schema_ids": {"orion:core:events": "CoreEventV1"},
        "stream_samples": {
            "orion:core:events": [_entry("data", {"event": "fine"})],
        },
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.schema_mismatches == []


def test_rollup_backward_compatible_without_schema_snapshot_keys() -> None:
    """Pre-existing callers (and the two tests above this one in the file)
    build a snapshot dict without catalog_schema_ids/stream_samples at all --
    must still build a valid rollup with no schema-mismatch atoms, not
    raise."""
    settings = Settings().model_copy(
        update={
            "bus_observer_node_id": "athena",
            "bus_observer_streams": "orion:evt:gateway",
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {"orion:evt:gateway": 1},
        "catalog_names": {"orion:evt:gateway"},
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.schema_mismatches == []


@pytest.mark.asyncio
async def test_fetch_snapshot_samples_only_cataloged_streams_bounded_by_setting() -> None:
    """XREVRANGE is called only for configured streams that have a
    schema_id in the catalog (uncataloged streams have nothing to validate
    against, and are already covered by catalog_names/uncataloged_streams),
    and always bounded by bus_observer_schema_sample_count -- this is the
    real Redis-cost knob."""
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:core:events,orion:no:schema",
            "bus_observer_schema_sample_count": 5,
        }
    )

    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=3)
    fake_client.xrevrange = AsyncMock(return_value=[("1-0", {"data": "{}"})])
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch(
            "app.bus_observer.load_channel_catalog_names",
            return_value={"orion:core:events", "orion:no:schema"},
        ),
        patch(
            "app.bus_observer.load_channel_catalog_schema_ids",
            return_value={"orion:core:events": "CoreEventV1"},
        ),
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    fake_client.xrevrange.assert_awaited_once_with("orion:core:events", count=5)
    assert "orion:core:events" in snapshot["stream_samples"]
    assert "orion:no:schema" not in snapshot["stream_samples"]


@pytest.mark.asyncio
async def test_fetch_snapshot_skips_sampling_when_sample_count_zero() -> None:
    settings = Settings().model_copy(
        update={
            "bus_observer_streams": "orion:core:events",
            "bus_observer_schema_sample_count": 0,
        }
    )
    fake_client = AsyncMock()
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.xlen = AsyncMock(return_value=3)
    fake_client.xrevrange = AsyncMock(return_value=[])
    fake_client.aclose = AsyncMock()

    with (
        patch("app.bus_observer.aioredis.from_url", return_value=fake_client),
        patch(
            "app.bus_observer.load_channel_catalog_names",
            return_value={"orion:core:events"},
        ),
        patch(
            "app.bus_observer.load_channel_catalog_schema_ids",
            return_value={"orion:core:events": "CoreEventV1"},
        ),
    ):
        snapshot = await _fetch_redis_snapshot(settings)

    fake_client.xrevrange.assert_not_awaited()
    assert snapshot["stream_samples"] == {}


# ── shipped-default regression (review finding, HIGH) ────────────
#
# The original 8 tests above all injected a synthetic catalog_schema_ids
# override, so none of them would have caught this: the REAL shipped default
# BUS_OBSERVER_STREAMS (orion:evt:gateway,orion:bus:out) had NEITHER stream
# cataloged in the real orion/bus/channels.yaml -- both exist purely for
# depth/backpressure monitoring, a different, older purpose. That meant
# catalog_schema_ids was always empty for the deployed config,
# count_schema_mismatches() never ran, schema_mismatch_stream_count stayed
# permanently 0, and contract_pressure stayed permanently 0.0 -- dead on
# arrival under the config that's actually deployed.
#
# A first attempted fix added orion:grammar:event (cataloged,
# schema_id=GrammarEventV1) to the default. That looked right on paper but
# was ALSO dead: verified live against bus-core Redis with TYPE/XLEN that
# orion:grammar:event is TYPE=none -- it is delivered purely via Redis
# PUBLISH (OrionBusAsync.publish(), never XADD'd anywhere in this repo), so
# XLEN/XREVRANGE structurally can never see it, no matter how much traffic
# flows through it. "Cataloged with a schema_id" is necessary but not
# sufficient -- the stream also has to genuinely be a Redis Stream (XADD'd),
# not a Pub/Sub-only channel that happens to be cataloged too. Kept in the
# default anyway (harmless) but it's not the fix.
#
# The real fix is orion:stream:world_pulse:run:result: cataloged
# (schema_id=WorldPulseRunResultV1, kind=stream in channels.yaml) AND
# genuinely XADD'd by orion-world-pulse via RedisStreamWorkQueue.enqueue()
# (WP_RUN_RESULT_STREAM_ENABLED=true by default). Verified live: TYPE=stream,
# XLEN=82 real entries, and a live-sampled XREVRANGE entry decodes via the
# "envelope" field exactly as count_schema_mismatches() expects.
#
# These tests use the real Settings() default and the real channels.yaml on
# disk -- no synthetic overrides of either -- so a regression back to "no
# genuinely-Stream cataloged stream in the default" fails loudly here.


def test_shipped_default_streams_includes_a_real_cataloged_schema_stream() -> None:
    settings = Settings()
    # The genuinely-fixing stream: cataloged AND (verified live, not
    # asserted here since that needs a Redis connection) a real XADD Stream.
    assert "orion:stream:world_pulse:run:result" in settings.observer_stream_list
    # Kept for the reasons above, but neither one is what makes sampling
    # fire -- see the module comment above this test.
    assert "orion:grammar:event" in settings.observer_stream_list
    assert "orion:evt:gateway" in settings.observer_stream_list
    assert "orion:bus:out" in settings.observer_stream_list

    schema_ids = load_channel_catalog_schema_ids(settings.channels_catalog_path)
    assert schema_ids.get("orion:stream:world_pulse:run:result") == "WorldPulseRunResultV1"
    assert schema_ids.get("orion:grammar:event") == "GrammarEventV1"
    # Confirm the finding itself, so this test stays honest if channels.yaml
    # is ever edited to catalog these two.
    assert "orion:evt:gateway" not in schema_ids
    assert "orion:bus:out" not in schema_ids


@pytest.mark.asyncio
async def test_shipped_default_produces_nonzero_mismatch_count_for_malformed_sample() -> None:
    """End-to-end with the REAL default BUS_OBSERVER_STREAMS and the REAL
    orion/bus/channels.yaml catalog on disk (no synthetic overrides of
    either) -- a malformed orion:stream:world_pulse:run:result entry (the
    genuinely-Stream-backed cataloged channel, not the Pub/Sub-only
    orion:grammar:event) must actually move schema_mismatch_stream_count off
    zero. Shaped after a real sampled entry from live bus-core Redis: the
    "envelope" field holds a JSON orion.envelope whose payload is
    WorldPulseRunResultV1-shaped."""
    settings = Settings()
    # WorldPulseRunResultV1.run is required (orion/schemas/world_pulse.py) --
    # this envelope's payload has no "run" key at all, a realistic
    # "producer sent something that doesn't match its declared contract"
    # case, same shape as the real entries seen live (schema="orion.envelope"
    # top-level, not the legacy-wrap path).
    malformed_envelope = json.dumps(
        {
            "schema": "orion.envelope",
            "schema_version": "2.0.0",
            "id": "11111111-1111-1111-1111-111111111111",
            "correlation_id": "22222222-2222-2222-2222-222222222222",
            "kind": "world.pulse.run.result.v1",
            "source": {"name": "orion-world-pulse"},
            "payload": {"not_run": "missing the required 'run' key"},
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {sk: 1 for sk in settings.observer_stream_list},
        "catalog_names": load_channel_catalog_names(settings.channels_catalog_path),
        "catalog_schema_ids": load_channel_catalog_schema_ids(settings.channels_catalog_path),
        "stream_samples": {
            "orion:stream:world_pulse:run:result": [
                ("1-0", {"envelope": malformed_envelope}),
            ],
        },
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.schema_mismatches, (
        "expected a schema mismatch for orion:stream:world_pulse:run:result "
        "under the real shipped BUS_OBSERVER_STREAMS default -- if this is "
        "empty, schema-mismatch sampling is dead on arrival again"
    )
    stream_key, mismatch_count, sampled_count = rollup.schema_mismatches[0]
    assert stream_key == "orion:stream:world_pulse:run:result"
    assert mismatch_count == 1
    assert sampled_count == 1


@pytest.mark.asyncio
async def test_shipped_default_no_mismatch_for_well_formed_sample() -> None:
    """Same real default + real catalog, but a well-formed
    WorldPulseRunResultV1 payload (minimal valid WorldPulseRunV1: only
    run_id/date/started_at are required, orion/schemas/world_pulse.py)
    must NOT be flagged -- proves the check discriminates, not just
    "always mismatch"."""
    settings = Settings()
    valid_envelope = json.dumps(
        {
            "schema": "orion.envelope",
            "schema_version": "2.0.0",
            "id": "11111111-1111-1111-1111-111111111111",
            "correlation_id": "22222222-2222-2222-2222-222222222222",
            "kind": "world.pulse.run.result.v1",
            "source": {"name": "orion-world-pulse"},
            "payload": {
                "run": {
                    "run_id": "test-run-id",
                    "date": "2026-07-17",
                    "started_at": "2026-07-17T00:00:00Z",
                }
            },
        }
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {sk: 1 for sk in settings.observer_stream_list},
        "catalog_names": load_channel_catalog_names(settings.channels_catalog_path),
        "catalog_schema_ids": load_channel_catalog_schema_ids(settings.channels_catalog_path),
        "stream_samples": {
            "orion:stream:world_pulse:run:result": [
                ("1-0", {"envelope": valid_envelope}),
            ],
        },
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.schema_mismatches == []
