"""Unit tests for RpcHealthAggregator (orion/core/bus/rpc_health.py).

Step 2 of docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md.
Pure, no asyncio/Redis required.
"""

from __future__ import annotations

from orion.core.bus.rpc_health import (
    MAX_DISTINCT_CHANNELS,
    MAX_SAMPLES_PER_BUCKET,
    RpcHealthAggregator,
)


def test_empty_snapshot_has_no_data() -> None:
    agg = RpcHealthAggregator()
    snap = agg.snapshot_and_reset()
    assert snap.success_count == 0
    assert snap.timeout_count == 0
    assert snap.success_latency_ms_p50 is None
    assert snap.success_latency_ms_p95 is None
    assert snap.success_latency_ms_max is None
    assert snap.timeout_elapsed_ms_max is None
    assert snap.channel_counts == {}
    assert snap.truncated is False


def test_record_success_counts_and_tracks_latency() -> None:
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:test:request", latency_ms=100.0)
    agg.record_success(request_channel="orion:test:request", latency_ms=200.0)
    snap = agg.snapshot_and_reset()
    assert snap.success_count == 2
    assert snap.timeout_count == 0
    assert snap.success_latency_ms_max == 200.0
    assert snap.channel_counts == {"orion:test:request": 2}


def test_record_timeout_counts_and_tracks_elapsed_separately_from_success() -> None:
    """A timeout's elapsed_ms must never leak into success latency stats -- same
    conflation bug already found and fixed in measure_rpc_health_baseline.py."""
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:test:request", latency_ms=100.0)
    agg.record_timeout(request_channel="orion:test:request", elapsed_ms=420000.0)
    snap = agg.snapshot_and_reset()
    assert snap.success_count == 1
    assert snap.timeout_count == 1
    assert snap.success_latency_ms_max == 100.0
    assert snap.timeout_elapsed_ms_max == 420000.0
    assert snap.channel_counts == {"orion:test:request": 2}


def test_snapshot_and_reset_clears_state_for_next_window() -> None:
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="c", latency_ms=1.0)
    first = agg.snapshot_and_reset()
    assert first.success_count == 1

    second = agg.snapshot_and_reset()
    assert second.success_count == 0
    assert second.channel_counts == {}


def test_snapshot_window_advances_between_drains() -> None:
    agg = RpcHealthAggregator()
    first = agg.snapshot_and_reset()
    second = agg.snapshot_and_reset()
    assert second.window_start == first.window_end


def test_percentiles_computed_over_success_latencies() -> None:
    agg = RpcHealthAggregator()
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        agg.record_success(request_channel="c", latency_ms=v)
    snap = agg.snapshot_and_reset()
    assert snap.success_latency_ms_p50 == 30.0
    assert snap.success_latency_ms_max == 50.0


def test_success_latency_bounded_and_flags_truncation() -> None:
    agg = RpcHealthAggregator()
    for i in range(MAX_SAMPLES_PER_BUCKET + 10):
        agg.record_success(request_channel="c", latency_ms=float(i))
    snap = agg.snapshot_and_reset()
    # count is real (not silently dropped), only the raw sample list is capped
    assert snap.success_count == MAX_SAMPLES_PER_BUCKET + 10
    assert snap.truncated is True


def test_timeout_elapsed_bounded_and_flags_truncation() -> None:
    agg = RpcHealthAggregator()
    for i in range(MAX_SAMPLES_PER_BUCKET + 5):
        agg.record_timeout(request_channel="c", elapsed_ms=float(i))
    snap = agg.snapshot_and_reset()
    assert snap.timeout_count == MAX_SAMPLES_PER_BUCKET + 5
    assert snap.truncated is True


def test_distinct_channel_count_bounded_and_flags_truncation() -> None:
    agg = RpcHealthAggregator()
    for i in range(MAX_DISTINCT_CHANNELS + 20):
        agg.record_success(request_channel=f"orion:test:request:{i}", latency_ms=1.0)
    snap = agg.snapshot_and_reset()
    assert len(snap.channel_counts) == MAX_DISTINCT_CHANNELS
    assert snap.truncated is True


def test_existing_channel_still_increments_past_distinct_cap() -> None:
    """Once the distinct-channel cap is hit, an *existing* channel key must still
    increment correctly -- only brand-new channel names get dropped."""
    agg = RpcHealthAggregator()
    for i in range(MAX_DISTINCT_CHANNELS):
        agg.record_success(request_channel=f"orion:test:request:{i}", latency_ms=1.0)
    # cap now hit; record more against an already-tracked channel
    agg.record_success(request_channel="orion:test:request:0", latency_ms=1.0)
    agg.record_success(request_channel="orion:test:request:0", latency_ms=1.0)
    snap = agg.snapshot_and_reset()
    assert snap.channel_counts["orion:test:request:0"] == 3


def test_empty_request_channel_ignored_without_crashing() -> None:
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="", latency_ms=1.0)
    snap = agg.snapshot_and_reset()
    assert snap.success_count == 1
    assert snap.channel_counts == {}
