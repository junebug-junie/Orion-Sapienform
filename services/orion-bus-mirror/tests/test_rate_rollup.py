"""TrafficRollupAccumulator: wall-clock 5-minute bucketing, flush, hop math."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.rate_rollup import (
    OVERFLOW_CHANNEL_KEY,
    TrafficRollupAccumulator,
    bucket_start_epoch,
)
from orion.schemas.bus_traffic import BusTrafficRollupV1

T0 = 1_763_000_100  # a multiple of 300: 2025-11-13T02:15:00Z
assert T0 % 300 == 0


def _acc(**kw) -> TrafficRollupAccumulator:
    kw.setdefault("node", "athena")
    return TrafficRollupAccumulator(**kw)


def test_bucket_start_is_floor_of_epoch_over_300() -> None:
    assert bucket_start_epoch(T0) == T0
    assert bucket_start_epoch(T0 + 299.9) == T0
    assert bucket_start_epoch(T0 + 300) == T0 + 300


def test_open_bucket_is_not_flushed() -> None:
    acc = _acc()
    acc.observe_message("orion:a", T0 + 10)
    assert acc.flush_closed(T0 + 299.99) == []
    assert len(acc) == 1


def test_closed_bucket_flushes_once_with_counts() -> None:
    acc = _acc()
    acc.observe_message("orion:a", T0 + 1)
    acc.observe_message("orion:a", T0 + 2)
    acc.observe_message("orion:b", T0 + 3, decode_failed=True)
    acc.observe_message("orion:a", T0 + 301)  # next bucket

    [event] = acc.flush_closed(T0 + 300)
    assert isinstance(event, BusTrafficRollupV1)
    assert event.bucket_start == datetime.fromtimestamp(T0, tz=timezone.utc)
    assert event.bucket_end == datetime.fromtimestamp(T0 + 300, tz=timezone.utc)
    assert event.rollup_id == f"athena:{event.bucket_start.isoformat()}"
    assert event.message_count == 3
    assert event.decode_failures == 1
    assert event.channel_counts == {"orion:a": 2, "orion:b": 1}
    assert event.hops == []
    # Flushed buckets are forgotten; the next one is still open.
    assert acc.flush_closed(T0 + 300) == []
    assert len(acc) == 1


def test_several_closed_buckets_flush_oldest_first_and_quiet_buckets_are_absent() -> None:
    acc = _acc()
    acc.observe_message("orion:a", T0 + 900)
    acc.observe_message("orion:a", T0 + 5)
    events = acc.flush_closed(T0 + 2000)
    assert [e.bucket_start.timestamp() for e in events] == [T0, T0 + 900]


def test_dynamic_reply_channel_collapses_to_catalog_wildcard() -> None:
    acc = _acc(catalog_names={"orion:exec:result:*"})
    acc.observe_message("orion:exec:result:LLMGatewayService:0cf1-uuid", T0)
    acc.observe_message("orion:exec:result:LLMGatewayService:9abc-uuid", T0)
    [event] = acc.flush_closed(T0 + 300)
    assert event.channel_counts == {"orion:exec:result:*": 2}


def test_distinct_channels_per_bucket_are_capped() -> None:
    acc = _acc(max_channels_per_bucket=2)
    for i in range(5):
        acc.observe_message(f"orion:uncataloged:{i}", T0)
    acc.observe_message("orion:uncataloged:0", T0)  # an existing key keeps counting
    [event] = acc.flush_closed(T0 + 300)
    assert event.channel_counts == {
        "orion:uncataloged:0": 2,
        "orion:uncataloged:1": 1,
        OVERFLOW_CHANNEL_KEY: 3,
    }
    assert event.message_count == 6


def test_hop_count_and_mean_latency() -> None:
    acc = _acc()
    acc.observe_hop("cortex-exec", "llm-gateway", 2.0, T0 + 1)
    acc.observe_hop("cortex-exec", "llm-gateway", 4.0, T0 + 2)
    acc.observe_hop("hub", "cortex-exec", 0.5, T0 + 3)
    acc.observe_hop("hub", "cortex-exec", -1.0, T0 + 4)  # clock step back: clamped to 0
    [event] = acc.flush_closed(T0 + 300)
    hops = {(h.from_organ, h.to_organ): h for h in event.hops}
    assert hops[("cortex-exec", "llm-gateway")].count == 2
    assert hops[("cortex-exec", "llm-gateway")].latency_mean_sec == pytest.approx(3.0)
    assert hops[("hub", "cortex-exec")].count == 2
    assert hops[("hub", "cortex-exec")].latency_mean_sec == pytest.approx(0.25)
    # A hop-only bucket still emits, with message_count 0.
    assert event.message_count == 0


def test_partial_startup_bucket_is_dropped() -> None:
    acc = _acc(started_at_epoch=T0 + 120)
    acc.observe_message("orion:a", T0 + 130)  # startup bucket: partial
    acc.observe_message("orion:a", T0 + 310)  # first full bucket
    events = acc.flush_closed(T0 + 600)
    assert [e.bucket_start.timestamp() for e in events] == [T0 + 300]


def test_start_exactly_on_a_boundary_keeps_that_bucket() -> None:
    acc = _acc(started_at_epoch=T0)
    acc.observe_message("orion:a", T0 + 1)
    [event] = acc.flush_closed(T0 + 300)
    assert event.bucket_start.timestamp() == T0


def test_event_round_trips_through_json() -> None:
    acc = _acc()
    acc.observe_message("orion:a", T0)
    acc.observe_hop("a", "b", 1.0, T0)
    [event] = acc.flush_closed(T0 + 300)
    again = BusTrafficRollupV1.model_validate(event.model_dump(mode="json"))
    assert again == event
