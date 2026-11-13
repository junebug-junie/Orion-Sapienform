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


def _steady(acc: TrafficRollupAccumulator, start: float, end: float, step: float = 10.0) -> None:
    t = start
    while t < end:
        acc.observe_message("orion:system:health", t)
        t += step


def test_startup_bucket_is_flagged_partial_and_the_next_is_not() -> None:
    acc = _acc(started_at_epoch=T0 + 120)
    _steady(acc, T0 + 125, T0 + 600)
    events = acc.flush_closed(T0 + 600)
    assert [(e.bucket_start.timestamp(), e.partial) for e in events] == [(T0, True), (T0 + 300, False)]


def test_start_exactly_on_a_boundary_is_not_partial() -> None:
    acc = _acc(started_at_epoch=T0)
    _steady(acc, T0 + 1, T0 + 300)
    [event] = acc.flush_closed(T0 + 300)
    assert event.partial is False


def test_slow_first_connect_flags_the_first_bucket_partial() -> None:
    # Started on a boundary, but the first message only arrived 45 s later.
    acc = _acc(started_at_epoch=T0)
    _steady(acc, T0 + 45, T0 + 300)
    [event] = acc.flush_closed(T0 + 300)
    assert event.partial is True


def test_silence_gap_flags_both_sides_and_skips_blind_buckets() -> None:
    acc = _acc(started_at_epoch=T0)
    _steady(acc, T0, T0 + 200)          # bucket T0 healthy until 190 s
    _steady(acc, T0 + 1000, T0 + 1500)  # resubscribed at 1000 s (bucket T0+900)
    events = acc.flush_closed(T0 + 1500)
    assert [(e.bucket_start.timestamp(), e.partial) for e in events] == [
        (T0, True), (T0 + 900, True), (T0 + 1200, False),
    ]


def test_short_silence_is_not_a_gap() -> None:
    acc = _acc(started_at_epoch=T0)
    acc.observe_message("orion:a", T0 + 1)
    acc.observe_message("orion:a", T0 + 29)
    [event] = acc.flush_closed(T0 + 300)
    assert event.partial is False


def test_grace_holds_a_just_closed_bucket() -> None:
    acc = _acc()
    acc.observe_message("orion:a", T0 + 299.5)
    assert acc.flush_closed(T0 + 303, grace_sec=5.0) == []
    assert len(acc.flush_closed(T0 + 305, grace_sec=5.0)) == 1


def test_late_message_after_flush_goes_to_the_next_open_bucket() -> None:
    # The mirror stamps `now` before its SQLite commit await; if the flush ran in
    # between, the message must not recreate the published bucket (duplicate rollup_id
    # -> dropped by the sql-writer -> message lost).
    acc = _acc()
    acc.observe_message("orion:a", T0 + 10)
    [first] = acc.flush_closed(T0 + 300)
    acc.observe_message("orion:a", T0 + 299.9)  # late, stamped in the published bucket
    [second] = acc.flush_closed(T0 + 600)
    assert first.message_count == 1
    assert second.bucket_start.timestamp() == T0 + 300
    assert second.message_count == 1
    assert second.rollup_id != first.rollup_id


def test_hop_overflow_key_and_negative_latency_clamp_together() -> None:
    acc = _acc(max_hops_per_bucket=1)
    acc.observe_hop("a", "b", 2.0, T0)
    acc.observe_hop("c", "d", -5.0, T0)  # overflow + clamped to 0
    acc.observe_hop("e", "f", 4.0, T0)   # overflow
    acc.observe_hop("a", "b", 4.0, T0)   # existing key keeps counting
    [event] = acc.flush_closed(T0 + 300)
    hops = {(h.from_organ, h.to_organ): (h.count, h.latency_mean_sec) for h in event.hops}
    assert hops == {("a", "b"): (2, 3.0), ("__overflow__", "__overflow__"): (2, 2.0)}


def test_event_round_trips_through_json() -> None:
    acc = _acc()
    acc.observe_message("orion:a", T0)
    acc.observe_hop("a", "b", 1.0, T0)
    [event] = acc.flush_closed(T0 + 300)
    again = BusTrafficRollupV1.model_validate(event.model_dump(mode="json"))
    assert again == event
