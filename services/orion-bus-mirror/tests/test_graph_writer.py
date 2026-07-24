from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from app.graph_writer import (
    BusEventFact,
    BusSynapticGraphWriter,
    ChainTracker,
    compute_ewma_update,
    extract_bus_event_fact,
)


class FakeGraphClient:
    """Minimal in-memory stand-in for FalkorGraphClient that actually tracks
    edge state across calls, so multi-call EWMA/z-score evolution can be
    tested for real instead of only checking which params were sent.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self._edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        self.calls.append((cypher, params))
        stripped = cypher.strip()

        if "PUBLISHES" in cypher:
            key = ("PUBLISHES", params.get("organ_id"), params.get("channel"))
        elif "CAUSALLY_FOLLOWED_BY" in cypher:
            key = ("CAUSALLY_FOLLOWED_BY", params.get("prior_organ_id"), params.get("organ_id"))
        else:
            return []

        if stripped.startswith("MATCH"):
            props = self._edges.get(key)
            return [props] if props else []

        # MERGE + SET write
        self._edges[key] = dict(params)
        return []


@pytest.fixture
def fake_client() -> FakeGraphClient:
    return FakeGraphClient()


class TestComputeEwmaUpdate:
    def test_first_observation_has_no_zscore(self) -> None:
        update = compute_ewma_update(prev_ewma=0.0, prev_variance=0.0, prev_count=0, value=5.0, alpha=0.2)
        assert update.zscore is None
        assert update.ewma == 5.0
        assert update.variance == 0.0

    def test_second_observation_computes_zscore_against_prior_baseline(self) -> None:
        update = compute_ewma_update(prev_ewma=5.0, prev_variance=1.0, prev_count=1, value=5.0, alpha=0.2)
        assert update.zscore == pytest.approx(0.0)

    def test_large_deviation_produces_large_zscore(self) -> None:
        update = compute_ewma_update(prev_ewma=1.0, prev_variance=0.01, prev_count=5, value=100.0, alpha=0.2)
        assert update.zscore is not None
        assert update.zscore > 50

    def test_zero_variance_does_not_divide_by_zero(self) -> None:
        update = compute_ewma_update(prev_ewma=1.0, prev_variance=0.0, prev_count=3, value=2.0, alpha=0.2)
        assert update.zscore is not None
        assert update.zscore == pytest.approx(1.0 / (1e-6 ** 0.5))


class TestChainTracker:
    def _fact(self, organ_id: str, correlation_id: Optional[str], epoch: float) -> BusEventFact:
        return BusEventFact(organ_id=organ_id, channel="orion:test", correlation_id=correlation_id, observed_at_epoch=epoch)

    def test_first_sighting_of_a_correlation_id_returns_none(self) -> None:
        tracker = ChainTracker(ttl_sec=60.0)
        result = tracker.observe(self._fact("cortex-exec", "corr-1", 100.0))
        assert result is None
        assert len(tracker) == 1

    def test_same_organ_repeat_returns_none_not_a_real_hop(self) -> None:
        tracker = ChainTracker(ttl_sec=60.0)
        tracker.observe(self._fact("cortex-exec", "corr-1", 100.0))
        result = tracker.observe(self._fact("cortex-exec", "corr-1", 101.0))
        assert result is None

    def test_different_organ_same_correlation_id_returns_prior_hop(self) -> None:
        tracker = ChainTracker(ttl_sec=60.0)
        tracker.observe(self._fact("cortex-exec", "corr-1", 100.0))
        result = tracker.observe(self._fact("llm-gateway", "corr-1", 117.0))
        assert result == ("cortex-exec", 100.0)

    def test_missing_correlation_id_is_never_tracked(self) -> None:
        tracker = ChainTracker(ttl_sec=60.0)
        result = tracker.observe(self._fact("cortex-exec", None, 100.0))
        assert result is None
        assert len(tracker) == 0

    def test_stale_entries_are_evicted_and_do_not_produce_a_hop(self) -> None:
        tracker = ChainTracker(ttl_sec=10.0)
        tracker.observe(self._fact("cortex-exec", "corr-1", 100.0))
        # 50s later, well past the 10s TTL -- this must NOT be treated as a
        # real causal hop (that would fabricate a huge, meaningless latency).
        result = tracker.observe(self._fact("llm-gateway", "corr-1", 150.0))
        assert result is None
        assert len(tracker) == 1  # only the new entry remains

    def test_eviction_bounds_table_size_under_sustained_traffic(self) -> None:
        tracker = ChainTracker(ttl_sec=5.0)
        for i in range(1000):
            tracker.observe(self._fact("organ-a", f"corr-{i}", float(i)))
        # Only entries within the last 5s of the final observed_at_epoch survive.
        assert len(tracker) <= 6


class TestBusSynapticGraphWriterPublish:
    def test_first_publish_on_an_edge_has_no_zscore_and_count_one(self, fake_client: FakeGraphClient) -> None:
        writer = BusSynapticGraphWriter(fake_client, alpha=0.2)
        fact = BusEventFact(organ_id="cortex-exec", channel="orion:system:health", correlation_id=None, observed_at_epoch=1000.0)

        writer.record_publish(fact)

        write_call = fake_client.calls[-1]
        assert write_call[1]["count"] == 1
        assert write_call[1]["gap_zscore"] is None

    def test_second_publish_computes_a_real_gap_and_zscore(self, fake_client: FakeGraphClient) -> None:
        writer = BusSynapticGraphWriter(fake_client, alpha=0.2)
        writer.record_publish(
            BusEventFact(organ_id="cortex-exec", channel="orion:system:health", correlation_id=None, observed_at_epoch=1000.0)
        )
        writer.record_publish(
            BusEventFact(organ_id="cortex-exec", channel="orion:system:health", correlation_id=None, observed_at_epoch=1010.0)
        )

        write_call = fake_client.calls[-1]
        assert write_call[1]["count"] == 2
        # First gap becomes the baseline (ewma=10.0, var=0.0) with no zscore;
        # this second call's gap (also 10.0) should read as unsurprising.
        assert write_call[1]["gap_zscore"] is not None

    def test_distinct_channels_from_the_same_organ_are_independent_edges(self, fake_client: FakeGraphClient) -> None:
        writer = BusSynapticGraphWriter(fake_client, alpha=0.2)
        writer.record_publish(
            BusEventFact(organ_id="cortex-exec", channel="orion:a", correlation_id=None, observed_at_epoch=1000.0)
        )
        writer.record_publish(
            BusEventFact(organ_id="cortex-exec", channel="orion:b", correlation_id=None, observed_at_epoch=1000.5)
        )

        assert fake_client._edges[("PUBLISHES", "cortex-exec", "orion:a")]["count"] == 1
        assert fake_client._edges[("PUBLISHES", "cortex-exec", "orion:b")]["count"] == 1


class TestBusSynapticGraphWriterCausalHop:
    def test_first_hop_between_two_organs_has_no_zscore(self, fake_client: FakeGraphClient) -> None:
        writer = BusSynapticGraphWriter(fake_client, alpha=0.2)
        fact = BusEventFact(organ_id="llm-gateway", channel="orion:exec:result:LLMGatewayService", correlation_id="corr-1", observed_at_epoch=1017.0)

        writer.record_causal_hop(prior_organ_id="cortex-exec", prior_epoch=1000.0, fact=fact)

        write_call = fake_client.calls[-1]
        assert write_call[1]["count"] == 1
        assert write_call[1]["latency_zscore"] is None
        assert write_call[1]["latency_ewma_sec"] == pytest.approx(17.0)

    def test_negative_clock_skew_is_clamped_to_zero_not_negative_latency(self, fake_client: FakeGraphClient) -> None:
        writer = BusSynapticGraphWriter(fake_client, alpha=0.2)
        fact = BusEventFact(organ_id="llm-gateway", channel="orion:exec:result:LLMGatewayService", correlation_id="corr-1", observed_at_epoch=999.0)

        writer.record_causal_hop(prior_organ_id="cortex-exec", prior_epoch=1000.0, fact=fact)

        write_call = fake_client.calls[-1]
        assert write_call[1]["latency_ewma_sec"] == 0.0


class TestExtractBusEventFact:
    @dataclass
    class _FakeSource:
        name: Optional[str]

    @dataclass
    class _FakeEnvelope:
        source: Any
        correlation_id: Any = None

    def test_extracts_organ_and_channel_and_correlation_id(self) -> None:
        envelope = self._FakeEnvelope(source=self._FakeSource(name="cortex-exec"), correlation_id="corr-1")
        fact = extract_bus_event_fact(envelope, channel="orion:cognition:trace", now=123.0)
        assert fact == BusEventFact(organ_id="cortex-exec", channel="orion:cognition:trace", correlation_id="corr-1", observed_at_epoch=123.0)

    def test_missing_source_name_fails_open_to_none(self) -> None:
        envelope = self._FakeEnvelope(source=self._FakeSource(name=None))
        fact = extract_bus_event_fact(envelope, channel="orion:x", now=1.0)
        assert fact is None

    def test_missing_source_entirely_fails_open_to_none(self) -> None:
        envelope = self._FakeEnvelope(source=None)
        fact = extract_bus_event_fact(envelope, channel="orion:x", now=1.0)
        assert fact is None

    def test_no_correlation_id_becomes_none_not_empty_string(self) -> None:
        envelope = self._FakeEnvelope(source=self._FakeSource(name="cortex-exec"), correlation_id=None)
        fact = extract_bus_event_fact(envelope, channel="orion:x", now=1.0)
        assert fact is not None
        assert fact.correlation_id is None
