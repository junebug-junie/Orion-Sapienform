from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from app.graph_writer import ChainTracker
from app.main import _record_graph_event


@dataclass
class _FakeSource:
    name: Optional[str]


@dataclass
class _FakeEnvelope:
    source: Any
    correlation_id: Any = None


@dataclass
class _FakeDecodeResult:
    envelope: Any
    ok: bool = True


class TestRecordGraphEventFailsOpen:
    @pytest.mark.asyncio
    async def test_writer_exception_is_swallowed_not_propagated(self) -> None:
        writer = MagicMock()
        writer.record_publish.side_effect = RuntimeError("falkor unreachable")
        chain_tracker = ChainTracker(ttl_sec=60.0)
        decoded = _FakeDecodeResult(envelope=_FakeEnvelope(source=_FakeSource(name="cortex-exec")))

        # Must not raise -- this is the whole point of the fail-open contract:
        # a graph-write failure can never block or crash the mirror's main loop.
        await _record_graph_event(writer, chain_tracker, decoded, "orion:test", 100.0)

        writer.record_publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_organ_id_skips_writer_entirely(self) -> None:
        writer = MagicMock()
        chain_tracker = ChainTracker(ttl_sec=60.0)
        decoded = _FakeDecodeResult(envelope=_FakeEnvelope(source=_FakeSource(name=None)))

        await _record_graph_event(writer, chain_tracker, decoded, "orion:test", 100.0)

        writer.record_publish.assert_not_called()
        writer.record_causal_hop.assert_not_called()
        assert len(chain_tracker) == 0

    @pytest.mark.asyncio
    async def test_chain_tracker_records_first_sighting_even_if_write_fails(self) -> None:
        # Regression test for a review finding: chain-tracker bookkeeping must
        # happen before the (possibly-failing) Falkor write, not after -- a
        # transient outage must not silently drop a correlation_id's first
        # sighting, or its later leg would never produce a causal-hop edge.
        writer = MagicMock()
        writer.record_publish.side_effect = RuntimeError("falkor unreachable")
        chain_tracker = ChainTracker(ttl_sec=60.0)
        decoded = _FakeDecodeResult(
            envelope=_FakeEnvelope(source=_FakeSource(name="cortex-exec"), correlation_id="corr-1")
        )

        await _record_graph_event(writer, chain_tracker, decoded, "orion:test", 100.0)

        assert len(chain_tracker) == 1

    @pytest.mark.asyncio
    async def test_causal_hop_recorded_on_second_organ_for_same_correlation_id(self) -> None:
        writer = MagicMock()
        chain_tracker = ChainTracker(ttl_sec=60.0)
        first = _FakeDecodeResult(
            envelope=_FakeEnvelope(source=_FakeSource(name="cortex-exec"), correlation_id="corr-1")
        )
        second = _FakeDecodeResult(
            envelope=_FakeEnvelope(source=_FakeSource(name="llm-gateway"), correlation_id="corr-1")
        )

        await _record_graph_event(writer, chain_tracker, first, "orion:a", 100.0)
        await _record_graph_event(writer, chain_tracker, second, "orion:b", 117.0)

        writer.record_causal_hop.assert_called_once()
        _, kwargs = writer.record_causal_hop.call_args
        assert kwargs["prior_organ_id"] == "cortex-exec"
        assert kwargs["prior_epoch"] == 100.0
