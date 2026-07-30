from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.substrate.bus_synaptic import query_real_bus_synaptic_raw_mean_abs_z


@pytest.mark.asyncio
async def test_query_parses_real_graph_query_reply_shape() -> None:
    # Real GRAPH.QUERY reply shape confirmed live 2026-07-28:
    # [header, [[value]], stats] -- result[1][0][0] is the scalar.
    fake_result = [["avg(abs(rel.gap_zscore))"], [[1.0858]], ["stats..."]]
    mock_client = AsyncMock()
    mock_client.execute_command.return_value = fake_result

    with patch("app.substrate.bus_synaptic.aioredis.from_url", return_value=mock_client):
        value = await query_real_bus_synaptic_raw_mean_abs_z("redis://fake:6379", "orion_bus_synapse")

    assert value == pytest.approx(1.0858)
    mock_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_falls_back_to_zero_on_connection_failure() -> None:
    # Confirmed live 2026-07-28: a synchronous CPU-bound absorb() burst
    # starving the event loop produced exactly this failure mode against a
    # real, otherwise-healthy FalkorDB -- this function must degrade to
    # "no reheat this tick", never raise into the dissipation loop.
    with patch("app.substrate.bus_synaptic.aioredis.from_url", side_effect=TimeoutError("boom")):
        value = await query_real_bus_synaptic_raw_mean_abs_z("redis://fake:6379", "orion_bus_synapse")

    assert value == 0.0


@pytest.mark.asyncio
async def test_query_falls_back_to_zero_on_malformed_reply() -> None:
    mock_client = AsyncMock()
    mock_client.execute_command.return_value = [["header"], [], []]  # empty result rows

    with patch("app.substrate.bus_synaptic.aioredis.from_url", return_value=mock_client):
        value = await query_real_bus_synaptic_raw_mean_abs_z("redis://fake:6379", "orion_bus_synapse")

    assert value == 0.0


@pytest.mark.asyncio
async def test_query_returns_zero_when_every_edge_has_aged_out() -> None:
    """Confirmed live 2026-07-30: `avg()` over an empty match returns a real
    SQL-style NULL (`[[null]]`), not 0.0 or an empty row list. Once the recency
    filter was added, "no edge has published in the last hour" became a
    reachable state -- and without an explicit None guard `float(None)` would
    raise every tick, get swallowed by the broad except below, and log a
    spurious `bus_synaptic_query_failed` warning for what is actually a
    correct, meaningful reading: no live traffic, therefore no reheat."""
    mock_client = AsyncMock()
    mock_client.execute_command.return_value = [["avg(...)"], [[None]], ["stats..."]]

    with patch("app.substrate.bus_synaptic.aioredis.from_url", return_value=mock_client):
        value = await query_real_bus_synaptic_raw_mean_abs_z("redis://fake:6379", "orion_bus_synapse")

    assert value == 0.0


def test_query_is_typed_recency_filtered_and_clamped_per_edge() -> None:
    """The three defects this query had until 2026-07-30, each of which alone
    was enough to pin the reheat signal at its ceiling. Asserted on the built
    Cypher because there is no way to unit-test FalkorDB's own evaluation here,
    and a silent regression on any one of them reproduces the original bug."""
    from app.substrate.bus_synaptic import (
        BUS_SYNAPTIC_ZSCORE_SATURATION,
        _MIN_EDGE_COUNT,
        _build_query,
    )

    cutoff = 1785447361.0
    q = _build_query(cutoff)

    # 1. Typed: an untyped MATCH also matched CAUSALLY_FOLLOWED_BY edges, which
    #    carry latency_zscore rather than gap_zscore and contributed NULL.
    assert "(:Organ)-[rel:PUBLISHES]->(:Channel)" in q
    # 2. Recency-filtered: the edge that dominated this reading (|z| = 7087.8)
    #    had not fired in 9 hours. orion-substrate-runtime's equivalent query
    #    gained this filter on 2026-07-25; heartbeat never did.
    assert f"rel.last_seen_epoch > {cutoff!r}" in q
    assert f"rel.count > {_MIN_EDGE_COUNT}" in q
    # 3. Clamped per edge BEFORE averaging, so one pathological edge cannot
    #    dictate the aggregate (live: mean 29.3 vs median 0.399).
    assert "CASE WHEN abs(rel.gap_zscore) >" in q
    assert str(BUS_SYNAPTIC_ZSCORE_SATURATION) in q
    assert "avg(" in q


def test_stale_cutoff_matches_substrate_runtimes_own_default() -> None:
    """Heartbeat and orion-substrate-runtime read the same graph for the same
    purpose; a divergent recency window would make their two readings
    inexplicably disagree. Mirrors SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC's
    1h default, kept in sync by comment cross-reference the same way
    BUS_SYNAPTIC_ZSCORE_SATURATION already is."""
    from app.substrate.bus_synaptic import _STALE_CUTOFF_SEC

    assert _STALE_CUTOFF_SEC == 3600.0
