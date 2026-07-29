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
