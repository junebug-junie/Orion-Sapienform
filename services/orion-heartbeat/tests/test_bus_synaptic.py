from __future__ import annotations

import time
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
    """Heartbeat and orion-substrate-runtime read the same graph over the same
    recency window (they deliberately read different EDGE SETS -- see the
    constant's own comment -- but a divergent *window* would make the two
    readings disagree for a reason no operator could diagnose).

    Review finding, 2026-07-30: the first version of this test asserted
    `_STALE_CUTOFF_SEC == 3600.0`, comparing heartbeat's constant against a
    literal in the same file. That is tautological -- it can never detect the
    drift it claims to guard, because it never reads substrate-runtime at all.
    It now parses that service's real default, so changing
    SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC without updating heartbeat fails
    here (CLAUDE.md §4: deterministic gate, not a comment cross-reference).

    substrate-runtime's setting is operator-tunable and heartbeat's is a
    compile-time constant; this asserts the DEFAULTS agree. A deployment that
    overrides the env key still diverges, which is called out in the PR report
    as a known, accepted asymmetry rather than silently tolerated here.
    """
    import re
    from pathlib import Path

    from app.substrate.bus_synaptic import _MIN_EDGE_COUNT, _STALE_CUTOFF_SEC

    settings_src = (
        Path(__file__).resolve().parents[2]
        / "orion-substrate-runtime"
        / "app"
        / "settings.py"
    ).read_text(encoding="utf-8")

    age = re.search(
        r"bus_synaptic_max_edge_age_sec:\s*float\s*=\s*Field\(\s*([0-9.]+)", settings_src
    )
    count = re.search(
        r"bus_synaptic_min_edge_count:\s*int\s*=\s*Field\(\s*([0-9]+)", settings_src
    )
    assert age is not None, "could not parse substrate-runtime's max_edge_age default"
    assert count is not None, "could not parse substrate-runtime's min_edge_count default"

    assert _STALE_CUTOFF_SEC == float(age.group(1))
    assert _MIN_EDGE_COUNT == int(count.group(1))


def test_built_query_is_valid_cypher_against_a_live_falkordb() -> None:
    """Review finding, 2026-07-30: the assertions above are substring matches
    and were confirmed to PASS against a deliberately malformed query (missing
    `END` and a closing paren) that FalkorDB rejects outright. That is the
    failure mode most likely to slip through, because the broad `except` in
    query_real_bus_synaptic_raw_mean_abs_z would swallow the parse error into a
    warning log and leave reheat silently pinned at 0.0 forever.

    Skipped rather than failed when no FalkorDB is reachable, so this stays a
    gate on developer/CI machines that have one without becoming a hard
    dependency for the rest of the suite.
    """
    import os

    import pytest as _pytest
    import redis as _redis

    from app.substrate.bus_synaptic import _STALE_CUTOFF_SEC, _build_query

    uri = os.getenv("FALKORDB_URI", "redis://127.0.0.1:6380")
    graph = os.getenv("FALKORDB_BUS_GRAPH", "orion_bus_synapse")
    try:
        client = _redis.Redis.from_url(uri, socket_connect_timeout=2, socket_timeout=5)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        _pytest.skip(f"no FalkorDB at {uri}: {exc}")

    result = client.execute_command(
        "GRAPH.QUERY", graph, _build_query(time.time() - _STALE_CUTOFF_SEC)
    )
    # A parse error raises above; reaching here means FalkorDB accepted it.
    raw = result[1][0][0]
    assert raw is None or isinstance(raw, (int, float, bytes, str))


def test_a_malformed_query_would_be_rejected_by_falkordb() -> None:
    """Companion proving the test above actually has teeth: the same malformed
    query that passed every substring assertion is rejected here."""
    import os

    import pytest as _pytest
    import redis as _redis

    uri = os.getenv("FALKORDB_URI", "redis://127.0.0.1:6380")
    graph = os.getenv("FALKORDB_BUS_GRAPH", "orion_bus_synapse")
    try:
        client = _redis.Redis.from_url(uri, socket_connect_timeout=2, socket_timeout=5)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        _pytest.skip(f"no FalkorDB at {uri}: {exc}")

    broken = (
        "MATCH (:Organ)-[rel:PUBLISHES]->(:Channel) WHERE rel.gap_zscore IS NOT NULL "
        "RETURN avg(CASE WHEN abs(rel.gap_zscore) > 3.0 THEN 3.0 ELSE abs(rel.gap_zscore)"
    )
    with _pytest.raises(Exception):
        client.execute_command("GRAPH.QUERY", graph, broken)
