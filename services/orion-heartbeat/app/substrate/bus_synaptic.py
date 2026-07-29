"""Real, live bus_synaptic query -- drives the dissipation ensemble's reheat
strength from actual ambient inter-service bus activity, not synthetic RNG.

Design doc: docs/superpowers/specs/2026-07-28-precision-weighted-attention-
organ-and-heartbeat-discrimination-design.md.

Deliberately reads the RAW mean(|gap_zscore|) across reliable orion_bus_synapse
edges, NOT the calm-floor-corrected version orion/substrate/prediction_error.py::
bus_synaptic_prediction_error() computes for its anomaly-detection use case.
That function subtracts _BUS_SYNAPTIC_CALM_FLOOR (sqrt(2/pi)) specifically so
a genuinely calm reading reads ~0 -- correct for "is something anomalous
happening," wrong for this module's job. The floor itself -- real,
structurally-guaranteed ambient bus timing jitter that never goes away even
when nothing anomalous is happening -- is exactly the "always some real
activity" signal a baseline reheat hum needs, read the opposite way.
Confirmed live 2026-07-28: raw mean(|gap_zscore|) sits around 1.0-1.1 during
normal operation, not the corrected metric's near-0 calm reading.

Does NOT import orion.substrate.prediction_error -- that package drags in
`requests` and heavier substrate-store machinery this service deliberately
avoids (see service.py's own "additive, read-only consumer" framing).
_BUS_SYNAPTIC_ZSCORE_SATURATION below is a local constant mirroring that
module's own value, not an import, kept in sync by comment cross-reference.
"""
from __future__ import annotations

import logging

import redis.asyncio as aioredis

logger = logging.getLogger("orion-heartbeat.substrate.bus_synaptic")

# Mirrors orion/substrate/prediction_error.py::_BUS_SYNAPTIC_ZSCORE_SATURATION.
# Reuses the same zscore_threshold=3.0 convention already live in
# services/orion-hub/scripts/bus_synaptic_graph_routes.py's anomalies()
# route, per that module's own docstring.
BUS_SYNAPTIC_ZSCORE_SATURATION = 3.0

_QUERY = (
    "MATCH ()-[rel]->() WHERE rel.count > 5 RETURN avg(abs(rel.gap_zscore))"
)


async def query_real_bus_synaptic_raw_mean_abs_z(falkordb_uri: str, graph_name: str) -> float:
    """Live GRAPH.QUERY against orion_bus_synapse. Never raises -- a query
    failure (FalkorDB unreachable, graph not yet populated) falls back to
    0.0 (no reheat this tick), an honest "no signal" rather than a crash of
    the dissipation loop. Logged at WARNING so a persistent failure is
    visible, not silent.
    """
    try:
        client = aioredis.from_url(falkordb_uri, decode_responses=True, socket_timeout=5.0)
        try:
            result = await client.execute_command("GRAPH.QUERY", graph_name, _QUERY)
            raw = result[1][0][0]
            return float(raw)
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001 - must not kill the dissipation loop
        logger.warning("bus_synaptic_query_failed err=%s falling back to 0.0 (no reheat)", exc)
        return 0.0
