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
import time

import redis.asyncio as aioredis

logger = logging.getLogger("orion-heartbeat.substrate.bus_synaptic")

# Mirrors orion/substrate/prediction_error.py::_BUS_SYNAPTIC_ZSCORE_SATURATION.
# Reuses the same zscore_threshold=3.0 convention already live in
# services/orion-hub/scripts/bus_synaptic_graph_routes.py's anomalies()
# route, per that module's own docstring.
BUS_SYNAPTIC_ZSCORE_SATURATION = 3.0

# 2026-07-30: three real defects fixed at once, all live-confirmed against the
# real graph. The previous query was:
#
#     MATCH ()-[rel]->() WHERE rel.count > 5 RETURN avg(abs(rel.gap_zscore))
#
# 1. **Untyped `MATCH ()-[rel]->()`** matched every relationship kind, so
#    CAUSALLY_FOLLOWED_BY edges (which carry `latency_zscore`, not
#    `gap_zscore`) were pulled in and contributed NULL.
# 2. **No recency filter.** An edge keeps its last-computed z-score forever
#    once `count` clears the cold-start floor. The single edge dominating this
#    reading (cortex-orch -> Channel, |z| = 7087.8) had not fired in 9 hours.
#    orion-substrate-runtime's equivalent query already gained exactly this
#    filter on 2026-07-25 for exactly this reason; heartbeat never got it.
# 3. **`avg()` over unbounded, heavy-tailed values.** Measured live: mean 29.3
#    vs median 0.399, p90 1.123 -- one edge contributed 28.6 of the 29.3. That
#    pinned `signal = min(1, z/3.0)` at a permanent 1.0, so the reheat term ran
#    at constant maximum and gated nothing, which is the opposite of this
#    module's entire purpose.
#
# Clamping per edge before averaging matches the same fix applied to
# orion/substrate/prediction_error.py::bus_synaptic_prediction_error. This
# query deliberately still does NOT subtract the calm floor -- see the module
# docstring above for why raw ambient hum is the right reading for reheat.
# Mirrors orion-substrate-runtime's SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC /
# _MIN_EDGE_COUNT defaults. Note the two services deliberately read DIFFERENT
# EDGE SETS from the same graph -- substrate-runtime aggregates PUBLISHES
# gap_zscore *plus* CAUSALLY_FOLLOWED_BY latency_zscore, this module only
# PUBLISHES gap, because reheat wants publish-cadence ambient hum specifically.
# Only the recency window and cold-start floor are synced; the populations are
# not, and are not meant to be.
_STALE_CUTOFF_SEC = 3600.0
_MIN_EDGE_COUNT = 5


def _build_query(stale_cutoff_epoch: float) -> str:
    """Raw ``GRAPH.QUERY`` takes no parameter map (that is a redis-py ``Graph``
    convenience this module deliberately does not depend on -- see the module
    docstring on staying off the heavier substrate machinery). Both
    interpolated values are locally-computed floats/ints, never caller input,
    so there is no injection surface here.
    """
    return (
        "MATCH (:Organ)-[rel:PUBLISHES]->(:Channel) "
        "WHERE rel.gap_zscore IS NOT NULL "
        f"AND rel.count > {_MIN_EDGE_COUNT!r} "
        f"AND rel.last_seen_epoch > {stale_cutoff_epoch!r} "
        "RETURN avg(CASE WHEN abs(rel.gap_zscore) > "
        f"{BUS_SYNAPTIC_ZSCORE_SATURATION!r} THEN {BUS_SYNAPTIC_ZSCORE_SATURATION!r} "
        "ELSE abs(rel.gap_zscore) END)"
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
            stale_cutoff_epoch = time.time() - _STALE_CUTOFF_SEC
            result = await client.execute_command(
                "GRAPH.QUERY", graph_name, _build_query(stale_cutoff_epoch)
            )
            raw = result[1][0][0]
            # avg() over an empty match returns NULL, not 0.0 -- "no live edges
            # right now" is a real state (every edge aged past the cutoff) and
            # must read as no reheat, not crash on float(None).
            if raw is None:
                return 0.0
            return float(raw)
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001 - must not kill the dissipation loop
        logger.warning("bus_synaptic_query_failed err=%s falling back to 0.0 (no reheat)", exc)
        return 0.0
