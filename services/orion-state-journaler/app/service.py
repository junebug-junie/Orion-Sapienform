"""orion-state-journaler: bus heartbeat only.

The spark-state rollup this service used to run was retired on 2026-10-10.
Its input, ``orion:spark:state:snapshot``, lost its only real producer when
orion-spark-introspector was deleted on 2026-07-28; from then on the rollup
loop wrote rows of 0.0 valence/arousal/coherence/novelty with
``pct_missing=1.0`` into ``spark_state_rollups`` every 30 seconds. The
``avg_distress`` column was still live (equilibrium snapshots) but had no
reader anywhere in the repo.

Kill means kill: no subscription, no rollup, no Postgres write, no fallback.
``spark_state_rollups`` is left in place as a frozen historical table (last
real spark row 2026-07-28 07:09 UTC). What remains is the standard
SystemHealthV1 heartbeat so equilibrium's expected-services check keeps
seeing this container until it is retired outright.
"""

from __future__ import annotations

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .settings import settings


def build_chassis() -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )
