# Orion Equilibrium Service

The equilibrium service normalizes system heartbeat signals into a distress/zen signal, publishes a versioned snapshot, and feeds Spark with `spark.signal.v1` for downstream modulation.

## Architecture

```mermaid
flowchart TD
    HB[system.health / system.health.v1] --> EQ[orion-equilibrium-service]
    EQ -->|equilibrium.snapshot.v1| SNAP[orion:equilibrium:snapshot channel]
    EQ -->|spark.signal.v1 (equilibrium)| SIG[orion:spark:signal channel]
    SIG -->|bias φ| SPARK[Spark ingestors\n(gateway + introspector)]
    SNAP --> JR[state-journaler\n(Postgres rollups)]
    SPARK --> STATE[state-service / UI]
```

## Contracts
- **Heartbeat (system.health.v1)**: includes `service`, `node`, `version`, `instance`, `boot_id`, `status`, `last_seen_ts`, `heartbeat_interval_sec`, and bounded `details`.
- **Equilibrium snapshot (equilibrium.snapshot.v1)**: per-service uptime, `down_for_ms`, per-window `uptime_pct`, `distress_score`, and `zen_score`.
- **Spark signal (spark.signal.v1)**: `signal_type=equilibrium`, normalized `intensity` (distress), optional deltas, TTL, and source metadata.

## Configuration
- `EQUILIBRIUM_EXPECTED_SERVICES` / `EQUILIBRIUM_EXPECTED_SERVICES_PATH`: explicit service roster.
- `EQUILIBRIUM_GRACE_MULTIPLIER`: heartbeat grace window (multiples of interval).
- `EQUILIBRIUM_PUBLISH_INTERVAL_SEC`: cadence for snapshots + signals.
- `EQUILIBRIUM_WINDOWS_SEC`: rolling uptime windows (e.g., `60,300,3600` seconds).

## Verification steps
1. **Heartbeat ingestion**
   - Publish a synthetic heartbeat:
     ```bash
     redis-cli -u "$ORION_BUS_URL" PUBLISH system.health.v1 \
       '{"schema":"orion.envelope","kind":"system.health.v1","source":{"name":"probe"},"payload":{"service":"probe","node":"athena","version":"1.0.0","boot_id":"demo","status":"ok","last_seen_ts":"'$(date -Iseconds)'","heartbeat_interval_sec":5}}'
     ```
   - Observe `equilibrium.snapshot.v1` on `orion:equilibrium:snapshot`.
2. **Spark signal impact**
   - Subscribe to `orion:spark:signal` and confirm `spark.signal.v1` frames include `signal_type: equilibrium` and the computed `intensity`.
   - Spark introspector snapshots (`spark.state.snapshot.v1`) should reflect the deltas (valence/coherence bias) in subsequent φ values.
3. **Rollup visibility**
   - Call `GET /rollups?window=300&hours=24` on `state-journaler` to see averaged φ + distress rollups for dashboards.
