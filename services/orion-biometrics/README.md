# Orion Biometrics

The **Biometrics** service collects hardware telemetry (CPU, memory, GPU usage, power consumption), normalizes it into bounded pressures, and publishes multiple payloads to the bus for storage and downstream cognition.

## Contracts

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:telemetry:biometrics` | `TELEMETRY_PUBLISH_CHANNEL` | `biometrics.telemetry` | Raw hardware metrics (legacy payload). |
| `orion:biometrics:sample` | `BIOMETRICS_SAMPLE_CHANNEL` | `biometrics.sample.v1` | Expanded raw sample (CPU/mem/disk/net/temps/power). |
| `orion:biometrics:summary` | `BIOMETRICS_SUMMARY_CHANNEL` | `biometrics.summary.v1` | Normalized pressures/headroom/composites. |
| `orion:biometrics:induction` | `BIOMETRICS_INDUCTION_CHANNEL` | `biometrics.induction.v1` | EWMA level/trend/volatility/spikes. |
| `orion:biometrics:cluster` | `BIOMETRICS_CLUSTER_CHANNEL` | `biometrics.cluster.v1` | Role-weighted cluster aggregate (hub mode). |
| `orion:spark:signal` | `SPARK_SIGNAL_CHANNEL` | `spark.signal.v1` | Bounded resource signal from cluster strain (hub mode). |
| `orion:grammar:event` | `GRAMMAR_EVENT_CHANNEL` | `grammar.event.v1` | Node-scoped grammar trace (one trace per observed node per tick). |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `TELEMETRY_PUBLISH_CHANNEL` | `orion:telemetry:biometrics` | Raw publish channel (legacy payload). |
| `BIOMETRICS_SAMPLE_CHANNEL` | `orion:biometrics:sample` | Sample publish channel. |
| `BIOMETRICS_SUMMARY_CHANNEL` | `orion:biometrics:summary` | Summary publish channel. |
| `BIOMETRICS_INDUCTION_CHANNEL` | `orion:biometrics:induction` | Induction publish channel. |
| `BIOMETRICS_CLUSTER_CHANNEL` | `orion:biometrics:cluster` | Cluster publish channel (hub mode). |
| `BIOMETRICS_MODE` | `agent` | `agent` (node), `hub` (aggregate), or `both`. |
| `CLUSTER_ROLE_WEIGHTS` | `{"atlas":0.7,"athena":0.3,"other":0.5}` | Role weighting for cluster aggregate. |
| `THERMAL_MIN_C` | `50.0` | Temperature floor for normalization. |
| `THERMAL_MAX_C` | `85.0` | Temperature ceiling for normalization. |
| `DISK_BW_MBPS` | `200.0` | Disk bandwidth scale (MB/s). |
| `NET_BW_MBPS` | `125.0` | Network bandwidth scale (MB/s). |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |
| `PUBLISH_BIOMETRICS_GRAMMAR` | `true` | Enable grammar trace publish after each biometrics tick. |
| `GRAMMAR_EVENT_CHANNEL` | `orion:grammar:event` | Grammar event publish channel. |
| `NODE_CATALOG_PATH` | `/app/config/biometrics/node_catalog.yaml` | Path to node catalog YAML (host: `config/biometrics/node_catalog.yaml`). |
| `DISK_CAPACITY_MOUNTS` | `{"docker":"/host_mnt/docker","scripts":"/host_mnt/scripts","postgres":"/host_mnt/postgres","graphdb":"/host_mnt/graphdb","telemetry":"/host_mnt/telemetry"}` | Mount name -> in-container path for disk-capacity heartbeat telemetry (see below). |

Node identity for grammar traces is resolved via `config/biometrics/node_catalog.yaml` (aliases canonicalize hostnames, e.g. `prometheous` → `prometheus`).

### Disk capacity telemetry (`details.disk_usage_pct`)

This service is one of the 25 already using `orion.core.bus.bus_service_chassis.BaseChassis`
(via `Clock`), so it already publishes a bus-native `SystemHealthV1` heartbeat to
`orion:system:health` every `HEARTBEAT_INTERVAL_SEC` (default 10s). `SystemHealthV1.details`
is a free-form dict; `app/main.py::_heartbeat_details()` folds real host disk-*capacity*
(not I/O throughput -- see below) into it via `app/metrics.py::collect_disk_capacity()`.

Key convention:

```json
{
  "disk_usage_pct": {"docker": 87.3, "scripts": 14.1, "postgres": 14.4, "graphdb": 0.3, "telemetry": 20.4},
  "disk_usage_errors": {"<mount_name>": "not_mounted"}
}
```

- `disk_usage_pct.<mount_name>` -- percent-used (0-100, 2dp) for that mount, from
  `shutil.disk_usage()` against the read-only bind mount configured in `docker-compose.yml`
  (`/mnt/<name>` on the host -> `/host_mnt/<name>:ro` in the container). Mount names/paths are
  configurable via `DISK_CAPACITY_MOUNTS` (JSON object, mount name -> in-container path).
- `disk_usage_errors` only appears when at least one configured mount is missing/inaccessible
  inside the container (e.g. not bind-mounted on this node yet). A missing mount is skipped, not
  fatal -- it never blocks the heartbeat itself.
- This is capacity (how full a filesystem is), not the same thing as the existing `disk` key in
  `biometrics.sample.v1`/`BiometricsSampleV1`, which is I/O *throughput* (`_collect_disk()` in
  `app/metrics.py`, read/write bytes/sec from `/proc/diskstats`) and feeds a separate, already-wired
  consumer chain (`BiometricsPipeline`'s `disk` pressure -> `grammar_emit.py`'s
  `disk_pressure_signal` -> field-digester's `disk_pressure` lattice channel). Do not conflate the
  two -- `collect_disk_capacity()` is intentionally a separate function/key, not an extension of
  `_collect_disk()`.
- Cross-node: because this same `services/orion-biometrics` codebase runs independently on
  athena/atlas/circe (each with its own `NODE_NAME`), redeploying this patch on each node gives
  real disk-capacity visibility per node with no new SSH/credential surface -- the bus already
  proven to carry heartbeats cross-node (`SystemHealthV1.node` distinguishes them).
- No alerting/threshold logic here by design -- this is pure telemetry collection, matching
  `_collect_disk()`'s existing scope (report numbers, don't act on them).

### Grammar node `pressure_hints` (consumed by `orion-field-digester`)

`app/grammar_emit.py::build_biometrics_node_grammar_events` emits one atom per
hardware-pressure signal on the `orion:grammar:event` trace. Each atom's
`salience` is later read by `orion/substrate/biometrics_loop/grammar_extract.py`
and surfaced on the `node_biometrics` projection's `pressure_hints` dict, which
`services/orion-field-digester/app/ingest/state_deltas.py`'s `node_biometrics`
block turns into lattice `Perturbation`s:

| Atom `semantic_role` | `pressure_hints` key | Lattice channel (`NODE_CHANNELS`) |
| :--- | :--- | :--- |
| `body_state` (composite `strain`) | `strain` | `cpu_pressure` |
| `capability_surface` (gated on `local_llm_heavy`) | `gpu` | `gpu_pressure` |
| `memory_pressure_signal` | `memory_pressure` | `memory_pressure` |
| `thermal_pressure_signal` | `thermal_pressure` | `thermal_pressure` |
| `disk_pressure_signal` | `disk_pressure` | `disk_pressure` |

`memory_pressure_signal`/`thermal_pressure_signal`/`disk_pressure_signal` carry
the individually-computed `mem`/`thermal`/`disk` values from
`orion/telemetry/biometrics_pipeline.py`'s `pressures` dict (2026-07-16 fix --
these were previously only folded into the `strain` composite and never
reached the field lattice, so the corresponding channels stayed pinned at
`0.0`). This is additive: `strain`/`gpu` are unchanged.

All five of these `Perturbation`s (`strain`/`gpu`/`memory_pressure`/
`thermal_pressure`/`disk_pressure`) use `mode="replace"`, not the library
default `mode="add"`: `orion/substrate/biometrics_loop/node_reducer.py`
emits one `StateDeltaV1` per *grammar event* in a trace (not just the atom
that sets a given hint -- `trace_started`/edges/`trace_ended` all produce
their own delta carrying the cumulative `pressure_hints` forward), so a
single trace yields well over a dozen deltas that each still contain these
hints once set. Under `"add"` mode that re-adds the same intensity that
many times per telemetry cycle and saturates the channel to the `1.0` clamp
almost immediately, independent of real load -- the same class of bug
`execution_run`/`chat_turn` already hit and fixed with `mode="replace"` for
their own `pressure_hints` snapshots.

`strain`/`gpu` (2026-07-17 fix) were the pre-existing pattern this repeated
-- and, per live verification, actually were saturating this way in
production. Querying `/mnt/telemetry/field_channels/corpus/field_channels.jsonl`
(133,968 rows spanning 2026-07-13..17) showed `cpu_pressure`/`gpu_pressure`
sitting exactly at their post-decay ceiling (`1.0 * BIOMETRICS_FIELD_DECAY_RATE`
= `0.92`) in 16.60%/12.98% of all rows -- vs. 0.01%/0.00% for
`execution_load`/`execution_friction`, which already used `mode="replace"`
-- and bit-identical to each other in 60.60% of rows despite deriving from
independent `strain`/`gpu` hints. Both signatures match repeated
re-saturation from add-mode duplicate deltas, not real, independent CPU/GPU
utilization, so `strain`/`gpu` were switched to `mode="replace"` too.

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-biometrics
```

### Smoke Test
```bash
scripts/smoke_biometrics.sh
# Expects one message on sample/summary/induction + state-service reply.
```
