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

Node identity for grammar traces is resolved via `config/biometrics/node_catalog.yaml` (aliases canonicalize hostnames, e.g. `prometheous` → `prometheus`).

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

Their `Perturbation`s use `mode="replace"`, not the `strain`/`gpu` default
`mode="add"`: `orion/substrate/biometrics_loop/node_reducer.py` emits one
`StateDeltaV1` per *grammar event* in a trace (not just the atom that sets a
given hint -- `trace_started`/edges/`trace_ended` all produce their own delta
carrying the cumulative `pressure_hints` forward), so a single trace yields
well over a dozen deltas that each still contain these hints once set. Under
`"add"` mode that re-adds the same intensity that many times per telemetry
cycle and saturates the channel to the `1.0` clamp almost immediately,
independent of real load -- the same class of bug `execution_run`/`chat_turn`
already hit and fixed with `mode="replace"` for their own `pressure_hints`
snapshots. `strain`/`gpu` still use `mode="add"` and are believed to already
be affected by this; that is pre-existing and out of scope for this change.

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
