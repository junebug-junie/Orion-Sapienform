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

### Which "how loaded is it" number to read

There are now three, they disagree on purpose, and picking the wrong one is how a saturated
machine goes unnoticed. Measured live on athena 2026-08-15, all at the same instant:

| field | value | what it answers |
| :--- | :--- | :--- |
| `composites.strain` | `0.11` | mean of **7 of the 11** pressures |
| `constraint` | `NONE` | peak channel's *name*, but only above 0.7 and only for the 8 channels in `CONSTRAINTS` |
| `peak_pressure` / `_channel` / `_node` | `1.00` `power` `athena` | the binding constraint: max over **all 11** |

athena's `power` was pegged at 1.000 and the first two both read "fine". Two independent
reasons:

- **`strain` is a mean.** One saturated channel is averaged away by six calm ones.
- **`strain` omits `gpu_mem`, `swap`, `disk_capacity` and `fan`** — which are the highest
  channel on two of three nodes. On 2026-08-15 athena's peak was `disk_capacity` at 0.772,
  over the alarm threshold, reported as `constraint=NONE` because that key is not in
  `CONSTRAINTS`.

**Read `peak_pressure` when you want to know whether something is about to stop.** Read
`strain` only if you specifically want the average, and know that it cannot see saturation.

`strain` is deliberately **not** fixed in place. It is written straight into the field
lattice's `cpu_pressure` channel (`orion-field-digester/app/ingest/state_deltas.py:125`,
`mode="replace"`) and feeds the substrate prediction-error diff, so changing its formula moves
Orion's substrate rather than a readout. `peak_pressure` was added alongside it instead —
nothing downstream of `strain` moved. See `_peak_pressure` in
`orion/telemetry/biometrics_pipeline.py`.

At cluster level `peak_pressure` is a **max across nodes**, never the role-weighted mean the
`pressures` dict uses — one saturated machine must not be smeared across three — and
`peak_pressure_node` travels with it so the number can be acted on.

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
  fatal -- it never blocks the heartbeat itself. `collect_disk_capacity()` checks
  `os.path.ismount()`, not just `os.path.isdir()` -- if a bind-mount source doesn't exist on the
  host, Docker silently creates an empty directory at the container target instead of failing,
  and `isdir()` alone can't tell that apart from a real mount (it would then report the
  *container's own* overlay filesystem usage as if it were the host mount's).
- The heartbeat loop runs `_heartbeat_details()` in a worker thread (`asyncio.to_thread`) with a
  bounded wait (`min(3.0, heartbeat_interval_sec / 2)`, floor 0.5s), not inline on the event loop.
  A wedged/stale mount blocking synchronously inside `shutil.disk_usage()` would otherwise stall
  every other task sharing this process's event loop, not just the heartbeat. A timeout drops that
  tick's details (`{}`) rather than hanging.
- This is capacity (how full a filesystem is), not the same thing as the existing `disk` key in
  `biometrics.sample.v1`/`BiometricsSampleV1`, which is I/O *throughput* (`_collect_disk()` in
  `app/metrics.py`, read/write bytes/sec from `/proc/diskstats`) and feeds a separate, already-wired
  consumer chain (`BiometricsPipeline`'s `disk` pressure -> `grammar_emit.py`'s
  `disk_pressure_signal` -> field-digester's `disk_pressure` lattice channel). Do not conflate the
  two -- `collect_disk_capacity()` is intentionally a separate function/key, not an extension of
  `_collect_disk()`.

### iLO/BMC hardware telemetry (`details.ilo_*`)

Real out-of-band hardware telemetry (thermal, fan, power) pulled from the node's iLO/BMC
RedFish API, piggybacked onto the same `SystemHealthV1.details` dict as disk capacity, above.
Node-level and optional -- a node with no `ILO_HOST` configured simply omits these keys
entirely (`IloPoller.details()` returns `{}`).

Key convention:

```json
{
  "ilo_fetched_at": 1721923200.123,
  "ilo_thermal_c": {"01-Inlet Ambient": 24.0, "02-CPU 1": 40.0},
  "ilo_fan_pct": {"Fan 1": 29.0, "Fan 2": 29.0},
  "ilo_power_watts": 310.0,
  "ilo_error": "connection timeout"
}
```

- `ilo_thermal_c`/`ilo_fan_pct` are keyed by the sensor's own RedFish `Name` (vendor-specific
  strings, e.g. HPE's `"01-Inlet Ambient"`) -- not normalized across vendors. Sensors reporting
  `Status.State != "Enabled"` (e.g. `"Absent"`, an unpopulated slot) are skipped rather than
  recorded as a fake `0`. Fans additionally require `ReadingUnits == "Percent"` -- RedFish fans
  can report Percent or RPM depending on vendor, and only HPE/athena is verified so far; an RPM
  reading is skipped rather than silently mislabeled as a percentage.
- `ilo_error` appears whenever the last poll failed (bad creds, network unreachable, non-RedFish
  BMC) or `ILO_HOST` isn't configured on this node (`"not_configured"`) -- never fatal to the
  heartbeat itself.
- Unlike disk capacity (a cheap local `shutil.disk_usage()` call, safe to run inline in the
  heartbeat's bounded per-tick hook), an iLO RedFish round-trip is a real network call to a
  comparatively weak out-of-band management processor -- not safe to run on every heartbeat tick
  (default 10s) or inside that hook's ~3s budget. `app/ilo.py::IloPoller` runs on its own
  `ILO_POLL_INTERVAL_SEC` cadence (default 60s) as a separate background task; the heartbeat hook
  just reads the last cached snapshot (`ilo_fetched_at` tells you how stale it is).
- Credentials (`ILO_HOST`/`ILO_USERNAME`/`ILO_PASSWORD`) are node-specific secrets -- set only in
  this node's local `.env`, never in `.env_example` (which ships them empty) or committed
  anywhere. Uses standard DMTF RedFish (`/redfish/v1/Chassis/` -> first chassis's `/Thermal/` and
  `/Power/`), confirmed live against athena's HPE iLO; unverified so far against Circe's Gigabyte
  BMC.
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
| `gpu_pressure_signal` | `gpu` | `gpu_pressure` |
<!-- The `strain` row is why `strain`'s formula is frozen: `mode="replace"` means the composite
     IS the lattice's `cpu_pressure` value, so redefining it moves Orion's substrate, not a
     readout. `peak_pressure` was added beside it rather than changing it. Note also that a
     whole-body composite feeding a channel named `cpu_pressure` is a pre-existing mislabel,
     independent of that. -->

| `memory_pressure_signal` | `memory_pressure` | `memory_pressure` |
| `thermal_pressure_signal` | `thermal_pressure` | `thermal_pressure` |
| `disk_pressure_signal` | `disk_pressure` | `disk_pressure` |

`gpu_pressure_signal`/`memory_pressure_signal`/`thermal_pressure_signal`/
`disk_pressure_signal` carry the individually-computed `gpu_util`/`mem`/
`thermal`/`disk` values from `orion/telemetry/biometrics_pipeline.py`'s
`pressures` dict. `memory_pressure_signal`/`thermal_pressure_signal`/
`disk_pressure_signal` were wired in by the 2026-07-16 fix -- these were
previously only folded into the `strain` composite and never reached the
field lattice, so the corresponding channels stayed pinned at `0.0`.
`gpu_pressure_signal` was wired in by a 2026-07-28 fix for the same class of
bug: `pressure_hints["gpu"]` previously came from `capability_surface`'s
`salience`, which is an **unconditional hardcoded literal (`0.8`)** in this
file, not a telemetry sample -- it had no relationship to real GPU load and
was gated on `local_llm_heavy` besides. `node:atlas`'s `gpu_pressure` field
channel read a flat `0.8` for ~42,400 consecutive real ticks (24h) as a
result. `gpu_pressure_signal` is emitted unconditionally for every node
(same as the memory/thermal/disk siblings, no capability gate) carrying the
real `gpu_util` value, which is `0.0` for nodes with no GPU samples --
`pressure_hints["gpu"]` is now populated for every node's tick, not just
`local_llm_heavy` ones. `capability_surface`'s own `salience` is unchanged
and still used for its own confidence/summary display; it is simply no
longer read as the source of the gpu pressure hint. This is additive to
`strain`: unaffected.

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

### When a node cannot reach the PDU: hub-side proxy polling

circe's **network card is dead**. It reaches the bus over Tailscale and has no LAN path to the
PDU, so its own poller fails every 65 s:

```text
orion.biometrics.pdu - WARNING - pdu_poll_failed error=5 second timeout exceeded on UDP transport.
```

athena can reach the PDU, so it reads circe's outlets **on circe's behalf**:

```bash
# athena only
PDU_HOST=192.168.1.39
PDU_OUTLETS=                                        # empty -> athena does NOT self-poll
PDU_PROXY_OUTLETS={"circe": [19,25,31], "atlas": [34,35]}
```

`PDU_HOST` means *"the PDU this node can reach"* — set it on any node that can talk to the PDU,
including a hub that only proxies. **Empty `PDU_OUTLETS` is what disables self-polling**, not an
empty `PDU_HOST`.

**Two rules keep this honest:**

1. **A proxy only fills a gap.** A node's own measurement always wins. Live: atlas keeps its iLO
   `chassis_watts` and gains only `pdu_watts` from the proxy, while circe (no BMC, no LAN) gets
   both.
2. **Provenance travels with the value.** The cluster carries `measurements_proxied`, e.g.
   `{"circe": ["chassis_watts","pdu_watts"], "atlas": ["pdu_watts"]}`. Without it, a future
   reader finding circe with a chassis figure would reasonably conclude its BMC came back.

Proxying is also **strictly better than self-polling for circe**: the outlets report its draw
whether circe is powered or not, so a shut-down circe reads a true ~0 W instead of vanishing
into `measurements_missing`.

A proxy that is itself failing contributes nothing — the node stays in `measurements_missing`,
so the fix for silent failure does not become a new way to fail silently.

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
