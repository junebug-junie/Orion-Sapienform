# orion-field-digester

Substrate field digestion worker that consumes committed reduction receipts (biometrics + cortex-exec execution trajectories) and compiles lattice field state:

```text
substrate_reduction_receipts → delta dedupe → perturb/decay/diffuse/suppress → substrate_field_state
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_field_digester_v1.sql
cp services/orion-field-digester/.env_example services/orion-field-digester/.env
```

## Run

```bash
cd services/orion-field-digester
docker compose up --build
```

Health: `GET http://localhost:8116/health`

## Idle tick (pacemaker)

`FIELD_DIGESTER_IDLE_TICK_ENABLED` (**default `true`**) keeps `tick_id` advancing on every poll even when there are no new receipts to consume. On a quiet poll the worker still loads/reconciles field state, runs decay + diffusion + suppression with an empty perturbation set, mints a new `tick_id`, and persists it via `save_field` — but it does not advance the receipt cursor or write pending deltas (that only happens via `commit_digest_tick` when receipts were actually consumed). This lets downstream free-running consumers (`orion-attention-runtime`, `orion-self-state-runtime`) keep advancing off the latest tick during quiet periods.

**Retention, the applied-deltas pruner, and the health monitor are all prerequisites, and are all live.** `save_field` mints a fresh `tick_id` every call, so every idle tick is a genuine new row in `substrate_field_state` — at the default 2s poll interval that's ~43k rows/day. This is bounded by an hourly batched pruner (`FIELD_STATE_RETENTION_HOURS`, default 72h) that never deletes the newest row. The same cascade happens downstream in `orion-attention-runtime`/`orion-self-state-runtime`, each of which independently prunes its own tables on the same 72h/hourly pattern.

`substrate_field_applied_deltas` (the delta dedup ledger) has no natural "latest row" reader, so instead of a time-based cutoff it is pruned by receipt existence: a dedup row is deleted only once its source receipt is confirmed gone from `substrate_reduction_receipts`, at which point that `delta_id` can structurally never be redelivered by `fetch_new_receipts()`. `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS` is only a small safety margin against racing the receipt pruner's own transaction, not a correctness bound.

A background health monitor (`FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC`, default 900s) watches for: `substrate_field_state`'s oldest row exceeding `FIELD_STATE_STALL_MULTIPLIER` × the retention window (pruner stalled), `substrate_field_applied_deltas` row count exceeding `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT`, and the `conjourney` database exceeding `FIELD_DIGESTER_DB_SIZE_ALERT_GB`. Checks are edge-triggered — an alert (via `orion-notify`'s `POST /attention/request`, surfacing in Hub's existing Pending Attention panel) fires only on a healthy→unhealthy transition, plus a lower-severity recovery note on the way back, so a persisting condition does not spam a fresh attention item every check.

This whole chain was originally left disabled after a prior unbounded-Postgres-growth incident on this host; the guardrails above (mirroring the existing `receipt_pruner.py` pattern in `orion-substrate-runtime`) are what made re-enabling it safe.

## Decay vs. injection-interval mismatch — confirmed root cause of the "accumulator-oscillation artifact" (2026-07-17)

The "Field channel glossary" section below (written 2026-07-16) flagged `cpu_pressure`/`gpu_pressure`
as *"a known accumulator-oscillation artifact (~60s beat); not confirmed whether that beat
reflects real hardware load or a polling-architecture artifact."* This section resolves that
question, found while tracing a separate, unrelated-looking symptom (a drive-pressure economy
saturation bug in `orion/spark/concept_induction/drives.py`, unpacked in full in
[`orion/autonomy/drives_and_autonomy_retrospective.md` §5b](../../orion/autonomy/drives_and_autonomy_retrospective.md)) all the way back to this service.

**It is a polling-architecture artifact, not real hardware load**, and it is a straightforward
consequence of two mechanisms in this pipeline working against each other:

- `apply_decay()` (`app/digestion/decay.py`) multiplies every channel in `NODE_DECAY_CHANNELS`
  (`cpu_pressure`, `memory_pressure`, `gpu_pressure`, `thermal_pressure`, `disk_pressure`,
  `staleness`, `failure_pressure`, `execution_friction`, `reliability_pressure`, and more) by
  `BIOMETRICS_FIELD_DECAY_RATE` (default `0.92`) on **every single `RECEIPT_POLL_INTERVAL_SEC`
  tick (default 2.0s), unconditionally** — whether or not fresh biometrics data arrived that
  tick.
- `apply_perturbations()` (`app/digestion/perturbation.py`) applies a fresh `node_biometrics`/
  `active_node_pressure` reading via `mode="replace"` (the default mode for most hardware
  channels — see the node-channel entries below): `node_vec[channel] = max(0.0, min(1.0,
  p.intensity))`, a **full overwrite**, not a blend, discarding whatever the decayed value had
  drifted to.

`orion-biometrics` only publishes fresh readings every `CLUSTER_PUBLISH_INTERVAL` (default 15s,
a re-broadcast of the latest received summary) or `TELEMETRY_INTERVAL` (default 30s, the actual
host re-measurement) — see `services/orion-biometrics/.env_example`. Between those publishes,
`apply_decay` erodes the channel continuously: `0.92^7 ≈ 0.56`, i.e. a ~44% loss over the ~15s
gap between publishes (7-8 ticks at 2s each). The next publish then snaps the value straight back
up via the full-overwrite `mode="replace"` path, with zero memory of the decayed trajectory.
**This produces a mechanical sawtooth regardless of whether the real underlying host metric is
stable or bursty** — it is an artifact of decay-every-tick-unconditionally plus
reset-via-full-replace, not a reflection of genuine host volatility.

Downstream, this channel-level sawtooth propagates into `SelfStateV1`'s `coherence` dimension
(`coherence_score()` in `orion/self_state/scoring.py` subtracts a penalty over the diffused
generic `pressure` channel, which is fed by `cpu_pressure`/`gpu_pressure`/`memory_pressure`/
`disk_pressure` via the topology edges in `config/field/orion_field_topology.v1.yaml`), producing
a ~16-second-period sawtooth in `coherence` (observed live: smooth rise 0.47→0.84, then a hard
snap back, repeating), which in turn drives `uncertainty` (derived directly from `coherence`) and
`agency_readiness` (a weighted composite of `coherence` and other dimensions) into their own
synchronized sawtooths. This is *not* organic cognitive signal — it mechanically fires
`tension.distress.v1` in `orion/spark/concept_induction/tensions.py` on a schedule tied to this
service's own publish/decay cadence, feeding artificially elevated tension volume into the
drive-pressure economy. See the autonomy retrospective §5b linked above for the full downstream
trace (self-state → tensions → `DriveEngine`) and why this is the same bug *class* — a decay
mechanism whose injection cadence isn't reconciled against its own decay rate — as two other bugs
found and fixed in the same investigation, just in the opposite parameter regime (decay here is
*strong enough* relative to the injection interval to swing dramatically, rather than pinning
flat).

**Blast radius is wider than the drive economy.** `NODE_DECAY_CHANNELS` feeds every consumer of
`FieldStateV1` node/capability vectors, not just self-state's `coherence`/`agency_readiness`/
`resource_pressure` — attention scoring and any other capability-vector consumer reads the same
channels and would show the same artifact.

**Not yet fixed.** This is confirmed but unpatched as of 2026-07-17 — it needs a decision (skip
decay on channels that received a fresh perturbation this tick, track true per-channel
time-since-last-real-update instead of decaying unconditionally every tick, or something else)
before anyone touches `apply_decay`/`apply_perturbations`, since both are foundational to this
service and used well beyond the biometrics/self-state path this was traced through.

## `capability_provenance` vs. attention salience (Phase 4, 2026-07-12)

`FieldStateV1.capability_provenance` records which edge source contributed the largest weighted amount to each capability channel this tick (`app/digestion/diffusion.py`) — a magnitude-comparison primitive, distinct from `orion-attention-runtime`'s salience scoring (which factors in novelty/urgency/confidence too, threshold-gated). Cross-checked against live data
(`docs/notes/2026-07-12-phase4-attention-provenance-crosscheck.md`): they corroborate on structurally-obvious nodes (100% agreement for the two single-source edges into `capability:transport`/`capability:graph`) and diverge in an explainable way on contested ones (`node:atlas` wins `capability:llm_inference`'s provenance 81.7% of ticks but only clears attention's top-5 salience list 52.9% of the time — expected, since salience is a stricter, multi-factor, capped signal, not the same question as "who contributed most"). Verdict: keep both, do not unify — they answer different questions by design.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URI` | (required) | Postgres connection string |
| `LATTICE_PATH` | `config/field/orion_field_topology.v1.yaml` | Node/capability lattice YAML (canonical) |

`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
| `RECEIPT_POLL_INTERVAL_SEC` | `2.0` | Receipt poll interval |
| `BIOMETRICS_FIELD_DECAY_RATE` | `0.92` | Per-tick pressure decay multiplier |
| `BIOMETRICS_FIELD_DIFFUSION_RATE` | `1.0` | Node→capability diffusion strength |
| `FIELD_DIGESTER_IDLE_TICK_ENABLED` | `true` | Keep minting ticks (decay/diffusion only) on quiet polls with no new receipts — see section above |
| `FIELD_STATE_RETENTION_HOURS` | `72.0` | `substrate_field_state` retention window (hourly batched prune) |
| `FIELD_STATE_PRUNE_INTERVAL_SEC` | `3600.0` | `substrate_field_state` prune cadence |
| `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS` | `1.0` | Safety margin before pruning an applied-delta row whose receipt is already gone |
| `FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC` | `900.0` | Health-monitor check cadence |
| `FIELD_STATE_STALL_MULTIPLIER` | `1.5` | Alert if `substrate_field_state`'s oldest row exceeds this × retention hours |
| `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT` | `5000000` | Alert if `substrate_field_applied_deltas` row count exceeds this |
| `FIELD_DIGESTER_DB_SIZE_ALERT_GB` | `60.0` | Alert if the `conjourney` database exceeds this size (observed baseline ~37.5GB as of 2026-07-12; default leaves real headroom, not a round guess) |
| `NOTIFY_BASE_URL` | `http://orion-athena-notify:7140` | `orion-notify` base URL for health-monitor attention alerts |
| `NOTIFY_API_TOKEN` | (empty) | `orion-notify` auth token, if configured |
| `LOG_LEVEL` | `INFO` | Python log level |
| `FIELD_DIGESTER_PORT` | `8116` | Host port for `docker compose` (compose-only) |
| `FIELD_CHANNEL_CORPUS_PATH` | (empty) | Field-channel raw-substrate corpus sink path — see section below. Empty/unset = disabled |
| `CORPUS_SINK_MAX_BYTES` | `200000000` | Corpus sink rotation threshold (bytes) |
| `CORPUS_SINK_ROTATED_KEEP` | `5` | Corpus sink rotated-file retention count |

v1 persists projections to Postgres only; bus emit is deferred (`orion/bus/channels.yaml` unchanged).

## Field-channel raw-substrate corpus collector (2026-07-13, roadmap item 1 v2)

`_tick()` optionally appends one `FieldChannelCorpusRowV1` row per tick to a
JSONL sink (`FIELD_CHANNEL_CORPUS_PATH`) — the flat, channel-name-keyed
output of `orion.self_state.scoring.collect_field_channel_pressures(state)`
(the merged `node_vectors`/`capability_vectors` pressure dict, e.g.
`cpu_pressure`/`gpu_pressure`/`memory_pressure`/`execution_load`/etc.,
typically 10-20 channels, not a fixed set). This is Item 1 v2 of
`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md` — the
corrected replacement target for `orion-spark-introspector`'s
`mood_arc_corpus.v1` sink (that sink is unaffected, still running; see the
spec doc's "Correction, 2026-07-13" note for why the original 4-scalar
post-composite design was found insufficient for detecting genuine
emergent structure).

**Off by default** (`FIELD_CHANNEL_CORPUS_PATH` empty = no-op, no file
written) — same convention as `MOOD_ARC_CORPUS_PATH`, and for the same
reason: a sibling corpus sink was found to grow unbounded (~104MB/36.8k
rows over 5 days) before rotation was added, so anything new stays inert
until an operator opts in.

The append happens unconditionally on every tick that produces a `state`
(right after the coherence-warning loop, before the `if not fetched: ...`
branch that only decides which store-write path runs) — it does not depend
on whether new receipts were actually fetched this tick.

**Rotation policy** (`CORPUS_SINK_MAX_BYTES`/`CORPUS_SINK_ROTATED_KEEP`) is
the same shared class (`orion.telemetry.corpus_sink.InnerStateCorpusSink`,
promoted out of `orion-spark-introspector`'s `app/` folder in this same
patch since a second service now needs it) and the same default values as
`orion-spark-introspector`'s corpus sinks — each service reads its own env
file independently.

**Dark by design (`REHEARSAL`)**: no bus publish, no cognition consumer —
see `field_channel_corpus.v1` in `orion/self_state/inner_state_registry.py`.
A future rework of `scripts/fit_mood_arc_encoder.py` to train against this
dict-shaped corpus instead of `mood_arc_corpus.v1` is separate, not-yet-
built work.

## Field channel glossary

This is the consolidated reference for all 29 channels in
`field_channel_corpus.v1` (`orion.schemas.telemetry.field_channel_corpus.FieldChannelCorpusRowV1`),
sourced from `app/tensor/channels.py`'s `NODE_CHANNELS` (23) + `CAPABILITY_CHANNELS`
(8), overlapping on `transport_pressure`/`contract_pressure` (23+8-2=29). This
corpus is the raw input for a future windowed autoencoder (roadmap item 2,
not yet trained) — but several of these same channels also feed
`SelfStateV1`'s live, cognition-facing dimensions today, via
`orion.self_state.scoring.collect_field_channel_pressures()` and
`config/self_state/self_state_policy.v1.yaml`. Written 2026-07-16 because no
centralized doc previously existed for what each channel means, how it's
calculated, or which self-state dimension (if any) it feeds.

**Pipeline order** (`app/tensor/update_rules.py::run_digestion_tick`, called
once per tick from `app/worker.py::_tick`): reconcile (seed any
lattice-declared node/capability not yet present with
`DEFAULT_NODE_VECTOR`/`DEFAULT_CAPABILITY_VECTOR`, `app/tensor/reconcile.py`)
→ `apply_perturbations` → `apply_decay` → `apply_diffusion` →
`apply_suppression` → `check_field_coherence` (sets `field_coherence_warning`
directly, outside the perturbation system) → `collect_field_channel_pressures`
(merges `node_vectors`+`capability_vectors` into the flat corpus row).

**Producer/calculation column key**: "target_kind" refers to
`StateDeltaV1.target_kind` as consumed by
`app/ingest/state_deltas.py::delta_to_perturbations()`, which has six blocks:
`active_node_pressure`, `node_biometrics`, `execution_run`, `chat_turn`,
`transport_bus`, `prediction_signal`. "mode" refers to `Perturbation.mode`
(`app/digestion/perturbation.py`): `replace` sets the channel directly
(clamped 0-1); the `availability` special case floors via `min(current,
intensity)` (can only decrease); every other default (`add`) accumulates via
`min(1.0, current + intensity)` (only increases within a tick, relies on
decay to come back down). `NODE_DECAY_CHANNELS`/`CAPABILITY_DECAY_CHANNELS`
are `app/digestion/decay.py`'s two decay sets — the capability set is
currently dead weight for every capability in the live topology, since
`apply_diffusion()` runs immediately after in the same tick and
unconditionally overwrites every diffusion-target channel with a fresh
memoryless recompute (`decay.py` lines 46-64, dated 2026-07-12). Topology
edges are `config/field/orion_field_topology.v1.yaml`. Note: that yaml file
also declares its own `node_channels`/`capability_channels` lists (lines
19-49) that look authoritative but are **not parsed by any code** (no
reference to either key anywhere under `app/`) — they are stale, incomplete
documentation inside the yaml itself (missing `egress_confidence_deficit`,
`prediction_error`, `field_coherence_warning` from the node list), not a
second source of truth. `channels.py`'s `NODE_CHANNELS`/`CAPABILITY_CHANNELS`
are the only enforced channel set.

**SelfState dimension column key**: from
`config/self_state/self_state_policy.v1.yaml`. `channel_dimension_map`
entries score a dimension directly. `evidence_channel_map` entries are
evidence/transparency only (surfaced in `dominant_evidence`/reasons) and do
**not** score anything — this exists specifically so
`orion-spark-introspector`'s tissue-viz bypass can read raw hardware-channel
names without reintroducing the double-counting bug that removing them from
`channel_dimension_map` fixed (2026-07-12). `stabilizing_channels` entries
carry an explicit weight used inside `scoring.py::coherence_score()`.
Verified against the full 138-line policy file, not a partial read:
`contract_pressure` and `catalog_drift_pressure` appear in **none** of
`channel_dimension_map`, `evidence_channel_map`, `stabilizing_channels`, or
`pressure_channels` — both are calculated and corpus-collected but feed no
live `SelfStateV1` dimension at all today.

**Live-data verdict column key**: verdicts below were re-verified against
the live corpus at `/mnt/telemetry/field_channels/corpus/field_channels.jsonl`
(123,332+ rows / 69h+ as of 2026-07-16) rather than taken on faith. Verdict
categories used: **real signal** (continuously or genuinely varying, no
known bug), **one-way ratchet** (mode=`add` + absent from
`NODE_DECAY_CHANNELS`, so it can only climb, never come back down),
**collateral damage** (a healthy-looking channel whose observed flatness is
actually caused by a *different* channel's bug), **masked by merge-polarity
bug** (real variance exists somewhere in the mesh but is hidden by
`collect_field_channel_pressures()`'s max()-merge, which is correct for
pressure-type channels but backwards for "higher-is-better" channels),
**saturating counter** (pinned by a fixed-size cap, not real activity
running out), **folded-away** (computed individually upstream but only ever
reaches the field as part of a different composite scalar, never perturbed
by its own name), **fully unproduced** (more extreme than folded-away: no
perturbation source and no diffusion source ever populates it, so it never
even reaches the merged corpus as a key, not even at 0.0), **exact
duplicate** (two channels observed byte-identical across every live row
checked), and **quiet/genuinely-wired** (correctly implemented, healthy
decay mechanism, simply has not received a real nonzero input in the
observed window).

Two mechanics worth knowing before reading the verdicts: (1)
`collect_field_channel_pressures()` (`orion/self_state/scoring.py:70,76`)
only adds a channel to the merged output when `channel in PRESSURE_CHANNELS
or value > 0` — `PRESSURE_CHANNELS` (scoring.py:7-27) is a 19-channel subset
of the 29; a channel outside that set only ever appears in the corpus once
it has received at least one genuinely nonzero perturbation, which is why
some channels (e.g. `transport_pressure`) never appear at all while others
with near-zero float noise (e.g. `observer_failure_pressure`, max observed
value `3e-323` — a floating-point subnormal, not a real signal) do appear.
(2) `DEFAULT_NODE_VECTOR`/`DEFAULT_CAPABILITY_VECTOR` (`channels.py:37-42`,
`availability`/`confidence`/`available_capacity` seeded to `1.0`) are
actively applied every tick via `reconcile_field_state_with_lattice()`
(`app/tensor/reconcile.py`, called from `app/worker.py:132`, before
`run_digestion_tick` at `app/worker.py:156`) — they are not vestigial.
`app/digestion/suppression.py:9`'s own `vec.get("availability", 1.0)`
default is redundant defensive coding on top of that, not the actual source
of the observed 1.0 baseline.

### Node channels (`NODE_CHANNELS`, 23)

#### `availability`
- **Meaning**: how available/reachable a compute node currently is.
- **Producer**: `active_node_pressure` delta, `"availability"` in
  `active_pressures` → special-cased in `perturbation.py`'s `apply_perturbations`
  (not through the generic add/replace path): `min(current, intensity)`,
  i.e. floor-only, can only decrease within a tick. Node-only; no topology
  edge diffuses `availability` to any capability. Not in `NODE_DECAY_CHANNELS`.
- **SelfState dimension fed**: `channel_dimension_map`: `availability` →
  `coherence`. Also `stabilizing_channels`: `availability` weight `0.50`
  (used inside `coherence_score()`).
- **Live-data verdict**: collateral damage. Floored to `≥0.85` on every tick
  by `apply_suppression()` (`suppression.py:8-9`) because
  `expected_offline_suppression` is latched at `1.0` essentially always;
  separately, its own perturbation path has never fired in the live corpus.
  Confirmed live 2026-07-16.

#### `staleness`
- **Meaning**: whether a node's most recent biometrics sample is too old to
  trust.
- **Producer**: `node_biometrics` delta, set to `0.5` (mode=`add`, default)
  when `availability_status == "stale"`. In `NODE_DECAY_CHANNELS`. Not a
  diffusion source/target anywhere in the topology.
- **SelfState dimension fed**: `channel_dimension_map`: `staleness` →
  `continuity_pressure`. Also referenced directly inside
  `coherence_score()`'s penalty loop (`("failure_pressure",
  "execution_friction", "staleness", "pressure")`, `0.25` weight each).
- **Live-data verdict**: collateral damage. Unconditionally zeroed every
  tick by `apply_suppression()` (`suppression.py:10`) whenever
  `expected_offline_suppression >= 1.0` — not natural rarity, actively
  suppressed. Confirmed live 2026-07-16.

#### `cpu_pressure`
- **Meaning**: how loaded a node's CPU currently is.
- **Producer**: `active_node_pressure` delta (`"strain"` in
  `active_pressures`, non-GPU nodes) and `node_biometrics` delta
  (`hints["strain"]`), both mode=`add`. In `NODE_DECAY_CHANNELS`. Diffuses
  into `capability:orchestration` (`pressure`, weight `0.90`),
  `capability:graph` (`pressure`, weight `0.70`), `capability:memory` via
  `node:prometheus` (`pressure`, weight `0.60`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly
  (removed 2026-07-12 — its diffused capability value is what's mapped
  instead). `evidence_channel_map`: `cpu_pressure` → `resource_pressure`
  (evidence/transparency only).
- **Live-data verdict**: real signal, continuous — but the "known
  accumulator-oscillation artifact" flagged 2026-07-16 is now **confirmed as
  a polling-architecture artifact, not real hardware load**. See "Decay vs.
  injection-interval mismatch" below for the full mechanism and the trace
  that resolved this (2026-07-17): `apply_decay()`'s unconditional
  `0.92`-per-2s decay against `orion-biometrics`' ~15-30s publish cadence
  produces a mechanical sawtooth (~16s period observed downstream in
  `coherence`) independent of whether the underlying CPU load is actually
  bursty.

#### `memory_pressure`
- **Meaning**: node RAM pressure.
- **Producer**: declared in `NODE_CHANNELS`, `NODE_DECAY_CHANNELS`, and as a
  diffusion target for `capability:storage` (`memory_pressure` → `pressure`,
  weight `0.75`, alongside `disk_pressure`) — but no `target_kind` block in
  `state_deltas.py` ever perturbs it by name, so its value never leaves its
  reconcile-seeded `0.0` default.
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `memory_pressure` → `resource_pressure`
  (evidence-only; in practice inert since the channel never carries a real
  value).
- **Live-data verdict**: folded-away, never produced. Computed individually
  as `mem_pressure` in `orion/telemetry/biometrics_pipeline.py:103` but only
  reaches the field as part of the composite `"strain"` scalar
  (`services/orion-biometrics/app/grammar_emit.py:159`); `state_deltas.py`'s
  `node_biometrics` block only reads `hints["gpu"]`/`hints["strain"]`, never
  a memory-named hint key.

#### `gpu_pressure`
- **Meaning**: how loaded a node's GPU currently is.
- **Producer**: `active_node_pressure` delta (`"strain"`, GPU nodes
  `{atlas, circe}`) and `node_biometrics` delta (`hints["gpu"]`), both
  mode=`add`. In `NODE_DECAY_CHANNELS`. Diffuses into
  `capability:llm_inference` from `node:atlas` (`pressure`, weight `0.85`,
  alongside `memory_pressure`) and `node:circe` (`pressure`, weight `0.50`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `gpu_pressure` → `resource_pressure`
  (evidence-only).
- **Live-data verdict**: real signal, continuous — same accumulator-
  oscillation mechanism as `cpu_pressure`, now confirmed as the
  decay/injection-interval mismatch described below, not real hardware load.

#### `thermal_pressure`
- **Meaning**: node thermal/temperature pressure.
- **Producer**: computed in `biometrics_pipeline.py:117` but no
  `target_kind` block perturbs it by name. In `NODE_DECAY_CHANNELS`. Not a
  diffusion source/target anywhere in the topology.
- **SelfState dimension fed**: `channel_dimension_map`: `thermal_pressure` →
  `resource_pressure` — this one **is** a direct entry (unlike
  `cpu_pressure`/`gpu_pressure`/`memory_pressure`/`disk_pressure`, which were
  demoted to evidence-only in the 2026-07-12 dedup pass), but since the
  channel is never produced this mapping is currently a dead entry in
  practice.
- **Live-data verdict**: folded-away, never produced — same mechanism as
  `memory_pressure`, composited into `"strain"` only.

#### `disk_pressure`
- **Meaning**: node disk I/O pressure.
- **Producer**: computed in `biometrics_pipeline.py:110` but no
  `target_kind` block perturbs it by name. In `NODE_DECAY_CHANNELS`.
  Diffusion target for `capability:storage` (`disk_pressure` → `pressure`,
  weight `0.75`, alongside `memory_pressure`) — inert since the source value
  is never nonzero.
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `disk_pressure` → `resource_pressure`
  (evidence-only).
- **Live-data verdict**: folded-away, never produced.

#### `expected_offline_suppression`
- **Meaning**: signals a node is expected to be offline (e.g. a scheduled or
  known-suppressed state), so its absence shouldn't be read as a failure.
- **Producer**: `active_node_pressure` delta sets `1.0` (mode=`add`) when
  `delta.operation == "suppress"`; `node_biometrics` delta sets `1.0`
  (mode=`add`) when `expected_online is False`. **Not** in
  `NODE_DECAY_CHANNELS` — mode=`add` with no decay means it can only
  increase, never come back down.
- **SelfState dimension fed**: `channel_dimension_map`:
  `expected_offline_suppression` → `coherence`. Also `stabilizing_channels`:
  weight `0.30`.
- **Live-data verdict**: one-way ratchet — latched at `1.0` for the entire
  observed corpus span. Confirmed live 2026-07-16.

#### `execution_load`
- **Meaning**: how much active execution work (agent runs, tool calls) is in
  flight on a node.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:orchestration`
  (`execution_pressure`, weight `0.90`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly
  (removed 2026-07-12). `evidence_channel_map`: `execution_load` →
  `execution_pressure` (evidence-only).
- **Live-data verdict**: real signal, continuous. Confirmed live 2026-07-16.

#### `execution_friction`
- **Meaning**: how much resistance (retries/backoff) execution is
  encountering on a node.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:orchestration`
  (`reliability_pressure`, weight `0.90`, `max()`'d against
  `failure_pressure`'s contribution to the same target).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `execution_friction` → `reliability_pressure`
  (evidence-only). Also referenced directly in `coherence_score()`'s
  penalty loop (`0.25` weight).
- **Live-data verdict**: real signal — part of a correlated sparse-event
  trio with `failure_pressure`/`reliability_pressure`; one real spike traced
  to `2026-07-16T02:15:08Z`, decaying cleanly afterward at the `0.92`/tick
  rate — genuine rare-event signal, not dead.

#### `reasoning_load`
- **Meaning**: how much active LLM-reasoning work a node is doing.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:llm_inference`
  (`reasoning_pressure`, weight `0.85`, from `node:atlas`) and
  `capability:orchestration` (`reasoning_pressure`, weight `0.90`, from
  `node:athena`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `reasoning_load` → `reasoning_pressure`
  (evidence-only).
- **Live-data verdict**: real signal — the cleanest channel in the corpus,
  continuously varying, small amplitude. Confirmed live 2026-07-16.

#### `failure_pressure`
- **Meaning**: recent execution failure rate/severity on a node.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:orchestration`
  (`reliability_pressure`, weight `0.90`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `failure_pressure` → `reliability_pressure`
  (evidence-only). Also referenced directly in `coherence_score()`'s
  penalty loop (`0.25` weight).
- **Live-data verdict**: real signal — part of the same correlated
  sparse-event trio as `execution_friction`, same `2026-07-16T02:15:08Z`
  spike, genuine rare-event signal.

#### `egress_confidence_deficit`
- **Meaning**: `1 - confidence` that an execution's output actually reached
  its destination.
- **Producer**: `execution_run` delta, `max(0.0, min(1.0, 1.0 -
  egress_raw))`, mode=`replace`. In `NODE_DECAY_CHANNELS`. Not a diffusion
  source/target anywhere in the topology.
- **SelfState dimension fed**: `channel_dimension_map`:
  `egress_confidence_deficit` → `introspection_pressure` (direct).
- **Live-data verdict**: real signal, sparse/low-duty-cycle but genuinely
  varying. Confirmed live 2026-07-16.

#### `repair_pressure`
- **Meaning**: how much conversational "repair" (corrections,
  re-explaining) is happening in chat.
- **Producer**: `chat_turn` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`.
  Not a diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`: `repair_pressure` →
  `social_pressure` (direct).
- **Live-data verdict**: real signal, sparse but genuinely varying.

#### `conversation_load`
- **Meaning**: how much active conversational load (turn volume/complexity)
  is occurring.
- **Producer**: `chat_turn` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`.
  Not a diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`: `conversation_load`
  → `social_pressure` (direct).
- **Live-data verdict**: real signal, sparse but genuinely varying.

#### `transport_pressure`
- **Meaning**: bus/transport-layer backpressure/congestion (a node channel
  and a capability channel, one of the two overlapping names between
  `NODE_CHANNELS` and `CAPABILITY_CHANNELS`).
- **Producer**: `transport_bus` delta, via `hints["transport_pressure"]`,
  `hints["stream_depth_pressure"]`, or `hints["backpressure"]` — all
  mode=`add` (default), targeting a **node** vector. In
  `NODE_DECAY_CHANNELS` and `CAPABILITY_DECAY_CHANNELS`. As a node channel
  it is a diffusion source for `capability:orchestration` (`pressure`,
  weight `0.90`) and `capability:transport` (`pressure`, weight `0.85`), and
  for the `capability:transport → capability:orchestration` cap-cap edge
  (`transport_pressure` → `transport_pressure`, weight `0.70`) — but that
  cap-cap edge's own source value (`capability:transport`'s own
  `transport_pressure` key) is only ever seeded `0.0` by
  `DEFAULT_CAPABILITY_VECTOR`, since the `node:athena → capability:transport`
  edge maps `transport_pressure` → `"pressure"`, not `"transport_pressure"`
  — no edge ever writes a channel literally named `transport_pressure`
  directly onto `capability:transport`.
- **SelfState dimension fed**: not in `channel_dimension_map` directly
  (removed 2026-07-12). `evidence_channel_map`: `transport_pressure` →
  `resource_pressure` (evidence-only).
- **Live-data verdict**: fully unproduced — confirmed absent (key entirely
  missing, not just `0.0`) from all 123,245+ live rows checked 2026-07-16
  (`jq -r '.channels.transport_pressure // "MISSING"' ... | sort -u` returns
  only `MISSING`). More extreme than "folded-away":
  `collect_field_channel_pressures()` only includes a channel when `channel
  in PRESSURE_CHANNELS or value > 0` (`scoring.py:70,76`), and
  `transport_pressure` is not in `PRESSURE_CHANNELS`, so even its
  reconcile-seeded `0.0` never reaches the corpus — it needs at least one
  real nonzero perturbation to ever appear at all, and none has occurred in
  this corpus.

#### `contract_pressure`
- **Meaning**: intended to represent pressure from bus/schema "contract"
  mismatches (the precise real-world condition isn't otherwise documented
  in code — it's perturbed from the same `transport_bus` hint dict as
  `catalog_drift_pressure`). A node channel and a capability channel (the
  other overlapping name).
- **Producer**: `transport_bus` delta, `hints["contract_pressure"]`,
  mode=`add` (default), onto a **node** vector. In `NODE_DECAY_CHANNELS` and
  `CAPABILITY_DECAY_CHANNELS`. At the capability level, `contract_pressure`
  is actually populated by a *different* node-level channel:
  `node:athena → capability:transport`'s edge maps `catalog_drift_pressure`
  → `"contract_pressure"` (weight `0.85`) — no edge maps the node-level
  `contract_pressure` channel itself into any capability field.
- **SelfState dimension fed**: **none.** Verified against the full 138-line
  `self_state_policy.v1.yaml` (not a partial read): `contract_pressure`
  appears in none of `channel_dimension_map`, `evidence_channel_map`,
  `stabilizing_channels`, or `pressure_channels`. Calculated and
  corpus-collected, feeds no live `SelfStateV1` dimension.
- **Live-data verdict**: exact duplicate of `catalog_drift_pressure` —
  byte-identical across every row checked (123,246 pairs, 0 mismatches,
  re-verified 2026-07-16). Root cause not yet determined by this patch
  (documentation-only): structurally, the field-digester code treats them
  as two independent perturbation targets from two separate hint keys, and
  the one diffusion path linking them (`catalog_drift_pressure` →
  capability-level `contract_pressure`, weight `0.85`) would scale the
  value down, not reproduce it byte-for-byte — which points more toward an
  upstream reducer setting both `hints["contract_pressure"]` and
  `hints["catalog_drift_pressure"]` to the same value in the same
  `transport_bus` delta, rather than a field-digester-side convergence. Not
  confirmed; a separate investigation is tracking the actual root cause.

#### `catalog_drift_pressure`
- **Meaning**: intended to represent drift/staleness in the bus event
  "catalog" (schema registry) relative to what's actually flowing.
- **Producer**: `transport_bus` delta, `hints["catalog_drift_pressure"]`,
  mode=`add` (default), onto a node vector. In `NODE_DECAY_CHANNELS`. This
  is the channel that actually feeds the capability-level `contract_pressure`
  field via the `node:athena → capability:transport` diffusion edge (weight
  `0.85`) — see `contract_pressure` above.
- **SelfState dimension fed**: **none** — same verification as
  `contract_pressure` above; absent from all four policy maps.
- **Live-data verdict**: exact duplicate of `contract_pressure` (see above);
  open root-cause question, not fixed by this patch.

#### `delivery_confidence`
- **Meaning**: confidence that bus messages are actually being delivered
  end-to-end.
- **Producer**: `transport_bus` delta, `hints["delivery_confidence"]`,
  mode=`add` (default). **Not** in `NODE_DECAY_CHANNELS` — one-way ratchet
  mechanism. Diffuses into `capability:transport`'s `"confidence"` field
  (weight `0.85`, from `node:athena`) — the only capability with a direct
  diffusion edge into `confidence`.
- **SelfState dimension fed**: not in `channel_dimension_map` directly (the
  capability-level `confidence` field it feeds is the one mapped — see
  `confidence` below). `evidence_channel_map`: `delivery_confidence` →
  `coherence` (evidence-only).
- **Live-data verdict**: one-way ratchet — same mechanism as
  `expected_offline_suppression` (mode=`add`, missing from
  `NODE_DECAY_CHANNELS`); currently benign since the bus is genuinely
  stable, but structurally could never show a real dip. Confirmed live
  2026-07-16.

#### `bus_health`
- **Meaning**: overall bus/transport subsystem health signal.
- **Producer**: `transport_bus` delta, `hints["bus_health"]`, mode=`add`.
  **Not** in `NODE_DECAY_CHANNELS` — same one-way-ratchet mechanism.
  Diffuses into `capability:transport`'s `"available_capacity"` field
  (weight `0.85`, from `node:athena`) — the only capability with a direct
  diffusion edge into `available_capacity`.
- **SelfState dimension fed**: not in `channel_dimension_map` directly (the
  capability-level `available_capacity` field it feeds is the one mapped —
  see `available_capacity` below). `evidence_channel_map`: `bus_health` →
  `coherence` (evidence-only).
- **Live-data verdict**: one-way ratchet, same mechanism as
  `delivery_confidence`/`expected_offline_suppression` — currently benign,
  structurally could never dip. Confirmed live 2026-07-16.

#### `observer_failure_pressure`
- **Meaning**: pressure from failures of the bus "observer" role
  (monitoring/subscriber-side failures).
- **Producer**: `transport_bus` delta, `hints["observer_failure_pressure"]`,
  mode=`add` (default). **Is** in `NODE_DECAY_CHANNELS` — a healthy decay
  mechanism, same as `cpu_pressure`/`gpu_pressure`. Diffuses into
  `capability:transport`'s `"reliability_pressure"` field (weight `0.85`,
  from `node:athena`).
- **SelfState dimension fed**: not in `channel_dimension_map` or
  `evidence_channel_map` by its own name — only its diffusion target
  (`reliability_pressure`) is mapped (see `reliability_pressure` below).
- **Live-data verdict**: genuinely quiet, correctly wired, no bug. Uses
  add-mode and is in the healthy decay set; simply has never received a
  real nonzero perturbation. Confirmed live 2026-07-16: present in 123,258
  rows, max observed value `3e-323` (a floating-point subnormal — decay
  noise, not a real signal).

#### `field_coherence_warning`
- **Meaning**: per-node incoherence score — fires when one channel is high
  and a paired channel that should track it is simultaneously low,
  suggesting two different reducers disagree about the node's actual
  state (rules: `execution_load`/`cpu_pressure`,
  `execution_load`/`gpu_pressure`, `failure_pressure`/`availability`,
  `transport_pressure`/`bus_health`, `reasoning_load`/`cpu_pressure` —
  `orion/field_coherence.py`).
- **Producer**: **not** one of `state_deltas.py`'s six `target_kind`
  blocks. Computed by `check_field_coherence(state)`
  (`orion/field_coherence.py:37`) and written directly onto
  `state.node_vectors[node_id]["field_coherence_warning"]`
  (`app/worker.py:163-164`) *after* `run_digestion_tick` (perturb/decay/
  diffuse/suppress) completes each tick — it bypasses the `Perturbation`
  add/replace mechanism entirely (direct dict assignment, effectively
  memoryless per tick). In `NODE_DECAY_CHANNELS` (largely moot since it's
  overwritten fresh every tick anyway). Not a diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`:
  `field_coherence_warning` → `coherence` (direct).
- **Live-data verdict**: real signal — present in all 123,332 rows checked,
  nonzero in all of them, max observed `0.4`. Sparse/low-duty-cycle in
  magnitude but genuinely, continuously computed. Confirmed live
  2026-07-16.

#### `prediction_error`
- **Meaning**: how much a recent prediction (from the prediction/surprise
  subsystem) missed reality.
- **Producer**: `prediction_signal` delta, `hints["prediction_error"]`,
  `max(0.0, min(1.0, ...))`, mode=`replace`. In `NODE_DECAY_CHANNELS`. Not a
  diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`: `prediction_error` →
  `uncertainty` (direct).
- **Live-data verdict**: quiet-so-far-but-correctly-wired, not dead. Flat
  (float noise on the order of `1e-18`, an artifact of repeated `0.92`
  decay applied to something already effectively zero) for the original
  69h scan window. A live re-check mid-session (2026-07-16) found real
  values starting to appear: first meaningful (`>1e-4`) value at row
  122,564, `2026-07-16T20:33:59.918142Z`, subsequently reaching `0.92`. This
  is a live-growing corpus; the channel was quiet during the scan window,
  not broken.

### Capability-only channels (`CAPABILITY_CHANNELS` not already listed above, 6)

#### `pressure`
- **Meaning**: overall load/pressure for a given capability
  (`llm_inference`, `orchestration`, `storage`, `graph`, `memory`,
  `transport`).
- **Producer**: capability-level only, populated purely by diffusion — most
  node→capability edges target `"pressure"` (see the per-channel entries
  above for the specific node-level sources feeding each capability). In
  `CAPABILITY_DECAY_CHANNELS`, but per `decay.py`'s 2026-07-12 comment this
  is dead weight: `apply_diffusion()` runs immediately after in the same
  tick and unconditionally overwrites it with a fresh memoryless recompute.
- **SelfState dimension fed**: `channel_dimension_map`: `pressure` →
  `resource_pressure` (direct).
- **Live-data verdict**: real signal, no bug — continuously varying (e.g.
  `capability:orchestration` observed mean `0.55`, std `0.20`). Confirmed
  live 2026-07-16.

#### `confidence`
- **Meaning**: how confident the system is that a given capability is
  currently reliable/available.
- **Producer**: capability-level. `capability:transport` has the only
  direct diffusion edge into `confidence` (from `delivery_confidence`,
  weight `0.85`). Every other capability
  (`orchestration`/`llm_inference`/`storage`/`graph`/`memory`) uses
  `apply_diffusion()`'s fallback formula when no direct edge fed it a real
  value this tick: `confidence = max(0.0, 1.0 - 0.5 * pressure)`
  (`diffusion.py:216-229`). Seeded `1.0` by `DEFAULT_CAPABILITY_VECTOR` via
  reconcile.
- **SelfState dimension fed**: `channel_dimension_map`: `confidence` →
  `coherence` (direct). Also `stabilizing_channels`: weight `0.50`.
- **Live-data verdict**: masked by merge-polarity bug. Real variance exists
  at `capability:orchestration` (derived from its own genuinely-varying
  `pressure`, mean `0.55`/std `0.20`) but is masked because
  `collect_field_channel_pressures()`'s max()-merge
  (`scoring.py:70,76`, `v >= out.get(channel, 0.0)`) is correct for
  pressure-type channels (higher = worse, worst-contributor wins) but
  backwards for a "goodness" channel like `confidence` — `capability:
  transport`'s near-1.0 constant value (itself a symptom of the
  `delivery_confidence` one-way ratchet) always wins the max(), permanently
  hiding `orchestration`'s real variation from the merged corpus. Confirmed
  live 2026-07-16.

#### `available_capacity`
- **Meaning**: how much spare operating capacity a capability has right
  now.
- **Producer**: capability-level. `capability:transport` has the only
  direct diffusion edge into `available_capacity` (from `bus_health`,
  weight `0.85`). Every other capability falls back to
  `available_capacity = max(0.0, 1.0 - pressure)` (`diffusion.py:216-227`).
  Seeded `1.0` by `DEFAULT_CAPABILITY_VECTOR` via reconcile.
- **SelfState dimension fed**: `channel_dimension_map`: `available_capacity`
  → `coherence` (direct). Also `stabilizing_channels`: weight `0.50`.
- **Live-data verdict**: masked by the same merge-polarity bug as
  `confidence` — `capability:transport`'s ratcheted near-1.0 constant
  always wins the max()-merge over `orchestration`'s real variation.

#### `execution_pressure`
- **Meaning**: capability-level execution-load pressure (currently only
  populated for `orchestration`).
- **Producer**: capability-level, diffused from `node:athena →
  capability:orchestration` (`execution_load` → `execution_pressure`,
  weight `0.90`). In `CAPABILITY_DECAY_CHANNELS`, same dead-weight caveat as
  `pressure` (overwritten by `apply_diffusion()` immediately after decay,
  every tick).
- **SelfState dimension fed**: `channel_dimension_map`: `execution_pressure`
  → `execution_pressure` (direct, 1:1 dimension name).
- **Live-data verdict**: real signal, continuous — same underlying trace as
  `execution_load`. Confirmed live 2026-07-16.

#### `reasoning_pressure`
- **Meaning**: capability-level reasoning-load pressure
  (`llm_inference`/`orchestration`'s reasoning burden).
- **Producer**: capability-level, diffused from `node:atlas →
  capability:llm_inference` (`reasoning_load` → `reasoning_pressure`,
  weight `0.85`) and `node:athena → capability:orchestration`
  (`reasoning_load` → `reasoning_pressure`, weight `0.90`). In
  `CAPABILITY_DECAY_CHANNELS`, same dead-weight caveat as `pressure`.
- **SelfState dimension fed**: `channel_dimension_map`: `reasoning_pressure`
  → `reasoning_pressure` (direct, 1:1).
- **Live-data verdict**: real signal — cleanest channel in the corpus per
  this investigation, continuously varying, small amplitude.

#### `reliability_pressure`
- **Meaning**: capability-level pressure representing risk to reliability
  (from execution friction, failures, or observer failures).
- **Producer**: capability-level only per `channels.py`'s declared lists
  (not in `NODE_CHANNELS`), diffused from `node:athena →
  capability:orchestration` (`execution_friction` → `reliability_pressure`
  and `failure_pressure` → `reliability_pressure`, both weight `0.90`,
  `max()`'d together per the 2026-07-12 diffusion fix) and `node:athena →
  capability:transport` (`observer_failure_pressure` →
  `reliability_pressure`, weight `0.85`). Note: `decay.py`'s
  `NODE_DECAY_CHANNELS` set *also* lists `reliability_pressure` (line 26)
  even though it's absent from `channels.py`'s `NODE_CHANNELS` — currently
  inert for `node_vectors` specifically, since no producer ever writes
  `reliability_pressure` onto a node vector by name; it's meaningful for
  `capability_vectors` via `CAPABILITY_DECAY_CHANNELS` (same dead-weight
  caveat as `pressure`, though, per the diffusion overwrite).
- **SelfState dimension fed**: `channel_dimension_map`: `reliability_pressure`
  → `reliability_pressure` (direct, 1:1).
- **Live-data verdict**: real signal — the third leg of the correlated
  sparse-event trio with `execution_friction`/`failure_pressure`, same
  `2026-07-16T02:15:08Z` spike, genuine rare-event signal, decaying cleanly
  afterward.
