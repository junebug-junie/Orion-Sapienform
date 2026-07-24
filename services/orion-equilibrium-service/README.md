# orion-equillibrium-service

System health + (currently) baseline Collapse Mirror metacognition ticks.

> Spelling matches repo (`equillibrium`).

---

## The Metacognition (double duty--refactor me into a new service!)
Equilibrium is doing **two** intentional jobs in your current implementation:

1) **Health aggregator**: “Which services are healthy/missing?”
2) **Metacognition tick emitter**: a periodic **Collapse Mirror baseline snapshot** with:
   - `trigger = "equilibrium.metacognition_tick"`
   - `snapshot_kind = "baseline"`
   - summary like “Periodic metacognition snapshot emitted by equilibrium monitor.”
   - published to `orion:event:equilibrium:snapshot`
   - envelope kind: `equilibrium.collapse.snapshot`

This is controlled by:
- `EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC` (default ~15s)

### The metacog trigger family — big picture

This section exists because design decisions for this system have historically only lived in one-off docs under `docs/superpowers/specs/`/`docs/superpowers/design/` that nobody goes back and reads once the branch merges. This README is meant to be the thing that stays current — the design docs below are the deep-dive/forensic record of *how* each decision got made, not the place to look first.

**The pipeline, same for every trigger kind:** a gate module in this service (`app/<kind>_metacog_gate.py`) evaluates real evidence and, if a real condition holds, builds a `MetacogTriggerV1` (`orion/schemas/telemetry/metacog_trigger.py`). `_publish_metacog_trigger()` (`app/service.py`) checks that kind's cooldown lane, then publishes to `CHANNEL_EQUILIBRIUM_METACOG_TRIGGER` (`orion:equilibrium:metacog:trigger`). `orion-cortex-orch`'s `dispatch_metacog_trigger()` picks it up and runs a fresh, independent `log_orion_metacognition` plan (`MetacogContextService` → `MetacogDraftService` → `MetacogEnrichService` → `MetacogPublishService`, all in `orion-cortex-exec`), which writes a real `MetacogEntryV1` row into the `orion_metacog` Postgres table via `orion-sql-writer`. Every trigger kind below reuses this exact mechanism unmodified — a new kind only ever adds a gate module + a dispatch branch here, never touches the draft/enrich/publish machinery.

**The design rule every trigger kind here follows, learned the hard way across several rounds:**
- **Ground every gate condition in a real, already-computed field.** Never invent a derived signal or guess a boolean from vibes — if the field doesn't exist yet, the condition doesn't ship yet (see `chat_turn`'s own history below, where two originally-proposed conditions got dropped for exactly this reason).
- **One gate module per trigger kind** (`app/chat_turn_metacog_gate.py`, `app/telemetry_anomaly_metacog_gate.py`, `app/transport_metacog_gate.py`, etc.) — never one shared mega-function.
- **Give a new kind its own cooldown lane if it fires on a fundamentally different cadence than the shared periodic/rare pattern.** `chat_turn` (fires on essentially every remarkable turn) originally shared the global `EQUILIBRIUM_METACOG_COOLDOWN_SEC` lane with `baseline`/`manual`/`pulse`/`relational`/`telemetry_anomaly` — a real bug, since a burst of `chat_turn` fires could silently starve the others. Fixed 2026-07-23 (`_publish_metacog_trigger`'s `_PER_KIND_COOLDOWN_SETTINGS_ATTR` dict, generalized from a hardcoded if/else the same day `transport` needed its own lane too). Any future kind that fires often should get its own `EQUILIBRIUM_METACOG_<KIND>_COOLDOWN_SEC` from day one, not retrofit it after shipping.
- **Ships disabled by default.** Every kind flips on only after a real, non-degenerate `orion_metacog` row has been observed post-deploy — "the flag is on" is not the same claim as "it's verified," and this repo's own history (`telemetry_anomaly`'s 2026-07-21 arsonist audit, in the design doc below) is a documented case of that distinction being missed for a while.
- **`orion_metacog` currently has no confirmed real consumer.** This is a standing open question, not resolved by adding more trigger kinds — see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s "Missing questions" for the full framing (`orion_metacog` vs. the separately-fed `orion_metacognitive_trace` table). Shipping a new trigger kind is real, verifiable progress on *evidence quality* regardless — but it is not, by itself, progress toward that open question, and shouldn't be reported as if it were.

**Current trigger kinds:**

| `trigger_kind` | Evidence source | Cooldown lane | Status |
|---|---|---|---|
| `baseline` | Scheduled tick, `EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC` | shared | live |
| `manual` | User-triggered Collapse Mirror event | shared | live |
| `dense` / `pulse` | Substrate self-state eventfulness score | shared | live |
| `relational` | Real `repair_pressure_v2` appraisal | shared | live |
| `telemetry_anomaly` | Trained autoencoder reconstruction-loss anomaly | shared | live (2026-07-21) |
| `chat_turn` | Correlated `ThoughtEventV1` + `HarnessRunV1` (or a governor/stance-react timeout) | own (`EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC`) | live (2026-07-23) |
| `transport` | `RpcHealthSnapshotV1` windows (Option A) + real per-call RPC timeout grammar events (Option C) | own (`EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC`) | **ships disabled**, not yet live-verified |

**A separate, parallel system this table's `transport` row borrows from, not the same pipeline:** `rpc_health` (`orion/core/bus/rpc_health.py` → `orion/core/bus/rpc_health_publish.py` → `orion:rpc_health:snapshot` → `orion-signal-gateway`'s `RpcHealthAdapter` → `OrionSignalV1`) is the `orion-signal-gateway` **organ-signal** pipeline (see that service's own README), completely independent of `orion_metacog`/`MetacogTriggerV1`. `transport`'s Option A subscribes to the same `orion:rpc_health:snapshot` channel as a *second* consumer, reading the same real data into a different destination table. Don't confuse the two pipelines when reading logs — a `rpc_health` organ signal in `orion-signal-gateway` and a `transport` metacog trigger in `orion_metacog` can both exist (or not) independently of each other.

**Deep-dive / forensic history**, if you need the "how did we get here" story behind any of the above (each is long — read the README first, these are for when you need the receipts):
- `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` — the original redesign (why `collapse_mirror`/`numeric_sisters` got replaced, the `relational`/`telemetry_anomaly`/`chat_turn` build history, the open `orion_metacog` consumer question).
- `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` + `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` — why the old `transport_pressure`/`bus_health` family was found narrowly-scoped/misleading, and the real `rpc_health` signal built to replace it.
- `docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md` — the `transport` trigger kind's own design (why it doesn't build on the old `bus.transport` grammar lane, Options A/B/C).

### Baseline metacog trigger

`_metacog_baseline_loop()` runs on `EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC` (default `1000`s) whenever `EQUILIBRIUM_METACOG_ENABLE=true`, and is the fallback trigger every other trigger type in this file effectively bypasses when it fires first: on each tick it first tries the substrate dense/pulse trigger (below); only if that doesn't fire does it fall through to a real `trigger_kind=baseline` (`reason="scheduled_check"`). `EQUILIBRIUM_METACOG_BASELINE_MAX_SKIPS` (default `3`) forces a baseline trigger anyway after that many consecutive ticks where the distress/zen scores haven't changed, so a genuinely quiet system still gets a periodic real trigger rather than skipping forever.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_ENABLE` | `true` | Master gate for the whole metacog trigger pipeline (baseline + every trigger type below) |
| `EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC` | `1000` | Baseline loop cadence |
| `EQUILIBRIUM_METACOG_BASELINE_MAX_SKIPS` | `3` | Force a real trigger after this many unchanged ticks |
| `EQUILIBRIUM_METACOG_COOLDOWN_SEC` | `30` | Global cooldown in `_publish_metacog_trigger()` -- applies across every trigger type in this file, not baseline-specific; a trigger firing during cooldown is silently dropped (logged, not queued) |

### Manual metacog trigger

Fires `trigger_kind=manual` (`reason="user_collapse_event"`) whenever a real user (not Orion itself) manually triggers a Collapse Mirror snapshot from the Hub UI, published on `CHANNEL_COLLAPSE_MIRROR_USER_EVENT` (`orion:collapse:intake`). Guarded against feedback loops: a payload with `observer=orion` is skipped outright (`elif channel == settings.channel_collapse_mirror_user_event` branch in `app/service.py`), so Orion's own collapse-mirror activity can never re-trigger itself through this path. No dedicated enable flag -- gated only by `EQUILIBRIUM_METACOG_ENABLE` above, same as every trigger type in this section.

| Env | Default | Purpose |
|-----|---------|---------|
| `CHANNEL_COLLAPSE_MIRROR_USER_EVENT` | `orion:collapse:intake` | Source channel (single consumer: this service) |

### Substrate-driven metacog triggers (dense / pulse)

When `EQUILIBRIUM_METACOG_ENABLE=true`, the baseline loop can emit **substrate-aware** triggers before falling back to scheduled baseline ticks. Equilibrium reads fresh Postgres projections (`substrate_self_state`, `substrate_execution_trajectory_projection`) via the shared felt-state reader and scores eventfulness.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_SUBSTRATE_TRIGGER_ENABLE` | `true` | Master gate for substrate dense/pulse triggers |
| `EQUILIBRIUM_METACOG_SUBSTRATE_DENSE_THRESHOLD` | `0.55` | Eventfulness score → `trigger_kind=dense` |
| `EQUILIBRIUM_METACOG_SUBSTRATE_PULSE_THRESHOLD` | `0.30` | Eventfulness score → `trigger_kind=pulse` |
| `ENABLE_SUBSTRATE_FELT_STATE_CTX` | `false` in code; `true` in `.env_example` | Must be on for Postgres hydration |
| `SUBSTRATE_FELT_STATE_DATABASE_URL` | conjourney Postgres URL | Reader DB target |
| `SUBSTRATE_FELT_STATE_MAX_AGE_SEC` | `120` | Stale rows ignored |

Docker compose wires all six keys from `.env`. Without `ENABLE_SUBSTRATE_FELT_STATE_CTX=true`, substrate triggers silently fall through to baseline.

### Relational metacog trigger

When `EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE=true`, equilibrium subscribes to `orion:repair_pressure:appraisal`, published by `orion-hub`'s `services/orion-hub/scripts/pre_turn_appraisal_wiring.py` whenever the `repair_pressure` paradigm actually runs for a turn (real repair_pressure_v2 evidence: rupture/repair detectors over the live turn window, not a self-report). `level >= EQUILIBRIUM_METACOG_RELATIONAL_LEVEL_THRESHOLD` and `confidence >= EQUILIBRIUM_METACOG_RELATIONAL_CONFIDENCE_THRESHOLD` fires `trigger_kind=relational`, carrying the full evidence breakdown (`evidence_kind`/`score`/`confidence` per detector) and `behavior_applied` in the trigger's `upstream` field.

As of 2026-07-18 this replaced the previous source, `orion/memory/turn_change_classify.py`'s SHIFT appraisal (NONE/TOPIC/STANCE/REPAIR) consumed off `orion:chat:history:spark_meta:patch` — see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` for the swap rationale. `trigger_kind=relational` is kept: same conceptual trigger category, different evidence source.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE` | `true` | Master gate for the relational trigger |
| `EQUILIBRIUM_METACOG_RELATIONAL_CONFIDENCE_THRESHOLD` | `0.7` | Minimum appraisal confidence to fire |
| `EQUILIBRIUM_METACOG_RELATIONAL_LEVEL_THRESHOLD` | `0.5` | Minimum repair_pressure level to fire |
| `CHANNEL_REPAIR_PRESSURE_APPRAISAL` | `orion:repair_pressure:appraisal` | Source channel (single consumer: this service) |

### Telemetry-anomaly metacog trigger

When `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE=true`, equilibrium subscribes to `orion:field_channel:anomaly_score`, published by `orion-field-digester`'s periodic anomaly-scoring loop (`app/anomaly_scorer.py`) whenever `FIELD_CHANNEL_ANOMALY_ENABLED=true` there. The score is reconstruction loss from a trained `orion/mood_arc/fit_encoder.py` autoencoder against the most recent rolling window of live `field_channel_corpus.v1` pressures.

Same design as the relational trigger above: the producer publishes the raw measurement (`recon_loss`, plus its own train-time `recon_error_p95` reference), this service applies its OWN `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER` rather than trusting the producer's embedded `anomalous` flag -- so trigger sensitivity is tunable here without redeploying `orion-field-digester`. Fires `trigger_kind=telemetry_anomaly` when `recon_loss > recon_error_p95 * threshold_multiplier`, carrying `recon_loss`/`recon_error_p95`/`threshold`/`window_start`/`window_end`/`encoder_id`/`encoder_version` in the trigger's `upstream` field.

Added 2026-07-21 -- see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s trigger taxonomy.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE` | `true` | Master gate for the telemetry-anomaly trigger |
| `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER` | `3.0` | Multiplier applied to the encoder's own `recon_error_p95` |
| `CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE` | `orion:field_channel:anomaly_score` | Source channel (single consumer: this service) |

### chat_turn metacog trigger

When `EQUILIBRIUM_METACOG_CHAT_TURN_TRIGGER_ENABLE=true`, equilibrium subscribes to `orion:thought:artifact` (`ThoughtEventV1`, published by `orion-thought` for every chat turn), `orion:harness:run:artifact` (`HarnessRunV1`, published by `orion-harness-governor` on every real `handle_harness_run_request` exit path), and `orion:grammar:event` filtered to two timeout atoms. A Redis-backed correlator (`app/chat_turn_metacog_gate.py::ChatTurnCorrelator`, key `orion:equilibrium:chat_turn_corr:<correlation_id>`, TTL `EQUILIBRIUM_METACOG_CHAT_TURN_CORRELATOR_TTL_SEC`) accumulates evidence per turn and fires once it's terminal.

Unlike the other triggers, this one reuses two already-registered schemas instead of a purpose-built payload -- no new bus contract, since the accumulator is ephemeral internal state, not a durable artifact. Fires `trigger_kind=chat_turn` when any of these real conditions hold (see the gate-condition table in `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s chat_turn spec section): `thought_event.disposition != "proceed"`, `thought_event.boundary_register is True`, `run_artifact.reflection.alignment_verdict != "aligned"`, `run_artifact.reflection.strain_unresolved is True`, `run_artifact.substrate_appraisal.surprise_level >= EQUILIBRIUM_METACOG_CHAT_TURN_SURPRISE_THRESHOLD`, `run_artifact.compliance_verdict != "completed"`, `run_artifact.exit_code not in (0, None)`, `run_artifact.finalize_degraded_reason is not None`, or a timeout (see below).

**Terminal evidence, four cases** (no more evidence will ever arrive for that `correlation_id`, so the correlator evaluates and clears immediately instead of waiting out the TTL):
- `run_artifact` arrived -- the turn ran to completion.
- `exec_turn_timeout` -- the harness-governor RPC never returned (`orion/hub/turn_orchestrator.py`'s `if run is None:` branch, Patch B / PR #1287).
- `stance_react_timeout` -- the *earlier* `ThoughtClient.react()` RPC itself never returned to Hub (`orion/hub/turn_orchestrator.py`'s `if thought is None:` branch); Hub never calls the harness governor on this path either.
- `thought_event.disposition in ("defer", "refuse")` -- Hub short-circuits before calling the harness governor.

Added 2026-07-22 (PR #1291), shipped disabled by default; **enabled 2026-07-23**. Live-data verification (real `orion_metacog` rows, non-degenerate `upstream`) is the acceptance check named in `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` and still needs to happen post-deploy -- watch for it, don't assume it's done just because the flag is on.

**Operational note (resolved 2026-07-23)**: unlike the other trigger kinds here (rare/periodic), `chat_turn` is designed to fire on essentially every remarkable chat turn. Sharing `EQUILIBRIUM_METACOG_COOLDOWN_SEC`'s single global cooldown timestamp with baseline/manual/pulse/relational/telemetry_anomaly would have let a burst of `chat_turn` fires silently starve those other trigger kinds too, not just drop `chat_turn`'s own excess -- so `chat_turn` now has its own separate cooldown lane (`EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC`, own timestamp, own setting). `_publish_metacog_trigger` still silently drops (logs only, does not queue) anything that fires within *its own kind's* cooldown window -- that part of the behavior is unchanged and intentional.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_CHAT_TURN_TRIGGER_ENABLE` | `true` | Master gate for the chat_turn trigger |
| `EQUILIBRIUM_METACOG_CHAT_TURN_CORRELATOR_TTL_SEC` | `600` | Correlator entry TTL (leak backstop for non-terminal evidence) |
| `EQUILIBRIUM_METACOG_CHAT_TURN_SURPRISE_THRESHOLD` | `0.7` | Minimum `substrate_appraisal.surprise_level` to fire |
| `EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC` | `30` | chat_turn's own cooldown window, separate from `EQUILIBRIUM_METACOG_COOLDOWN_SEC` |
| `CHANNEL_THOUGHT_ARTIFACT` | `orion:thought:artifact` | Source channel (wildcard consumers) |
| `CHANNEL_HARNESS_RUN_ARTIFACT` | `orion:harness:run:artifact` | Source channel (wildcard consumers) |
| `CHANNEL_GRAMMAR_EVENT` | `orion:grammar:event` | Source channel, filtered to `semantic_role in ("exec_turn_timeout", "stance_disposition")` |

### transport metacog trigger

Full design: `docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md`. Two independent evidence sources, both feeding `trigger_kind=transport` (`app/transport_metacog_gate.py`), no correlator needed -- each source fires on its own real evidence directly:

- **(A) `RpcHealthSnapshotV1` windows** on `orion:rpc_health:snapshot` (published every `RPC_HEALTH_PUBLISH_INTERVAL_SEC` by `orion-cortex-exec`/`orion-cortex-orch`, `orion/core/bus/rpc_health_publish.py`, PR #1313/#1315, live-verified). Fires when `timeout_count > 0` (real evidence, no threshold) or `success_latency_ms_p95 >= EQUILIBRIUM_METACOG_TRANSPORT_LATENCY_P95_THRESHOLD_MS` (a starting default, unvalidated). An empty window (no real calls) fires nothing -- absence of traffic isn't evidence of trouble, same rule `orion-signal-gateway`'s `rpc_health` organ adapter already applies.
- **(C) `rpc_transport_timeout` grammar atoms** on `orion:grammar:event` (published by `orion/core/bus/async_service.py::_emit_rpc_timeout_grammar`, fired from both of `rpc_request()`'s real timeout branches -- generalizes `chat_turn`'s own `exec_turn_timeout`/`stance_timeout` markers, scoped to one harness/thought RPC each, to every one of the 37+ real `rpc_request()` call sites sharing that one client). Terminal by construction -- a real RPC already timed out by the time this atom exists, no threshold to evaluate.

Own cooldown lane from day one (`EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC`) -- not sharing the global lane, avoiding the exact bug `chat_turn` had to fix after the fact (see above).

Ships disabled (`EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE=false`). Needs the same live-verification standard as every other trigger in this family (real `orion_metacog` rows, non-degenerate `upstream`) before flipping on, and Option A's latency threshold needs real data before it can be trusted.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE` | `false` | Master gate for the transport trigger |
| `EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC` | `30` | transport's own cooldown window, separate from `EQUILIBRIUM_METACOG_COOLDOWN_SEC` |
| `EQUILIBRIUM_METACOG_TRANSPORT_LATENCY_P95_THRESHOLD_MS` | `5000` | Option A's latency-spike threshold; unvalidated starting default |
| `CHANNEL_RPC_HEALTH_SNAPSHOT` | `orion:rpc_health:snapshot` | Option A's source channel |
| `CHANNEL_GRAMMAR_EVENT` | `orion:grammar:event` | Option C's source channel, filtered to `semantic_role=="rpc_transport_timeout"` |

---

## Quick start (copy/paste)

### Bus URL
```bash
BUS=redis://100.92.216.81:6379/0
```

### Run
```bash
docker compose up -d orion-equillibrium-service
docker logs -f orion-equillibrium-service
```

### Watch health snapshots
```bash
redis-cli -u "$BUS" SUBSCRIBE "orion:equilibrium:snapshot"
```

### Watch baseline Collapse Mirror snapshots (the metacognition tick)
```bash
redis-cli -u "$BUS" SUBSCRIBE "orion:event:equilibrium:snapshot"
```

---

## What it does
### A) Health aggregation
- Tracks expected services
- Computes healthy/degraded/missing over a time window
- Emits `orion:equilibrium:snapshot`

### B) Baseline Collapse Mirror tick (currently embedded here)
- Constructs `CollapseMirrorStateSnapshot` + `CollapseMirrorEntryV2`
- Emits it as a system “self-awareness” baseline snapshot

---

## How to use it (practical)
1) Start equilibrium
2) Start a handful of services
3) Stop one service and watch equilibrium mark it missing
4) If collapse ticks are enabled, confirm `orion:event:equilibrium:snapshot` emits every interval

---

## Architectural note (placement)
Two valid stances:

### Clean
- Equilibrium stays only health.
- A dedicated `baseline-snapshotter` / `state-service` emits Collapse Mirror baseline snapshots.

### Pragmatic / evolved
- Keep baseline tick inside Equilibrium.
- Make it opt-in and quiet by default.

---

## Preferred workflow (no channel memorization)
Use logs first:

```bash
docker logs -f orion-equillibrium-service
```

Future stub: equilibrium should print on startup:
- expected services list
- window + publish interval
- collapse tick interval + output channel

---

## Future stubs we should add
- `GET /healthz`, `/readyz`, `/stats`
- A “disable collapse tick” mode via `EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC=0`
- Standard bus summary logging flags (`ORION_LOG_BUS_IN/OUT`)
- Clear schema separation:
  - `EquilibriumSnapshotV1` for health
  - `CollapseMirrorEntryV2` for metacognition tick

---

## Common failure modes
- Publishing to a channel not registered in Titanium channel catalog (enforcer error)
- Expected services list out of sync with actual compose deploy
- Window/grace tuning too aggressive → false missing
