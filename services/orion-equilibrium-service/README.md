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
