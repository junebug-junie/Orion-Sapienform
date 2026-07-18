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

When `EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE=true`, equilibrium also subscribes to `orion:chat:history:spark_meta:patch` (published by `orion-memory-consolidation`) and reads its `turn_change_appraisal` — the live `orion/memory/turn_change_classify.py` SHIFT appraisal (NONE/TOPIC/STANCE/REPAIR), already theory-anchored (REPAIR maps to rupture-and-repair; Safran & Muran) and already scored via real per-token logprob confidence. `REPAIR` or `TOPIC` above the confidence floor fires `trigger_kind=relational`; `STANCE` is intentionally excluded for now (see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s "Still open" section — undecided, not forgotten). The resulting entry's `causal_density` also blends in this trigger's `novelty_score`/`confidence` as evidence (see `orion/collapse/service.py::_relational_evidence_score`), reusing the appraisal's own severity signal instead of a new metric.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE` | `true` | Master gate for the relational trigger |
| `EQUILIBRIUM_METACOG_RELATIONAL_CONFIDENCE_THRESHOLD` | `0.7` | Minimum appraisal confidence to fire |
| `CHANNEL_CHAT_HISTORY_SPARK_META_PATCH` | `orion:chat:history:spark_meta:patch` | Source channel (fan-out; also consumed by `orion-sql-writer`, `orion-spark-introspector`) |

Live-data check (2026-07-18, 7-day window, `chat_history_log.spark_meta`): 74 real appraisals — TOPIC 33 (avg confidence 0.87), NONE 24, STANCE 17, REPAIR 0. Zero REPAIR events in this window; TOPIC alone is enough to validate the wiring, but REPAIR's real firing rate is still unverified live.

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
