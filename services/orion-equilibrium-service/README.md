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
