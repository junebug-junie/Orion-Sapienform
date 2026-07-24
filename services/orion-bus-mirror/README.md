# orion-bus-mirror

A “wiretap + relay” for the Orion Titanium bus. Use it to copy a slice of the bus into a debug stream/namespace and/or record it for replay.

---

## Status: disabled, ran into exactly the failure mode this README warns about

Live incident, 2026-07-24: this service was left running with
`MIRROR_PATTERN=orion:*`, which matches all 265 channels registered in
`orion/bus/channels.yaml` -- the entire bus, not the narrow debug slice this
README recommends below. With no retention/rotation cap, the recording grew
to **98GB** before anyone noticed, on the `/mnt/scripts` disk (shared with
the codebase and all worktrees).

Current state:
- The container is stopped and excluded from `mesh-utilities/common/
  exclude_services.txt`'s auto-rebuild list, so it will not restart on its
  own.
- The 98GB recording was archived (not deleted) to
  `/mnt/postgres/dumps/bus-mirror/bus_mirror.sqlite`.
- `MIRROR_DATA_DIR` now points at `/mnt/postgres/bus-mirror/data` (moved off
  the `/mnt/scripts` disk entirely) -- if this service is ever re-enabled,
  it starts fresh and empty there, not resuming the archived recording.

**If you re-enable this service**, use a narrow pattern (see "Recommended
patterns" below) and set a real retention/rotation limit -- there still
isn't a `MIRROR_SAMPLE_RATE` or automatic rotation built in (see "Future
stubs we should add"), so an unbounded pattern will recreate this exact
failure.

---

## Quick start (copy/paste)

### Set your bus URL once
```bash
BUS=redis://100.92.216.81:6379/0
```

### Run the service
```bash
docker compose up -d orion-bus-mirror
docker logs -f orion-bus-mirror
```

### Prove it’s mirroring
Depending on configuration, look for one of these:

```bash
redis-cli -u "$BUS" PSUBSCRIBE "orion:mirror:*"
# or (single aggregated debug stream)
redis-cli -u "$BUS" SUBSCRIBE "orion:debug:mirror"
```

---

## What it does
- Subscribes to configured patterns (e.g., `orion:pad:*`, `orion:spark:*`)
- Forwards messages to a mirror destination
  - same bus (debug channels)
  - another bus
  - files (recording)

This is how you capture “what actually happened” without changing upstream services.

---

## How to use it (practically)

### Use case A — Debug stream for humans
1) Configure a narrow pattern set (avoid `orion:*`)
2) Mirror into `orion:debug:mirror`
3) Watch that one channel

### Use case B — Capture a session for replay
1) Mirror into file recording
2) Re-run a downstream service against recorded envelopes (future stub)

---

## Recommended patterns (do NOT mirror the entire bus)
Use a lens tied to what you’re debugging:

- Health lens: `orion:equilibrium:*`, `orion:event:equilibrium:*`
- Perception lens: `orion:pad:*`, `orion:vision:*`
- Cognition lens: `orion:cortex:*`, `orion:spark:*`

---

## Preferred workflow (no channel memorization)
Use container logs as the first-line tool:

```bash
docker compose up -d orion-bus-mirror
docker logs -f orion-bus-mirror
```

The mirror should log:
- subscribed patterns
- mirror target
- message counts/rates

---

## Future stubs we should add
- `GET /healthz`, `/readyz`, `/stats`
- `MIRROR_SAMPLE_RATE` for high-volume streams
- A “replay mode” that reads recorded envelopes and republishes them at controlled speed
- Standard bus summary logging flags (`ORION_LOG_BUS_IN/OUT`)

---

## Common failure modes
- Pattern too broad → overload
- Mirror publishes to channels not in the Titanium channel catalog → enforcement error
- Destination bus unreachable → retry/backoff needed
