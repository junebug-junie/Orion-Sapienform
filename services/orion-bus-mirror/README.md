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

## Bus synaptic graph (Phase 1)

An additive, independently-gated capability alongside the SQLite raw-log path above:
`MIRROR_GRAPH_ENABLED=true` writes bounded, aggregated edges into FalkorDB instead of
an ever-growing log. This is the architecture that makes running against the full
`MIRROR_PATTERN=orion:*` firehose actually sustainable — state size is bounded by mesh
topology (organs × channels), not by message count, so it does not repeat the 98GB
failure described above. Off by default; the two paths (SQLite, graph) are controlled
by separate settings and can run independently or together.

Design doc: `docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md`,
"Big-swing direction: a live bus synaptic graph (FalkorDB)". This is Phase 1 of that
direction only — the foundation, not the full scope described there.

**Graph shape** (graph name `FALKORDB_BUS_GRAPH`, default `orion_bus_synapse`, on the
shared FalkorDB instance at `FALKORDB_URI`):
- `(:Organ {organ_id})-[:PUBLISHES {count, last_seen_epoch, gap_ewma_sec, gap_var, gap_zscore}]->(:Channel {channel})`
  — every publish this wiretap observes. `gap_ewma_sec`/`gap_var` are a rolling EWMA
  baseline of the inter-arrival gap; `gap_zscore` is how surprising the *most recent*
  gap was against that baseline.
- `(:Organ)-[:CAUSALLY_FOLLOWED_BY {count, last_seen_epoch, latency_ewma_sec, latency_var, latency_zscore}]->(:Organ)`
  — derived when a second envelope carrying a `correlation_id` already seen from a
  *different* organ arrives within `MIRROR_GRAPH_CHAIN_TTL_SEC` (default 120s). Same
  EWMA/z-score shape, over hop latency instead of publish gap.

**What this deliberately does NOT build** (honesty over completeness):
- **No `CONSUMES` edge.** A passive wiretap only ever observes publishes — it cannot
  see who actually received a pub/sub message. A live `CONSUMES` edge would need each
  consumer to self-report, the same pattern the RPC-health arc already built for 2
  services (PR #1313) — not something this service can infer by eavesdropping. Do not
  fake this edge from `channels.yaml`'s static `consumer_services` metadata; that is
  config truth, not runtime truth.
- **No in-flight/still-open chain tracking.** `CAUSALLY_FOLLOWED_BY` only records a hop
  once the *second* envelope of a pair arrives — it says nothing about a chain that's
  still running right now. That's Phase 2.
- **No per-verb latency slicing from payload content**, no anomaly propagation across
  edges, no node centrality, no chain-shape clustering. All named as Phase 2+ in the
  design doc, not built here.

**A real cold-start caveat**: an edge's first `gap_zscore`/`latency_zscore` is always
`null` (no baseline exists yet to be anomalous against — see `compute_ewma_update`'s
docstring in `app/graph_writer.py`), and the *second* observation on a fresh edge can
read as an extreme z-score (the variance floor is tiny until a few real observations
have accumulated). Don't trust a z-score on an edge with `count < ~5` yet.

**Known gaps** (found in review, acceptable for a Phase 1 foundation, not fixed here):
- **Single-instance only.** `BusSynapticGraphWriter`'s read-then-write pattern (read
  prior EWMA state, compute the update, write it) has no CAS/version check. Running
  more than one `orion-bus-mirror` container against the same graph concurrently would
  race and corrupt edge state. Nothing in `docker-compose.yml` enforces this today.
- **No write-health visibility.** Graph-write failures (e.g. FalkorDB unreachable) are
  caught and logged at `warning`, never surfaced anywhere else — an operator flipping
  `MIRROR_GRAPH_ENABLED=true` against a broken config has no signal short of grepping
  logs or querying the graph directly. A consecutive-failure counter feeding `/stats`
  (itself still a stub) is the natural follow-up, not built here.

**Query it directly** (once `MIRROR_GRAPH_ENABLED=true` has been running a while):
```cypher
MATCH (o:Organ)-[e:PUBLISHES]->(c:Channel)
WHERE abs(e.gap_zscore) > 3 AND e.count > 5
RETURN o.organ_id, c.channel, e.gap_zscore, e.count
ORDER BY abs(e.gap_zscore) DESC
```

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
- `GET /healthz`, `/readyz`, `/stats` — `/stats` is now partially answerable by querying
  the bus synaptic graph directly (see above) once `MIRROR_GRAPH_ENABLED=true`; an HTTP
  surface over that query is still a stub.
- `MIRROR_SAMPLE_RATE` for high-volume streams (still relevant to the SQLite path)
- A “replay mode” that reads recorded envelopes and republishes them at controlled speed
- Standard bus summary logging flags (`ORION_LOG_BUS_IN/OUT`)

---

## Common failure modes
- Pattern too broad → overload
- Mirror publishes to channels not in the Titanium channel catalog → enforcement error
- Destination bus unreachable → retry/backoff needed
