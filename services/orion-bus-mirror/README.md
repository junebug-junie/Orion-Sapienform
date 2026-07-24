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
failure described above. `MIRROR_GRAPH_ENABLED` defaults to `true` in `.env_example`;
the two paths (SQLite, graph) are controlled by separate settings and can run
independently or together. Note this service itself is currently stopped (see
"Status" above) — enabling the graph writer here does not restart the container.

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

**Channel node collapsing (fixed 2026-07-24, found via the Hub hot-path view):** `Channel`
node identity is normalized against `orion/bus/channels.yaml`'s wildcard catalog entries
(`orion.bus.census.normalize_channel_name`, the same wildcard-prefix matching
`compute_census()` already used for Phase 2's census diff) before being written. Without
this, every dynamic per-request reply channel (e.g.
`orion:exec:result:LLMGatewayService:<uuid>`) got its own permanent node — live-found:
~9K `Channel` nodes against a 264-entry declared catalog. Now they collapse to the
matching catalog wildcard entry (narrowest/most-specific match wins when more than one
wildcard matches the same channel; an exact literal catalog entry always short-circuits
wildcard matching, even if a wildcard sibling would also prefix-match it -- otherwise
already-declared, already-bounded channels like `orion:exec:result:LLMGatewayService`
would themselves get incorrectly merged into the broader `orion:exec:result:*` bucket,
exactly backwards -- caught in review). Catalog is loaded once at process startup (static
for the process lifetime), not re-read per message -- a channel added to `channels.yaml`
while this service is running won't be normalized (or a channel removed will keep being
normalized against the stale entry) until the next restart. **Known gap, not fixed here:**
this only affects *new* writes — the ~9K already-inflated `Channel` nodes from before this
fix are not retroactively merged or cleaned up. A one-off migration/cleanup pass is a
real, separate follow-up, not silently assumed unnecessary.

**What this deliberately does NOT build** (honesty over completeness):
- **No `CONSUMES` edge.** A passive wiretap only ever observes publishes — it cannot
  see who actually received a pub/sub message. A live `CONSUMES` edge would need each
  consumer to self-report, the same pattern the RPC-health arc already built for 2
  services (PR #1313) — not something this service can infer by eavesdropping. Do not
  fake this edge from `channels.yaml`'s static `consumer_services` metadata; that is
  config truth, not runtime truth.
- **No anomaly propagation across edges, no node centrality, no chain-shape
  clustering.** Named as Phase 3+ in the design doc, not built here. (In-flight chain
  tracking and per-verb latency slicing *are* now built — see "Phase 2" below.)

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

**Or via Hub's read-only debug API** instead of raw Cypher: `services/orion-hub/scripts/bus_synaptic_graph_routes.py`
(`GET /api/bus-synaptic-graph/summary|hot-organs|hot-edges|anomalies`) — see that service's README
"Bus synaptic graph debug routes" section.

---

## Bus synaptic graph (Phase 2)

Two additions on top of Phase 1's foundation, both gated by the same
`MIRROR_GRAPH_ENABLED`, no new settings required to enable them.

**In-flight / long-running chain tracking.** `ChainTracker` now remembers when each
`correlation_id` was *first* seen, not just its most recent hop. Every
`MIRROR_GRAPH_INFLIGHT_LOG_INTERVAL_SEC` (default 30s), a background task logs a
summary of every chain still being tracked:
```text
bus synaptic graph in-flight chains: open=3 long_running=1 max_duration_sec=47.2
```
A chain counts as "long-running" once its duration exceeds
`MIRROR_GRAPH_LONG_RUNNING_THRESHOLD_SEC` (default 30s). **This is log-only** — not
wired into `NODE_CHANNELS`, the field-digester, or any other consumer yet; the
smallest real step that makes the signal observable before deciding how it should be
consumed further (Phase 3+, not decided).

Real limitation, not solved here: **there is no terminal marker in the envelope
schema.** "In-flight" means "has had activity within `MIRROR_GRAPH_CHAIN_TTL_SEC`,"
not "known to still be genuinely running" — a chain that finishes and one that's
abandoned both just stop appearing once they go quiet past the TTL. Don't read
`open_count` as "N operations currently executing"; read it as "N flows have had
recent activity."

**Fixed post-deploy, found in real traffic (2026-07-24):** `summarize_open_chains`
only counts chains with `real_hop_count >= 1` (a `min_real_hop_count` parameter,
configurable, default `1`). Most bus envelopes carry a fresh, never-repeated
`correlation_id` (`BaseEnvelope`'s `default_factory=uuid4` — only *propagated*
correlation_ids ever produce a second hop), so an unfiltered count was dominated by
one-shot messages sitting in the tracker until TTL eviction. Live before the fix:
~6800 tracked entries, only 51 ever became a real `CAUSALLY_FOLLOWED_BY` edge, yet
~5000 were being logged as `long_running` — schema-valid, completely misleading.
Caught in review: the first cut of this fix filtered on `hop_count` (total
observations), which still credits same-organ repeats that never touch a different
organ — `real_hop_count` counts only genuine cross-organ hops, the kind that produce
a real `CAUSALLY_FOLLOWED_BY` edge. `hop_count` is still reported on `OpenChainInfo`
for visibility, just no longer used as the filter.
Also fixed in the same pass: `ChainTracker`'s internal store is now an `OrderedDict`
with `move_to_end()` on every touch, so eviction pops only the genuinely-stale prefix
instead of scanning the full (thousands-of-entries) table on every single message —
a real, measured contributor to CPU usage at full mesh traffic volume.

**Per-verb latency** (`EXECUTES_VERB` edges, `(:Organ)-[:EXECUTES_VERB {count,
last_seen_epoch, last_node, latency_ewma_sec, latency_var, latency_zscore}]->(:Verb
{verb_name})`): mined from any envelope whose payload has a `steps[]` array with real
measured `latency_ms` per step (the `cognition.trace` shape) — same EWMA/z-score
mechanism as Phase 1's edges, applied per (organ, verb) pair.

Real limitation, not solved here: **`last_node` is captured but does not partition
the baseline.** If the same organ runs the same verb on more than one physical host,
their latencies blend into one EWMA — this does not yet cleanly separate "this verb
is slow" from "this specific machine is loaded," despite that being the original
motivation for wanting a per-node slice. Also not built: slicing by preceding verb in
a chain, by `model_used`, or by concurrent in-flight-chain count at the same moment —
all named as real ideas in the design doc, none implemented.

**Fixed in review, worth knowing about**: unlike Phase 1's facts (derived only from
trusted envelope metadata and wall-clock timestamps), `extract_verb_step_facts` reads
arbitrary nested payload content, so it guards against two things Phase 1 never had
to: `steps[]` is capped at `MIRROR_GRAPH_MAX_VERB_STEPS` (default 200, truncated not
errored) so a buggy/adversarial producer sending a huge array can't stall the mirror's
message loop, and non-finite `latency_ms` (`NaN`/`Infinity`) is rejected rather than
silently accepted — a single NaN observation would otherwise permanently poison that
edge's EWMA baseline with no self-healing (verified live in review).

```cypher
MATCH (o:Organ)-[e:EXECUTES_VERB]->(v:Verb)
WHERE abs(e.latency_zscore) > 3 AND e.count > 5
RETURN o.organ_id, v.verb_name, e.latency_ewma_sec, e.latency_zscore, e.last_node
ORDER BY abs(e.latency_zscore) DESC
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
