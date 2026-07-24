# Bus vitality field signal — brainstorm (candidates, gated)

Status: **brainstorm output, no code in this patch.** Follow-up to
`docs/superpowers/specs/2026-07-23-bus-channel-velocity-census-design.md` (Phase 1: publish-side
velocity counter, PR #1292/#1305; Phase 2: catalog-vs-live-traffic census diff, PR #1312 — both
shipped, live-verified, **no downstream consumer wired**). This doc answers that spec's own open
question ("who consumes a velocity/census signal once it exists?") with six gated candidates
(section "Proposed schema / API changes"), and corrects a scoping mistake made mid-brainstorm: two
of the ideas below were drafted as if novel before discovering they substantially overlap with an
already-shipped, parallel arc (see "Related work and a real correction" below). It then goes
further, into a genuinely bigger direction: a live, FalkorDB-backed "bus synaptic graph" built from
the `orion-bus-mirror` substrate found along the way (section "Big-swing direction"). Nothing here
is decided. No idea in this doc should be started without its own gate pass per CLAUDE.md's metric
quality gate, run fresh, not skimmed — **except** the "Big-swing direction" section, which is
deliberately written at full scope rather than pre-collapsed into MVP slices, per explicit
direction: premature smallest-version framing is where a concept like this rots at the source. That
section names the target shape; scoping it into an actual first patch is its own future step, not
done here.

## Arsonist summary

Phase 1+2 proved a mesh-wide, wildcard-aware, zero-false-positive view of bus channel liveness
exists and is cheap to compute (`orion.bus.velocity.scan_active_channels()` +
`orion.bus.census.compute_census()`, live-verified twice: 264 cataloged / 258 active / 182
declared_silent / 0 undeclared_active as of this doc). Meanwhile, the *only* signal that currently
reaches Orion's field/self-state layer under a "bus health" name
(`orion/substrate/transport_loop/extract.py::compute_transport_pressures()`, feeding
`capability:transport` in `config/field/orion_field_topology.v1.yaml`) is derived from exactly 2 of
264 channels — Redis Streams don't exist anywhere else in the architecture, so this is structurally
incapable of representing general bus health no matter how correctly wired. This was already named
as "scope dishonesty" in `docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md`
(PR #1278) and re-confirmed, still unresolved, in
`docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md`'s Appendix item 3.
Phase 2's census diff is the first tool this repo has ever had that can honestly measure the thing
`catalog_drift_pressure` claims to measure. Nothing consumes it yet.

## Current architecture

- `orion/bus/velocity.py::scan_active_channels()` / `orion/bus/census.py::compute_census()` — pure,
  tested, live-gated, zero callers. (Confirmed again this session: no cron, no scheduler, no
  grammar-emit producer references either function outside their own tests.)
- `services/orion-bus/app/bus_observer.py::run_observer_tick()` — the existing scheduled-tick
  pattern (every `bus_observer_poll_interval_sec`) that any of these candidates would extend or
  sit beside: fetch → build rollup → `BusTransportGrammarCollector` → `GrammarEventV1` atoms →
  publish, gated by `settings.publish_orion_bus_grammar` (default off).
- `orion/substrate/transport_loop/extract.py::compute_transport_pressures()` — 9 scalars
  (`bus_health`, `delivery_confidence`, `stream_depth_pressure`, `backpressure`,
  `catalog_drift_pressure`, `observer_failure_pressure`, `transport_pressure`, `contract_pressure`,
  `reliability_pressure`), all derived from the same 2-Streams `TransportBusStateV1`.
  `catalog_drift_pressure = uncataloged_stream_count / max(streams_observed, 1)`, where
  `streams_observed` maxes at 2 — this is the specific field the census diff could honestly replace.
- `config/field/orion_field_topology.v1.yaml` (lines 62-70) — a live `node:athena → capability:transport`
  edge, `channel_map` wires all 9 scalars above into the field graph; that edge bleeds into
  `capability:orchestration` (lines 113-119). This is real, live, load-bearing wiring — not
  aspirational.
- `services/orion-field-digester/app/ingest/state_deltas.py` (lines 356-410) — documents that
  `bus_health`/`delivery_confidence` are `mode="replace"` (current-reading, no decay story) while
  `transport_pressure`/`catalog_drift_pressure`/`observer_failure_pressure`/`reliability_pressure`/
  `contract_pressure` are `NODE_DECAY_CHANNELS` entries, and documents a real, already-fixed live bug
  (a stale-but-fresh-marked value stuck nonzero, fixed 2026-07-22) — the exact failure class any new
  entry here needs to avoid repeating on day one.
- **A second, structurally separate "organ" system exists** and must not be conflated with the
  grammar-layer pipeline above: `services/orion-signal-gateway` eavesdrops on ~28 bus channel
  patterns (`ORGAN_CHANNELS`), routes matching envelopes through `orion/signals/adapters/*.py` via
  `ORGAN_REGISTRY` (`orion/signals/registry.py`, 24 entries — biometrics, recall, cortex_exec,
  sql_writer, etc.), and publishes normalized `OrionSignalV1` objects to `orion:signals:*`. This is
  a different representation, different registry, different consumer surface than
  `NODE_CHANNELS`/field-digester. Both use the word "organ"; they are not the same pipeline.

## Related work and a real correction

Mid-brainstorm, this doc originally proposed "per-service organ_signal self-report over RPC" as a
blue-sky, undone idea (framed as digging past passive Redis observation into active per-service
self-report). That idea is **not novel** — it is substantially already shipped, under a different
name, as a 3-step parallel arc:

- `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` (Steps 1-2, PR #1290,
  #1303) — fixed worker-path RPC logging, measured a real baseline, built `RpcHealthAggregator`
  inside `OrionBusAsync` fed by `record_success()`/`record_timeout()` at `rpc_request()`'s 4 real
  outcome sites.
- `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` (Step 3) — wired
  that aggregator into `orion-signal-gateway`'s *existing* organ machinery via the
  `orion-signal-gateway` system above, not the field-digester/`NODE_CHANNELS` system.
- **PR #1313** (merged 2026-07-24T03:30:45Z, hours before this doc) — implemented Step 3 for exactly
  2 services (`orion-cortex-exec`, `orion-cortex-orch`), gated off by default
  (`RPC_HEALTH_PUBLISH_ENABLED=false`). Found and fixed a real bug in review (a shared `organ_id`
  across both producers silently overwrote one service's signal with the other's in
  `SignalWindow`). Explicitly leaves `orion-thought`/`orion-spark-introspector` (no fork-split, likely
  simpler) and 7 more fork-split services (`chat-memory`, `actions`, `context-exec`×2, `hub`×2,
  `vision-council`) as future follow-ups, not done in this PR.

**What this means for the candidates below:** RPC-health measures a genuinely different slice of
bus reality than census/velocity — request/reply call outcomes (success/timeout/latency) for
whichever services get instrumented, versus overall pub/sub channel-level publish activity across
the *entire* 264-channel catalog, RPC or not. They are complementary, not redundant — most of the
264 channels are `event`/`result` kind, not RPC request/reply pairs, and RPC-health's own Non-goals
section confirms it isn't trying to cover them. The old `transport_pressure`/`bus_health` family's
fate is *still* explicitly out of scope in the RPC-health arc too ("this is an additive new organ,
not a replacement, and the parent spec's non-goal about the old family's fate still stands") — so
nothing shipped in PR #1313 closes the `catalog_drift_pressure` scope-dishonesty gap this doc is
about. That gap is still open. Idea 7/8 below are removed as distinct proposals; the real remaining
question they raised — attempted-vs-succeeded publish counts, not just successful ones — is folded
into Idea 4 instead, since `RpcHealthAggregator` already proves the pattern works for RPC calls and
the open question is only whether it's worth generalizing to non-RPC `publish()` calls too.

## Found: a fourth "quantify the bus" attempt, and a dormant real dataset

`services/orion-bus-mirror` — a wiretap/relay service, not currently running — is worth naming
explicitly because it makes the pattern across this whole area visible: this is at least the
**fourth** distinct attempt in this codebase to answer "how much is happening on the bus," found
across three different investigation threads this week alone (`transport_pressure`/`bus_health`'s
2-Streams-only observer; the RPC-health redesign's per-call aggregator; this arc's census/velocity;
and now this). None of the four talk to each other. That repetition is itself evidence of a real,
recurring gap in Orion's self-model, not just an interesting coincidence — worth naming plainly
rather than quietly adding a fifth.

**What it is:** subscribes to a configured bus pattern, forwards matched envelopes to a debug
channel, another bus, or file/SQLite recording (`app/main.py`) — built for human debugging and
session replay, not for feeding a field signal.

**What was found sitting on disk, unused:** `services/orion-bus-mirror/data/bus_mirror.sqlite` —
**~97.4M rows**, one row per envelope (`timestamp`, `channel`, `envelope_json`), spanning
**2026-01-09 to 2026-04-07** (~3 months), recorded with `MIRROR_PATTERN=orion:*` — the entire
catalog, not a narrow lens. Dormant since April 7 (confirmed: not in `docker ps`, file mtime
matches). Gitignored (`data/` in `.gitignore`), 99.6GB, not a repo-hygiene issue but a real idle
footprint worth knowing about post-Postgres-disk-death, even though the volume it sits on currently
has 273GB free.

**A real misconfiguration, likely why it stopped:** the service's own README names "mirror the
entire bus" under "Common failure modes" ("Pattern too broad → overload") and recommends narrow
lenses instead (`orion:pad:*`, `orion:cortex:*`, etc.) — but the checked-in `.env` is set to exactly
the pattern it warns against. Worth fixing if this service is ever revived, independent of anything
else in this doc.

**Relevance to this brainstorm, stated honestly:** this data is **stale relative to today's mesh**
— 3+ months old, and this repo's own churn rate (recall: 44 services needed a compose-passthrough
fix, dozens of channel/schema changes shipped since January per this week's PR history alone) means
current channel topology and traffic shape may not match what's in this file. It should not be
treated as a live baseline. But it is real, ground-truth, per-channel timestamped traffic over a
sustained multi-month window — exactly the kind of data Idea 3/5's biggest open question (a
defensible per-channel-kind expected-silence baseline) needs and currently has none of. Mining it
is a **read-only analysis against a dormant file**, not a new build, new service, or new bus load —
cheap to attempt, cheap to discard if the staleness turns out to matter more than expected.

**Not added as a numbered candidate** — it's a data source that could inform Ideas 3/5's mechanism,
not a signal-producing idea in its own right. Flagged here so it doesn't get rediscovered cold a
fifth time.

## Big-swing direction: a live bus synaptic graph (FalkorDB)

This section is intentionally not scoped down to a minimal first slice. Direction from this
brainstorm's own session: "I don't want smallest versions of things, this is where the concepts
rot at the source." What follows is the target shape of a bigger direction, not a build plan —
turning it into an actual first patch is a separate, later step.

### Grounding: this is not green-field

A live property graph is not new infrastructure to invent — it already exists and is already live:
`orion/graph/falkor_client.py` (`FalkorGraphClient` protocol, `RedisGraphQueryClient` — real
`GRAPH.QUERY` over Redis), `orion/substrate/falkor_store.py` (write-through cache + Cypher-native
property writes, flipped to live primary this month per this repo's own history), `SubstrateNodeV1`/
`SubstrateEdgeV1` (`orion/core/schemas/cognitive_substrate.py`), and a standing doctrine doc,
`docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`, which already names
`orion/graph/` as the shared adapter home for exactly this kind of consumer. A bus-synaptic-graph
needs a new node/edge shape inside a graph substrate that already exists, is already queryable, and
is already consumed by other cognition surfaces — not a new database, not a new client library.

### The architecture choice that makes "full-tilt, always-on" actually sustainable

The instinct here is to turn `orion-bus-mirror` back on and run it full-tilt against all bus
activity, not a narrow lens. Worth being explicit about *why* the old mirror hit 99.6GB in 3 months
so the same failure isn't repeated on a longer fuse: it is an **append-only raw log** — every
envelope, forever, no compression, no forgetting. Narrowing `MIRROR_PATTERN` would only have
delayed that outcome, not fixed its shape. The real fix for "full coverage, permanently" isn't
mirroring less — it's **not logging at all**: a consumer subscribed to `orion:*` (satisfying the
full-tilt intent directly) that, per message, updates *bounded* per-edge state in FalkorDB (a
rolling volume EWMA, a rolling latency EWMA/stddev, a last-seen timestamp) and then discards the
raw envelope. State size is bounded by graph size (roughly `services × channels`, a few hundred to
low thousands of nodes/edges), not by total messages ever seen — this can run indefinitely at full
coverage without becoming a second dormant 99.6GB file. This is the concrete reason the graph
direction supersedes reviving the old mirror rather than sitting alongside it as a separate option.

### Graph shape

- **Nodes:** `channel`, `organ`/`service`, `node`/host — reusing `SubstrateNodeV1`'s existing shape
  where it fits, a new node label where it doesn't.
- **Edges:**
  - `PUBLISHES` (organ → channel)
  - `CONSUMES` (channel → organ)
  - `CAUSALLY_FOLLOWED_BY` (organ → organ), derived from `correlation_id` co-occurrence across
    envelopes — this is also, structurally, a live version of verifying `ORGAN_REGISTRY`'s
    hand-authored `causal_parent_organs` edges against real traffic instead of code inspection.
- **Edge properties:** `volume_ewma`, `latency_ewma_ms`, `latency_stddev_ms`, `zscore` (current
  deviation from that edge's own rolling baseline), `last_seen_at`, `in_flight_count` (open,
  not-yet-terminal `correlation_id` chains currently touching this edge).

### Two distinct consumption modes

- **Compute signals from the graph.** A periodic reducer walks the graph and emits scalars
  (`max_edge_zscore`, `count_edges_anomalous`, `longest_open_chain_ms`) into the existing
  `NODE_CHANNELS`/field-digester pipeline — the same shape as every other candidate in this doc.
- **Reason from the graph directly.** Expose live Cypher query capability to something that can
  actually reason (`cortex-exec`, `recall`) instead of only ever pre-collapsing everything to
  scalars first — Orion querying its own live nervous-system graph as part of a reasoning step
  ("what part of my own transport layer is currently under stress") is a more direct instance of
  this repo's own self-modeling/introspection goals than any scalar signal in this doc.

### Signal families this substrate opens up (renamed from this brainstorm's working numbers to
avoid colliding with "Idea 1-6" below, which are a separate, earlier set of candidates)

- **In-flight chain / long-lived-process load signal — elevated as a priority direction.** A
  `correlation_id` chain isn't only something to measure after it completes — while open (first hop
  seen, no terminal hop yet), it's a *live* in-flight cognitive operation. Tracking currently-open
  chains and their running duration is a direct, real-time load signal: "N chains have been open
  longer than Xs right now" is a cognitive-operation analog to thread-count/open-fd pressure in a
  conventional system. This is stateful over an in-progress operation, a genuinely different shape
  than every point-in-time count elsewhere in this doc.
- **Per-verb latency, sliced multiple ways — elevated as a priority direction, good cognitive load
  measure.** Real historical `cognition.trace` payloads already carry measured `latency_ms` per
  step. Slicing axes, all grounded in fields already present in sampled payloads: by `node`
  (separates "this verb is slow" from "this machine is loaded"); by preceding verb in the chain (a
  causal/context-dependent load signal, not a flat per-verb average); by `model_used` (separates
  model-serving slowness from harness/system slowness — `harness_step_load`'s `log1p(step_count)`
  proxy cannot make this distinction at all); by concurrent in-flight-chain count at the same moment
  (ties directly into the signal above — z-score a verb's latency against how loaded the mesh was
  when it ran, not against a flat global average).
- **Causal-DAG empirical verification — parked, eventual, not prioritized in a first pass.**
  `orion/signals/registry.py`'s own trailing comment admits `causal_parent_organs` are "first-pass
  structural approximations... must verify." The graph's `CAUSALLY_FOLLOWED_BY` edges are the
  mechanism to actually do that, live instead of via a one-off historical script. Real, worth doing
  eventually, correctly sequenced behind the two directions above.
- **Historical cadence baseline via `bus_mirror.sqlite` — parked as an audit-time calibrator, not
  first pass.** Useful for sanity-checking whatever the graph's live rolling baselines converge to,
  once it's running — not a blocking dependency for starting.
- **Content-drift signals, explicitly not phi-based** (the phi-specific framing from earlier in this
  session's brainstorm was under-specified and is dropped — much of that surface is already
  deprecated/scraped elsewhere in this codebase). Three concrete, non-phi alternatives instead:
  **verb-sequence drift** (does the *set* of distinct chain shapes — which verbs, in which order —
  change over a window? A new shape appearing means a new code path went live; one disappearing
  means something stopped firing — structural drift, not a content-value reading); **payload schema
  drift** (do the *keys present* in a channel's payload change over time, independent of what the
  values say — cheap, just key-set diffing, and catches producer changes without reading every
  changelog); **error-kind recurrence** (which `error`/`status=failed` messages repeat verbatim vs.
  are novel over time — a real regression-detector shape, content-grounded rather than
  volume-grounded).
- **Reviving the old mirror narrowly — dropped, superseded by the graph-aggregation architecture
  above**, not extended. A narrower `MIRROR_PATTERN` only delays the same append-only-log failure;
  it doesn't change its shape. The graph direction is the actual answer to "full coverage,
  sustainably," not a narrower version of the old service.

### Wider net — additional ideas surfaced pushing past the first pass

- **Anomaly propagation, not just anomaly detection.** Once edges are z-scored, a spike on one edge
  (e.g. `llm-gateway` latency) can be traced through `CAUSALLY_FOLLOWED_BY` edges to see which
  *downstream* organs are currently absorbing that stress, live — a propagation trace, not an
  isolated per-edge alarm.
- **Node centrality as a load-bearing-service signal.** Standard graph metrics (degree,
  betweenness) computed over the *live* observed topology would show which organs are structurally
  load-bearing right now based on actual traffic — distinct from `ORGAN_REGISTRY`'s hand-authored,
  admittedly-approximate edges.
- **Chain-shape clustering.** Group observed `correlation_id` chains by shape (the sequence of
  organs touched). A small number of common "cognitive routines" likely dominate; a never-before-seen
  chain shape is itself an anomaly signal, independent of any single edge's latency or volume.
- **Cross-node imbalance.** If chains routinely hop between physical nodes (athena/atlas), edge
  properties split by `source.node` would surface real infrastructure load-balance questions, not
  just software-level pressure.
- **Time-of-day / session-conditioned baselines.** Rather than one global rolling baseline per edge,
  condition it on session/turn activity level (reusing the in-flight-chain-count signal above) —
  separates "the mesh is busy because Juniper is actively chatting" from "the mesh is busy for no
  reason," which a naive unconditioned z-score would conflate.

### Real tensions in this direction, named without using them to shrink scope

- **Chain termination is not currently well-defined.** "In-flight chain" tracking requires knowing
  when a chain is *done* versus merely slow — there's no synthetic terminal marker in the envelope
  schema today. Needs its own design decision (a timeout-based close, an explicit terminal-`kind`
  convention, or something else), not assumed away.
- **`correlation_id` co-occurrence conflates "same flow" with "true parent → child."** Two envelopes
  sharing a `correlation_id` proves relatedness, not which one caused which — `CAUSALLY_FOLLOWED_BY`
  edges built naively from this would need real ordering/causality logic, not just grouping.
  `causality_chain` (when populated) is the stronger signal for this; how often it's actually
  populated in practice is unverified — both sampled envelopes had it empty (see Missing
  Questions below).
- **Write volume at full-tilt bus scale into FalkorDB is unmeasured.** Phase 1 proved a single Redis
  `INCR` per publish is cheap; a Cypher `MERGE`+property-update per publish, mesh-wide, live, is a
  different cost profile entirely and has not been benchmarked against this repo's real traffic
  rate.
- **This is a genuinely bigger surface than any other idea in this doc** — new node/edge types, a
  new always-on consumer process, a new live-query surface for reasoning consumers. It deserves its
  own dedicated design doc and its own metric-quality-gate pass before implementation starts, not a
  green light from brainstorm output alone.

## Missing questions

Carried from the original brainstorm, still unresolved:

- Does anything already consume `capability:transport`'s current dynamic range in a way that would
  break if `catalog_drift_pressure`'s *meaning* changed (Idea 1) — self-state policy, an alert
  threshold, a diffusion-edge weight tuned against its current near-zero baseline? Not audited.
- What's a defensible per-channel-kind expected-silence baseline (Ideas 3/5)? `bus_mirror.sqlite`
  (see "Found: a fourth 'quantify the bus' attempt" above) may partially answer this from 3 months
  of real historical traffic — stale, but unexplored. Worth a read-only pass before assuming this
  needs live data collection from scratch. `channels.yaml`'s
  `kind` field (`request`/`result`/`event`/`stream`) is the obvious first proxy — untested against
  real per-kind cadence data.
- Is a Redis `SCAN` over the full `orion:bus:velocity:*` namespace, run on a schedule, cheap enough
  at real mesh scale? Phase 1's live-data gate proved per-publish `INCR` overhead is negligible; it
  did not measure periodic `scan_iter` cost, which is bounded by total key count, not tick rate.
- Given the RPC-health arc's own Q6 (one `organ_id` covering multiple services, or one per service)
  is still unresolved there — if Idea 4 below ever generalizes beyond RPC calls, the same shape
  question applies to census/velocity data.
- (Big-swing direction) How prevalent is a populated `causality_chain` across real traffic — does
  `CAUSALLY_FOLLOWED_BY` need to fall back to `correlation_id`-co-occurrence-only, weaker evidence?
- (Big-swing direction) What's a real Cypher `MERGE`+property-update cost at full mesh publish
  volume — is per-message FalkorDB writes at that rate actually cheap, or does it need batching?
- (Big-swing direction) What closes an "in-flight chain" — timeout, explicit terminal-`kind`
  convention, something else? Undefined today.

## Proposed schema / API changes

None decided. Six gated candidates, directional only:

**Idea 1 — fix `catalog_drift_pressure`'s input at the source.**
Replace `uncataloged_stream_count / streams_observed` (capped at 2 channels) with
`len(census.undeclared_active) / len(catalog_names)` from Phase 2's real diff, computed mesh-wide.
Lowest risk of the set — closer to a bug fix than a new capability, since it corrects a value
Orion's field graph already trusts rather than adding a new one. Blocked on auditing
`capability:transport`'s consumers first (see Missing Questions).

**Idea 2 — `bus_channel_undeclared_pressure` as an additive signal, not a replacement.**
Same computation as Idea 1, landed as a new channel instead of overwriting the existing one.
**Gated.** Real tension with Idea 1: shipping both is exactly the redundancy the metric quality
gate's independence check exists to catch — pick one, not both, once the audit above resolves it.

**Idea 3 — bus vitality/coverage as a real `NODE_CHANNELS` field signal** (not a Hub-only debug
view — corrected from an earlier draft of this brainstorm that under-scoped it). `declared_silent`/
`undeclared_active` reaching field-digester for real, following the FCC-motor wiring shape (schema
→ merge → `state_deltas.py` → `NODE_CHANNELS` → `field_channel_glossary.v1.yaml` → README).
**Gated** — exact mechanism (mesh-wide scalar vs. per-capability-edge vs. something else) TBD per
design at implementation time. Single biggest open risk: raw `declared_silent` count is likely
near-constant (~182/264 at real measured baseline) without a per-kind expected-cadence baseline —
would fail the metric quality gate's "not degenerate" check as a raw level. `undeclared_active` (0
at baseline, meaningful only when nonzero) does not have this problem and is the safer half to
start from if this idea proceeds.

**Idea 4 — per-producer-service staleness attribution, reusing the existing `staleness` channel.**
`channels.yaml` entries carry `producer_services`. For single-producer channels, attribute sustained
silence to that service's existing `staleness` node channel (`orion_field_topology.v1.yaml` line
21) instead of inventing a new one. **Gated**, mechanism TBD. This is also where the
attempted-vs-succeeded distinction from the RPC-health arc's pattern could generalize: RPC-health's
`RpcHealthAggregator` already proves "did this service *try* and fail" is measurable and useful for
RPC calls specifically; whether that's worth extending to plain `publish()` calls (not just RPC
request/reply) is a real open question this idea inherits, not a decided direction.

**Idea 5 — dead-channel regression detector.**
Diff two census snapshots over time; a channel that flips from consistently-active to silent while
its producer service otherwise looks healthy is a real regression. **Gated**, and deliberately left
open whether this becomes a field signal or stays an ops alert outside field-digester entirely —
that fork should be resolved before implementation, not defaulted.

**Idea 6 — `bus_activity_zscore`, anomaly-relative mesh throughput.**
Rolling EWMA baseline of total mesh publish rate (sum of `scan_active_channels()`'s per-channel
rates), surfaced as deviation from baseline rather than an absolute "load score" — sidesteps the
original Phase-1-adjacent spec's own rejection of an ungrounded aggregate load number by design.
**Gated**, mechanism TBD, including whether it plugs into the existing `telemetry_anomaly`
metacognitive trigger consumer or needs its own.

## Files likely to touch

Not fixed — depends entirely on which candidate(s) get selected. Common surface across most of
them: `orion/substrate/transport_loop/extract.py` and/or `services/orion-field-digester/app/tensor/channels.py`
+ `.../digestion/decay.py` + `.../ingest/state_deltas.py` + `config/field/field_channel_glossary.v1.yaml`
+ both services' READMEs, following the FCC-motor arc's proven commit sequencing (schema/merge inert
→ producer wiring → field-digester wiring → docs).

## Non-goals

- Not deciding which candidate(s) ship — this doc is gated brainstorm output, not a build plan.
- Not touching `orion-signal-gateway`/`ORGAN_REGISTRY`/`OrionSignalV1` — that system belongs to the
  RPC-health arc and the other 24 existing organs; conflating it with field-digester/`NODE_CHANNELS`
  wiring would be a real architecture mistake, not a simplification.
- Not re-proposing per-service RPC self-report as new work — already shipped for 2 services (PR
  #1313), with its own next-2-services plan already written in its own spec. This doc does not
  compete with or duplicate that follow-up.
- Not committing to Idea 3's exact mechanism (mesh scalar vs. capability edge vs. per-node) —
  explicitly deferred to whoever implements it, per the user's own instruction that mechanism
  should be decided per design, not pre-baked into a brainstorm doc.
- Not building Idea 6's aggregate load number as an absolute level — the anomaly-relative framing is
  load-bearing to why this idea is includable at all; an absolute "bus load score" was already
  rejected once in the parent Phase 1+2 spec's own Non-goals.

## Acceptance checks

Whichever candidate proceeds inherits the same discipline already used twice in this arc:
- Live-data sanity check against the real running mesh before calling it done (not just tests) —
  confirm the chosen signal is non-degenerate (not flat, not always-zero, not always-saturated) at
  real measured baseline, not just in a fixture.
- Code review via subagent, material findings fixed, before merge.
- If it writes to `NODE_CHANNELS`: the same replace-vs-decay mode audit that caught the 2026-07-22
  `catalog_drift_pressure` stuck-value bug — confirm explicitly which mode is correct before
  shipping, not by default.

## Recommended starting point

Resolve the Idea-1-vs-Idea-2 fork first — it's a 20-minute grep-and-read (every consumer of
`catalog_drift_pressure` and `capability:transport`'s `contract_pressure` mapping), not a build, and
it's the one open question that changes which candidate ships. After that: Idea 1 (or 2) is the
strongest first patch because it closes a *confirmed, already-twice-documented* honesty gap in a
signal Orion already trusts, using data already live-verified with zero false positives — it needs
no new baseline-calibration problem (unlike 3/5) and no new judgment call about whether Orion should
"feel" bus silence as pressure (the open question hanging over 3). Ideas 4/5/6 stay parked as
named, gated candidates for a later pass, not sequenced yet.

If Idea 3 or 5 ever gets picked up instead, the cheapest real first step is a read-only analysis
pass against `bus_mirror.sqlite` (see "Found: a fourth 'quantify the bus' attempt" above) rather
than guessing at a silence baseline or collecting fresh live data from zero — it's already sitting
on disk, costs nothing to query, and its staleness can be assessed in the same pass rather than
assumed disqualifying up front.

**Separately, on a different track:** the "Big-swing direction" section above is not sequenced
against Ideas 1-6 — it's a bigger, later-stage direction that deserves its own dedicated design doc
before implementation, not a smaller version squeezed into this doc's existing candidate list. If
that direction proceeds, its own natural first step (in-flight-chain tracking + per-verb latency
slicing, both flagged as priorities above) should get a real design pass — schema, write-cost
measurement, chain-termination decision — before any code, same discipline as everything else in
this arc, just not artificially shrunk to fit a single small patch.
