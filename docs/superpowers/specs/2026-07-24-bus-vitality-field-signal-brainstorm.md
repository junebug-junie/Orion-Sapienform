# Bus vitality field signal — brainstorm (candidates, gated)

Status: **brainstorm output, no code in this patch.** Follow-up to
`docs/superpowers/specs/2026-07-23-bus-channel-velocity-census-design.md` (Phase 1: publish-side
velocity counter, PR #1292/#1305; Phase 2: catalog-vs-live-traffic census diff, PR #1312 — both
shipped, live-verified, **no downstream consumer wired**). This doc answers that spec's own open
question ("who consumes a velocity/census signal once it exists?") with six gated candidates, and
corrects a scoping mistake made mid-brainstorm: two of the ideas below were drafted as if novel
before discovering they substantially overlap with an already-shipped, parallel arc (see
"Related work and a real correction" below). Nothing here is decided. No idea in this doc should
be started without its own gate pass per CLAUDE.md's metric quality gate, run fresh, not skimmed.

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

## Missing questions

Carried from the original brainstorm, still unresolved:

- Does anything already consume `capability:transport`'s current dynamic range in a way that would
  break if `catalog_drift_pressure`'s *meaning* changed (Idea 1) — self-state policy, an alert
  threshold, a diffusion-edge weight tuned against its current near-zero baseline? Not audited.
- What's a defensible per-channel-kind expected-silence baseline (Ideas 3/5)? `channels.yaml`'s
  `kind` field (`request`/`result`/`event`/`stream`) is the obvious first proxy — untested against
  real per-kind cadence data.
- Is a Redis `SCAN` over the full `orion:bus:velocity:*` namespace, run on a schedule, cheap enough
  at real mesh scale? Phase 1's live-data gate proved per-publish `INCR` overhead is negligible; it
  did not measure periodic `scan_iter` cost, which is bounded by total key count, not tick rate.
- Given the RPC-health arc's own Q6 (one `organ_id` covering multiple services, or one per service)
  is still unresolved there — if Idea 4 below ever generalizes beyond RPC calls, the same shape
  question applies to census/velocity data.

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
