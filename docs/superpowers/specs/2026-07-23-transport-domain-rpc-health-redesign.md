# Transport domain redesign: real cross-service RPC health

Status: **design mode, not implemented.** Answers Juniper's explicit choice ("build real
cross-service bus telemetry from scratch") after confirming the current `transport_pressure`/
`bus_health` family measures one unrelated queue, not the bus. Per CLAUDE.md §0A, changes
touching the Sentience Striving Program's Predictive-Processing/Active-Inference substrate
need explicit sign-off before implementation -- this document proposes, it does not build.

## Arsonist summary

`transport_pressure`, `bus_health`, `stream_depth_pressure`, `backpressure`,
`delivery_confidence`, `contract_pressure`, `catalog_drift_pressure`,
`observer_failure_pressure`, `reliability_pressure`, and `transport_prediction_error()` are
all derived entirely from `BUS_OBSERVER_STREAMS` (`services/orion-bus/app/settings.py`),
which is configured to exactly one real producer's Redis Stream:
`orion:stream:world_pulse:run:result` (a periodic news-digest job from `orion-world-pulse`)
plus its dead-letter queue. Confirmed live (2026-07-22/23): that queue's consumer group
(`cg:concept-induction`) has `pending=0, lag=0` -- fully healthy, fully consumed -- and its
`XLEN=91` is the stream's *entire un-trimmed lifetime message count* since 2026-07-07, not a
backlog depth. Everything else on the bus is Pub/Sub, which has no `XLEN`/backlog concept at
all. Documented across five READMEs in PR #1278 (`docs/superpowers/pr-reports/` /
`services/orion-bus/README.md`, `services/orion-field-digester/README.md`,
`services/orion-substrate-runtime/README.md`, `orion/mood_arc/README.md`,
`orion/sentience_striving_program/README.md`) as a narrow-scope finding, not re-derived here.

**Audited whether this is systemic across all five Predictive-Processing domains (it is
not):** `execution_prediction_error` (2 real producers: `orion-cortex-exec`,
`orion-harness-governor`), `biometrics_prediction_error` (1 codebase, deployed per real
physical node, fleet-wide by design), `chat_prediction_error` (1 producer, `orion-hub` --
the only chat surface that exists, so there is no broader "chat" being missed), and
`route_prediction_error` (1 producer, `orion-cortex-orch` -- the single arbitration point,
structurally correct to be singular) are all honestly scoped. Transport is the only domain
where the name promises whole-bus visibility that the wiring cannot deliver.

**The real fix is not a new instrument invented from scratch -- it's capturing data that
already exists and is currently thrown away.** `orion/core/bus/async_service.py`'s
`OrionBusAsync.rpc_request()` -- the shared async bus client's request/reply RPC method,
called from **37+ distinct real files across nearly every service in the architecture**
(`orion-cortex-orch`, `orion-cortex-exec`, `orion-hub`, `orion-embodiment`,
`orion-chat-memory`, `orion-spark-introspector`, `orion-actions`,
`orion-memory-consolidation`, `orion-topic-foundry`, `orion-context-exec`, `orion-mind`,
`orion-vision-host`, `orion-self-experiments`, `orion-dream`, `orion-agent-council`,
`orion-cortex-gateway`, `orion-vision-council`, `orion-thought`, `orion/harness/`,
`orion/autonomy/`, `orion/cognition/`, `orion/memory_graph/`, and more) -- already measures
real round-trip latency via `perf_counter()` on every single call, on every success/timeout
path (`async_service.py:326,340,348,364,377,385,394,413`). Every one of those measurements is
currently written only to `logger.info`/`logger.error` and discarded. This is genuinely
cross-service (the opposite of transport's problem): whichever real service makes an RPC
call, its latency is measured, by the one shared client every one of them uses.

## Current architecture

- `OrionBusAsync.rpc_request(request_channel, envelope, *, reply_channel, timeout_sec=60.0)`
  (`orion/core/bus/async_service.py:315-413`): publishes to `request_channel`, awaits the
  first reply on `reply_channel`, times out after `timeout_sec`. Two code paths depending on
  whether a long-lived RPC worker task is running (`path=worker` vs `path=inline`), both
  measure elapsed time via `perf_counter()` from `started = perf_counter()` at entry.
- No schema, channel, or persistence exists for this timing data today. Grepped
  `orion/bus/channels.yaml` for any `rpc`-related entry: only `orion:pad:rpc:request` and
  `orion:pad:rpc:reply:*` exist (lines 930, 939) -- these are a *different*, narrower FCC/PAD
  mechanism, not the generic `rpc_request()` this spec is about. No grammar event is
  registered or emitted for a generic RPC round-trip anywhere in the codebase.
- The five existing prediction-error domains (`orion/substrate/{execution,transport,
  biometrics,chat,route}_loop/`) establish the working pattern this spec should mirror:
  `grammar_extract.py` parses real events into a typed projection, a reducer persists it,
  `services/orion-substrate-runtime/app/worker.py`'s per-domain `_*_tick()` method computes
  a prediction-error delta and writes it onto a `FieldStateV1` node via the shared
  `_write_prediction_error_node()` writer, gated behind `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`.
- `check_single_consumer_channels.py` already does live Redis introspection (`NUMSUB`) as a
  CI gate, not a runtime signal -- confirms this kind of Redis-native inspection is already a
  precedented pattern in this repo, just not wired as an ongoing measurement.

## Missing questions

1. **Capture point: instrument `rpc_request()` itself, once, at the shared-client level.**
   This is the obvious answer given the whole point is "one seam, every caller" -- but needs
   confirming there's no meaningfully different second RPC path (`rpc_legacy_dict`,
   `async_service.py:417+`) that would also need instrumenting to get full coverage.
   `UNVERIFIED` -- not read in this pass.
2. **Real distribution, unknown.** No live measurement of actual RPC latency/timeout rate
   exists yet. Per the "measure before minting" rule (`orion/sentience_striving_program/
   README.md` §7) that already caught `autonomy`'s dead origination signal, this needs a
   read-only measurement pass -- e.g., tailing/parsing real `[rpc] ...elapsed_ms=` log lines
   across services for a real window -- *before* any schema/threshold gets designed, not
   after.
3. **Volume and aggregation cadence.** `rpc_request()` is called extremely frequently across
   nearly every service. Emitting one grammar event per call would be a volume explosion,
   unlike the other five domains' tick-scoped batches. This needs an in-memory rollup
   (count/p50/p95/max latency, success/timeout/error counts) flushed on a fixed interval or
   call-count threshold, not a per-call event. Exact window size needs real volume data from
   item 2 above to choose responsibly, not an assumed default.
4. **Per-service-pair breakdown vs. one blended number.** `request_channel`/`reply_channel`
   already identify which real service pair an RPC spans. This could produce a genuinely
   richer signal than transport's single scalar ever was -- per-pair health, not just one
   number -- but multiplies the aggregation-state surface. Decide after item 2's real data
   shows how many distinct channel pairs actually see traffic in a representative window.
5. **Old `transport_pressure` family's fate: not decided here.** Options are (a) rename to
   reflect its real, narrow, still-useful meaning ("world_pulse queue health") and keep it
   running for that purpose, or (b) deprecate once this new signal takes over the charter's
   "transport domain" role. This spec does not decide between them -- that's a separate call
   once the new signal is real and live-verified, not a default outcome of building it.
6. **Hot-path overhead risk.** `rpc_request()` is one of the hottest paths in the entire
   system. Any instrumentation added here must be synchronous, in-memory, and effectively
   free (a counter/histogram update, not I/O) -- a real risk to name explicitly, not assume
   away. Needs a before/after latency benchmark as part of any implementation, not just a
   code read.

## Proposed schema / API changes

Sketched, not finalized -- each item below still needs its own metric-quality-gate pass at
implementation time, same as every other signal built this session:

- A new lightweight in-memory aggregator (module TBD, e.g.
  `orion/core/bus/rpc_health.py`), updated synchronously and cheaply inside
  `rpc_request()`'s existing success/timeout/error branches -- no new I/O on the hot path.
- A new schema, e.g. `BusRpcHealthV1` (fields TBD from item 3's real volume data:
  likely `window_start`/`window_end`, `sample_count`, `success_count`, `timeout_count`,
  `error_count`, `p50_latency_ms`, `p95_latency_ms`, `max_latency_ms`, optionally
  per-channel-pair breakdown).
- A periodic flush -- likely from `services/orion-substrate-runtime`'s existing tick
  infrastructure, mirroring the other five domains' `_*_tick()` pattern -- writing a new
  `node:substrate.bus_rpc` (or similar) `FieldStateV1` node via the existing
  `_write_prediction_error_node()`-style shared writer, gated behind its own explicit flag
  (not silently piggybacking on `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`, since this is a
  new kind of signal, not a sixth instance of the existing prediction-error shape, until
  proven to fit that shape).
- Registered in `orion/schemas/registry.py`, cataloged in `orion/bus/channels.yaml` if
  published as a bus event rather than written directly to the field-state table.

## Files likely to touch (at implementation time, not this patch)

- `orion/core/bus/async_service.py` -- the actual instrumentation inside `rpc_request()`.
- New aggregator module (location TBD above).
- New schema file under `orion/schemas/` + `orion/schemas/registry.py` entry.
- `services/orion-substrate-runtime/app/worker.py` -- new tick/consumer to flush + persist
  the rollup, mirroring the existing five `_*_tick()` methods.
- `orion/bus/channels.yaml` -- new channel entry, if this ships as a published event.
- `scripts/analysis/measure_rpc_health_baseline.py` (new) -- the real-data measurement this
  spec's Missing Question 2 requires, built and run *before* the schema is finalized, same
  sequencing this whole session has used for every other signal.
- `orion/sentience_striving_program/README.md` §9b item 3 -- correct the "transport" domain
  claim once (and only once) this signal is real, live-verified, and its relationship to the
  old `transport_pressure` family is decided (Missing Question 5).

## Non-goals

- Not deleting, renaming, or recalibrating `transport_pressure`/`bus_health`/`stream_depth_
  pressure`/etc. in this patch -- that decision (Missing Question 5) is explicitly deferred.
- Not touching `execution_prediction_error`/`biometrics_prediction_error`/
  `chat_prediction_error`/`route_prediction_error` -- confirmed honestly scoped, out of
  scope here.
- Not wiring this new signal into the Hub lattice console, `DriveEngine`, or any live
  consumer -- shadow-measurement only, per the charter's own §7 process rule, until real
  data validates it the same way every other signal in this program has been validated.
- Not implementing any code in this patch -- design mode only, per Juniper's explicit framing
  and CLAUDE.md §0A's proposal-mode requirement for changes touching this substrate.
- Not designing the exact rollup window, field set, or per-pair breakdown -- all explicitly
  deferred to Missing Questions 2-4, which need real measurement data first.

## Acceptance checks (for a future implementation pass, not this doc)

- A real, read-only measurement (`scripts/analysis/measure_rpc_health_baseline.py` or
  equivalent) of actual RPC latency/timeout distribution across a real window, run *before*
  any schema field is finalized.
- A before/after latency benchmark on `rpc_request()` itself, proving the instrumentation
  adds no measurable overhead to the hottest RPC path in the system.
- The resulting rollup shows real, non-degenerate variance across a real historical window
  (CLAUDE.md's metric-quality-gate step 4) -- not another always-flat signal.
- Confirmed genuinely cross-service: real data shows more than one distinct
  `request_channel`/`reply_channel` pair represented in the rollup, not one dominant caller
  standing in for "the bus" the same way world_pulse did.
- Shadow-measured only at first -- no live consumer wired -- per charter §7, until the above
  checks pass.

## Recommended next patch

1. Build the read-only measurement first (Missing Question 2) -- capture real RPC
   latency/timeout data over a representative window before designing the schema around
   assumed shapes.
2. Once real variance is confirmed non-degenerate and volume/cadence questions (Missing
   Questions 3-4) are answered with real numbers, build the in-memory aggregator inside
   `rpc_request()`, benchmarked for zero added hot-path overhead.
3. **Superseded 2026-07-23** -- wire the periodic flush into `orion-signal-gateway`'s
   existing organ-adapter machinery, not `orion-substrate-runtime`'s tick loop as originally
   written here. Reasoning: `orion-signal-gateway` already ingests 24 organs via
   eavesdropping on bus channels they publish for their own operation, and reuses that
   registry/normalization path rather than adding a second, parallel ingestion mechanism
   through substrate-runtime. See
   `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` for the
   full design, including a real gotcha found while writing it: 9 services (including both
   `cortex-exec` and `cortex-orch`) route real RPC traffic through a `fork_rpc_client()`
   child `OrionBusAsync` instance, not the chassis's own `svc.bus` -- naively draining
   `svc.bus.get_rpc_health_snapshot()` would silently report an always-empty aggregator.
4. Only after real, live-verified data exists, decide the old `transport_pressure` family's
   fate (Missing Question 5) and whether/how to correct the charter's §9b item 3 claim.

## Related work

- `docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md`
  (PR #1275, merged) + its implementation (PR #1277, merged) -- the measurement work on the
  *old* `transport_pressure` family that led to discovering the narrow-scope problem this
  spec responds to. That work (incident logging, historical baseline, cadence, correlation
  probe) remains valid regardless of this spec's outcome -- it measures a real, if narrowly
  scoped, signal.
- PR #1278 (merged) -- documents the narrow-scope finding itself across five READMEs; this
  spec does not re-derive that finding, only responds to it.
- `docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md` (branch
  `docs/fcc-motor-field-digester-signals-design`, open) -- a same-day, independently-scoped
  spec whose Appendix item 3 reaches the identical two-option framing for the
  `transport_pressure` family (invent a real proxy vs. rename honestly) and explicitly defers
  fixing it to "a separate follow-up effort" -- this spec is that follow-up. Zero file/scope
  overlap otherwise (that spec's Patch A/B touch `orion/harness/`, `execution_loop/`, and
  field-digester's execution-domain wiring; nothing here does).
- `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` (Step 3,
  design-only) -- the forward continuation of this spec's Recommended-next-patch item 3,
  now scoped to `orion-signal-gateway` instead of `orion-substrate-runtime`.

  **Checked for a hidden data-source collision, found none, but surfaced a real scope
  disclosure worth stating plainly:** that spec's Patch B (`harness_rpc_timeout`) detects Hub's
  RPC to `orion-harness-governor` never returning -- the same *category* of blind spot
  (RPC-adjacent failure with zero bus-observable trace) this spec's Missing Question 6 and
  Acceptance Checks are built to close generically. Verified directly:
  `HarnessGovernorClient.run()` (`services/orion-hub/scripts/harness_governor_client.py`) does
  **not** call `OrionBusAsync.rpc_request()` -- it has its own bespoke long-poll RPC
  implementation with mid-run liveness checks (FCC motor runs can take a long time and need
  streaming step-relay support, which a plain single-shot `rpc_request()` can't provide). So
  Patch B's signal and this spec's proposed `bus_rpc_health` rollup would be structurally
  disjoint, not double-counting the same event -- confirmed, not assumed.

  This does mean the "37+ callers of `rpc_request()`" scope claimed in this spec's Arsonist
  Summary is not full RPC-traffic coverage: `orion-harness-governor` traffic -- and any other
  service using a similarly bespoke long-poll pattern, not yet audited -- stays invisible to
  the signal this spec proposes, even after implementation. Not a reason to change approach
  (instrumenting the one shared client is still the correct thin seam for everything that
  actually uses it), but stated here plainly rather than implied as complete coverage.

## Revision, 2026-07-25 -- bus synaptic graph as a candidate third evidence source

Status: **proposed, design-only, not decided.** Does not resolve Missing Question 5 (old
`transport_pressure`/`bus_health` family's fate -- now renamed `stream_backlog_pressure`/
`stream_backlog_health`, PR #1331, a plain rename per that PR, not a product of this redesign).
Written the same week the `transport` metacog trigger (Options A+C, RpcHealthSnapshotV1 +
`rpc_transport_timeout` grammar) shipped -- disabled at first (`95db26ba9`,
`EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE=false`), then flipped on about an hour later the
same day (`40cd21f80`, "live verification is the next step") -- confirmed live in this worktree's
own `services/orion-equilibrium-service/.env_example`, which reads `=true`. Re-check this flag's
live value before relying on its state in any future revision; don't assume this note stays
current. Also the same week `orion-bus-mirror`'s bus synaptic graph (`orion_bus_synapse`,
FalkorDB) came back online after a `distutils`/Python-3.12 crash-loop fix (`redis` bumped to
5.2.1, commit `2e1fa7f2`).

**Arsonist read.** This spec's own "Related work" section already names the blind spot: Options
A+C both depend on a service self-reporting (`RpcHealthSnapshotV1`) or calling the one
instrumented shared client (`OrionBusAsync.rpc_request()`). `orion-harness-governor` does neither
-- it has its own bespoke long-poll RPC (mid-run liveness checks for long FCC motor turns), so it
is invisible to both existing evidence sources, permanently, even after full implementation of
this spec's Recommended-next-patch. The bus synaptic graph does not have this limitation: it is a
passive wiretap on envelope `correlation_id` co-occurrence, agnostic to which RPC mechanism (or
no RPC mechanism at all -- ordinary pub/sub) produced the traffic. Verified live, same day:

```text
MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
WHERE a.organ_id CONTAINS 'harness' OR b.organ_id CONTAINS 'harness'
RETURN a.organ_id, b.organ_id, e.count, e.latency_zscore
```
`hub -> orion-harness-governor`: `count=5, latency_zscore=-2.10`. `cortex-exec ->
orion-harness-governor`: `count=34, latency_zscore=-0.25`. Real edges, real z-scores, on the exact
organ this spec's Related-work section names as structurally unreachable by its own proposed
signal.

This is not a proposal to replace Options A/C -- `RpcHealthSnapshotV1`/`rpc_transport_timeout`
measure something the graph cannot (an actual timeout/error outcome, not just a latency
deviation) -- but a candidate **third** input specifically for the coverage gap those two leave
open, using infrastructure that already exists, is already live, and required zero new
instrumentation to produce the numbers above.

**Relationship to the bus synaptic graph's own arc.** This is a narrow, transport-domain-specific
use case, not a duplicate of work already scoped there. The graph's brainstorm doc
(`docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md`) and its already-built
Phase-3+ Ideas (labeled 1, 4, 5 in that doc's "Signal families this substrate opens up" section --
**not** the same numbers as that doc's own earlier "Idea 1-6" list under "Proposed schema / API
changes", which the doc itself flags as a labeling collision; catalog-drift fix, live recall/chat
anomaly awareness, and Hub debug routes, respectively) do not include feeding the Sentience
Striving Program's transport prediction-error/metacog-trigger gap; the closest sibling, Phase-3+
Idea 4
(`docs/superpowers/specs/2026-07-24-bus-synaptic-graph-reasoning-consumer-design.md`, merged), is a
different consumer entirely -- unconditional per-chat-turn awareness fed into `orion-recall`'s
fragment fusion, not a metacog trigger or a `FieldStateV1` node write. The Cypher this proposal
would reuse (`anomalies` -- already live-verified and tested via
`services/orion-hub/scripts/bus_synaptic_graph_routes.py`, the same query Idea 4's adapter reuses)
is shared infrastructure; the consumption target (transport metacog trigger / prediction-error
substrate) is new and out of scope for both existing Ideas.

**Missing questions, additive to the ones already open above:**

- Should this feed the `transport` metacog trigger (Option A/C's `trigger_kind`) as a third
  evidence branch, or feed a new `node:substrate.bus_synaptic` `FieldStateV1` node directly (the
  "compute signals from the graph" consumption mode the brainstorm doc names, as opposed to Idea
  4's "reason from the graph directly" mode)? Real trade-off: the metacog-trigger route reuses an
  already-shipped, already-enabled dispatch mechanism (live since `40cd21f80`, re-check before
  relying on this) -- if anything this favors reusing it over building a new path; the
  `FieldStateV1` route matches the other four domains' shape but would need its own periodic reducer (mirroring
  `orion-substrate-runtime`'s `_*_tick()` pattern, per this spec's own "Proposed schema / API
  changes" section above) reading a graph instead of Postgres. Not decided here.
- What z-score/count thresholds distinguish "worth feeding a cognition-adjacent signal" from "Hub
  debug noise"? The recall consumer design doc flags this exact question as unresolved for its own
  use case (Hub's `zscore_threshold=3.0, min_count=5` were tuned for a human reading a table) --
  applies here too, not re-derived.
- Cold-start reliability: this session's own numbers above (`count=5`) sit right at the graph's
  own documented `count < ~5` unreliable-z-score floor. Needs a longer live window before treating
  any single edge's z-score as trustworthy, not just this one example.
- Does `orion-harness-governor` traffic reliably produce a `CAUSALLY_FOLLOWED_BY` edge at all, or
  did this session's example only work because `hub` also happens to route other traffic through
  channels the graph tracks? Needs checking across more of the 37+ real RPC-adjacent services this
  spec's Arsonist Summary names, not generalized from one organ pair.

**Non-goals, additive:** not deciding Missing Question 5 (old channel's rename-vs-deprecate fate)
here or as a side effect of this revision. Not building the metacog-trigger wiring, the
`FieldStateV1` reducer, or any adapter in this patch -- proposal only, per CLAUDE.md §0A's
proposal-mode requirement for changes touching this substrate, same standard the rest of this doc
already holds itself to.

**Recommended next patch (revised):** before choosing between the two consumption modes above,
run a real-data pass over a longer window (a day or more) confirming `CAUSALLY_FOLLOWED_BY` edges
involving RPC-health-invisible organs (`orion-harness-governor` and any others sharing its
bespoke-RPC pattern, not yet audited) are a recurring, non-degenerate signal -- not a one-off
artifact of this session's snapshot.

## Revision, 2026-07-25 (same day, continued) -- one of the two consumption modes built

Status: **built, shadow-only, off by default.** Juniper approved both consumption-mode directions
in principle ("1 and 2 sound good"), then asked the two open questions to collapse into one shared
foundation rather than two bespoke builds ("what about reducers back into grammar substrate?"):
real precedent already exists (`OrionBusAsync._emit_rpc_timeout_grammar`,
`orion/core/bus/async_service.py:363`, emits `GrammarEventV1(semantic_role=
"rpc_transport_timeout", ...)` -- Option C from the earlier transport-metacog-trigger patch).

**What actually got built is simpler than a new grammar-event emission, on inspection.** The five
existing domains' `prediction_error` functions all diff a `prev`/`curr` projection pair built from
grammar events accumulated in Postgres. The bus synaptic graph doesn't need that shape at all: its
own EWMA baseline per edge (`services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update`)
already *is* the "prev expectation," continuously maintained as new traffic arrives -- there is no
separate prev/curr snapshot to diff. So `bus_synaptic_prediction_error()`
(`orion/substrate/prediction_error.py`) takes a flat list of current `|zscore|` values (queried
fresh from FalkorDB each tick, not accumulated in Postgres via a new grammar event type), and a
new `_bus_synaptic_tick`/`_bus_synaptic_tick_loop` (`services/orion-substrate-runtime/app/
worker.py`, mirroring `_dynamics_tick_loop`'s periodic-poll shape, not the four grammar-driven
`_*_tick` methods) reads both `PUBLISHES.gap_zscore` and `CAUSALLY_FOLLOWED_BY.latency_zscore`
edges (filtered to `count > SUBSTRATE_BUS_SYNAPTIC_MIN_EDGE_COUNT`, mirroring Hub's own
`min_count=5` cold-start floor), aggregates `mean(|zscore|)`, and writes a new sixth domain node
(`node:substrate.bus_synaptic`) via the same shared `_write_prediction_error_node()` the other
five domains use. No new `GrammarEventV1` type, no new reducer, no new Postgres table -- this
reuses `RedisGraphQueryClient` (`orion/graph/falkor_client.py`, already a proven dependency of
this service via `orion.substrate.falkor_store`, already pinned to `redis==5.2.1` so it doesn't
hit the `distutils` crash the rest of the bus-synaptic-graph arc found) pointed at a second graph
name (`FALKORDB_BUS_GRAPH=orion_bus_synapse`) on the same FalkorDB instance
`SUBSTRATE_STORE_BACKEND=falkor` already connects to.

**This resolves as originally analyzed: the "no threshold needed" consumption mode is the one
that's buildable now, the metacog-trigger dispatch mode is not.** Saturation uses a fixed 3.0
z-score constant (`_BUS_SYNAPTIC_ZSCORE_SATURATION`, reusing Hub's own `zscore_threshold=3.0`
convention) rather than a tuned "is this notable" threshold -- the continuous magnitude score
itself is the signal, same as the other five domains' continuous `prediction_error` values. The
genuinely open question (what z-score/count is worth *interrupting cognition* for, via the
metacog trigger) remains untouched and explicitly tabled, per Juniper: "not sure about zscore
thresholds, probably table without more data."

**Shadow-only, off by default** (`SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED=false`) -- CLAUDE.md
metric-quality-gate step 4 (live-data sanity check) has not yet been run against this specific
aggregate. Real edge z-scores were confirmed live earlier this session (see this doc's prior
revision), but `bus_synaptic_prediction_error()`'s own aggregate output (mean over both edge
kinds, saturated) has never run against live traffic. Do not flip the flag before running that
check.

Tests: `orion/substrate/tests/test_prediction_error.py::TestBusSynapticPredictionError` (6 tests,
pure function) and `services/orion-substrate-runtime/tests/test_worker_bus_synaptic_tick.py` (7
tests, disabled-is-noop, fail-open on client-init/query error, aggregation-and-write wiring,
client caching). Full suite run clean against a stashed baseline of the same worktree: same 13
pre-existing failures present with or without this patch (unrelated -- `test_cursor_reset_auth.py`
and `test_quarantine_truth.py` need a real operator-token fixture not available in this sandbox;
`test_worker_independent_reducers.py::test_start_spawns_independent_reducer_poll_tasks` asserts a
stale poll-task count of 3 against a codebase that already has 5, unrelated to this patch's new
task which isn't even named `*-poll`).

**Non-goals, additive:** not building the metacog-trigger wiring in this patch (still needs the
threshold decision, still tabled). Not folding `node:substrate.bus_synaptic` into
`_aggregate_prediction_error_confidence`'s existing four-domain mean (PR #1329) -- that's a
separate call about whether a sixth domain changes that formula's semantics, not a side effect of
writing the node. Not flipping `SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED` to `true` -- shadow-only until
the live-data sanity check above runs.

**Recommended next patch:** flip `SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED=true` on a real deployment,
run `measure_ast_hot_reducer.py`-style replay against real `node:substrate.bus_synaptic` history
to confirm non-degenerate variance (not flat/always-0/always-saturated), then decide the
`_aggregate_prediction_error_confidence` integration question above with real numbers in hand.
