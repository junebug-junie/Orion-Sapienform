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
3. Wire the periodic flush into `orion-substrate-runtime`'s existing tick infrastructure,
   shadow-only -- no live consumer.
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
