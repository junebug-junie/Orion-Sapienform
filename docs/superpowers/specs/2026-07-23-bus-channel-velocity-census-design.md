# Bus channel velocity & census — design spec

Status: **proposed, design-only**. No code in this patch. Sequencing: build Phase 1, verify
against real live data, gate before Phase 2; build Phase 2, verify, gate before anything in the
Later/Parked section is even scoped as a next patch. Do not skip a gate because a prior phase
"seems obviously fine" — CLAUDE.md's metric quality gate applies per phase, every time.

**Cross-reference:** `docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md`
covers a related but distinct question — the *calibration and trustworthiness* of the six
existing `compute_transport_pressures()` signals (stream depth, backpressure, catalog drift,
contract mismatch, observer failure, reliability), all derived from the **2-channel** allowlist
`bus_observer.py` already polls (`orion:stream:world_pulse:run:result` and its `:dlq` sibling).
That spec found `stream_depth_pressure` miscalibrated ~1,000x and confirmed live traffic reads
near-zero on those two channels. This spec does not touch that calibration work and does not
propose changing `stream_depth_pressure`, `backpressure`, or any of the other five signals. It
asks a different question: of the **264** channels declared in `orion/bus/channels.yaml`, almost
all of them Redis pub/sub rather than Streams, how many are carrying real traffic right now, and
at what rate — a question the existing observer structurally cannot answer, because pub/sub has
no backlog for `XLEN`/`XREVRANGE` to sample after the fact.

---

## Arsonist summary

`services/orion-bus/app/bus_observer.py` polls exactly 2 of 264 cataloged channels, because
`XLEN`/`XREVRANGE` only work against Redis Streams, and confirmed (per the sibling spec above)
almost the entire mesh runs on Redis pub/sub, which has no persistence to poll. There is no
concept of message velocity (messages/sec) anywhere in this codebase — `record_stream_depth()`
records a single point-in-time count, not a rate. There is no concept of "how many of the 264
declared channels are actually alive right now" — the catalog is static, and observation coverage
is a hardcoded 2-channel allowlist (`BUS_OBSERVER_STREAMS`), not derived from the catalog.

Polling all 264 channels the way the observer polls its current 2 would not work for pub/sub
channels at all (nothing to poll), and would multiply Redis read volume ~130x for the channels it
could reach. The fix is to stop polling and start counting at the source: every producer in the
mesh funnels through exactly one method, `OrionBusAsync.publish()`
(`orion/core/bus/async_service.py:242`), whether the channel is stream or pub/sub, cataloged or
not. Instrumenting that single seam gives real per-channel traffic visibility across the whole
mesh at the cost of one counter increment per publish, not one poll per channel per tick.

## Current architecture

- `orion/bus/channels.yaml`: 264 channel entries, each with `name`, `kind`
  (`request`/`result`/`event`/`stream`), `schema_id`, `producer_services`, `consumer_services`,
  `stability`, `since`. Purely declarative — nothing in the repo counts how many of these 264 are
  presently producing or consuming traffic.
- `orion/core/bus/async_service.py::OrionBusAsync.publish()` (line 242): the single choke point
  for all outbound bus traffic. `async def publish(self, channel: str, msg: BaseEnvelope | Dict)`.
  No sync duplicate exists elsewhere in `orion/core/bus/`. Every producer service in the mesh
  calls through this one method.
- `services/orion-bus/app/bus_observer.py`: runs `run_observer_tick()` every
  `BUS_OBSERVER_POLL_INTERVAL_SEC` (default 10s). Per tick: 1 Redis `PING`, then `XLEN` on each
  entry in `settings.observer_stream_list` (currently `orion:stream:world_pulse:run:result` +
  `:dlq`, i.e. 2 channels — both real `kind="stream"` XADD channels per the settings.py comment
  trail documenting how the two former placeholder defaults, `orion:evt:gateway`/`orion:bus:out`,
  were found to point at Redis keys that structurally never existed). Also runs a bounded
  `XREVRANGE` sample per stream for schema-mismatch checking
  (`count_schema_mismatches()`), and diffs the polled set against
  `load_channel_catalog_names()` to flag `uncataloged_streams` — but only within that same
  2-channel universe, not across the full 264-channel catalog.
- `ObserverRollup` (dataclass, `bus_observer.py:134`): `stream_lengths: dict[str, int]`,
  `uncataloged_streams: list[str]`, `backpressure: list[tuple]`, `schema_mismatches: list[tuple]`.
  No `velocities` field exists. `to_collector()` feeds all of this into
  `BusTransportGrammarCollector` (`services/orion-bus/app/grammar_emit.py`), which emits grammar
  trace events via `publish_bus_transport_grammar_trace()`, gated by
  `PUBLISH_ORION_BUS_GRAMMAR` (default `False`).
- `orion/bus/consumer_readiness.py::check_bus_consumer_readiness()`: exists, handles a different
  concern (consumer-side readiness), not surveyed in depth for this spec — relevant to the
  Phase-6 (Parked) consumer-liveness idea below, not to Phases 1–2.
- Downstream of the observer's 2-channel depth data: `orion/substrate/transport_loop/extract.py`
  computes the six pressure signals covered by the sibling spec. This spec's new velocity/census
  data has **no downstream consumer yet** — that is explicitly out of scope until Phase 1+2 are
  live-verified (see Non-goals).

## Missing questions

Carried over from the brainstorm, unresolved, and relevant to sequencing:

- What is the real live publish rate on this mesh — tens/sec, thousands/hour? Determines whether
  a per-second or per-minute counter bucket is the right granularity, and whether a naive `INCR`
  per publish is measurable overhead at all. **Phase 1's own live-data gate answers this
  directly** — no need to guess before building, just before trusting the result.
  **Update, day-of (2026-07-23):** the sibling calibration spec already establishes that the
  2-channel Streams subset the current observer polls reads near-zero traffic. That is evidence
  about those 2 channels specifically, not about the other 262 pub/sub channels this spec's
  Phase 1 would newly instrument — it does not answer this question, but it is a reason not to
  assume mesh-wide traffic is heavy either.
- Is there an existing Redis pipelining convention in `OrionBusAsync` that a velocity counter
  should follow, or would this be the first use of one? Affects whether Phase 1's `INCR` should be
  pipelined with the existing `publish()` call or issued as a genuinely separate fire-and-forget
  round-trip.
- Who consumes a velocity/census signal once it exists — a Hub dashboard tile, a future
  pressure/arousal input, an ops alert? Left open on purpose: Phase 1+2 are read-only and produce
  no consumer-facing output, so this doesn't block starting, but it should be answered before
  Phase 3+ (aggregate load score) gets scoped as a real patch.
- "Declared silent" needs a per-kind or per-channel expectation, not one global threshold — a
  `request`-kind channel that only fires on rare user action is correctly silent most of the time.
  Phase 2's census function should surface silence as a fact, not a verdict, until this is
  resolved.

## Proposed schema / API changes

None in Phase 1 or Phase 2. Both are additive, internal-only:

- Phase 1 adds a new Redis key namespace (`orion:bus:velocity:{channel}:{minute_bucket}`, short
  TTL) — not a schema, not a bus channel, not a contract. No `channels.yaml` entry, no
  `orion/schemas/registry.py` entry: this is transport-internal telemetry about the bus, not a
  message carried on it.
- Phase 2 adds one pure function (name TBD, e.g. `orion.bus.census.compute_census()`) taking the
  Phase 1 counters and the existing `load_channel_catalog_names()` output. No new schema.
- If Phase 3+ (grammar-trace emission of velocity, per the brainstorm's idea 3) is later scoped as
  its own patch, *that* would extend `BusTransportGrammarCollector` and would need its own gate
  pass at that time — deliberately not designed here.

## Files likely to touch

**Phase 1 (velocity counter):**
- `orion/core/bus/async_service.py` — instrument `publish()`, behind a settings flag defaulting
  off.
- `orion/core/bus/tests/` (or wherever this module's tests live) — unit test for the counter
  behavior, mockable Redis client.
- Possibly a new `orion/bus/velocity.py` for the read-side helper (sum buckets over a trailing
  window → msgs/sec), kept separate from the write-side instrumentation in `async_service.py`.

**Phase 2 (census diff):**
- New `orion/bus/census.py` (or extend `bus_observer.py` if that reads better once Phase 1 exists)
  — pure function, `catalog_names: set[str]`, `active_channels: dict[str, int]` →
  `{declared_silent: list[str], undeclared_active: list[str]}`. Testable with zero Redis
  dependency.
- `orion/bus/tests/test_census.py` — pure-function unit tests, fixture-driven, no live Redis
  needed for correctness.

**Not touched by Phase 1 or 2** (listed so a reviewer can confirm scope stayed thin):
`orion/bus/channels.yaml`, `orion/schemas/registry.py`, `services/orion-bus/app/bus_observer.py`
(deferred — see Non-goals), `services/orion-bus/app/grammar_emit.py`,
`orion/substrate/transport_loop/`.

## Non-goals

- **Not** recalibrating `stream_depth_pressure` or any of the six existing pressure signals — that
  is the sibling spec's job, explicitly.
- **Not** wiring velocity or census data into `bus_observer.py`'s grammar-trace emission path in
  this patch. That's a real next step (brainstorm idea 3) but deliberately deferred until Phase
  1+2 have live numbers to justify it — building the emission path before confirming the counters
  themselves behave sanely under real traffic would be exactly the kind of ungated build-on-top
  the user asked to avoid.
- **Not** building an aggregate "bus load score" (brainstorm idea 4) yet. Flagged in the brainstorm
  itself as the idea most at risk of becoming ungrounded ceremony if built before 1–2 exist.
- **Not** building the dead-channel/never-fired detector (idea 5) or consumer-side liveness (idea
  6) yet — both depend on Phase 1's counters existing and being trusted first.
- **Not** expanding `BUS_OBSERVER_STREAMS` / polling more channels via `XLEN`. That approach does
  not work for pub/sub channels at all and was rejected during brainstorming in favor of the
  publish-side counter.
- **Not** touching `orion/bus/consumer_readiness.py`'s existing logic.

## Phased plan (build 1 → gate → build 2 → gate → stop)

### Phase 1 — publish-side velocity counter

**What:** Instrument `OrionBusAsync.publish()` to increment a per-channel, per-minute-bucket
Redis counter on every call (`INCR orion:bus:velocity:{channel}:{minute}` + short `EXPIRE`),
behind a settings flag defaulting **off**. Add a read helper that sums the trailing N buckets for
a channel into messages/sec.

**Why it matters for Orion's development toward sentience:** this is a precondition, not a
cognition feature by itself — Orion cannot have any grounded sense of its own bus as a live,
moving substrate (as opposed to a static wiring diagram in a YAML file) without first being able
to observe that substrate's actual throughput. Every later idea in this spec, and every future
claim about transport "arousal" or "activity," depends on this existing and being trustworthy
first.

**Smallest buildable version:** one `INCR`+`EXPIRE` pair added to `publish()`, one pure read-side
summation function, one settings flag. No new service, no schema change, no bus channel.

**Gate before Phase 2 starts** (per CLAUDE.md's metric quality gate, run in full, not skimmed):
1. Trace provenance — confirm the counter increments only happen on real `publish()` calls in a
   live environment, not on retries/reconnects that would double-count.
2. Independence — confirm this isn't just a monotonic transform of something `bus_observer.py`
   already computes (it isn't: `stream_lengths` is a point-in-time depth on 2 channels; this is a
   rate on up to 264 channels — different quantity, different coverage, but state explicitly why
   before moving on, not just assert it).
3. Theory anchor — "messages per channel per minute measures publish activity" is close to
   definitionally true, but still write down what it's *for* before Phase 2 assumes it's usable:
   a coverage census.
4. Live-data sanity check — pull real counter values after this runs against production traffic
   for a real window. Confirm it is not degenerate (flat zero on every channel, or saturating
   instantly on all of them). This is the single most important gate given the missing-questions
   section above flags real publish rate as unknown.
5. Existing-mechanism check — done as part of this spec's own research (confirmed: nothing else
   in the repo computes a rate/velocity on bus traffic).
6. Reversibility — cheap: one settings flag flips it off, one Redis key namespace with TTLs
   expires on its own, no schema/manifest bakes it in anywhere.

Do not start Phase 2 until step 4 has been run against real traffic and the results look sane.

### Phase 2 — catalog vs. live-traffic census

**What:** A pure function comparing `set(264 catalog channel names)` against
`set(channels with nonzero Phase-1 velocity in the last N minutes)`, returning two lists:
channels declared but silent, and channels carrying traffic but not in the catalog (generalizing
`bus_observer.py`'s existing `uncataloged_streams` logic from its current 2-channel scope to the
full catalog).

**Why it matters:** this is the direct answer to "x count of channels in yaml, this is how many
are actually running" — turning a static registry into a live census. It's also a mechanical,
deterministic catch for the exact failure mode already found by hand once
(`orion:evt:gateway`/`orion:bus:out` — placeholder channel names that were never real, undetected
for months because nothing checked).

**Smallest buildable version:** the pure function itself plus tests, fed by fixture data — no live
Redis required to validate the logic. Wiring it to real Phase-1 data for a live report is a
separate, small follow-up once the function itself is reviewed.

**Gate before anything past this point is scoped as a next patch:** run the same six-point
metric-quality-gate pass against the *census output itself* once fed real Phase-1 data — in
particular, confirm the "declared silent" list isn't dominated by low-frequency-by-design channels
(the per-kind-expectation gap flagged in Missing Questions), which would make the census
technically correct but practically useless without further work.

## Later / parked (from the brainstorm, not scoped as next patches)

Recorded here so nothing from the session is lost, explicitly **not** committed to as sequenced
work. Each needs its own gate pass, informed by Phase 1+2's real results, before it becomes a
patch:

- **Grammar-trace emission of velocity** — extend `ObserverRollup`/`BusTransportGrammarCollector`
  with a `velocities` field once Phase 1 counters are trusted, reusing the already-wired emission
  path instead of inventing a new one.
- **Aggregate bus load score** — one normalized "how loud is the mesh right now" scalar. Flagged
  as highest keyword-cathedral risk of the set; do not start here.
- **Dead-channel / never-fired detector** — track last-publish timestamp per channel (a Redis hash
  set on every `publish()` call), surface channels that have never fired since counter start or
  haven't fired in > X hours. Distinguishes "quiet because nothing to say" from "wired but
  structurally dead," generalizing the exact bug class this spec's Current Architecture section
  documents (`orion:evt:gateway`/`orion:bus:out`) into an ongoing check instead of a one-off catch.
- **Consumer-side liveness** — "running" arguably needs both a publisher *and* a receiving
  consumer. Extend the census with `PUBSUB NUMSUB`/consumer-group presence checks, folding into
  `orion/bus/consumer_readiness.py` or the new census module. Needs its own investigation into
  what `check_bus_consumer_readiness()` currently does and doesn't cover before scoping.

## Acceptance checks

**Phase 1 done when:**
- `publish()` instrumented behind a flag; flag documented in `.env_example` per CLAUDE.md env
  parity rules.
- Unit test proves counter increments correctly and does not fire when the flag is off.
- Flag flipped on in a real environment for a real window; live Redis values pulled and inspected
  by a human, not just asserted by a test — the actual live-data gate, not a substitute for it.
- Live values are not degenerate (not flat-zero everywhere, not saturating on every channel).

**Phase 2 done when:**
- Census function has unit tests covering: all-silent catalog, all-active catalog, mixed, and an
  undeclared-active channel appearing.
- Run once against real Phase-1 data; output reviewed by a human for whether "declared silent"
  reads as meaningful or as noise dominated by low-frequency-by-design channels.

## Recommended next patch

Start Phase 1 only. Branch/worktree per CLAUDE.md section 2, thin patch: instrument
`OrionBusAsync.publish()`, add the settings flag (default off), add the read-side summation
helper, add tests. Do not touch `bus_observer.py`, grammar emission, or Phase 2's census function
in the same patch — Phase 2 depends on Phase 1's gate passing with real data first, per the
phased plan above.
