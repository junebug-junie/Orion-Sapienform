# Transport bus signal quality — measurement design

Status: **proposed, design-only**. No code in this patch. Sequenced follow-up to
`docs/superpowers/specs/2026-07-22-chat-route-prediction-error-audit.md` and the
`fix/delete-self-state-module`/`fix/hub-lattice-remove-self-state` PRs (#1266, #1272), which
removed the Hub lattice console's dead L6 self-state layer for the transport lane. This spec
covers measurement only; the question of what (if anything) replaces L6 is deliberately left
for a follow-up turn once this measurement work has real numbers to answer it with.

**Cross-reference (added after PR #1276 opened, same day):** PR #1276
(`docs/superpowers/specs/2026-07-22-l6-self-model-ast-hot-active-inference-design.md`) answers a
related but higher-altitude question — what should replace `SelfStateV1` as the *system-wide*
canonical Layer 6 (self-model), via AST/HOT + Active Inference — where this spec asks the
narrower, lane-specific "is transport's real data even trustworthy" question. They are the same
underlying layer, approached from opposite ends (top-down theory-fit vs. bottom-up
data-quality), and they share one already-confirmed collision:
- Both threads independently found `orion-athena-self-state-runtime` still running in production
  after its source was deleted in PR #1266. **Resolved same day**: the container has been
  stopped and removed directly (`docker stop`/`docker rm` — its compose file no longer exists to
  bring it down cleanly via `docker compose down`); confirmed via `docker ps` and a growing
  `substrate_self_state` write-lag post-stop. PR #1276's "Missing Question 1" and "Recommended
  next patch" step 1 are both satisfied by this.
- PR #1276's item 3 (a confidence signal aggregating `dynamic_pressure` volatility across all
  five `prediction_error` domains) must not skip this spec's own findings when its required
  metric-quality-gate pass runs: transport's channel is real but calibrated ~1,000x off the
  system's actual operating range (see "Current architecture" below), and route's channel is a
  still-unexplained subnormal float per the chat/route audit doc. An aggregate that blends these
  two known-compromised domains in with the three healthy ones (biometrics, execution, and
  soon-fixed chat) risks inheriting their brokenness at a higher level of abstraction unless
  explicitly accounted for.

---

## Arsonist summary

`transport_prediction_error()` (`orion/substrate/prediction_error.py`) reads flat `0.0` in
production. Unlike `chat_prediction_error()` (a proven structural bug, fixed in `a98854a2`) or
`route_prediction_error()` (an unexplained subnormal-float corruption, still open), transport's
flatness traces to something more mundane and better-evidenced: the bus really has been quiet.
Six real, independently-computed substrate signals already exist at the M3 reducer level
(`orion/substrate/transport_loop/extract.py::compute_transport_pressures()`) — stream depth,
backpressure, catalog drift, contract/schema mismatch, observer failure, and a composite
reliability pressure — all derived from real Redis-stream-observer grammar events, not invented.
Live data confirms all six read as zero or near-zero over every window checked, and — separately
— one of them (`stream_depth_pressure`) is calibrated against a threshold roughly 1,000x the
system's real observed operating range. Neither problem has been fixed or even measured with a
real historical baseline before now.

This spec proposes four read-only measurement steps (a fifth, wiring the four unused signals
into the Hub lattice console, is deferred to the appendix as it's a different kind of change —
surfacing existing data, not new measurement) to establish real ground truth before anything
gets recalibrated, composed, or wired into a live consumer. Per
`orion/sentience_striving_program/README.md` §7's "measure before minting" rule — the same
discipline that caught `autonomy`'s dead origination signal (PR #1156) before two more weeks
were spent on it.

## Current architecture

**What's built and real:**

- `orion/substrate/transport_loop/extract.py::compute_transport_pressures()` computes six
  distinct pressure signals from real grammar events (`bus_stream_depth_observed`,
  `bus_backpressure_observed`, `bus_configured_stream_uncataloged`, `bus_schema_validation_failed`,
  `bus_observer_tick_failed`, all emitted by `services/orion-bus`'s bus observer):
  - `stream_depth_pressure = min(max_stream_depth / DEFAULT_STREAM_DEPTH_CRITICAL, 1.0)`,
    where `DEFAULT_STREAM_DEPTH_CRITICAL = 100_000` (`orion/substrate/transport_loop/constants.py`)
  - `backpressure = min(backpressure_count / streams_observed, 1.0)`
  - `catalog_drift_pressure = min(uncataloged_stream_count / streams_observed, 1.0)`
  - `contract_pressure = min(schema_mismatch_stream_count / streams_observed, 1.0)`
  - `observer_failure_pressure = 1.0 if observer_failure_count > 0 else 0.0`
  - `reliability_pressure = max(observer_failure_pressure, 1.0 - delivery_confidence)`
- Only two of the six (`stream_depth_pressure`, `backpressure`) feed the aggregate
  `transport_pressure = max(stream_depth_pressure, backpressure)`
  (`extract.py:105`), which is one of `transport_prediction_error()`'s three diffed inputs
  (`orion/substrate/prediction_error.py:65-79`). The other four exist as real, computed fields on
  `TransportBusStateV1` (`orion/schemas/transport_projection.py`) but are not read out anywhere
  beyond the raw DB row.
- Real live data checked 2026-07-22 (both `substrate_transport_bus_projection`, a **singleton**
  upsert table with no history, and `substrate_reduction_receipts`, append-only but
  **short-TTL** — only ~35 minutes / 195 receipts survived at query time despite a 24h window
  request):
  - `max_stream_depth`: exactly `91` on every single receipt in the observed window (zero
    variance).
  - `backpressure_count`, `observer_failure_count`, `uncataloged_stream_count`,
    `schema_mismatch_stream_count`: all `0` across every receipt checked.
  - `transport_pressure`/`stream_depth_pressure`: both read `0.00091` (91 / 100,000) —
    real, non-fabricated, but numerically indistinguishable from noise given the threshold's
    scale.
- Separately, `orion/field/pressure.py` (formerly `orion/self_state/scoring.py`) and
  `services/orion-field-digester/README.md`'s own per-channel audit found `transport_pressure`
  "fully unproduced" in the field-digester's downstream channel corpus specifically — consistent
  with the above: a value that has only ever read as `~0` never clears
  `collect_field_channel_pressures()`'s `channel in PRESSURE_CHANNELS or value > 0` inclusion
  test (`orion/field/pressure.py`), so it never appears in the corpus at all, even though it is
  being computed correctly upstream at M3.

**What this is not**: not a data-quality bug on the scale of `chat_prediction_error()`'s (which
structurally can never diff a new turn against anything) or `route_prediction_error()`'s (an
unexplained subnormal-float corruption). The transport bus signals read as flat because the
bus has, as far as every window checked shows, genuinely been quiet — combined with a
stream-depth threshold that was never checked against real operating data.

## Missing questions

1. **Has `substrate_transport_bus_projection` ever been given an append-only companion table**,
   the same fix already applied to `substrate_attention_broadcast_projection`
   (`services/orion-substrate-runtime/app/store.py::save_attention_broadcast_history()`,
   `orion/sentience_striving_program/README.md` §6 item 2)? Without one, no measurement here can
   see further back than `substrate_reduction_receipts`' short TTL allows (~35 minutes observed).
   This is very likely the actual blocking prerequisite for idea 1 (percentile baseline) and
   idea 3 (reducer-cadence gaps).
2. **Does `orion-bus`'s Redis layer expose longer-lived metrics independent of the grammar-event
   pipeline** (`XINFO STREAM`, `XLEN` history, or similar) that could backstop a real baseline
   faster than waiting for a new append-only table to accumulate history from scratch?
3. **Has this deployment had a real transport incident, ever** — a Redis restart, an actual
   backpressure event, a genuine catalog/schema drift? Or is "it genuinely has not, not once"
   itself the finding — the same shape as `autonomy`'s dead origination signal (PR #1156, fired
   zero times across 84,511 replayed ticks)? This determines whether any of the ideas below can
   ever be validated against ground truth, or whether they remain permanently theoretical until
   an incident occurs naturally.
4. **Is `DEFAULT_STREAM_DEPTH_CRITICAL = 100_000` actually the value used in production**, or is
   it overridden by config/env anywhere not yet checked? (`compute_transport_pressures()`'s
   `stream_depth_critical` parameter defaults to this constant but is a real parameter — grep for
   call-site overrides before assuming the default is what's live.)

## Proposed schema / API changes

None. Every item below is a read-only analysis script or a single log line — no schema, no bus
channel, no consumer wiring. Per the metric quality gate (CLAUDE.md §0A), no new signal is
proposed as "real substrate" in this patch; this is entirely the trace-provenance and
live-data-sanity steps, run before any such proposal would be made.

## Files likely to touch

Four items, in recommended sequence (see "Recommended next patch"):

1. **Real-incident capture** (lowest risk, do first): one log line in
   `services/orion-substrate-runtime/app/worker.py`'s transport tick, firing when any of the six
   raw counters/pressures goes nonzero for the first time in a run, so the next real incident —
   whenever it occurs — gets noticed instead of scrolling past unlogged on an otherwise-flat
   dashboard.
   - Files: `services/orion-substrate-runtime/app/worker.py`.
2. **Real historical baseline for `stream_depth_pressure`'s threshold**: a read-only script,
   same shape as `scripts/analysis/measure_origination_gate.py` and
   `scripts/analysis/measure_emergent_clustering_probe.py`, reporting real percentiles of
   `max_stream_depth` and totals of the other five raw counters over the longest available
   window. Blocked on Missing Question 1/2 if the receipts TTL turns out to be the only source —
   may need to start an append-only capture now and re-run this script in a week rather than
   trying to retroactively mine history that was never retained.
   - Files: new `scripts/analysis/measure_transport_bus_signal_history.py`.
3. **Reducer-cadence gap analysis**: read-only script measuring the real inter-arrival gap
   between consecutive `transport_bus_reducer` receipts (`substrate_reduction_receipts`,
   `reducer_name = 'transport_bus_reducer'`) against its expected ~10s cadence (observed live) —
   an orthogonal "is the reducer keeping up with its own perception" signal, independent of what
   it's reducing.
   - Files: new script, or a mode added to item 2's script if the query shape overlaps enough
     to share a file — decide once item 2 is scoped.
4. **Correlation probe against a known-live domain**: reuses
   `scripts/analysis/measure_emergent_clustering_probe.py`'s methodology to test whether
   transport's (currently tiny, currently-constant) real values correlate with anything else
   real, before any recalibration or composition work is justified as worthwhile.
   - Files: new sibling script alongside `measure_emergent_clustering_probe.py`, or an added
     mode to it if the correlation machinery is directly reusable — decide once item 2/3 give a
     real data shape to correlate.

## Non-goals

- **Not recalibrating `DEFAULT_STREAM_DEPTH_CRITICAL` in this patch.** Changing a threshold
  without first measuring a real percentile baseline is the same hand-tuned-guess anti-pattern
  the self-state/drives burns already corrected for elsewhere — this spec proposes the
  measurement, not the new number.
- **Not composing a new `transport_pressure`-replacement formula in this patch.** Any recombination
  of the six real signals into something new is explicitly deferred until items 1-4 above produce
  real data to design it from.
- **Not wiring anything into the Hub lattice console, `transport_prediction_error()`, or any L6
  replacement.** That question is explicitly deferred to a follow-up turn per Juniper's own
  sequencing ("let's answer the L6 replacement strategy... in our next turn").
- **Not touching `DriveEngine`, `tensions.py`, or any halted drives-system code.** Per
  `orion/sentience_striving_program/README.md` §8, that system receives no further development;
  none of the ideas above route through it.
- **Not adding an append-only table for `substrate_transport_bus_projection` in this patch**,
  even though Missing Question 1 flags it as a likely prerequisite — that's its own scoped
  change (mirroring the `substrate_attention_broadcast_log` migration) and needs its own
  sign-off once the measurement scripts above confirm it's actually needed.

## Acceptance checks

- Item 1 (incident-capture log line): a manual test firing a synthetic nonzero value through
  `compute_transport_pressures()` produces the expected log line; no behavior change when all
  six signals are zero (today's steady state).
- Item 2 (historical baseline script): runs read-only against real Postgres data, produces a
  percentile report, and explicitly states whether the observed window ever contained a nonzero
  value for any of the six signals — `UNVERIFIED`/absent is an acceptable, honest output, not a
  reason to fabricate a number.
- Item 3 (cadence gap script): reports a real gap distribution against the observed ~10s
  cadence; flags (not silently drops) any gap exceeding some multiple of the median as a
  candidate reducer-stall event.
- Item 4 (correlation probe): reports real correlation coefficients or explicitly states "no
  correlation computable, insufficient real variance in transport's signal" if that's what the
  data shows — same honesty standard as the emergent-clustering probe's own AMBIGUOUS/NOT MET
  verdicts.
- None of the four scripts write to any table, publish any bus event, or change any live
  consumer's behavior. All are inspectable via their own `/tmp/<script-name>/report.md` output,
  per CLAUDE.md §14's read-only-analysis monitoring convention.

## Recommended next patch

1. Item 1 (incident-capture log line) first — lowest risk, immediate value, and the single
   highest-priority gap (right now a real incident would pass completely unnoticed).
2. Item 2 (historical baseline) in parallel or immediately after — but check Missing Question 1
   first; if the receipts TTL really is the only available history, the honest output of this
   script for a while will be "insufficient history, needs N more days of accumulation," which
   is itself useful information, not a failure.
3. Items 3 and 4 after item 2 gives a real data shape to correlate against and measure cadence
   from — sequencing them earlier would mean measuring against a baseline that doesn't exist yet.
4. Re-open the L6-replacement question (deferred per this spec's own scope) once items 1-2 have
   run for at least a few days and either (a) show a real incident, in which case there's real
   ground truth to design a signal against, or (b) confirm the quiet period continues, in which
   case the honest answer for L6 may simply be "there is currently nothing to show here, and that
   itself is disclosed" rather than any constructed signal.

---

## Appendix: deferred idea — surface the four unused signals in the Hub lattice console

Not included in this patch's scope; recorded here so the idea isn't lost.

**What**: `catalog_drift_pressure`, `contract_pressure`, `observer_failure_pressure`, and
`reliability_pressure` are already computed and stored on `TransportBusStateV1` every tick
(`orion/substrate/transport_loop/extract.py`) but nothing reads them out beyond the raw
`substrate_transport_bus_projection`/`substrate_reduction_receipts` rows. `services/orion-hub/
scripts/substrate_lattice_routes.py`'s M3 layer (`_load_transport_proof_chain()`) already reads
the projection row and could include these four fields in its `values` dict with no new
computation — they exist, they're just not surfaced.

**Why it's an appendix, not part of this patch**: this is a "expose existing data" change, not a
measurement step — it doesn't need a read-only analysis script, a historical baseline, or any
new provenance tracing (the provenance is already established: real grammar events, already
audited above). It's a different kind of unit of work than items 1-4, and doing it now would mean
displaying four signals that have, per this spec's own live-data findings, never once read
nonzero in any observed window — informative as an honest "here's the full picture, and it's
currently all quiet" disclosure, but not blocking or blocked by the measurement work above.

**Smallest buildable version**: add four keys to the `l6_values`... no — to the **M3** `values`
dict in `_load_transport_proof_chain()` (M3, not L6 — L6 itself is gone per PR #1272). No schema
change, no new query — the data is already in the row being read.

**Files likely to touch**: `services/orion-hub/scripts/substrate_lattice_routes.py`,
`services/orion-hub/static/js/substrate-lattice.js` (render the four additional fields in the
existing M3 card), `services/orion-hub/tests/test_substrate_lattice_routes.py` (assert the new
keys are present in the M3 values dict).
