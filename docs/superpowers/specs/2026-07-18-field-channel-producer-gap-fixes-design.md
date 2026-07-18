# Field-channel producer gap fixes — a design spec, not a taxonomy re-audit

Status: design mode. No code changes proposed in this document. Catalogs the real,
confirmed-via-live-data producer gaps found across this investigation
(`scripts/analysis/measure_capability_channel_health.py`, PR #1171, and
`services/orion-field-digester/README.md`'s Field channel glossary) — distinct from the
larger set of findings that turned out to be by-design, already-benign, or already
explained. This is not a re-litigation of PR #1171; it's the fix list that survives after
cross-checking every "dead" finding against the glossary.

## Arsonist summary

PR #1171 found most channels feeding `capability:transport`/`capability:llm_inference` are
structurally dead. Cross-checking against `services/orion-field-digester/README.md`'s
channel glossary showed almost all of that is by-design (a channel simply has no diffusion
edge into that particular capability) or already-documented-benign (one-way ratchets, a
genuinely-rare-but-correctly-wired event). Four items survive that filter as real, unfixed
gaps — a producer that should exist and doesn't, not a channel that was never meant to fire
for that target. This spec names them, root-caused against live data, prioritized by
blast radius and effort.

## Current architecture

Pipeline order (`app/tensor/update_rules.py::run_digestion_tick`): reconcile → 
`apply_perturbations` → `apply_decay` → `apply_diffusion` → `apply_suppression` → 
`check_field_coherence` → `collect_field_channel_pressures`. Perturbations originate from
`app/ingest/state_deltas.py`'s six `target_kind` blocks: `active_node_pressure`,
`node_biometrics`, `execution_run`, `chat_turn`, `transport_bus`, `prediction_signal`. A
channel is only ever populated if some upstream producer actually emits a delta targeting it
by its exact hint key — the topology YAML's diffusion edges describe how a value propagates
*once it exists*, they don't create the value.

## Missing questions

1. Is `node:atlas`'s `reasoning_load` gap (item 1 below) already being worked on elsewhere?
   Juniper has flagged a separate agent actively on "the pressure thing" this session —
   confirm scope before starting, this spec does not claim ownership of that fix.
2. Does `node:atlas` (the GPU-hosting node per the topology, feeding `capability:llm_inference`)
   actually run LLM-reasoning workloads that should be emitting `execution_run` deltas with
   reasoning-load telemetry, or is `reasoning_load` simply the wrong signal to expect from
   that node at all? Needs tracing at the `GrammarEventV1`/execution-trajectory producer
   level, not assumed.
3. **Resolved 2026-07-18, differently than originally framed.** The hint keys are already
   populated correctly by a real, live pipeline — the gap is that `BUS_OBSERVER_STREAMS`
   watches two nonexistent Redis keys (`orion:evt:gateway`, `orion:bus:out`), confirmed via
   direct `XLEN`/`SCAN`. New open question: which of the mesh's real active streams
   (`orion:stream:world_pulse:run:result`, `orion:pad:events`/`orion:pad:frames`,
   `orion:queue:spark:introspection` are live candidates, none yet vetted) actually
   represent genuine bus congestion, as opposed to normal persistent-queue size? See Fix 2
   for full detail — needs domain input, not a guess from this investigation.

## Proposed fixes, prioritized

### Fix 1 (highest blast radius, flag scope first): `node:atlas`'s `reasoning_load` never populated

**Confirmed live** (fresh 1h window, 2026-07-18): `node:atlas`'s `reasoning_load` is exactly
`0.0`, `std=0` — never instrumented, not decayed from a real value. This is the direct root
cause of `capability:llm_inference`'s `reasoning_pressure` reading permanently `0.0`
(confirmed in the same check), which is one of the two channels most responsible for
`capability:llm_inference`'s low field-attention salience (PR #1169's original finding).

**Fix approach**: trace whether `node:atlas` actually runs LLM-inference workloads that
should be surfacing `GrammarEventV1`s with reasoning-load telemetry, and if so, find where
that telemetry currently dead-ends before reaching `state_deltas.py`'s `execution_run` block.
Do not assume the fix is in `orion-field-digester` itself — the gap is more likely upstream,
in whatever service should be emitting the grammar event in the first place
(`orion-llamacpp-host`/`orion-llm-gateway` are the likely candidates given `node:atlas`'s
role in the topology).

**Files likely to touch**: whichever service actually runs inference on `node:atlas`
(grammar emission), `orion/substrate/execution_loop/reducer.py` (if the reducer itself is
dropping the field rather than the producer never sending it).

### Fix 2: `capability:transport`'s `pressure` channel dead — root cause corrected (2026-07-18)

**Correction to the original framing above.** This was not a producer gap in the sense of
"nothing computes this channel." Traced end-to-end: `orion/substrate/transport_loop/
reducer.py`'s `reduce_transport_trace_events()` runs continuously (`ENABLE_TRANSPORT_BUS_REDUCER=true`,
180 real receipts confirmed over 15 days), and `extract.py`'s `compute_transport_pressures()`
does correctly compute `transport_pressure = max(stream_depth_pressure, backpressure)` from
real, live `bus_stream_depth_observed`/`bus_backpressure_observed` grammar events (488,268+
of the former exist). The pipeline is not broken.

**The real root cause**: `services/orion-bus/.env`'s `BUS_OBSERVER_STREAMS=orion:evt:gateway,
orion:bus:out` names two Redis stream keys that do not exist. Confirmed directly —
`redis-cli XLEN orion:bus:out` and `redis-cli XLEN orion:evt:gateway` both return `0`, and
neither key appears in a live `redis-cli --scan --pattern "orion:*"` at all. Every
`bus_stream_depth_observed` event sampled from these keys honestly reports
`stream_length=0` because the keys are empty/nonexistent, not because the bus is healthy —
confirmed by checking the full 15-day receipt history: `transport_pressure`/
`stream_depth_pressure`/`backpressure` have never once been nonzero, zero variance across
180 receipts from 2026-07-03 to 2026-07-18. This matches a pre-existing finding already on
the agent board from earlier this session ("`orion:evt:gateway`/`orion:bus:out`
(`BUS_OBSERVER_STREAMS` original defaults) point at non-existent Redis keys, depth/
backpressure monitoring likely dead since it shipped") — this is that finding, confirmed
and root-caused, not a new discovery.

**A second, separate issue found in the same config**: `services/orion-bus/.env_example`
declares `BUS_OBSERVER_STREAMS=orion:evt:gateway,orion:bus:out,orion:grammar:event,
orion:stream:world_pulse:run:result` (4 keys) — live `.env` only has the first two (2 keys).
Real drift, independent of the dead-key issue. Restoring parity alone is not sufficient
though: `orion:grammar:event` was also checked live (`XLEN=0`) — also currently empty.

**Real, active, nonzero streams confirmed live** (candidates, not a decision):
`orion:stream:world_pulse:run:result` (XLEN 83), `orion:pad:events`/`orion:pad:frames`
(500 each — possibly `MAXLEN`-capped, worth checking before treating 500 as a real depth
signal), `orion:queue:spark:introspection` (XLEN 22,077 — large; could be a genuine
backlog/congestion signal worth monitoring, or could be an expected-size persistent queue
that would misrepresent normal operation as backpressure if watched naively).

**Fix approach — two parts, not guessed together as one patch**:
1. Restore env parity (`.env` → the 4 keys already declared in `.env_example`) — cheap,
   safe, uncontroversial, but not sufficient alone since two of those four are also
   currently empty.
2. Determine which Redis streams actually represent genuine bus congestion/backpressure
   semantics — this needs domain input on `orion-bus`'s real architecture, not a guess from
   this investigation. Picking an active-but-semantically-wrong stream (e.g. treating
   `orion:queue:spark:introspection`'s size as bus congestion when it might just be a normal
   persistent queue) would relocate the same class of bug rather than fix it.

**Files likely to touch**: `services/orion-bus/.env` (parity fix, immediate),
`services/orion-bus/.env_example` (if the real correct stream set differs from what's
already declared there — needs part 2's answer first), whichever service/doc defines what
`orion-bus`'s actual message-passing streams are (not yet traced in this investigation).
`StateDeltaV1`s (likely `orion-substrate-runtime` or a bus-observer-adjacent service — not
yet traced).

### Fix 3: One-way-ratchet channels have no decay path (`expected_offline_suppression`, `delivery_confidence`, `bus_health`)

**Confirmed** (glossary): all three are `mode=add` with no entry in `NODE_DECAY_CHANNELS`,
so they can only increase, never decrease. "Currently benign since the bus is genuinely
stable" per the glossary, but structurally this means a real outage/degradation could never
be reflected as improving once it resolves — and it has a real downstream cost already:
`availability`/`staleness` are both permanently suppressed/floored because
`expected_offline_suppression` is latched near `1.0`.

**Fix approach**: add these three channels to `NODE_DECAY_CHANNELS` with an appropriate
decay rate, or (if a hard reset is more semantically correct than gradual decay for a
suppression-style flag) add an explicit reset condition when the underlying condition
resolves (e.g. a node confirmed back online clears `expected_offline_suppression`
immediately rather than waiting on decay). The second option avoids reproducing the
decay-vs-injection-interval mismatch bug class this service has already hit twice.

**Files likely to touch**: `services/orion-field-digester/app/digestion/decay.py`
(`NODE_DECAY_CHANNELS`), possibly `app/ingest/state_deltas.py` if a reset condition is added
instead of pure decay.

### Fix 4 (lower priority, already separately tracked — named for completeness, not detailed here)

`contract_pressure`/`catalog_drift_pressure`'s exact-duplicate values — the glossary already
flags this as "not confirmed... a separate investigation is tracking the actual root cause."
Not re-scoped in this document; do not duplicate that work.

### Fix 5 (lower priority): folded-away biometrics channels never reach the field by their own name

`memory_pressure`, `thermal_pressure`, `disk_pressure` are computed individually in
`orion/telemetry/biometrics_pipeline.py` but only ever reach the field as part of the
composite `"strain"` scalar — `state_deltas.py`'s `node_biometrics` block only reads
`hints["gpu"]`/`hints["strain"]`, never a memory/thermal/disk-named hint key, even though
the pipeline already computes those three values separately.

**Fix approach**: add three explicit hint keys (mirroring how `"gpu"` already gets its own
key alongside the composite `"strain"`) in `biometrics_pipeline.py`'s emitted hints dict, and
confirm `state_deltas.py`'s `node_biometrics` block reads them.

**Files likely to touch**: `orion/telemetry/biometrics_pipeline.py`,
`services/orion-field-digester/app/ingest/state_deltas.py`.

## Non-goals

- Not touching anything already explained as by-design in the glossary (e.g. `execution_pressure`
  only applying to `orchestration`, not `transport`/`llm_inference` — intentional topology).
- Not touching the halted drives system (`orion/spark/concept_induction/`).
- Not re-investigating Fix 4's root cause — explicitly tracked elsewhere.
- Not implementing any fix in this document — sign-off per `CLAUDE.md` §0A required first,
  same as every other proposal in this program.

## Acceptance checks (per fix, once implemented)

- Fix 1: `node:atlas`'s `reasoning_load` shows real, nonzero, varying values in
  `substrate_field_state` over a live window; `capability:llm_inference`'s
  `reasoning_pressure` correspondingly moves off `0.0`.
- Fix 2: with `BUS_OBSERVER_STREAMS` pointed at real, vetted-as-congestion-relevant stream
  keys, `bus_stream_depth_observed`/`bus_backpressure_observed` events report nonzero
  `stream_length` at least some of the time over a live window (not permanently `0`);
  `transport_pressure`/`capability:transport`'s `pressure` channel show real variance,
  breaking the 15-day zero-variance streak confirmed 2026-07-18.
- Fix 3: `availability`/`staleness` are no longer permanently floored/zeroed;
  `expected_offline_suppression`/`delivery_confidence`/`bus_health` show real movement in
  both directions over a live window that includes a real degrade-then-recover event.
- Fix 5: `memory_pressure`/`thermal_pressure`/`disk_pressure` appear as real values in the
  live corpus, not folded into `strain`.

Each acceptance check should be run the same way every measurement in this investigation
has been — live data, not test-only, using the existing analysis scripts
(`measure_capability_channel_health.py`, `measure_capability_salience_coupling.py`) as the
verification instrument, not a new one per fix.

## Recommended next patch

Fix 1, if not already claimed by the agent Juniper has working "the pressure thing" —
highest blast radius (directly caps `capability:llm_inference`'s salience, the thread this
whole investigation started from), and the root cause is already precisely traced, not a
guess. If Fix 1 is already spoken for, Fix 3 is the next best pick: lower effort (a
`NODE_DECAY_CHANNELS` config change plus, possibly, a reset-condition addition), self-
contained, and has a clean, checkable acceptance test (a real degrade/recover cycle showing
up correctly in both directions).

## Source material

- `scripts/analysis/measure_capability_channel_health.py` (PR #1171) — the measurement that
  found the raw dead-channel list this spec filters down.
- `services/orion-field-digester/README.md`'s "Field channel glossary" — the ground truth
  this spec cross-checks every item against; re-read before touching any of the above.
- `orion/sentience_striving_program/README.md` — the program this fix work serves; Objective
  2 (capability-budget-to-salience coupling) is directly blocked on Fix 1 and Fix 2 landing.
