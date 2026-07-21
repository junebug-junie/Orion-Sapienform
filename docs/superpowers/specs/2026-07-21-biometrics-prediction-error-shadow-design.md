# Biometrics prediction-error shadow instrument

Status: design mode + implemented in the same PR (root `CLAUDE.md`'s implementation-mode
follow-through requirement — this doc captures the decision record, the code lands alongside
it, not as a separate follow-up). Shadow-measure only, per the Sentience Striving Program
charter (`orion/sentience_striving_program/README.md`) §6 item 3: "Phased: shadow-measure
one producer domain before migrating any live."

## Arsonist summary

Charter §9b item 3 (Predictive Processing/Active Inference) named `execution_prediction_
error()`/`transport_prediction_error()` as the real field-native substrate for this theory
thread, and left an open question: "whether the other three reducers (biometrics, chat,
route) have equivalent instrumentation, or whether coverage is genuinely incomplete." This
patch answers that for one of the three — `biometrics` — by adding a third instrument,
`biometrics_prediction_error()`, that mirrors the already-live pattern exactly: diff two
successive `NodeBiometricsProjectionV1` snapshots, produce a 0-1 surprise score, and (when
`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true`) write it onto a durable `node:substrate.
biometrics` node using the same shared `_write_prediction_error_node()` writer the other two
already use. Chat and route remain open.

## Naming-mistake context (why this matters here)

A prior draft of this same task named the new signal `coherence_prediction_error`, sourced
from `turn_effect` — a `SelfStateV1`/`concept_induction`-adjacent field, in direct violation
of charter §7's standing rule ("field-native only — no `SelfStateV1`-anchored substrate for
new instrumentation") and §6 item 3's explicit reframing ("not a port of `tensions.py`'s
hand-classified kind vocabulary onto field channels"). That draft was corrected before this
patch. This instrument is named for its actual source domain (`biometrics`, the reducer key
in `REDUCER_SPECS[0]`), reads only `NodeBiometricsProjectionV1` (a reducer projection built
from `orion-biometrics` grammar events, never `SelfStateV1`), and the diff was run to confirm
zero references to `concept_induction`, `turn_effect`, `SelfStateV1`, or `self_state_id`
anywhere in the new code.

## Current architecture

- **Producer:** `services/orion-substrate-runtime/app/worker.py`'s `_tick()`
  (`REDUCER_SPECS[0]`, `reducer_key="biometrics"`) — fetches `orion-biometrics` grammar
  events, runs `process_biometrics_grammar_events()`, saves via `self._store.
  save_node_biometrics`, projection type `NodeBiometricsProjectionV1`
  (`orion/schemas/biometrics_projection.py`), loaded via a local `load_node_bio()` closure
  keyed on `NODE_BIOMETRICS_PROJECTION_ID`.
- **Existing pattern (unchanged):** `orion/substrate/prediction_error.py`'s
  `execution_prediction_error()`/`transport_prediction_error()`, wired into `_execution_tick`/
  `_transport_tick` (lines ~1704-1758, ~1824-1880 pre-patch). Both diff two successive
  projection snapshots per-entity (per `trace_id` / per `bus_id`) over a *fixed* small key
  set, average the absolute deltas, and scale by `min(1.0, mean/0.30)`.
  `_write_prediction_error_node()` (`app/worker.py` ~line 691) is the shared durable writer:
  it internally checks `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`
  (`_prediction_error_nodes_enabled()`, line 727 — the flag check lives inside the shared
  method, not duplicated by each caller) and is otherwise fail-open (never raises out of a
  tick).
- **Current gap:** no third instrument existed for the biometrics reducer before this patch.

## Why a dynamic key set, not a fixed four-key list like execution's

Read `orion/substrate/biometrics_loop/grammar_extract.py::extract_node_state_from_events()`
(the actual producer of `pressure_hints` for biometrics nodes) before deciding this — the
task's own instruction was not to assume. Unlike execution's four fixed keys
(`execution_load`, `execution_friction`, `failure_pressure`, `reasoning_load`, present on
every run), biometrics `pressure_hints` keys are populated *conditionally per node role*:

- `strain` — set whenever a `body_state` atom carries a `salience` value (most nodes).
- `gpu` — set only when the node's catalog profile has `capabilities.local_llm_heavy`
  (GPU inference nodes).
- `memory_pressure` / `thermal_pressure` / `disk_pressure` — each set only when the matching
  pressure-signal atom (`memory_pressure_signal` / `thermal_pressure_signal` /
  `disk_pressure_signal`) is present in the event batch.

Confirmed live against real data (`substrate_node_biometrics_projection`, queried directly via
`psql -h localhost -p 55432 -U postgres -d conjourney`, 2026-07-21T03:29 UTC row):

- `atlas` (role `inference_gpu`, `local_llm_heavy` capability): `pressure_hints = {"gpu": 0.8,
  "strain": 0.076}`.
- `circe` (role `burst_gpu`): `pressure_hints = {"gpu": 0.8, "strain": 0.099}`.
- `athena` (role `orchestration`, no `local_llm_heavy`): `pressure_hints = {"strain": 0.183,
  "disk_pressure": 0.063, "memory_pressure": 0.254, "thermal_pressure": 0.429}`.

No single fixed key list covers every node's real key set. `biometrics_prediction_error()`
therefore diffs the *union* of keys present on either side of a given node
(`set(prev.pressure_hints) | set(curr.pressure_hints)`), defaulting an absent key to `0.0` —
the same default-to-zero behavior `execution_prediction_error` already uses for a key that
happens to be missing on one side, just applied to a key set that is itself dynamic instead
of fixed. This is documented directly in the function's docstring, not just here.

## Metric quality gate (root `CLAUDE.md` §0A) — findings

Run fresh for this new metric, not inherited from execution/transport's prior pass.

1. **Trace provenance to real code.** `biometrics_prediction_error(prev, curr)` in
   `orion/substrate/prediction_error.py` (new function). Called from `_tick()` in
   `services/orion-substrate-runtime/app/worker.py`, which loads `prev_projection` via
   `load_node_bio()` *before* `process_biometrics_grammar_events()` runs and
   `curr_projection` via the same closure immediately after, when `last_id is not None`. The
   underlying `pressure_hints` values are traced one level further back to
   `extract_node_state_from_events()` (`orion/substrate/biometrics_loop/grammar_extract.py`
   lines 105-125), which reads `atom.salience` off real `orion-biometrics` grammar events.
2. **Independence check.** Not redundant with `execution_prediction_error`/
   `transport_prediction_error`: those diff `ExecutionTrajectoryProjectionV1` (per-`trace_id`
   execution-load/friction/failure/reasoning pressure from `orion-cortex-exec` grammar
   events) and `TransportBusProjectionV1` (per-`bus_id` bus health/delivery/transport
   pressure from `orion-bus` grammar events), respectively. `NodeBiometricsProjectionV1` is
   built from an entirely separate event stream (`orion-biometrics`, hardware/infra
   telemetry — CPU/GPU/memory/thermal/disk), a separate reducer (`REDUCER_SPECS[0]`, not
   `[1]`/`[2]`), and a separate node-keying scheme (physical/logical node id, not trace/bus
   id). The causal chain from "this host got hot" to "an execution run's reasoning_load
   changed" is real but long and indirect (mediated by, at minimum, thermal throttling
   affecting inference latency) — not a monotonic transform of an already-included signal.
   This is a genuinely separate sensor and a genuinely separate upstream computation, not
   redundant signal.
3. **Theory anchor.** Same anchor already used for execution/transport, charter §9b item 3:
   Predictive Processing / Active Inference — a 0-1 surprise score computed as the magnitude
   of change in a reducer's own pressure-hint state between successive observations, written
   onto `FieldStateV1` nodes so `orion/substrate/dynamics.py::prediction_error_pressure()`
   can seed activation pressure from real, measured surprise rather than a hand-asserted
   value. The biometrics domain is a natural fit for this same anchor: infra load spiking or
   dropping unexpectedly is exactly the kind of "prediction violated" event the theory
   anchors to, no new theoretical justification needed beyond what already grounds the other
   two instruments.
4. **Live-data sanity check.** Confirmed non-degenerate: the three real node rows quoted
   above carry distinct, non-zero, non-saturated float values across different key sets —
   not flat, not always-null, not always-1.0. `substrate_node_biometrics_projection` is a
   **singleton upsert row** (`projection_id` primary key), same storage shape as execution/
   transport's own projection tables — there is no history table to replay multiple
   historical deltas against (the same structural gap the AST/HOT reducer's Phase 1 status
   note in the charter §6 item 2 already disclosed for a different singleton table). This
   patch does not fabricate a historical replay: the live-data check here is "current
   snapshot is real and non-degenerate," not "N historical deltas were replayed and looked
   reasonable." Honest limitation, stated plainly rather than glossed over.
5. **Existing-mechanism check.** `grep -rn "biometrics_prediction_error"` across the repo
   before this patch: zero hits outside this patch's own new code. No prior instrument for
   this signal exists.
6. **Reversibility.** Fully reversible: gated behind the existing
   `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` flag (no new flag — see below), writes to a
   single fixed `node_id` (`node:substrate.biometrics`) that collapses on repeat writes (no
   unbounded growth), and the function itself is a pure diff with no persisted schema/
   manifest dependency. Deleting `biometrics_prediction_error()` and its two call sites in
   `_tick()` would leave execution/transport untouched and would not break any consumer,
   since `orion/substrate/dynamics.py::prediction_error_pressure()` reads generically off
   `node.metadata.get("prediction_error")` for whatever nodes exist — removing this node
   simply removes one more seed source, not a hardcoded dependency.

## Shadow-only confirmation

`orion/substrate/dynamics.py::_compute_pressures()` (called from `SubstrateDynamicsEngine.
tick()`) iterates `nodes.values()` generically and calls `prediction_error_pressure(node,
...)` for every node in the graph — it is **not** hardcoded to `node:substrate.execution`/
`node:substrate.transport`. This was confirmed by reading the function body (`orion/
substrate/dynamics.py` lines 201-220), not assumed. This means `node:substrate.biometrics`
is automatically picked up by the same already-live mechanism the other two instruments use
once written — this is the existing wire, not a new consumer being added. No change was made
to `capability_policy.py`, `top_down.py`, or `goal_context.py`, and no new bus consumer was
wired.

## Proposed schema / API changes

- `orion/substrate/prediction_error.py`: new function `biometrics_prediction_error(prev:
  NodeBiometricsProjectionV1, curr: NodeBiometricsProjectionV1) -> float`. No schema changes
  — `NodeBiometricsStateV1.pressure_hints: dict[str, Any]` already exists.
- `services/orion-substrate-runtime/app/worker.py`'s `_tick()`: captures `prev_projection =
  load_node_bio()` before `process_batch` runs; after a non-`None` `last_id`, reloads
  `curr_projection`, computes `error`, and on `error > 0.0` saves a `_prediction_error_
  receipt(reducer_key="node_biometrics", node_id="node:substrate.biometrics", ...)` and calls
  `self._write_prediction_error_node(node_id="node:substrate.biometrics", ...,
  reducer_key="node_biometrics", contributing_id=last_id)`. `_tick()`'s return signature
  (`tuple[str | None, list[GrammarEventV1]]`) is unchanged.
- No new env key. Reuses `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` (already `true` in
  `.env_example`) — this is the same generic "write prediction-error nodes" switch execution/
  transport already share, not a per-domain flag.

## Files likely to touch (this patch's actual diff)

- `orion/substrate/prediction_error.py` — new `biometrics_prediction_error()`.
- `orion/substrate/tests/test_prediction_error.py` — new unit test file (no prior dedicated
  test file for this module existed to mirror; found via repo-wide grep before writing).
- `services/orion-substrate-runtime/app/worker.py` — `_tick()` wiring, import update.
- `services/orion-substrate-runtime/README.md` — document the third instrument alongside
  execution/transport in the "Dynamics tick" / "Unified turn bus listeners" sections.
- `orion/sentience_striving_program/README.md` — §6 item 3 status note, §9b item 3 open-
  question line updated (biometrics now covered, chat/route remain open).
- This doc.

## Non-goals

- Not migrating any producer off `tensions.py`'s bucket-vote layer — that is a separate,
  later step in charter §6 item 3, gated on every producer domain first being shadow-
  measured.
- Not adding a chat or route instrument — explicitly out of scope per the task; the charter's
  open-question line is updated to say only biometrics is now covered.
- Not touching `capability_policy.py`, `top_down.py`, `goal_context.py`, or any bus consumer.
- Not adding a new env key.
- Not building a historical-replay script for this instrument (unlike `measure_origination_
  gate.py`/`measure_ast_hot_reducer.py`) — the singleton-table limitation named above makes a
  meaningful replay impossible today; a follow-up companion to the append-only
  `substrate_attention_broadcast_log` pattern (charter §6 item 2) would be needed first.

## Acceptance checks

- `pytest orion/substrate/tests/test_prediction_error.py -q` passes (zero delta, partial
  delta, disjoint key sets, multi-node averaging, empty projections, node absent from prev,
  clamp-to-1.0).
- `_tick()`'s existing return signature and poison-isolation behavior are unchanged for
  callers (`_biometrics_poll_loop`) — confirmed by running the full
  `services/orion-substrate-runtime/tests/` suite (excluding the always-broken, pre-existing
  `test_grammar_consumer_integration.py` module-collection error, unrelated to this service's
  own code) with this patch applied and with it reverted: both produce the identical
  `13 failed, 137 passed, 9 errors`. This patch changes zero pre-existing pass/fail outcomes —
  see the PR report for the full pre-existing-failure list, none of which reference
  `biometrics_prediction_error`, `_tick`, or `NodeBiometricsProjectionV1`.
- Grep of the final diff for `concept_induction`, `turn_effect`, `SelfStateV1`,
  `self_state_id`: zero hits.
- Live check: `node:substrate.biometrics` write is gated behind
  `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` like the other two lanes; no live write was forced
  as part of this patch (shadow-only, per the task). `UNVERIFIED` for end-to-end live
  durability until a real biometrics batch with a non-zero delta lands post-deploy — same
  honesty standard the harness-closure lane's own design doc used.

## Recommended next patch

- After deploy, confirm a live `node:substrate.biometrics` write via `redis-cli GRAPH.QUERY`
  against the running FalkorDB backend, the same manual-check pattern used for
  `node:substrate.harness_closure`.
- Build the equivalent shadow instrument for `chat_grammar` and `route_grammar` (the two
  remaining reducers named in charter §9b item 3's open question), closing that question
  fully rather than partially.
