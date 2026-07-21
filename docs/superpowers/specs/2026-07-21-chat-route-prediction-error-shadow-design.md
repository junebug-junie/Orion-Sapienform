# Chat + route prediction-error shadow instruments

Status: design mode + implemented in the same PR (root `CLAUDE.md`'s implementation-mode
follow-through requirement — this doc captures the decision record, the code lands alongside
it, not as a separate follow-up). Shadow-measure only, per the Sentience Striving Program
charter (`orion/sentience_striving_program/README.md`) §6 item 3: "Phased: shadow-measure
one producer domain before migrating any live."

## Arsonist summary

Charter §9b item 3 (Predictive Processing/Active Inference) named `execution_prediction_
error()`/`transport_prediction_error()` as the real field-native substrate for this theory
thread, and (as of the 2026-07-21 biometrics patch) left two of the five named producer
domains open: "Chat and route remain open — not yet checked, coverage still genuinely
incomplete for those two." This patch closes both, adding `chat_prediction_error()` and
`route_prediction_error()` to `orion/substrate/prediction_error.py`, wired into `_chat_tick()`
and `_route_tick()` in `services/orion-substrate-runtime/app/worker.py`, using the same
shared `_write_prediction_error_node()` durable writer and the same
`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` flag execution/transport/biometrics already use.
This closes the producer-instrumentation half of charter §6 item 3 for all five named
domains (execution, transport, biometrics, chat, route) — it does **not** close item 3
itself, which also requires migrating each domain off `tensions.py`'s bucket-vote layer live
and proving item 2's reducer as a `dominant_drive` replacement, per the charter's own phased
language.

## Naming-mistake context (why this matters here)

A prior draft of this same task (see `docs/superpowers/specs/2026-07-21-biometrics-
prediction-error-shadow-design.md`'s own "Naming-mistake context" section) named a signal
`coherence_prediction_error`, sourced from `turn_effect` — a `SelfStateV1`/
`concept_induction`-adjacent field, in direct violation of charter §7's standing rule
("field-native only — no `SelfStateV1`-anchored substrate for new instrumentation") and §6
item 3's explicit reframing ("not a port of `tensions.py`'s hand-classified kind vocabulary
onto field channels"). Both new functions in this patch read only real reducer projections
(`ChatSessionProjectionV1`, `RouteArbitrationProjectionV1`) built from `orion-cortex-exec`/
`orion-cortex-orch` grammar events, never `SelfStateV1` or `concept_induction`. Grep of the
final diff for `concept_induction`, `turn_effect`, `SelfStateV1`, `self_state_id`: zero hits
(confirmed below).

## Current architecture

- **Chat producer:** `services/orion-substrate-runtime/app/worker.py`'s `_chat_tick()`
  (`REDUCER_SPECS[3]`, `reducer_key="chat_session"`) — fetches `orion-cortex-exec` chat
  grammar events, runs `process_chat_grammar_events()`
  (`orion/substrate/chat_loop/pipeline.py`), saves via `self._store.
  save_chat_session_projection`, projection type `ChatSessionProjectionV1`
  (`orion/schemas/chat_projection.py`), `.turns: dict[str, ChatTurnStateV1]`, loaded via a
  local `_load_chat_projection()` closure keyed on `CHAT_SESSION_PROJECTION_ID`.
  `reduce_chat_trace_events()` (`orion/substrate/chat_loop/reducer.py` line ~146:
  `updated.turns[turn_id] = turn`) revises a turn's state in place across ticks as more
  evidence arrives for the same `turn_id` — create/update, not append-only.
- **Route producer:** `_route_tick()` (`REDUCER_SPECS[4]`, `reducer_key="route_arbitration"`)
  — fetches `orion-cortex-orch` route grammar events, runs `process_route_grammar_events()`
  (`orion/substrate/route_loop/pipeline.py`), saves via `self._store.
  save_route_arbitration`, projection type `RouteArbitrationProjectionV1`
  (`orion/schemas/route_projection.py`), `.runs: dict[str, RouteArbitrationRunStateV1]`,
  loaded via a local `load_projection()` closure keyed on
  `ROUTE_ARBITRATION_PROJECTION_ID`. `reduce_route_trace_events()`
  (`orion/substrate/route_loop/reducer.py` line ~146-147) has the same create/update-by-
  `trace_id` semantics as chat.
- **Existing pattern (unchanged):** `orion/substrate/prediction_error.py`'s
  `execution_prediction_error()`/`transport_prediction_error()`/`biometrics_prediction_
  error()`, wired into `_execution_tick`/`_transport_tick`/`_tick`. All three diff two
  successive projection snapshots per-entity, average absolute deltas, and scale by
  `min(1.0, mean/0.30)`. `_write_prediction_error_node()` (`app/worker.py` ~line 715) is the
  shared durable writer: it internally checks `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`
  (`_prediction_error_nodes_enabled()`) and is otherwise fail-open (never raises out of a
  tick).
- **Current gap (pre-patch):** no instrument existed for the chat or route reducers.

## Why chat mirrors execution/transport's shape, but route does not

Read `orion/substrate/chat_loop/grammar_extract.py::compute_chat_pressure_hints()` (line
114) and `orion/schemas/route_projection.py::RouteArbitrationRunStateV1` before deciding
this — the task's own instruction was not to assume either function's shape transfers.

**Chat**: `compute_chat_pressure_hints(turn: ChatTurnStateV1) -> dict[str, float]` returns
exactly three keys (`conversation_load`, `repair_pressure`, `topic_coherence`), all derived
from numeric fields (`word_count`, `repair_pressure_level`) already on `ChatTurnStateV1`. This
is structurally identical to `execution_prediction_error`'s fixed four-key continuous-
magnitude diff — the only wrinkle is that the hints are *not* persisted on the projection
(unlike execution's `pressure_hints` dict, which the reducer itself writes onto each run).
`compute_chat_pressure_hints()` is a pure function, already tested (see its own test file),
that only gets called transiently at reduction time to build a receipt's `after` payload. So
`chat_prediction_error()` calls it directly on both `prev_turn` and `curr_turn` for each
shared `turn_id`, then diffs the resulting dicts the same way execution diffs its persisted
`pressure_hints`.

**Route**: `RouteArbitrationRunStateV1`'s fields that describe the actual arbitration
decision — `lane: str`, `lane_reason: str`, `output_mode: str`, `mind_requested: bool` — are
categorical/discrete, not continuous. There is no numeric magnitude to subtract between
`"background"` and `"spark"`. Two options were considered and rejected before landing on a
mismatch-rate score:

1. **Hand-assign numeric codes to each category** (e.g. `lane: {"background": 0, "chat": 1,
   "spark": 2}`) and diff those. Rejected: this is exactly the kind of hand-authored
   taxonomy-on-top-of-taxonomy charter §6 item 3 explicitly warns against ("not a port of
   `tensions.py`'s hand-classified kind vocabulary onto field channels") — an arbitrary
   ordinal encoding implies a false distance metric (is `"spark"` "twice as far" from
   `"background"` as `"chat"` is? no principled answer exists) and would need updating every
   time a new lane is added.
2. **Skip route entirely, leave it unmeasured.** Rejected: the task explicitly named route as
   one of the two remaining domains to close, and a categorical mismatch rate is a real,
   principled, if different, prediction-error signal — "did the arbitration decision change
   at all, and how many of its dimensions changed" is itself a meaningful surprise measure
   for Predictive Processing/Active Inference (see theory anchor below).

`route_prediction_error()` therefore computes a genuinely different-shaped score: for each
matched run (`trace_id` present in both `prev.runs`/`curr.runs`), compare `lane`,
`lane_reason`, `output_mode`, `mind_requested` — 1.0 per field if it differs, 0.0 if it
doesn't — and average across the four fields, then average across matched runs. This is
disclosed prominently in the function's own docstring (not just this doc), including an
explicit "do not scale by `_THRESHOLD`" warning, because a mismatch rate is already bounded
to `[0, 1]` by construction (mean of N values each in `{0.0, 1.0}`) — dividing an
already-bounded `[0, 1]` value by 0.30 would push most non-zero mismatches to the 1.0 ceiling
and destroy the very distinction (one field flipped vs. all four) that makes the signal
informative. `test_route_prediction_error_not_saturated_by_threshold_scaling` is an explicit
regression guard against a future "fix" that re-applies the `_THRESHOLD` scale here.

**Fields deliberately excluded from route's comparison:** `correlation_id`, `session_id`,
`turn_id`, `evidence_event_ids`, `last_updated_at` (bookkeeping fields that change on every
revision by construction — including them would saturate every batch's score to near-1.0,
making the signal useless), and `mind_skip_reason` (a free-text explanation that is non-null
only when `mind_requested` is already false — including it would double-count the same
underlying decision `mind_requested` already captures).

## Metric quality gate (root `CLAUDE.md` §0A) — findings

Run fresh for both new metrics, not inherited from execution/transport/biometrics' prior
passes.

### `chat_prediction_error()`

1. **Trace provenance to real code.** `chat_prediction_error(prev, curr)` in
   `orion/substrate/prediction_error.py` (new function). Called from `_chat_tick()` in
   `services/orion-substrate-runtime/app/worker.py`, which now captures `prev_projection =
   _load_chat_projection()` before `process_batch` runs and `curr_projection` via the same
   closure immediately after, when `last_id is not None` (this capture did not exist
   pre-patch — `_chat_tick()` previously returned immediately after
   `_process_events_with_poison_isolation`, with no pre/post projection diff). The underlying
   values are traced one level further: `compute_chat_pressure_hints()`
   (`orion/substrate/chat_loop/grammar_extract.py:114`) reads `turn.word_count` and
   `turn.repair_pressure_level`, both populated by `extract_chat_turn_state()`
   (same file, line 35-111) from real `orion-cortex-exec` chat grammar events (word-count
   regex over message summaries, repair-pressure-level atoms).
2. **Independence check.** Not redundant with execution/transport/biometrics: chat's
   projection is built from an entirely separate event stream (`orion-cortex-exec` chat
   grammar events specifically tagged with `CHAT_TRACE_PREFIX`, distinct from execution's
   general trajectory events), a separate reducer (`REDUCER_SPECS[3]`, not `[0]`/`[1]`/`[2]`),
   and a separate keying scheme (`turn_id`, not `trace_id`/`bus_id`/`node_id`). The causal
   chain from "conversation got long and repair-heavy" to, say, "execution reasoning_load
   changed" is indirect at best (mediated by, at minimum, a human/Orion turn triggering a
   downstream execution run) — not a monotonic transform of an already-included signal. Not
   redundant with `route_prediction_error()` either: chat measures *how a conversational turn
   itself is going* (load, repair, coherence), route measures *how the arbitration layer
   routing that turn behaved* (lane, decision) — different projections, different reducers,
   different event streams (`orion-cortex-exec` vs `orion-cortex-orch`).

   **Intra-instrument redundancy found on review, disclosed rather than fixed silently.**
   Code review of this patch caught a redundancy *inside* `chat_prediction_error()` itself,
   not against another instrument: `compute_chat_pressure_hints()` defines `topic_coherence =
   max(0.0, 1.0 - repair_pressure_level)`, an affine (monotonic) transform of the same
   `repair_pressure_level` that also drives `repair_pressure` directly. A change in
   `repair_pressure_level` therefore moves both `repair_pressure` and `topic_coherence` by
   the same magnitude, giving that one underlying signal roughly 2x the weight of
   `conversation_load` in the 3-key mean — a direct instance of CLAUDE.md §0A's own
   independence-check language ("a monotonic transform of something already included... not
   independent"). Kept rather than fixed by dropping a key: the three keys diffed are
   `compute_chat_pressure_hints()`'s full, already-tested output contract, not a subset
   hand-picked for this instrument — silently dropping `topic_coherence` here would be a
   second, quieter instance of exactly the "hand-classified vocabulary" problem charter §6
   item 3 exists to avoid, just one layer down (deciding which of an existing reducer's
   already-computed signals "count"). Documented explicitly in the function's own docstring.
   If this weighting proves to be a real problem against live data, the correct fix is
   upstream in `compute_chat_pressure_hints()` itself (which several other consumers besides
   this instrument may also read), not a silent key-drop in this diff function.
3. **Theory anchor.** Same anchor already used for execution/transport/biometrics, charter
   §9b item 3: Predictive Processing/Active Inference — a 0-1 surprise score computed as the
   magnitude of change in a reducer's own pressure-hint state between successive
   observations, written onto `FieldStateV1` nodes so `orion/substrate/dynamics.py::
   prediction_error_pressure()` can seed activation pressure from real, measured surprise.
   Chat is a natural fit: a conversation's load/repair/coherence shifting unexpectedly
   between two observations is exactly the kind of "prediction violated" event the theory
   anchors to.
4. **Live-data sanity check.** Queried directly via `psql -h localhost -p 55432 -U postgres -d
   conjourney` against `substrate_chat_session_projection` (single live row, `generated_at =
   2026-07-21T03:43:33Z`, `total_turn_count = 230`, 8 distinct sessions). Sampled turn field
   values: `word_count` ranges 1-587 across turns (not flat), `repair_pressure_level` averages
   ~0.094 with non-zero spread (not always-zero, not always-saturated). Confirmed
   non-degenerate. Same singleton-upsert-table limitation as the other three instruments
   (`projection_id` primary key, one row) — no history table to replay multiple historical
   deltas against, so this is "current snapshot is real and non-degenerate," not "N
   historical deltas were replayed."
5. **Existing-mechanism check.** `grep -rn "chat_prediction_error"` across the repo before
   this patch: zero hits outside this patch's own new code.
6. **Reversibility.** Fully reversible: gated behind the existing
   `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` flag (no new flag), writes to a single fixed
   `node_id` (`node:substrate.chat`) that collapses on repeat writes, pure diff function with
   no persisted schema/manifest dependency. Deleting `chat_prediction_error()` and its call
   site in `_chat_tick()` would leave the other four instruments untouched and would not break
   any consumer, since `prediction_error_pressure()` reads generically off
   `node.metadata.get("prediction_error")`.

### `route_prediction_error()`

1. **Trace provenance to real code.** `route_prediction_error(prev, curr)` in
   `orion/substrate/prediction_error.py` (new function). Called from `_route_tick()`, which
   now captures `prev_projection = load_projection()` before `process_batch` runs and
   `curr_projection` after, when `last_id is not None` (same new-capture pattern as chat —
   `_route_tick()` previously had no pre/post diff either). The underlying field values are
   traced to `process_route_grammar_events()` (`orion/substrate/route_loop/pipeline.py`),
   which populates `lane`/`lane_reason`/`output_mode`/`mind_requested` on
   `RouteArbitrationRunStateV1` from real `orion-cortex-orch` route grammar events (decision
   router output, `services/orion-cortex-orch/app/decision_router.py`).
2. **Independence check.** Not redundant with the other four instruments — separate event
   stream (`orion-cortex-orch`, not `orion-cortex-exec`/`orion-bus`/`orion-biometrics`),
   separate reducer (`REDUCER_SPECS[4]`), separate keying (`trace_id` for arbitration runs,
   distinct namespace from execution's `trace_id` — route's is prefixed `orch.route:`,
   confirmed against live data below). The causal chain from "this turn's lane changed" to
   any of the other four projections' own state is real (a lane change plausibly follows from
   or precedes a load/pressure shift) but indirect, mediated by upstream routing logic, not a
   direct transform.
3. **Theory anchor.** Same Predictive Processing/Active Inference anchor. A categorical
   mismatch rate is a legitimate variant of "surprise" under this theory: active inference
   formalizes surprise as (roughly) the divergence between predicted and observed state:
   for a discrete decision variable, the natural divergence measure is exactly a mismatch/
   disagreement rate, not an ill-defined "distance" between category labels. This is not a
   stretch of the theory to fit the data shape — categorical prediction error is a standard
   treatment in predictive-coding literature for discrete/symbolic states, distinct from but
   consistent with the continuous case the other four instruments use.
4. **Live-data sanity check.** Queried directly via `psql` against
   `substrate_route_arbitration_projection` (single live row, `generated_at =
   2026-07-21T04:03:36Z`, 105 runs). Distinct `(lane, lane_reason, output_mode,
   mind_requested)` combinations observed: `(background, verb_background, direct_answer,
   false)` ×87, `(spark, explicit_options, direct_answer, false)` ×15, `(chat, mode_chat,
   direct_answer, false)` ×2, `(chat, verb_chat, direct_answer, false)` ×1. Confirmed
   non-degenerate: real variation in `lane`/`lane_reason` across the live sample (not a
   single constant value), even though `mind_requested` and `output_mode` happen to be
   constant in this particular snapshot (real — `mind_requested` is gated off in the current
   deployment per `mind_skip_reason: "mind_enabled_not_true"`, not a data-collection bug).
   This means a *live* prediction-error signal for this domain right now would be driven
   almost entirely by `lane`/`lane_reason` transitions until mind-escalation is enabled
   elsewhere — disclosed honestly rather than glossed over. Same singleton-table limitation
   as the other four instruments.
5. **Existing-mechanism check.** `grep -rn "route_prediction_error"` across the repo before
   this patch: zero hits outside this patch's own new code.
6. **Reversibility.** Fully reversible, same argument as chat: existing flag, single
   collapsing `node_id` (`node:substrate.route`), pure function, no schema dependency.
   Deleting it and its call site would not break any consumer.

## Shadow-only confirmation

`orion/substrate/dynamics.py::_compute_pressures()` calls `prediction_error_pressure(node,
...)` (defined `orion/substrate/pressure.py:36`) for every node in the graph generically —
confirmed by reading the function body: it reads `node.metadata.get("prediction_error")` and
`node.temporal.observed_at` off whatever `BaseSubstrateNodeV1` it is handed, with no
`node_id`-prefix branching anywhere in the function. This means `node:substrate.chat` and
`node:substrate.route` are automatically picked up by the same already-live mechanism the
other three instruments use once written — this is the existing wire, not a new consumer
being added. No change was made to `capability_policy.py`, `top_down.py`, or
`goal_context.py`, and no new bus consumer was wired.

## Proposed schema / API changes

- `orion/substrate/prediction_error.py`: two new functions, `chat_prediction_error(prev:
  ChatSessionProjectionV1, curr: ChatSessionProjectionV1) -> float` and
  `route_prediction_error(prev: RouteArbitrationProjectionV1, curr:
  RouteArbitrationProjectionV1) -> float`. No schema changes to `ChatTurnStateV1` or
  `RouteArbitrationRunStateV1` — both new functions read existing fields only.
- `services/orion-substrate-runtime/app/worker.py`'s `_chat_tick()`/`_route_tick()`: each now
  captures a `prev_projection` before `process_batch` runs; after a non-`None` `last_id`,
  reloads `curr_projection`, computes `error`, and on `error > 0.0` saves a
  `_prediction_error_receipt(...)` and calls `self._write_prediction_error_node(node_id=
  "node:substrate.chat"|"node:substrate.route", ..., contributing_id=last_id)`. Both
  functions' return type is unchanged (`str | None`).
- No new env key. Reuses `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` — the same generic switch
  execution/transport/biometrics already share.

## Files likely to touch (this patch's actual diff)

- `orion/substrate/prediction_error.py` — new `chat_prediction_error()`,
  `route_prediction_error()`.
- `orion/substrate/tests/test_prediction_error.py` — new unit tests for both functions,
  appended to the existing execution/transport/biometrics test file.
- `services/orion-substrate-runtime/app/worker.py` — `_chat_tick()`/`_route_tick()` wiring,
  import update.
- `services/orion-substrate-runtime/README.md` — document the fourth and fifth instruments
  alongside execution/transport/biometrics.
- `orion/sentience_striving_program/README.md` — §6 item 3 status note updated (all five
  producer domains now shadow-measured, item 3 itself still open pending live migration),
  §9b item 3's open-question line closed (chat/route no longer unmeasured).
- This doc.

## Non-goals

- Not migrating any producer off `tensions.py`'s bucket-vote layer — that is a separate,
  later step in charter §6 item 3, gated on every producer domain first being shadow-measured
  AND item 2's reducer being proven a `dominant_drive` replacement.
- Not touching `capability_policy.py`, `top_down.py`, `goal_context.py`, or any bus consumer.
- Not adding a new env key.
- Not building a historical-replay script for either instrument (unlike
  `measure_origination_gate.py`/`measure_ast_hot_reducer.py`) — same singleton-table
  limitation named in the biometrics design doc.
- Not claiming charter §6 item 3 itself is complete. Only the producer-instrumentation half
  is done; the migration-off-bucket-vote half is untouched.

## Acceptance checks

- `pytest orion/substrate/tests/test_prediction_error.py -q` passes — 22 tests (9 pre-existing
  biometrics tests unchanged, 6 new chat tests, 7 new route tests, including an explicit
  `test_route_prediction_error_not_saturated_by_threshold_scaling` regression guard for the
  documented `_THRESHOLD`-scaling deviation).
- `_chat_tick()`/`_route_tick()`'s existing return signature and poison-isolation behavior
  are unchanged for callers — confirmed by running the full
  `services/orion-substrate-runtime/tests/` suite (excluding the always-broken, pre-existing
  `test_grammar_consumer_integration.py` module-collection error) with this patch applied and
  with it reverted (`git stash`): both produce the identical `16 failed, 143 passed`. This
  patch changes zero pre-existing pass/fail outcomes.
- Grep of the final diff for `concept_induction`, `turn_effect`, `SelfStateV1`,
  `self_state_id`: zero hits.
- Live check: both `node:substrate.chat` and `node:substrate.route` writes are gated behind
  `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` like the other three lanes; no live write was
  forced as part of this patch (shadow-only). `UNVERIFIED` for end-to-end live durability
  until a real chat/route batch with a non-zero delta lands post-deploy — same honesty
  standard the biometrics/harness-closure lanes' own design docs used.

## Recommended next patch

- After deploy, confirm live `node:substrate.chat`/`node:substrate.route` writes via
  `redis-cli GRAPH.QUERY` against the running FalkorDB backend, the same manual-check
  pattern used for `node:substrate.biometrics`/`node:substrate.harness_closure`.
- All five named producer domains (execution, transport, biometrics, chat, route) now have
  shadow instrumentation. The next real step in charter §6 item 3 is migrating one domain off
  `tensions.py`'s bucket-vote layer live — gated on item 2's reducer being proven a
  `dominant_drive` replacement first, per the charter's own phased sequencing.
