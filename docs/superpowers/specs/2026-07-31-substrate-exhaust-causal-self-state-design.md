# Substrate exhaust, daydream material, and causal self-state: design notes

Status: **design/brainstorm, not proposal-mode-approved**. One small, additive,
non-invasive patch shipped alongside this doc (see "Recommended next patch,
shipped"); everything else here is investigation and open direction, not a
committed roadmap. Any future work that touches memory, autonomy, or
cognition-loop decision logic still needs explicit proposal-mode scoping per
`AGENTS.md` before implementation.

## Arsonist summary

Juniper asked what Orion computes and throws away -- "exhaust" -- and whether
that exhaust could feed a daydream/reverie stream, distinct from re-dispatch
autonomy. The investigation ruled out the obvious candidate (AI Town's
spontaneous-thought reverie loop -- known, already-fixed repetition-bug
infrastructure junk, not interesting exhaust) and moved into the
substrate-runtime's own "grammar" subsystem. That surfaced two structurally
different discard mechanisms, neither of which alone matches what Juniper
was actually after: real per-topic causal narrative ("I was doing X, which
backed up Y, which is why I feel Z"). Checking for that directly found the
existing grammar-ledger edges are templated fan-out, not discovered
causality, and surfaced two more, separate, real "causal" mechanisms
elsewhere in the repo that need reconciling before any new one gets built.
Along the way, a fourth, smaller, exhaust instance was found and fixed: a
per-domain prediction-error breakdown that gets computed every tick, folded
into one confidence scalar, and discarded.

## Current architecture

### Two real discard paths in the grammar subsystem, different timescales

**`orion:grammar:accepted-pressure`** (`orion/bus/channels.yaml` ~L2241) --
produced only by `orion-substrate-runtime`'s `_tick()`
(`services/orion-substrate-runtime/app/worker.py`), which runs biometrics
grammar events through a reducer (`min_confidence`/staleness accept-gate)
and republishes the accepted subset. `consumer_services: []`. Confirmed live
2026-07-31: `redis-cli XLEN orion:grammar:accepted-pressure` = 0 -- this is a
bare `PUBLISH`, not a stream, so nothing is even buffered; content evaporates
in the same tick it's produced. `services/orion-substrate-runtime/app/
grammar_truth.py` explicitly documents it as `"not canonical grammar
ingress"`. This is the purest structural match to "exhaust" found this
session, but its one confirmed direct producer
(`orion/substrate/biometrics_loop/candidate_events.py::
build_pressure_candidate_events()`) is mechanical: `summary=f"{semantic_role}
for {node_id}"`, `text_value=node_id`. Real and gated, but not narratively
rich.

**Canonical `orion:grammar:event`** -> Postgres `grammar_atoms`/
`grammar_events` ledger -- has real consumers (`orion-sql-writer` for
persistence, `orion-equilibrium-service` for the chat-turn and
transport-timeout metacog triggers, `orion-heartbeat` as a v0 research
consumer). Live-queried 2026-07-31 (last 3h window): genuinely rich content
-- `exec_plan_started` ("Execution plan started for verb=
log_orion_metacognition; step_count=3"), `route_arbitration_decided`
("lane=background, mind_requested=false..."), `exec_recall_gate_observed`
("Recall policy resolved: run=False, profile=reflect.v1, reason=
disabled_by_client"), `exec_result_assembled`, `capability_surface`. This
*does* get discarded, just slowly: `services/orion-sql-writer/app/
grammar_truth.py::apply_grammar_events_retention()` runs a batched `DELETE
FROM grammar_events WHERE created_at < cutoff` at service startup,
`GRAMMAR_EVENTS_RETENTION_DAYS=30` by default (`services/orion-sql-writer/
.env_example`), with FK cascade into `grammar_atoms`. Every plan, arbitration
decision, and declined-recall moment Orion ever generated quietly ages out
past 30 days with nothing summarizing or reflecting on it first.

### Grammar-ledger edges are templated, not causal

Checked directly (live Postgres, 2026-07-31, 3h window) because Juniper's
own instinct was that the ledger is "so abstracted from 'X topic caused this
Y'" that it probably lacks real trace provenance. Confirmed: `grammar_edges`
has 7,956 `influenced` edges in-window, **100% same-`trace_id`** (`0`
cross-trace edges), and the actual pairing is a fixed per-tick template --
every biometrics pressure-signal atom type (`disk_pressure_signal`,
`gpu_pressure_signal`, `thermal_pressure_signal`, `memory_pressure_signal`,
etc.) points to the same `capability_surface` atom, each pairing appearing
exactly 886 times in the window. That's mechanical fan-out wiring, not
discovered cross-domain causality. The `RelationType`/`TemporalHopType`
enums in `orion/schemas/grammar.py` already anticipate richer use --
`dream_reentry`, `counterfactual`, `future_candidate` hop types exist in the
schema -- but grepping the full `orion/`/`services/` tree found zero
consumers or producers of those three values anywhere. Schema-only, never
built.

### A real composite "mood" instrument already exists, one layer up

`orion/substrate/attention_self_model.py::_aggregate_prediction_error_confidence()`
takes live prediction-error from 5 real domains (execution, biometrics,
chat, route, bus_synaptic) and computes `confidence = 1 - mean(prediction_error
across domains)`. This is structurally close to what Juniper described --
"I have X things going on, I'm backed up here, combined they make me feel
chaotic" -- a single scalar synthesized from concurrent pressure across
unrelated subsystems. It's real, live-verified (2026-07-23, 6h window:
biometrics carries almost all real variance, other four domains present but
tiny), and the code is honest about the resulting dilution rather than
hiding it. Deliberately `mean()` not `max()`, specifically so it doesn't
just restate whichever single domain attention-selection already surfaced
as loudest (CLAUDE.md's metric-quality-gate independence check, applied at
build time).

Until this session's patch (below), the per-domain values that composed
that mean were discarded the instant the scalar was computed -- the exact
same "real content, computed, thrown away" shape as the grammar-exhaust
channels above, just one level of abstraction up and un-gated by any
staleness/retention window at all (it never touched storage in the first
place).

### Three separate existing "causal" mechanisms, not yet reconciled

Before building anything new that claims to give Orion causal
self-understanding, these three need to be checked against each other --
this repo's metric-quality-gate "existing-mechanism check" applies here as
much as it does to any new metric:

1. **Grammar-ledger edges** (`grammar_edges.relation_type`) -- see above.
   Same-trace, templated, not discovered causality.
2. **`orion/signals/causal_helpers.py::with_missed_parent_notes()`** -- a
   real, live consumer (`services/orion-signal-gateway/app/processor.py`,
   plus test coverage in `orion/signals/adapters/tests/`). Checks a
   *registry-declared* structural parent-organ hierarchy
   (`OrionOrganRegistryEntry.causal_parent_organs`) against which parent
   organs actually produced a recent signal in `prior_signals`; if a
   declared parent is missing, appends an audit note to the signal. This is
   more principled than the grammar edges (declared structure, not
   per-tick mechanical fan-out) but it only flags *absence* -- it records
   "expected causal parent didn't fire," never "this specific value caused
   that specific downstream value." No positive causal attribution, no
   content.
3. **Causal Geometry v1** (PR #1087, #1102-ish spec closeout, 2026-07-16/17)
   -- a different scope than either of the above, despite the name overlap:
   causal geometry of *substrate structural mutation proposals* (a
   `proposal -> trial -> HITL adopt` pipeline with divergence snapshots),
   not causal explanation of moment-to-moment self-state. Phase A
   (snapshot persistence, hub Snapshot/History panels) is live end-to-end,
   bus-routed through `orion-sql-writer` per the corrected design. Phase B
   (`SubstrateTrialRunner`) is wired and genuinely exercised per proposal,
   but always resolves `"inconclusive"` today -- no replay corpus is
   registered for its mutation class yet. Not dead, just unproven in
   practice. Worth knowing this exists and what its real status is before
   assuming "causal self-state" needs a new mechanism from scratch --  but
   it answers a different question than Juniper's "why do I feel this way"
   framing.

None of the three currently produces "I feel uncertain because route
arbitration and biometrics disagreed on X within the last minute" --  the
closest existing building block for that is `causal_helpers.py`'s
registry-declared parent-organ pattern, because it's the only one of the
three based on *declared* structure rather than mechanical per-tick fan-out
or a differently-scoped mutation-trial pipeline.

### Self-modeling ladder placement convention

Per prior work (Reverie/Dream substrate-native design), new reflective/
narrative cognition is supposed to land ON the existing self-modeling
ladder (layers 5-11), not as a parallel shadow mechanism. All ladder rungs
are merged; ignition flags are off; rung 5 is awaiting sign-off. Any future
daydream/exhaust-reflection consumer should be scoped against this ladder's
existing rung structure rather than invented as a new standalone pipeline.

## Missing questions

- Why did Causal Geometry v1's Phase B trial runner never get a registered
  replay corpus -- is that a small, unblockable gap, or does it reveal a
  harder structural problem with running trials against live mutation
  classes?
- Would extending `causal_helpers.py`'s registry-declared parent-organ
  pattern (already live, already tested) to the grammar-atom layer be
  cheaper than inventing new edge-discovery logic for `grammar_edges`?
- Where on the self-modeling ladder (which rung, if any) does "explain my
  own composite confidence" belong?
- Is `orion:grammar:accepted-pressure`'s zero-consumer status worth fixing
  on its own, independent of the richer causal-provenance work -- i.e. is
  a thin, mechanical exhaust stream still useful as raw "sprinkled" daydream
  material even without narrative richness?

## Proposed schema / API changes

None proposed yet at the daydream/causal-bridge level -- this stays
investigation-only per proposal-mode discipline (memory/cognition-loop
changes need explicit scoping before implementation). The one schema change
in this patch is additive-only and observability-only (see below).

## Files likely to touch (future work, not this patch)

- `orion/schemas/grammar.py` / grammar-atom producers, if `grammar_edges`
  ever grows real cross-domain causal discovery instead of templated
  fan-out.
- `orion/signals/causal_helpers.py`, if its registry-declared pattern gets
  extended to another domain.
- Whatever rung of the self-modeling ladder ends up owning a "why do I feel
  this way" reflective consumer.
- `docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md` and its
  closeout PR report, to check Phase B's replay-corpus gap in detail before
  reviving it.

## Non-goals

- Not rebuilding `grammar_edges` into a general causal-inference engine in
  this patch.
- Not touching `orion:grammar:accepted-pressure`'s zero-consumer status in
  this patch -- flagged as a separate, smaller, optional thread.
- Not reviving Causal Geometry Phase B without first understanding why its
  replay corpus was never registered.
- Not building any new daydream/reverie consumer yet -- that's proposal-mode
  work, not this session's scope.

## Acceptance checks

For this doc's own shipped patch (see below): tests pass, the new field is
additive (no existing consumer's behavior changes), no schema/column-list
drift risk (the persisted row is a single JSONB blob, not an explicit
column list -- confirmed via `services/orion-substrate-runtime/app/
store.py::save_attention_self_model()`).

For any future work building on this doc: must show a live trace (not just
a passing test) that a specific self-state readout can be explained by
naming the specific atoms/signals that composed it, per CLAUDE.md's
"runtime truth beats config truth."

## Recommended next patch, shipped in this same branch

**Expose `prediction_error_by_domain` on `AttentionSelfModelV1`.** The
per-domain prediction-error snapshot that `_aggregate_prediction_error_
confidence()`/`_unconditional_prediction_error_confidence()` already receive
and average into `prediction_error_confidence` every tick were being
discarded the moment the scalar was computed -- never stored on the output
model, never persisted. This is the same "real, gated content computed and
thrown away" shape as the grammar-exhaust channels above, at the cheapest
possible fix point.

- `orion/schemas/attention_self_model.py`: added `prediction_error_by_domain:
  dict[str, float] | None = None`, additive field, no `extra="forbid"`
  conflict.
- `orion/substrate/attention_self_model.py::reduce_attention_self_model()`:
  populates the new field with the same `ACTIVE_INFERENCE_DOMAINS`-filtered
  dict that composed `prediction_error_confidence` (so the exposed
  breakdown always matches exactly what the scalar actually averaged, never
  a superset).
- `orion/substrate/tests/test_attention_self_model.py`: 4 new tests
  (`TestPredictionErrorByDomainExposed`) covering the mirrored-domain case,
  the transport-exclusion case, and both honest-`None` cases.
- Persistence: `services/orion-substrate-runtime/app/store.py::
  save_attention_self_model()` writes the whole model as one JSONB blob
  (`model.model_dump(mode="json")`), so this field round-trips into
  `substrate_attention_self_model` automatically -- no migration, no
  column-list update, no drift risk.
- Scope check: `services/orion-hub/scripts/attention_organ_routes.py`
  already independently re-derives a per-domain breakdown live, for the UI,
  by calling `_brain_frame_prediction_error_by_domain()` directly against
  the fresh brain-frame -- so this patch does not add new *live* UI
  capability. What it fixes is the **persisted historical record**:
  `substrate_attention_self_model` rows older than the current tick never
  carried this breakdown before, so replay/history could never show *why* a
  past confidence dip happened, only that it happened. Wiring this into the
  hub UI's history view (reading the persisted field instead of only ever
  recomputing live) is a natural, still-small follow-up, not done in this
  patch.

Tests: `orion/substrate/tests/test_attention_self_model.py` (52 passed,
including the 4 new), `services/orion-substrate-runtime/tests/
test_worker_attention_self_model_tick.py` (16 passed) -- both run via the
main checkout's `.venv` from inside this task's worktree.

## Second patch, shipped in this same branch: real evidence on prediction-error receipts

Juniper asked to implement "the 1547 proposal." Of the four deferred threads
this doc names (accepted-pressure's zero-consumer status, Causal Geometry
Phase B, extending `causal_helpers.py` to grammar atoms, or a new daydream
consumer), none turned out to be the cheapest real next step once traced
further. Following `_aggregate_prediction_error_confidence()`'s inputs
upstream (the same digging that produced the first patch above) found a
closer, already-real seam: **`_prediction_error_receipt()`**
(`services/orion-substrate-runtime/app/worker.py`) builds a
`ReductionReceiptV1`/`StateDeltaV1` every time a domain's prediction-error
rises above zero -- the exact receipt trail behind
`prediction_error_confidence`/`prediction_error_by_domain` -- and it was the
**sole holdout in the entire reducer family** hardcoding
`caused_by_event_ids=[]`. Every sibling reducer (`orion/substrate/
chat_loop/reducer.py`, `execution_loop/reducer.py`, `route_loop/reducer.py`,
`transport_loop/reducer.py`) already threads its batch's real `event_id`s
into this same `StateDeltaV1` field. And it isn't a dead field:
`orion/substrate/receipts/retention.py::primary_event_id()` and
`services/orion-hub/scripts/substrate_biometrics_routes.py`'s
`/biometrics-node/{node_id}/latest` route both already read
`caused_by_event_ids` off every other receipt -- prediction-error receipts
were the one blind spot in an existing, live, already-consumed evidence
mechanism, not a place needing a new one invented.

This is the concrete first slice of "why do I feel this way": for the four
grammar-event-driven domains, a receipt now names the actual grammar event
IDs processed in the tick whose projection diff produced that error reading
-- inspectable today via the same hub route above once
`node_id=node:substrate.<domain>` is queried, no new UI work required.

- `services/orion-substrate-runtime/app/worker.py`:
  - `_prediction_error_receipt()` gained a `caused_by_event_ids: Sequence[str]
    = ()` param, threaded into `StateDeltaV1.caused_by_event_ids`, capped by
    a new `_PREDICTION_ERROR_EVIDENCE_CAP = 50` module constant -- same
    convention as `chat_loop/reducer.py`'s existing `_EVIDENCE_CAP`, guarding
    against the same unbounded-list-if-batch-limit-is-raised shape already
    fixed elsewhere in this repo (`evidence_event_ids` in the pressure and
    execution-merge reducers).
  - The four grammar-event-driven tick methods (`_tick` / biometrics,
    `_execution_tick`, `_chat_tick`, `_route_tick`) now pass
    `[e.event_id for e in events if e.atom]` -- the real batch each tick
    already fetched and processed -- matching the exact filter the sibling
    reducers already use.
  - `bus_synaptic` (`_bus_synaptic_tick`) and `codebase`
    (`_handle_codebase_mass_delta` and friends) call sites are **left
    unchanged, on purpose**: `bus_synaptic` has no `GrammarEventV1` batch in
    scope at all (it reads FalkorDB edge z-scores fresh each call), so there
    is nothing real to name -- a comment now says so explicitly rather than
    leaving the gap to look like an oversight. `codebase` already has an
    explicit, dated Phase-3-deferral comment in the same file explaining why
    it stays excluded from downstream wiring; extending evidence-population
    there is a drive-by fix on top of a deliberate prior scoping decision,
    not this patch's job.
- `services/orion-substrate-runtime/tests/
  test_worker_prediction_error_receipt_evidence.py` (new): 4 tests against
  the pure `_prediction_error_receipt()` helper -- default stays `[]`
  (backward compatible with the two untouched call sites), real IDs thread
  through, the cap actually caps, and a non-list `Sequence` input works.

Tests: `services/orion-substrate-runtime/tests/
test_worker_prediction_error_receipt_evidence.py` (4 passed, new).
Broader regression: `pytest tests/ -k "worker or prediction_error"`
(102 passed / 3 pre-existing failures unrelated to this patch, confirmed via
`git stash` on the same baseline -- `test_worker_falkor_routed_store.py::
test_write_prediction_error_node_preserves_dynamics_state_on_rewrite`,
`test_worker_independent_reducers.py::
test_start_spawns_independent_reducer_poll_tasks`,
`test_worker_reducer.py::
test_advance_cursor_records_commit_failure_when_created_at_missing`; the
`test_grammar_consumer_integration.py` collection error is a live-Postgres
dependency unavailable in this environment, also pre-existing).

Still open, still deferred (unchanged from this doc's original scope): fixing
`accepted-pressure`'s zero-consumer status, reviving Causal Geometry Phase B,
extending `causal_helpers.py` to the grammar-atom layer, and any new
daydream/reverie consumer. This patch closes one small, real evidence gap; it
does not build a general causal-inference engine.

## Third patch, separate PR: durable evidence on the self-model row itself

An adversarial pass on the second patch above found the fix was
incomplete: `caused_by_event_ids` reaches `substrate_reduction_receipts`,
which prunes on `ORION_RECEIPT_RETENTION_SUCCESS_MINUTES` -- **30 minutes
by default, live-confirmed 2026-08-11** (`SELECT min(created_at) FROM
substrate_reduction_receipts WHERE ...` returned a row exactly 31 minutes
old, no older). Meanwhile `substrate_attention_self_model` -- the row that
already carries `prediction_error_by_domain` from the first patch above --
lives for `attention_self_model_log_retention_hours` (7 days by default).
Any consumer triggered on a reflective/batch cadence rather than watching
the bus in real time would find the evidence already gone almost every
time it looked. There was also no join key at all: `AttentionSelfModelV1`
had no `receipt_id`/`delta_id` field, so even a real-time consumer had to
correlate by timestamp + domain name across two independently-ticking
loops, not a real foreign key.

Fix: stamp the same evidence onto the durable FalkorDB node the scalar
itself already lives on (`node:substrate.<domain>`), then read it back
into the self-model row alongside `prediction_error_by_domain`. No new
storage, no new retention window to manage, no join required -- the scalar
and its evidence now live on the same object.

- `services/orion-substrate-runtime/app/worker.py`:
  - `_write_prediction_error_node()` gained an `evidence_event_ids:
    Sequence[str] = ()` param, written to
    `metadata['prediction_error_evidence_event_ids']`. Unlike
    `contributing_turn_ids` (accumulate + cap across writes), this fully
    REPLACES on every write -- it names the events behind *this* write's
    own `prediction_error` value, so it must move in lockstep with that
    value, never lag it the way an accumulated list would.
  - The four grammar-event-driven tick methods (biometrics/execution/chat/
    route) now compute their evidence list once per tick and pass it to
    *both* `_prediction_error_receipt()` (unchanged, still the 30-minute
    copy) and `_write_prediction_error_node()` (the new durable copy) --
    same evidence, two destinations with very different lifespans.
  - New `_brain_frame_prediction_error_evidence_by_domain()`, a direct
    companion to the existing `_brain_frame_prediction_error_by_domain()`
    -- same node snapshot, zero extra I/O, reads
    `metadata['prediction_error_evidence_event_ids']` instead of
    `metadata['prediction_error']`. Wired into both live callers of
    `reduce_attention_self_model()` (`_attention_self_model_tick()` and
    `_brain_frame_tick()`).
- `orion/schemas/attention_self_model.py`: added
  `prediction_error_evidence_by_domain: dict[str, list[str]] | None`,
  additive, no `extra="forbid"` conflict.
- `orion/substrate/attention_self_model.py::reduce_attention_self_model()`:
  new `prediction_error_evidence_by_domain` param, populated filtered
  against `model.prediction_error_by_domain`'s own keys (not just
  `ACTIVE_INFERENCE_DOMAINS` directly) -- a domain's evidence can never
  appear without its own scalar also appearing, closing a gap a first pass
  of this filter missed (caught by the reducer's own unit tests, not
  review).
- 12 new tests across three files: 5 on the pure reducer
  (`test_attention_self_model.py`), 4 on `_write_prediction_error_node()`'s
  new param (including a replace-not-accumulate regression test and a cap
  test), 6 on the new `_brain_frame_prediction_error_evidence_by_domain()`
  helper (malformed-input tolerance, unknown node_ids, empty list).

Tests: `orion/substrate/tests/test_attention_self_model.py` (57 passed),
`services/orion-substrate-runtime/tests/
test_worker_prediction_error_node.py` + `test_worker_prediction_error_
evidence_by_domain.py` + `test_worker_prediction_error_receipt_evidence.py`
(29 passed), `test_worker_attention_self_model_tick.py` (16 passed).
Broader regression: `pytest tests/ -k "worker or prediction_error or
brain_frame"` (139 passed / same 3 pre-existing unrelated failures as the
second patch above).

### Review found the patch didn't actually work against the real backend

The first review pass found this shipped a no-op against live FalkorDB:
`prediction_error_evidence_event_ids` was never added to
`orion/substrate/falkor_codec.py`'s closed `DYNAMICS_METADATA_KEYS`
allowlist, so `encode_node_properties()`/`decode_concept_node()` silently
dropped it from every real Cypher write and read. It only *appeared* to
work because `FalkorSubstrateStore`'s in-process cache retains the full
Python object within the same long-lived process, and this patch's own
tests all used hand-rolled fake stores that record the raw node object
without round-tripping through the real codec -- so the gap was invisible
to the test suite that shipped with it. Directly contradicted the patch's
stated goal.

Fixed by adding the key to `DYNAMICS_METADATA_KEYS`,
`_dynamics_properties_from_metadata()` (encode), and
`_dynamics_metadata_from_row()` (decode) -- same "omit when absent/empty"
convention `contributing_turn_ids` already established. Also fixed a
second, previously-undiscovered instance of the *exact same class of bug*
one file over: `orion/substrate/falkor_store.py`'s raw-key -> encoded-key
translation for `EXTERNALLY_OWNED_METADATA_KEYS` was a hardcoded ternary
that only knew about `contributing_turn_ids`'s `_json` suffix -- adding
the new key to `EXTERNALLY_OWNED_METADATA_KEYS` without updating that
ternary would have let `SubstrateDynamicsEngine.tick()`'s own periodic
node re-writes silently clobber fresh evidence back to stale/absent
shortly after `_write_prediction_error_node()` wrote it, the same
"frozen field" incident already documented (and fixed) for
`prediction_error`/`contributing_turn_ids` on 2026-07-29. Caught by this
repo's own pre-existing drift-guard test
(`test_externally_owned_metadata_keys_translation_matches_real_encoding`,
`orion/substrate/tests/test_falkor_store.py`) failing immediately once the
new key was added to the allowlist -- confirmed live before the fix, not
assumed. Fixed by moving the raw->encoded suffix mapping into a real
constant (`JSON_SUFFIXED_EXTERNALLY_OWNED_METADATA_KEYS`,
`falkor_codec.py`) that `falkor_store.py` imports, instead of a
hand-maintained ternary -- the next key added to
`EXTERNALLY_OWNED_METADATA_KEYS` can't silently repeat this.

Also fixed (lower severity, both from the same review pass):
- `_write_prediction_error_node()`/`_brain_frame_prediction_error_evidence_by_domain()`
  only surface a domain's evidence when it's genuinely non-empty -- a
  calm domain's `0.0` scalar reading has no matching `[]` evidence entry.
  This is a real, intentional asymmetry (not a bug): the Falkor codec's
  own established convention already omits empty lists on decode, so
  chasing full key-parity with `prediction_error_by_domain` would fight
  that convention for no functional gain (every current consumer treats
  "key absent" and "key present as `[]`" identically). Docstrings/tests
  corrected to state this honestly rather than overclaim a mirror
  guarantee that doesn't hold end-to-end.
- `_brain_frame_prediction_error_by_domain()` and
  `_brain_frame_prediction_error_evidence_by_domain()` were being called
  back-to-back over the identical node snapshot at both live call sites,
  walking it twice for no reason. Both call sites now use a new
  `_brain_frame_prediction_error_and_evidence_by_domain()` single-pass
  helper; the two individually-named methods still exist (they have their
  own direct unit tests) but now delegate to it.

Additional tests from this fix-up pass: 8 in
`test_falkor_codec.py`/`test_falkor_store.py` (encode/decode round-trip,
3 fail-open/omit cases, the corrected drift-guard mapping), 2 in
`test_worker_prediction_error_evidence_by_domain.py` (the combined helper
matches its two individual siblings). Full regression re-run after this
fix-up: `pytest orion/substrate/tests/` (508 passed),
`pytest services/orion-substrate-runtime/tests/ -k "worker or
prediction_error or brain_frame"` (139 passed, same 3 pre-existing
unrelated failures) -- plus a `git stash` comparison confirming a fuller,
unfiltered `services/orion-substrate-runtime/tests/` run's additional
failures (`test_quarantine_truth.py`, `test_reducer_health.py`,
`test_cursor_reset_auth.py` -- auth-token/env-dependent, not this patch)
pre-exist on the branch baseline too, equal or worse.

This closes the retention and join-key gaps the adversarial pass found.
Still not done: the "co-occurrence, not per-event causation" language
constraint (documented on the new schema field itself, not enforced in
code -- there is no consumer yet to enforce it against), and any actual
narrator/consumer that reads this field. Both remain future,
proposal-mode-scoped work.
