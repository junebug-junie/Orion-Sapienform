# Chat-level attention ground-truth gap finding

GitHub issue #1512 (Sentience Striving Program) asked whether chat-level/coalition open-loop
attention salience (`orion/substrate/attention/salience.py`, `SEED_WEIGHTS`,
`weights_version="seed-v1"`) is redundant with Layer 5 field attention
(`orion/attention/field_attention/`, Candidate A/B, PR #1484/#1488) or the AST/HOT reducer
(`AttentionSelfModelV1`), per the charter's (`orion/sentience_striving_program/README.md`)
sec 7 rule: "Reuse the live pipeline, don't parallel it... before being built."

Script: `scripts/analysis/measure_chat_attention_ground_truth_gap.py`. Read-only, no
writes/events/flags/consumer changes. 19 unit tests, all passing, no DB required. Run
2026-07-30, real Postgres data, 168h (7-day) window.

## What was checked

1. **Target-universe disjointness**: does chat-level salience's `attention_salience_trace.
   theme_key` id space overlap Layer 5's `substrate_attention_frames` `target_id` space
   (`FieldAttentionFrameV1`'s `dominant_targets`/`node_targets`/`capability_targets`/
   `system_targets`/`suppressed_targets`)?
2. **AST/HOT winner traceability**: does every `AttentionSelfModelV1.broadcast_selected_
   open_loop_id` value correspond to a real `theme_key` that was already being scored at or
   before that self-model tick -- i.e. is AST/HOT narrating a real, independently-tracked
   loop, or a second scorer producing its own ids?
3. **Ground-truth outcome gap**: how many rows exist in `attention_loop_outcome` (the
   operator Resolve/Dismiss verdict table, written only by `services/orion-hub/scripts/
   attention_loops_store.py::persist_loop_outcome`, called only from `attention_loops_
   routes.py`'s Resolve/Dismiss routes) in the same window `attention_salience_trace`
   covers?
4. **Which hand-weighted formula is live**: `salience.py`'s `SEED_WEIGHTS` blend (v2, gated
   by `ORION_ATTENTION_SALIENCE_V2_ENABLED`) or `scoring.py::score_loop`'s separate inline
   fallback formula, read the same way `salience.py`'s own `salience_v2_enabled()` does (same
   flag, same truthy set, imported directly).

## Real numbers (168h window, 2026-07-30)

- `attention_salience_trace`: 304 rows, 5 distinct `theme_key` values (open loops) since
  2026-07-24.
- `substrate_attention_frames`: 124,816 rows, 19 distinct `target_id` values (7
  `capability:*`, 1 `field:*`, 11 `node:*`) since 2026-07-27.
- **Overlap: 0.** Confirmed disjoint, computed live, not assumed.
- `substrate_attention_self_model`: 4,823 rows in its own ~168h retention window; 394 name a
  non-null `broadcast_selected_open_loop_id`.
- **Winner traceability: 394/394 (100%).** Every named winner is a real `theme_key` whose
  earliest `attention_salience_trace` sighting is at or before the self-model tick that names
  it.
- **`attention_loop_outcome`: 0 rows, ever, in this window.** Re-checked live, not carried
  forward from an earlier session's estimate.
- **Live formula: `salience.py::compute_salience` (SEED_WEIGHTS blend, v2)** --
  `ORION_ATTENTION_SALIENCE_V2_ENABLED` raw value confirmed `"true"` at run time, resolved
  with the same truthy check `salience_v2_enabled()` itself uses.

## Interpretation

**Issue #1512's redundancy premise does not hold, on two independent grounds:**

1. Chat-level salience and Layer 5 field attention score disjoint real object types
   (conversational open loops vs. physical/capability/system targets) -- there is no shared
   id space for either mechanism to be "redundant" with the other at the target level.
2. AST/HOT is not a second, competing scorer of open loops. Its
   `broadcast_selected_open_loop_id` field is a narrative wrapper around whatever the
   coalition (`salience.py`/`scoring.py`) already decided -- confirmed by the 100%
   traceability result, not just by reading `AttentionBroadcastProjectionV1`'s source. No
   redesign or retirement is justified by a redundancy argument; there is no redundancy to
   resolve.

**The load-bearing finding is the ground-truth gap, not the redundancy question.**
`attention_loop_outcome` has zero rows across the entire real history `attention_salience_
trace` covers (2026-07-24 onward, 6+ days, 304 real salience-scoring events across 5
distinct loops). Neither of the two hand-weighted formulas in this codebase --
`salience.py`'s live `SEED_WEIGHTS` blend or `scoring.py`'s dormant inline fallback -- has
ever been checked against a real human or automated verdict on whether a surfaced loop was
actually worth surfacing. This is structurally the same shape as `measure_autonomy_gate.py`'s
finding that the autonomy origination signal never fired once in 84,511 ticks: a mechanism
that computes continuously but has never touched real ground truth.

## What this does NOT decide

- Does not decide whether to retire, merge, or redesign chat-level attention salience.
- Does not validate either salience formula's real-world accuracy -- there is no ground-truth
  data to validate against.
- Does not itself close the outcome-logging gap.

## Recommended next step

A design/proposal-mode decision by Juniper on how to close the outcome-logging gap -- e.g.
wiring the operator Resolve/Dismiss UI (`services/orion-hub/scripts/
attention_loops_routes.py`) into real, regular operator usage, or building an automatic
ground-truth proxy (e.g. inferring an implicit "resolved" signal from later chat turns that
reference the same loop without re-raising it). Not attempted in this patch -- this script's
own scope is establishing whether ground truth exists (it does not), not building the fix.

## Source material

- `scripts/analysis/measure_chat_attention_ground_truth_gap.py` -- the measurement itself.
- `orion/substrate/attention/salience.py` -- the live SEED_WEIGHTS formula.
- `orion/substrate/attention/scoring.py` -- the dormant inline fallback formula
  (`score_loop`).
- `orion/schemas/attention_self_model.py` -- `AttentionSelfModelV1`, confirming
  `broadcast_selected_open_loop_id` is sourced from the GWT-dispatch/broadcast lane, not a
  second scoring pass.
- `orion/schemas/field_attention_frame.py` -- `FieldAttentionFrameV1`'s real target-id
  convention (`node:*`/`capability:*`/`field:*`/`system:*`).
- `services/orion-hub/scripts/attention_loops_routes.py` /
  `services/orion-hub/scripts/attention_loops_store.py` -- the operator Resolve/Dismiss API,
  `attention_loop_outcome`'s only writer.
- `orion/sentience_striving_program/README.md` sec 7 -- "Reuse the live pipeline, don't
  parallel it" rule this measurement is answering.
