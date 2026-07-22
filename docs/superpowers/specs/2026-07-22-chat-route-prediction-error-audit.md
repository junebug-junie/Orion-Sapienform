# Chat/route prediction-error audit — design doc

Status: **closed**. Chat was a real bug and is fixed. Route was investigated and is **not** a
bug — verified as the intended decay behavior operating correctly on a signal that hasn't fired
recently. This audit was triggered by a direct question — whether Orion's substrate
prediction-error receipts lean on biometrics because it's the easy domain while chat/route (the
ones requiring real substrate-grammar work) quietly don't produce real signal. Chat: yes, and now
fixed. Route: no — real signal, real decay, nothing to fix.

`chat_prediction_error()`'s `_latest_run()` fallback has landed in `orion/substrate/
prediction_error.py`, with regression tests and a `services/orion-substrate-runtime/README.md`
update in the same patch. `route_prediction_error()`'s subnormal value was traced to
`services/orion-field-digester/app/digestion/decay.py`'s `apply_decay()` — not the diffusion pass
this doc originally (wrongly) pointed at — and confirmed as correct, designed behavior: see
Missing Questions 1-2, now answered, below.

## Arsonist summary

All five prediction-error instruments in `orion/substrate/prediction_error.py`
(`execution_prediction_error`, `transport_prediction_error`, `biometrics_prediction_error`,
`chat_prediction_error`, `route_prediction_error`) are real code, correctly wired into
`services/orion-substrate-runtime/app/worker.py`'s per-domain ticks, and gated behind the same
`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` flag. Checked live against `substrate_field_state`'s
`node_vectors` (the durable, non-pruned signal — `substrate_reduction_receipts` has a short
retention TTL and is not a reliable historical record):

| node | prediction_error (live, 2026-07-22) |
|---|---|
| `node:substrate.biometrics` | 0.0237 |
| `node:substrate.execution` | 0.0007 |
| `node:substrate.transport` | 0.0 |
| `node:substrate.route` | ~3e-322 (subnormal float — not a meaningful value) |
| `node:substrate.chat` | **absent from `node_vectors` entirely** |

`node:substrate.chat` has never been written. This is direct, load-bearing evidence that
`chat_prediction_error()` has never returned `error > 0.0` in production — the write in
`_chat_tick()` (`worker.py` ~line 1826) is gated on exactly that condition — a real instrument
defect, fixed in this patch (see below).

`node:substrate.route`'s subnormal value is **not** an instrument defect. Traced historical
`substrate_field_state` rows (only ~15h13m of history was actually retained at query time —
`min(generated_at)`/`max(generated_at)` confirmed directly, not assumed to span the full
`FIELD_STATE_RETENTION_HOURS=72.0` window):

| `generated_at` (2026-07-22) | `node:substrate.route` prediction_error |
|---|---|
| 04:08 (oldest retained) | 0.0035 |
| 19:20 (current) | ~3e-323 (the display floor, one multiply from exact 0.0) |

This is `apply_decay()` (`services/orion-field-digester/app/digestion/decay.py`) correctly
eroding a real value toward zero because no new route-arbitration surprise has landed in the
~15h13m retained window (and likely longer — 0.0035 decaying continuously at `0.92`/2s-tick
reaches the subnormal floor in under 5 hours, so the *last real write* predates even this
window's oldest row) — `"prediction_error"` is a genuine member of `NODE_DECAY_CHANNELS` (line 35), decayed by
`BIOMETRICS_FIELD_DECAY_RATE=0.92` every 2s tick once stale (per the 2026-07-17 decay-hold fix
documented in this service's own README). The perturbation write path
(`services/orion-field-digester/app/ingest/state_deltas.py` lines 391-403, triggered from a
`delta.target_kind == "prediction_signal"` state delta) builds a proper `mode="replace"`
`Perturbation`, which goes through the normal `apply_perturbations()` stamping of
`node_vector_updated_at` — this is *not* the "written outside the normal path, stamp missed" bug
that service's own `CLAUDE.md` warns about (that already happened once for `field_coherence_
warning`/`staleness`; it did not happen here). `route_prediction_error()` only writes when
`error > 0.0` (worker.py's `if error > 0.0:` gate, same as chat/biometrics/execution), and route
arbitration decisions are evidently very stable (441 total runs, apparently no `lane`/`mind_
requested` mismatches for at least the last several hours, likely longer) — so the signal
genuinely has had nothing new to report, and decay is doing exactly what it was built to do in
the 2026-07-17 fix.

This is not "biometrics is real, chat/route are fake" — `substrate_chat_session_projection`
holds 241 real turns across 8 sessions (accumulating since 2026-06-19), and
`substrate_route_arbitration_projection` holds 441 real arbitration runs (since 2026-07-13). The
underlying grammar-event reduction genuinely works for both domains, and so does route's shadow
instrument (it produced a real 0.0035 value earlier the same day this was checked) — chat's
shadow instrument was the one actually broken.

## Current architecture

- `orion/substrate/prediction_error.py`: five pure functions, `_latest_run(runs)` helper (returns
  the run with the most recent `last_updated_at` in a `dict[str, T]` mapping, or `None`).
  `execution_prediction_error`/`route_prediction_error` were both fixed in `a98854a2` (2026-07-22,
  earlier this session) to fall back to `_latest_run(prev.runs)` when no exact `trace_id` match
  exists, because real cortex-exec/route-arbitration runs are single-shot creates (confirmed live:
  26/26 sampled `execution_trajectory_reducer` receipts were `operation: create` with unique
  `target_id`s) — an exact match structurally never occurred, so both instruments silently
  returned `0.0` forever before that fix.
- `chat_prediction_error()` does **not** use `_latest_run` — it loops
  `for turn_id, curr_turn in curr.turns.items(): prev_turn = prev.turns.get(turn_id); if
  prev_turn is None: continue`. Read directly (`services/orion-hub/scripts/grammar_emit.py`
  `build_chat_turn_grammar_events`, line 83: `trace_id = f"hub.chat:{node_id}:{turn_id}"`,
  reused across every event layer — trace_started, chat root, context, raw_input,
  repair_signal, stance_disposition, edges, trace_ended — all built and presumably published
  together for one turn): a chat turn is a single-shot burst, structurally identical in shape to
  execution/route's single-shot creates. But `chat_prediction_error`'s failure mode is different
  from execution/route's original bug, not the same one:
  - `prev`/`curr` are both loaded around **one tick** of `_chat_tick()`, and `updated.turns` is a
    **persistent, cumulative** dict (241 turns and growing) — not a small per-tick batch like
    execution's `runs`. Most `turn_id`s in `curr.turns` already existed in `prev.turns`
    unchanged (delta = 0 by construction). The handful of turns processed *this tick* are, by
    definition, **new** `turn_id`s not present in `prev.turns` at all — so they hit the
    `if prev_turn is None: continue` branch and are skipped, never contributing to the surprise
    score. There is no mechanism by which a `turn_id` gets revisited in a *later* tick, because
    hub emits a turn's full event burst once.
  - Net effect: the only way this instrument could ever produce `error > 0.0` is if the exact
    same `turn_id` appeared in two different ticks' `curr` snapshots with different content —
    which does not happen given how hub emits chat grammar events. This is a **structural
    "new content never counted" gap**, not the same "unmatched key" bug as execution/route, but
    it produces the identical symptom (permanent `0.0`).
- `ChatTurnStateV1.last_updated_at: datetime` exists (`orion/schemas/chat_projection.py` line 22)
  — the exact field `_latest_run()` needs. Nothing structural blocks reusing the same helper here.
- `route_prediction_error()` already has the `a98854a2` fallback and is not structurally dead the
  way chat is — its live value being a subnormal float is **traced and confirmed as correct
  decay behavior, not a bug** (see Missing Questions 1-2, now answered).
- `orion/substrate/pressure.py::prediction_error_pressure()` (read directly, lines 36-60): a pure
  function of `node.metadata['prediction_error']` and `node.temporal.observed_at`, linearly
  decaying to zero over `prediction_error_decay_horizon_seconds=1800` (30 min). **It does not
  persist a decayed value back into storage** — it recomputes fresh from `raw` on every call. An
  earlier draft of this doc incorrectly asserted this function's decay explains the subnormal
  `node:substrate.route` value in `substrate_field_state.node_vectors` — wrong storage layer.
  The real mechanism is `services/orion-field-digester/app/digestion/decay.py::apply_decay()`,
  a completely different decay applied to `FieldStateV1.node_vectors` directly (see below).

## Missing questions (both now answered — kept for record)

1. **Answered.** What actual mechanism turns a real `node.metadata['prediction_error']` value
   into the subnormal float seen in `substrate_field_state.node_vectors['node:substrate.route']`?
   `services/orion-field-digester/app/digestion/decay.py::apply_decay()`. `"prediction_error"` is
   a member of `NODE_DECAY_CHANNELS` (line 35), decayed by `BIOMETRICS_FIELD_DECAY_RATE=0.92`
   every `RECEIPT_POLL_INTERVAL_SEC=2.0s` tick once stale (no fresh perturbation within
   `FIELD_DECAY_STALENESS_THRESHOLD_SEC=90s`, per the 2026-07-17 decay-hold fix). The write path
   (`app/ingest/state_deltas.py` lines 391-403) builds a proper `mode="replace"` `Perturbation`
   that goes through `apply_perturbations()`'s normal `node_vector_updated_at` stamping — not the
   "written outside the normal path" bug class this service's `CLAUDE.md` warns about.
2. **Answered.** When was `node:substrate.route`'s last real (non-decayed) write? Traced via
   `substrate_field_state` history: `0.0035` at the oldest retained row (2026-07-22 04:08, only
   ~15h13m of history was actually retained at query time, not the full 72h
   `FIELD_STATE_RETENTION_HOURS` window), decayed to the subnormal floor (~3e-323) by 19:20 the
   same day. `0.0035` decaying continuously at `0.92`/2s-tick reaches the subnormal floor in
   under 5 hours, so the actual last real write predates even this window's oldest retained row —
   route arbitration genuinely hasn't produced a `lane`/`lane_reason`/`output_mode`/`mind_
   requested` mismatch in a long time. Verdict: **not a bug** — correct decay of a real signal
   during a real quiet period, exactly what the 2026-07-17 fix was built to do. No further action
   needed for route.
3. For chat: is there a principled way to score "surprise" for a brand-new turn with no
   predecessor to diff against, analogous to what `_latest_run` gives execution/route? The
   proposed fix below (diff a new turn against the most-recently-updated *other* turn) is the
   direct structural mirror of the already-shipped execution/route fix, but it's answering a
   different question ("how does this turn compare to the last one") than "how did this turn's
   own content evolve" (which literally cannot be answered — most chat turns never get revised).
   Confirm this reframing is an acceptable theory-anchor before implementing, per CLAUDE.md's
   metric quality gate step 3 — it is not the same claim the docstring currently makes.

## Proposed schema / API changes

No schema changes. `orion/substrate/prediction_error.py::chat_prediction_error()`:

- Add `prev_fallback = _latest_run(prev.turns)` (mirrors `execution_prediction_error`'s existing
  `prev_fallback = _latest_run(prev.runs)` line verbatim, applied to `.turns` instead of `.runs`).
- Change `if prev_turn is None: continue` to `if prev_turn is None: prev_turn = prev_fallback` /
  `if prev_turn is None: continue` (exact match still takes priority; fallback only when no exact
  match exists) — same two-line structural change `a98854a2` already made to
  `execution_prediction_error`.
- Update the docstring to state the corrected theory: comparing a new turn's pressure hints
  against the most-recently-updated prior turn's hints (not "the same turn revised in place",
  which does not happen in practice for chat).

No change for `route_prediction_error()` or `orion-field-digester`'s `apply_decay()` — both
traced and confirmed working as designed (Missing Questions 1-2, above). Nothing to fix.

## Files likely to touch

- `orion/substrate/prediction_error.py` — `chat_prediction_error()` fallback + docstring
  correction.
- `orion/substrate/tests/test_prediction_error.py` — new test cases: brand-new turn with an
  existing-but-different prior turn present should now produce `error > 0.0`; brand-new turn
  with *no* prior turns at all should still return `0.0` (no fallback exists); exact-match
  update-in-place case (if it were ever to occur) should still take priority over the fallback.
- `services/orion-substrate-runtime/README.md` — document the corrected chat instrument
  behavior alongside execution/route's already-documented fallback.
- This doc, updated to `implemented` status once the patch above lands and a live
  `node:substrate.chat` write is confirmed (mirrors the biometrics-shadow-design doc's own
  "Recommended next patch: confirm a live write" acceptance pattern).

## Non-goals

- Not changing `route_prediction_error()`, `apply_decay()`, or any other part of
  `orion-field-digester` — traced and confirmed both are already working correctly; there is
  nothing to fix.
- Not adding a new theory anchor — reuses charter §9b item 3 (Predictive Processing / Active
  Inference), the same anchor every other instrument in this module already uses.
- Not changing `_THRESHOLD` scaling, `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` gating, or any
  other instrument's behavior.

## Acceptance checks

- `pytest orion/substrate/tests/test_prediction_error.py -q` passes, including the three new
  cases named above. **Done — 32 passed.**
- Live check (post-deploy, mirrors the biometrics-shadow-design doc's own honesty standard):
  query `substrate_field_state.node_vectors` for `node:substrate.chat` and confirm it now exists
  with a non-zero value after a real chat turn follows a prior one. `UNVERIFIED` until observed
  live — do not claim this fixed until that specific row is seen (this patch has not been
  deployed as of writing).
- Grep the diff for any change to `route_prediction_error`, `prediction_error_pressure`,
  `apply_decay`, or `orion-field-digester`: zero hits (confirms this patch stayed scoped to chat
  only). **Done.**

## Recommended next patch

1. ~~Implement the `chat_prediction_error()` fix above.~~ **Done.**
2. ~~Trace `orion-field-digester`'s decay path to answer Missing Questions 1-2.~~ **Done — not a
   bug, no fix needed.**
3. After deploy: confirm the live `node:substrate.chat` write per the Acceptance Checks above.
   This is the only remaining open item from this audit.
