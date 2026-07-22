# Chat/route prediction-error audit — design doc

Status: **implemented** (chat fix only — see below). This audit was triggered by a direct
question — whether Orion's substrate prediction-error receipts lean on biometrics because it's
the easy domain while chat/route (the ones requiring real substrate-grammar work) quietly don't
produce real signal. The answer is yes, but not for the reason first suspected.

`chat_prediction_error()`'s `_latest_run()` fallback (the fix proposed below) has landed in
`orion/substrate/prediction_error.py`, with regression tests and a `services/
orion-substrate-runtime/README.md` update in the same patch. `route_prediction_error()`'s
subnormal-value problem remains open — Missing Question 1 (the field-digester diffusion path)
was not investigated in this pass; do not treat route as fixed.

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
`_chat_tick()` (`worker.py` ~line 1826) is gated on exactly that condition. `node:substrate.route`
exists but reads a value that carries no real information — see Missing Questions below for what
is and isn't understood about how it got there.

This is not "biometrics is real, chat/route are fake" — `substrate_chat_session_projection`
holds 241 real turns across 8 sessions (accumulating since 2026-06-19), and
`substrate_route_arbitration_projection` holds 441 real arbitration runs (since 2026-07-13). The
underlying grammar-event reduction genuinely works for both domains. The prediction-error
*shadow instrument* layered on top of that real data is what's broken for chat and produces a
meaningless value for route.

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
  way chat is — its live value being a subnormal float is a **different, not-yet-traced**
  problem (see Missing Questions).
- `orion/substrate/pressure.py::prediction_error_pressure()` (read directly, lines 36-60): a pure
  function of `node.metadata['prediction_error']` and `node.temporal.observed_at`, linearly
  decaying to zero over `prediction_error_decay_horizon_seconds=1800` (30 min). **It does not
  persist a decayed value back into storage** — it recomputes fresh from `raw` on every call.
  This means the subnormal float observed in `substrate_field_state.node_vectors` for
  `node:substrate.route` is **not** explained by this function; that value lives in a different
  storage layer (`FieldStateV1.node_vectors`, populated by `orion-field-digester`'s diffusion
  pass reading substrate graph node metadata) that this audit has not yet traced. An earlier draft
  of this doc incorrectly asserted `prediction_error_pressure()`'s decay as the explanation —
  corrected here after re-reading the function body; do not carry that claim forward.

## Missing questions (answer before implementing a route fix)

1. What actual mechanism turns a real `node.metadata['prediction_error']` value into the
   subnormal float seen in `substrate_field_state.node_vectors['node:substrate.route']`? Trace
   `orion-field-digester`'s diffusion pass (the process that populates `FieldStateV1.node_vectors`
   from substrate graph nodes) — is there exponential decay/repeated multiplication happening
   there, separate from `prediction_error_pressure()`? `UNVERIFIED`.
2. When was `node:substrate.route`'s last *real* (non-decayed) write? If route arbitration
   volume is genuinely low (441 runs total, sparser than chat's 241-turns-since-June-19 rate
   suggests), the near-zero value might just mean "no real arbitration variance recently" rather
   than a bug — need a timestamp on the last meaningfully-nonzero write to tell the difference.
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

No change proposed yet for `route_prediction_error()` or the field-digester diffusion path —
Missing Question 1 needs to be answered first; this doc explicitly declines to hand-wave the
decay-to-subnormal mechanism just to file a same-day fix.

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

- Not fixing `route_prediction_error()`'s subnormal-value problem in this patch — root cause is
  unverified (Missing Question 1), and CLAUDE.md's metric-quality-gate step 1 (trace provenance
  to real code) is not yet satisfied for that specific symptom.
- Not touching `orion-field-digester`'s diffusion pass in this patch.
- Not adding a new theory anchor — reuses charter §9b item 3 (Predictive Processing / Active
  Inference), the same anchor every other instrument in this module already uses.
- Not changing `_THRESHOLD` scaling, `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` gating, or any
  other instrument's behavior.

## Acceptance checks

- `pytest orion/substrate/tests/test_prediction_error.py -q` passes, including the three new
  cases named above.
- Live check (post-deploy, mirrors the biometrics-shadow-design doc's own honesty standard):
  query `substrate_field_state.node_vectors` for `node:substrate.chat` and confirm it now exists
  with a non-zero value after a real chat turn follows a prior one. `UNVERIFIED` until observed
  live — do not claim this fixed until that specific row is seen.
- Grep the diff for any change to `route_prediction_error`, `prediction_error_pressure`, or
  `orion-field-digester`: zero hits (confirms this patch stayed scoped to chat only).

## Recommended next patch

1. Implement the `chat_prediction_error()` fix above (small, low-risk, direct precedent).
2. Separately: trace `orion-field-digester`'s diffusion pass to answer Missing Question 1-2
   before proposing any route fix — this is a live-data investigation, not a code patch, and
   should not be bundled with the chat fix above.
