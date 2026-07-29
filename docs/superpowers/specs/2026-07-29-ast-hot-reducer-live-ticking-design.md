# AST/HOT reducer live-ticking — design spec (prerequisite for CollapseMirror's "insight" trigger)

Status: **design mode, not implemented.** Touches self-modeling instrumentation (AST/HOT), which
CLAUDE.md §0A requires explicit proposal mode for before implementation. This document proposes; it
does not build.

## Arsonist summary

`docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`'s Missing
Question 1 asked whether `reduce_attention_self_model()` ticks live anywhere. Answer, confirmed this
session by grep across both that doc's arc and this one: **no.** It's a pure function with two real
callers, both offline (`test_attention_self_model.py`, `measure_ast_hot_reducer.py`). This doc scopes
the prerequisite patch: what it takes to make it tick live, safely, given this repo's specific and
repeated history of new/existing ticks clobbering fields they don't own.

**Correction, review finding 2026-07-29 (after implementation): this premise was wrong.**
`_brain_frame_tick()` (`services/orion-substrate-runtime/app/worker.py`,
`SUBSTRATE_BRAIN_FRAME_ENABLED` default `true`) already calls `reduce_attention_self_model()` live —
a narrower computation (`field_frame=None`, no trend) whose result is embedded inline into one
brain-frame UI dimension, never persisted as its own durable artifact. The grep that produced "no"
above missed it (root cause not re-investigated — possibly a stale search state mid-session, possibly
a concurrent commit landing between the grep and this doc being written). This doesn't change the
patch's core conclusion — a *complete, durable* self-model (real field lane + real trend, persisted
for future consumers like CollapseMirror's "insight" trigger) still didn't exist and still needed
building — but "reduce_attention_self_model() ticks live anywhere" should have been answered "yes,
narrowly," not "no." See `_attention_self_model_tick()`'s own docstring
(`services/orion-substrate-runtime/app/worker.py`) for the full reasoning on why the two live
computations are left intentionally un-unified rather than refactored to share one call.

## Current architecture

- **Field lane — already live.** `services/orion-attention-runtime` polls `substrate_field_state`
  every `ATTENTION_POLL_INTERVAL_SEC` (2s default), builds `FieldAttentionFrameV1`, persists to
  `substrate_attention_frames`. Idempotent per `tick_id`. No bus traffic besides its own
  `SystemHealthV1` heartbeat. Single-table owner — nothing else writes `substrate_attention_frames`.
- **Broadcast lane — already live.** `orion-substrate-runtime::_attention_broadcast_tick()`
  (worker.py:1545), 30s cadence. Writes a singleton (`substrate_attention_broadcast_projection`,
  `save_attention_broadcast`) plus an append-only companion
  (`substrate_attention_broadcast_log`, `save_attention_broadcast_history`).
- **`prediction_error_by_domain` — already live**, per-domain, via the FalkorDB
  `node:substrate.<domain>` nodes (`_write_prediction_error_node`, six domains including
  `bus_synaptic`, `orion-substrate-runtime`). This session fixed the last known clobber bug on this
  path (PR #1449).
- **`prediction_error_trend_by_domain` — not live anywhere.** Only
  `measure_ast_hot_reducer.py::compute_prediction_error_trend()` computes it, over historical
  `substrate_field_state` rows in an offline replay. No live producer exists.
- **`AttentionSelfModelV1` (the reducer's output) — has no durable home.** Registered in
  `orion/schemas/registry.py` as a known schema kind (`"attention.self_model.v1"`), but before this
  patch: zero tables, zero bus channels, zero standalone consumers. `_brain_frame_tick()` computes
  one live (see correction above) but only embeds it inline in a brain-frame region, never persists
  the model itself.
- **`reduce_attention_self_model()` itself — pure, no I/O, already correct.** Takes already-parsed
  inputs, never raises, handles any input being `None`. No changes needed to the reducer to make it
  tick live; the gap is entirely in assembly (who gathers the inputs) and persistence (where the
  output goes), not the reduction logic itself.

### This repo's specific, repeated failure pattern (why this design is conservative)

Three confirmed live incidents, same root cause, different services:

1. **`execution_load` cross-lane stomp** (PR #1338, [[project_execution_load_rename_and_schema_drift_gate]]):
   a lane's structurally-always-zero delta overwrote cortex-exec's real value via `mode="replace"`,
   because the emitting loop didn't know it didn't own that field on that lane.
2. **`orion-field-digester`'s generic decay clobber** (documented directly in this repo's CLAUDE.md
   metric-quality-gate section): `NODE_DECAY_CHANNELS`' generic staleness-decay loop silently
   multiplied `prediction_error` by 0.92 every tick for 48+ hours because nothing told it that field
   was externally owned and already fresh.
3. **`SubstrateDynamicsEngine.tick()` clobbering `bus_synaptic`'s `prediction_error`** (PR #1449,
   fixed earlier in this same session): a generic write-guard (fires on activation decay alone,
   which is nearly every tick) re-persisted a stale snapshot-time copy of a field it only ever read,
   never computed, clobbering the real writer's fresh values for 3+ hours and causing real false
   alerts.

Every one of these is the same shape: **a periodic tick touches a field/row/table it does not
exclusively own, using a generic/unconditional write path that has no concept of field ownership.**
Not three unrelated bugs — one recurring architectural gap in this codebase. Any new tick proposed
here must be designed to make this bug class structurally impossible, not just avoided by discipline.

## Missing questions

1. ~~Which service owns the new tick?~~ **Resolved: `orion-substrate-runtime`.** Checked whether
   `orion-attention-runtime`/`orion-policy-runtime`/`orion-proposal-runtime`'s narrow
   "one-service-per-L-layer" pattern or `orion-substrate-runtime`'s broad catch-all is the real repo
   convention — it's the latter: dynamics, bus_synaptic, brain_frame, attention_broadcast,
   drive-state materialization, perception ingestion, and prediction-error writes already all
   converge there. "That's where substrate ticks go" is a real, documented pattern, not an
   assumption. The safety property that actually matters — *this tick only reads existing state and
   writes only to a table it exclusively owns* — is orthogonal to which service hosts it; enforced
   that way, the three-incident failure class above becomes structurally impossible regardless of
   host. Hosting in `orion-substrate-runtime` also has a real practical win: the broadcast lane and
   FalkorDB prediction-error client are already in-process there, so only the field lane (a
   cross-service Postgres read, already how the offline script itself does it) needs new plumbing —
   less new surface than starting fresh in `orion-attention-runtime`.

2. ~~How does `prediction_error_trend_by_domain` get computed live?~~ **Resolved: in-process
   rolling buffer, not a lookback query.** A bounded in-memory deque of the last N
   `prediction_error_by_domain` snapshots (N sized to match
   `PREDICTION_ERROR_TREND_WINDOW_TICKS`), computing `mean(prior half) - mean(recent half)` itself
   each tick. Cheaper than a repeated DB round-trip, and trivially safe — private per-process state,
   nothing to race against. On restart it starts empty and `predicted_shift` is honestly absent for
   one window, which the reducer already handles gracefully (never crashes on missing input,
   matches its existing "honest absence" convention). A lookback query would only buy "survives
   restart," which isn't worth a real query every tick here.
3. **Where does `AttentionSelfModelV1` get persisted?** New Postgres table
   (`substrate_attention_self_model`, matching every other lane's append-or-latest pattern) is the
   safe default — it is a **brand-new table that nothing else in the repo has ever written to**,
   which is precisely what makes this new tick immune to the three-incident failure class by
   construction: there is no pre-existing writer to race against, no field to accidentally not-own.
   A FalkorDB node was considered and rejected for this specific artifact: FalkorDB nodes are where
   the ownership bugs above actually happened (2 of 3 incidents), and `AttentionSelfModelV1` doesn't
   need graph relationships — it's a flat snapshot, a plain table is the honest fit.
4. ~~Cadence.~~ **Resolved: ride `_attention_broadcast_tick()`'s existing 30s timer directly, call
   the reducer at its tail — no new independent timer.** Now that this is hosted in the same
   service as the broadcast tick, inventing a third timer has no benefit: broadcast is already the
   slowest of the two real inputs (30s vs. field lane's 2s), so there's no resolution to gain by
   ticking AST/HOT any faster, and piggybacking avoids a timer that would mostly just replay the
   same broadcast/trend state between real changes.
5. **Does this new tick need write access to anything at all, ever?** Explicitly: no. Every one of
   its inputs (field lane, broadcast lane, FalkorDB prediction-error nodes) is read-only from this
   tick's perspective. This should be enforced structurally, not just by convention — the assembly
   function that gathers inputs should use existing read-only getters only (no new `upsert_node`/
   `save_*` calls added anywhere except the one new table this tick exclusively owns).
6. **Does the CollapseMirror doc's own Missing Question 2 (real historical shape of confidence
   recovery) become answerable once this ships, or does it still need its own separate measurement
   pass against the new live table?** Almost certainly still needs its own pass — this patch makes
   the *data* available live; it does not itself analyze whether confidence-recovery events are
   discrete/meaningful. Flagging so this patch isn't mistaken for closing MQ2 by itself.

## Proposed schema / API changes

- New Postgres table `substrate_attention_self_model` (exact DDL TBD at implementation time,
  mirroring `substrate_attention_frames`' shape: `tick_id`/`generated_at`/JSONB payload column,
  append-only, no upsert-by-fixed-key — avoids reintroducing the singleton-row schema-drift class
  `check_substrate_projection_schema_drift.py` was built to catch).
- No changes to `reduce_attention_self_model()` itself, `AttentionBroadcastProjectionV1`,
  `FieldAttentionFrameV1`, or any FalkorDB node shape — this patch is additive assembly + persistence
  only.
- No bus channel yet. Matches CollapseMirror doc's own Missing Question 6 reasoning and this
  session's own precedent (`orion-heartbeat`'s ensemble mechanism stayed read-only until Acceptance
  Check 1 was independently re-confirmed live) — publishing a signal before its live shape is
  verified risks CLAUDE.md §0A's "empty-shell cognition" failure.

## Files likely to touch

- `services/orion-substrate-runtime/app/worker.py` — call `reduce_attention_self_model()` at the
  tail of the existing `_attention_broadcast_tick()`, assembling `prediction_error_by_domain` from
  the same in-process FalkorDB store already used by `_write_prediction_error_node`/`_dynamics_tick`,
  plus a new bounded in-process deque for `prediction_error_trend_by_domain`, plus one new read of
  the field lane's `substrate_attention_frames` (cross-service Postgres read — the one genuinely new
  plumbing this patch needs), plus one new write call for `substrate_attention_self_model`
  (exclusively owned by this tick, no other writer added anywhere).
- `services/orion-substrate-runtime/app/store.py` — new read-only getter for the latest
  `FieldAttentionFrameV1` row, and one new write path for `substrate_attention_self_model`.
- `services/orion-sql-db/manual_migration_attention_self_model_v1.sql` (new) — matches this repo's
  existing manual-migration convention for `substrate_attention_frames`/`substrate_attention_broadcast_log`.
- `services/orion-substrate-runtime/.env_example` + synced local `.env` — new enable flag +
  trend-window-size setting (no new cadence setting needed — rides the existing broadcast-tick
  interval).
- `services/orion-substrate-runtime/README.md` — document the new tick alongside the existing ones.
- `orion/substrate/attention_self_model.py` — no logic changes; this module's own docstring never
  claimed "no live caller" (that was this doc's own now-corrected premise, not the module's).
- `scripts/analysis/measure_ast_hot_reducer.py` — no changes required for this patch, but its
  historical-replay role becomes partially redundant once live data accumulates; worth a follow-up
  note once this ships, not scoped here.

## Non-goals

- Not building CollapseMirror's `insight`/`flow` triggers themselves — that's the parent doc's own
  scope, gated on its own Missing Questions 2-6, which this patch does not answer.
- Not publishing `AttentionSelfModelV1` to any bus channel.
- Not touching `orion-substrate-runtime`'s existing ticks' own write paths (dynamics, bus_synaptic,
  brain_frame, attention_broadcast, drive-state, perception ingestion) — this patch adds one new
  read-only consumer of their already-live outputs plus one new exclusively-owned output table, not
  a change to any existing tick's behavior.
- Not deciding the exact trend-window size here — needs a live-data sanity check before being
  hardcoded (Missing Question 2's resolution fixes the *mechanism*, not the window-size constant).
- Not building `scripts/analysis/measure_attention_self_model_confidence_baseline.py` (CollapseMirror
  doc's own Missing Question 2 tooling) — separate patch, separate doc.

## Acceptance checks

1. The new tick runs for a real live window (several hours minimum) without any exception loop
   (fail-open, matching every other tick in this repo — a failure here must never crash the host
   service, same as `_dynamics_tick`'s own `try/except` wrapping).
2. `substrate_attention_self_model` accumulates real per-tick history with non-degenerate variance in
   `confidence`/`prediction_error_confidence` — same live-data sanity check this session already ran
   offline via `measure_ast_hot_reducer.py`, now checked against the *live* table instead of a replay.
3. Confirmed via direct query (same discipline as this session's `bus_synaptic` freeze investigation):
   no other writer ever touches `substrate_attention_self_model` — grep for the table name repo-wide
   after implementation finds exactly one writer.
4. No regression to `orion-substrate-runtime`'s existing `_attention_broadcast_tick()` cadence/
   behavior — the appended reducer call must not measurably extend that tick's own runtime past its
   30s interval (verify via the same tick-duration logging pattern already used elsewhere in that
   file), and must fail open (a reducer-call exception must not break the broadcast tick's own
   existing writes).
5. `prediction_error_trend_by_domain`'s live-computed value is checked against the offline script's
   historical formula for the same real window and confirmed to agree (same number, same sign
   convention) — a determinism/parity check before trusting the live version as a replacement for
   the offline one.

## Recommended next patch

1. Implement the new tick in `orion-substrate-runtime` per this doc's resolved Missing Questions
   1/2/4: appended to `_attention_broadcast_tick()`, in-process trend buffer, read-only against
   every existing live input, writing only to the new `substrate_attention_self_model` table.
2. Live-verify for several hours (Acceptance Checks 1-3) before considering this "done" — matching
   this session's own standard for every fix shipped today (the dynamics.py clobber fix wasn't
   considered closed until live ticks were watched directly, not just tests passing).
3. Only then: return to `docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`'s
   own Missing Question 2 (confidence-recovery shape measurement) — now against real live data
   instead of an offline replay.
