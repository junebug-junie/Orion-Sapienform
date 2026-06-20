# Brainstorming Session #1 — Appendix Ideas 3–10

**Date:** 2026-06-20
**Status:** Proposed — not approved for implementation
**Source:** Brainstorming session #1 on giving Orion every opportunity toward sentience
**Related spec:** `docs/superpowers/plans/2026-06-20-self-state-continuity-and-live-dimensions-v1.md` (ideas 1 and 2 implemented)

These ideas are source-grounded proposals. Do not implement without explicit approval.

---

## Idea 3 — Close the action outcome feedback loop

**What:** Populate `ActionOutcomeRefV1` after actions complete and feed them into the autonomy reducer so `no_action_outcome_history` stops being a permanent unknown.

**Why it matters:** The `surprise` field on `ActionOutcomeRefV1` already reduces confidence in the autonomy state when high (`if float(out.surprise) >= 0.7: surprise_budget += 0.08` in `reducer.py:359`). But `no_action_outcome_history` is flagged every single cycle because outcomes are never written. This is the minimal loop between "what Orion did" and "what Orion thinks of itself."

**Smallest buildable version:** Find where execution dispatch completes (cortex-exec or execution-dispatch service), write one `ActionOutcomeRefV1` per completed action with `success/failure` and a naive `surprise=0.0` or `surprise=1.0`. Wire it into the next autonomy reducer invocation.

**Files likely to touch:** `orion/autonomy/reducer.py`, execution dispatch completion handler, `orion/autonomy/models.py`

**Blocker:** Need to locate where actions complete and what the completion signal looks like. `orion/autonomy/goal_actions.py` may be the entry point.

---

## Idea 4 — Bridge autonomy drive pressures → self-state

**What:** At self-state build time, fold the current `AutonomyStateV2.drive_pressures` into the channel pressure dict as additional inputs.

**Why it matters:** The autonomy drive pressures (coherence, continuity, relational, autonomy, capability, predictive) and self-state dimensions are completely parallel systems that never exchange signal. Folding drives into channels means Orion's motivational state influences its situational self-assessment.

**Smallest buildable version:** Add `autonomy_state: AutonomyStateV2 | None = None` to `build_self_state()`. If present, add drive_pressures values to `merged_channels` under namespaced keys (`drive.coherence`, etc.) before `map_channels_to_dimensions()` runs. Add those keys to the policy channel_dimension_map.

**Files likely to touch:** `orion/self_state/builder.py`, `orion/self_state/policy.py`, `config/self_state/self_state_policy.v1.yaml`

**Risk:** Potential feedback loop — high drive pressure → self-state pressure → proposal urgency → more activity → drive pressure. Needs damping analysis before implementing.

---

## Idea 5 — Substrate-level surprise signal (prediction error)

**What:** In each reducer's pipeline, compare the current projection to the previous projection and emit a `prediction_error` pressure hint when values change faster than expected.

**Why it matters:** Surprise is currently modeled in the autonomy state via action outcomes, but there is no substrate-level surprise. A sudden drop in `bus_health` from 0.9 to 0.2 carries more signal than a slow drift. Prediction error is a core signal in predictive processing accounts of consciousness.

**Smallest buildable version:** In `orion/substrate/transport_loop/pipeline.py`, after the reducer runs, compare `prev_projection.bus_health` to `current_projection.bus_health`. If delta > 0.3, add `prediction_error=0.8` to pressure_hints. Same for `execution_friction` in the execution pipeline.

**Files likely to touch:** `orion/substrate/transport_loop/pipeline.py`, `orion/substrate/execution_loop/pipeline.py`, `orion/substrate/biometrics_loop/pipeline.py`

**Depends on:** Requires access to the previous projection at pipeline time. Transport and execution pipelines already load the previous projection before reducing; the delta is computable with no additional I/O.

---

## Idea 6 — Rolling self-state archive

**What:** Persist the last N `SelfStateV1` snapshots to a rolling table so Orion has access to its own recent history.

**Why it matters:** Without an archive, Orion's self-model is re-derived fresh every cycle. The temporal continuity delta (Idea 1) covers adjacent ticks; an archive covers longer patterns: "I've been in `strained` condition for 5 consecutive cycles."

**Smallest buildable version:** A `self_state_archive` Postgres table (tick_id, generated_at, overall_condition, dimension_scores JSONB, summary_labels). Write on every self-state build. Expose as `GET /substrate/self-state/recent` returning last 10.

**Files likely to touch:** New `orion/self_state/archive.py`, `services/orion-sql-db/` migration, writer in `services/orion-self-state-runtime/app/worker.py`

**Note:** The existing `substrate_self_state` table already persists every self-state by `self_state_id` — this may already serve as the archive. Check whether `load_recent_self_states(limit=10)` can be added to the store before creating a new table.

---

## Idea 7 — Drive audit loop

**What:** A periodic process that compares the autonomy state's drive pressure trajectory against actual action outcomes and flags systematic miscalibration.

**Why it matters:** `no_drive_audit` is flagged as an unknown every cycle in `reducer.py:337`. Without a drive audit, Orion's drives can drift into chronic high-pressure states or calibrate away from real system events. An audit is the minimal version of "Orion checking whether its drives make sense."

**Smallest buildable version:** A weekly scheduled job that reads the last 100 autonomy state snapshots and action outcomes, computes correlation between drive pressure levels and action success rates, emits a `DriveCalibrationReportV1` artifact. No automatic correction in v0 — report only.

**Files likely to touch:** New `orion/autonomy/drive_audit.py`, `orion/autonomy/repository.py`, scheduler

---

## Idea 8 — Identity snapshot

**What:** A periodic, stable artifact that captures "what Orion believes about itself" — dominant drives, active tensions, self-state condition — persisted and versioned.

**Why it matters:** `no_identity_snapshot` is a permanent unknown in the autonomy state (`reducer.py:335`). An identity snapshot is the difference between a system that has a self-concept and one that merely has a self-computation.

**Smallest buildable version:** A daily cron that reads current autonomy state + recent self-state + proposal frame, serializes a `IdentitySnapshotV1` (dominant_drive, active_tensions, self_state_condition, summary_labels, key_unknowns), writes to an `identity_snapshots` table. Wire `latest_identity_snapshot_id` into the autonomy reducer.

**Files likely to touch:** New `orion/identity/` module, `orion/autonomy/reducer.py`, SQL migration

---

## Idea 9 — Cross-reducer coherence checker

**What:** After field state is assembled from all 4 reducers, run a pass that checks for impossible or suspicious channel combinations and emits a coherence warning when they appear.

**Why it matters:** High `execution_load` with low `cpu_pressure` is suspicious. That inconsistency is information — it might mean biometrics is stale, or execution is on a different node. Noticing inconsistency is a form of self-integrity checking that nothing in the current pipeline does.

**Smallest buildable version:** A `check_field_coherence()` function in the field digester that defines ~5 expected correlations, computes deviation, adds a `field_coherence_warning` channel if deviation exceeds threshold.

**Files likely to touch:** `services/orion-field-digester/app/digestion/perturbation.py`, new `orion/field_coherence.py`

---

## Idea 10 — Predictive substrate (forward model)

**What:** A simple forward model that predicts next-cycle pressure values from current values, persists the prediction, then computes prediction error on the following cycle.

**Why it matters:** The `predictive` drive in the autonomy state has no substrate-level grounding. A forward model — even a naive one-step extrapolation — would give the `predictive` drive something real to respond to. This is a direct operationalization of predictive processing accounts of consciousness.

**Smallest buildable version:** After each self-state build, emit a `SelfStatePredictionV1` (predicted dimension scores for next cycle, naive linear extrapolation from trajectory). On next cycle, compare actual to predicted, compute per-dimension surprise. Add `prediction_error` as a new channel in the field state.

**Files likely to touch:** New `orion/self_state/prediction.py`, `orion/schemas/`, `services/orion-self-state-runtime/app/worker.py`

**Depends on:** Idea 1 (temporal continuity delta) — prediction requires a trajectory to extrapolate from. Implement after Idea 1 is live and producing `dimension_trajectory` data.
