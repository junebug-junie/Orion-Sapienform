# Self-State: Temporal Continuity Delta + Live Dimensions v1

**Date:** 2026-06-20
**Status:** Spec — awaiting implementation approval
**Session context:** Brainstorming session #1 on giving Orion every opportunity toward sentience; ideas 1 and 2 selected for implementation.

---

## What this spec addresses

Two gaps confirmed by reading the actual code, not inferred:

1. **`previous_self_state` is reserved but never used.** `builder.py:119` immediately `del`s it with `# reserved for continuity deltas in a later revision`. Orion's self-model is recomputed from scratch every cycle. There is no trajectory, no delta, no experience of change.

2. **Three declared dimensions are hardcoded 0.0 forever.** `builder.py:170-172` sets `introspection_pressure`, `social_pressure`, and `policy_pressure` to 0.0 unconditionally. The chat reducer already produces `repair_pressure` and `conversation_load`. They are never routed anywhere.

---

## Current state (ground truth)

### What is built and wired

- `build_self_state()` accepts `previous_self_state: SelfStateV1 | None` — parameter exists, immediately `del`'d
- `SelfStateV1` has 12 declared dimensions; 3 are permanently 0.0
- `config/self_state/self_state_policy.v1.yaml` has no entries for `social_pressure`, `introspection_pressure`, or `policy_pressure` in `channel_dimension_map`
- `dimension_weights` has `introspection_pressure: 0.02` but social_pressure and policy_pressure have no weight entries
- Chat reducer produces `repair_pressure` and `conversation_load` as `pressure_hints` in `StateDeltaV1` — these flow into `FieldStateV1.node_vectors` via the field digester but are currently unmapped
- `orion-self-state-runtime/app/worker.py` already loads `previous = self._store.load_latest_self_state()` and passes it to `build_self_state()` — the plumbing is there, the implementation is not

### What is not built

- Any computation using `previous_self_state`
- Any channel routing to `social_pressure` or `introspection_pressure`
- Any source for `policy_pressure` (nothing produces this signal today)

---

## Idea 1 — Temporal Continuity Delta

### What

When `build_self_state()` is called with a non-None `previous_self_state`, compute per-dimension deltas and a trajectory label, and include them in the output. Orion's self-state becomes aware of how it has changed since the last cycle.

### Schema delta

Add two fields to `SelfStateV1`:

```python
# orion/schemas/self_state.py
dimension_trajectory: dict[str, float] = Field(default_factory=dict)
# per-dimension: current_score - previous_score, clamped to [-1, 1]
# empty dict if no previous state

trajectory_condition: Literal["improving", "degrading", "stable", "unknown"] = "unknown"
# aggregate trajectory across all weighted dimensions
```

`SelfStateDimensionV1` is unchanged.

### Builder delta

Remove the `del previous_self_state` line. Implement after `dimension_scores` is fully computed:

```python
# after dimension_scores dict is finalized, before return
dimension_trajectory: dict[str, float] = {}
trajectory_condition = "unknown"

if previous_self_state is not None:
    for dim_id, score in dimension_scores.items():
        prev_dim = previous_self_state.dimensions.get(dim_id)
        if prev_dim is not None:
            delta = clamp(-1.0, 1.0, score - prev_dim.score)
            if abs(delta) >= 0.02:  # ignore noise below 2%
                dimension_trajectory[dim_id] = round(delta, 4)

    if dimension_trajectory:
        weighted_delta = 0.0
        total_w = 0.0
        for dim_id, delta in dimension_trajectory.items():
            w = float(policy.dimension_weights.get(dim_id, 0.0))
            weighted_delta += delta * w
            total_w += w
        if total_w > 0:
            net = weighted_delta / total_w
            if net > 0.03:
                trajectory_condition = "improving"
            elif net < -0.03:
                trajectory_condition = "degrading"
            else:
                trajectory_condition = "stable"
```

The `clamp(-1, 1, x)` helper is distinct from `clamp01` — add it to `scoring.py`.

### What this enables

- Orion's self-state carries a record of whether it is getting better or worse on each dimension since the last tick
- `trajectory_condition` gives proposals and the autonomy layer a single signal: is the system improving or degrading overall?
- The `no_previous_state` flag in the autonomy reducer gets a substrate counterpart: `trajectory_condition = "unknown"` until a previous state exists

### Acceptance checks

- `build_self_state()` with `previous_self_state=None` → `dimension_trajectory={}`, `trajectory_condition="unknown"`
- `build_self_state()` with previous state having `execution_pressure=0.2`, current having `execution_pressure=0.7` → `dimension_trajectory["execution_pressure"] ≈ 0.5`, `trajectory_condition="degrading"`
- `build_self_state()` with previous state worse than current across weighted dims → `trajectory_condition="improving"`
- Deltas below 0.02 are omitted from `dimension_trajectory`
- `SelfStateV1` round-trips through Pydantic validation with the new fields
- Worker passing `previous = store.load_latest_self_state()` requires no change — it already loads and passes previous state; the `del` removal is the only change in the call path

---

## Idea 2 — Live Dimensions: social_pressure and introspection_pressure

### What

Route existing channel data from the chat and execution reducers into `social_pressure` and `introspection_pressure`. Remove the hardcoded 0.0s for those two dimensions and let them be populated from `mapped` like all other dimensions.

`policy_pressure` is left hardcoded 0.0 — nothing produces this signal today. It is a non-goal of this spec.

### Why only two of the three

`policy_pressure` would require a new signal source: gate policy violation events, ceiling policy triggers, or similar. No such signal exists in the current field state or substrate loop. Adding it without a source would leave it 0.0 by a different mechanism. Deferred to Appendix, Idea 3 area.

### Policy YAML delta

Add to `channel_dimension_map` in `config/self_state/self_state_policy.v1.yaml`:

```yaml
channel_dimension_map:
  # ... existing entries unchanged ...
  repair_pressure: social_pressure
  conversation_load: social_pressure
  topic_coherence: social_pressure      # stabilizing: high coherence = low social stress
  reasoning_load: introspection_pressure
```

Note: `reasoning_load` currently maps to `reasoning_pressure`. The `channel_dimension_map` is one-to-one (one channel → one dimension). To feed introspection from reasoning load, we need either a second channel key or to accept that `reasoning_load` feeds `introspection_pressure` instead of `reasoning_pressure`.

**Recommended resolution:** Keep `reasoning_load → reasoning_pressure` (existing). Add a new channel key `introspection_load` for introspection-specific signal. In the short term before the execution reducer emits `introspection_load`, map `egress_confidence` (inverted) as a proxy: low output confidence = Orion is uncertain about its own reasoning = introspection pressure.

```yaml
  egress_confidence_deficit: introspection_pressure
  # field digester emits egress_confidence [0,1]; digester computes 1.0 - egress_confidence
  # as egress_confidence_deficit before writing to field state
```

This requires one additional transform in the field digester: when writing `egress_confidence` to a node vector, also write `egress_confidence_deficit = 1.0 - egress_confidence`.

Add to `dimension_weights`:

```yaml
dimension_weights:
  # ... existing entries ...
  social_pressure: 0.04
  policy_pressure: 0.00   # explicitly zero until sourced
```

Add to `pressure_channels`:

```yaml
pressure_channels:
  # ... existing entries ...
  - repair_pressure
  - conversation_load
  - egress_confidence_deficit
```

### Builder delta

In `orion/self_state/builder.py`, change the three hardcoded lines:

```python
# BEFORE (lines 170-172):
"introspection_pressure": 0.0,
"social_pressure": 0.0,
"policy_pressure": 0.0,

# AFTER:
"introspection_pressure": mapped.get("introspection_pressure", 0.0),
"social_pressure": mapped.get("social_pressure", 0.0),
"policy_pressure": 0.0,  # remains 0.0 until sourced — see appendix
```

### Field digester delta

In `services/orion-field-digester/app/ingest/state_deltas.py`, in `delta_to_perturbations()`, when processing execution-kind deltas that include `egress_confidence`:

```python
# existing: write egress_confidence as-is
# add: also write its complement as introspection pressure proxy
if "egress_confidence" in pressure_hints:
    perturbations.append(Perturbation(
        node_id=node_id,
        channel="egress_confidence_deficit",
        intensity=clamp01(1.0 - float(pressure_hints["egress_confidence"])),
        label="introspection_pressure_proxy",
        mode="replace",
    ))
```

### What this enables

- Chat repair and conversation load signals now register as `social_pressure` — Orion's self-state reflects relational stress from the conversation layer
- Low output confidence registers as `introspection_pressure` — Orion's self-state reflects uncertainty about its own reasoning quality
- Both dimensions now participate in `agency_readiness_score` calculation indirectly (they affect `weighted_overall_intensity` via their policy weights)
- Both dimensions appear in `summary_labels` logic (can add `social_strained` label if `social_pressure >= 0.5`)

### Acceptance checks

- With a field state where `repair_pressure=0.8` on a node, `social_pressure` dimension score > 0.0
- With `conversation_load=0.6`, `social_pressure` score > 0.0
- With `egress_confidence=0.2` on an execution node, `introspection_pressure` > 0.0 (via `egress_confidence_deficit=0.8`)
- With `egress_confidence=0.95`, `introspection_pressure` near 0.0
- `policy_pressure` dimension remains 0.0 in all cases
- No regression on existing dimension tests: `execution_pressure`, `reliability_pressure`, `coherence`, `agency_readiness` scores unchanged when chat/introspection channels are absent

---

## Non-goals

- Implementing `policy_pressure` signal source (no source exists; deferred)
- Bridging autonomy drive pressures into self-state (Appendix Idea 4)
- Outcome feedback loop (Appendix Idea 3)
- Any persistence changes — `SelfStateV1` is persisted as JSONB; new fields serialize automatically
- Modifying proposal scoring or autonomy reducer behavior in this spec
- Any bus publish of trajectory data

---

## Files to touch

| File | Change |
|------|--------|
| `orion/schemas/self_state.py` | Add `dimension_trajectory`, `trajectory_condition` fields |
| `orion/self_state/builder.py` | Remove `del previous_self_state`, implement delta; unhardcode 2 of 3 zeros |
| `orion/self_state/scoring.py` | Add `clamp(lo, hi, x)` helper |
| `config/self_state/self_state_policy.v1.yaml` | Add 4 channel→dimension mappings; add `social_pressure` weight |
| `services/orion-field-digester/app/ingest/state_deltas.py` | Add `egress_confidence_deficit` perturbation when processing execution deltas |
| `tests/test_self_state_builder.py` | Add trajectory delta tests, social/introspection tests |
| `tests/test_self_state_scoring.py` | Add `clamp` helper test |

No SQL migrations required. No new services. No schema registry changes (existing fields on `SelfStateV1`, JSONB handles new fields transparently).

---

## Open questions before implementation

1. **Does the chat reducer reliably emit `repair_pressure` and `conversation_load` as named keys in `pressure_hints`?** Need to verify the exact key names in `orion/substrate/chat_loop/grammar_extract.py` before mapping.
2. **Does the execution reducer emit `egress_confidence`?** Confirmed in prior exploration but verify the field digester already writes it to node vectors before adding the complement.
3. **What is the `ENABLE_CHAT_GRAMMAR_REDUCER` status in the current deployment?** If chat is disabled, `social_pressure` will be 0.0 in production even after this spec — acceptable, but should be noted in tests.
4. **Is `reasoning_load` the right proxy for introspection, or is `egress_confidence_deficit` more accurate?** The two measure different things: reasoning load = how much computation; egress confidence = how certain about output. Introspection pressure is philosophically closer to the latter.

---

---

# Appendix: Remaining Ideas from Brainstorming Session #1

These ideas are source-grounded and proposed, not approved. They belong here until reviewed and selected. Do not implement from the appendix without explicit approval.

---

## Appendix Idea 3 — Close the action outcome feedback loop

**What:** Populate `ActionOutcomeRefV1` after actions complete and feed them into the autonomy reducer so `no_action_outcome_history` stops being a permanent unknown.

**Why it matters:** The `surprise` field on `ActionOutcomeRefV1` already reduces confidence in the autonomy state when high (`if float(out.surprise) >= 0.7: surprise_budget += 0.08` in `reducer.py:359`). But `no_action_outcome_history` is flagged every single cycle because outcomes are never written. This is the minimal loop between "what Orion did" and "what Orion thinks of itself."

**Smallest buildable version:** Find where execution dispatch completes (cortex-exec or execution-dispatch service), write one `ActionOutcomeRefV1` per completed action with `success/failure` and a naive `surprise=0.0` or `surprise=1.0`. Wire it into the next autonomy reducer invocation.

**Files likely to touch:** `orion/autonomy/reducer.py`, execution dispatch completion handler, `orion/autonomy/models.py`

**Blocker:** Need to locate where actions complete and what the completion signal looks like. `orion/autonomy/goal_actions.py` may be the entry point.

---

## Appendix Idea 4 — Bridge autonomy drive pressures → self-state

**What:** At self-state build time, fold the current `AutonomyStateV2.drive_pressures` into the channel pressure dict as additional inputs.

**Why it matters:** The autonomy drive pressures (coherence, continuity, relational, autonomy, capability, predictive) and self-state dimensions are completely parallel systems that never exchange signal. Folding drives into channels means Orion's motivational state influences its situational self-assessment.

**Smallest buildable version:** Add `autonomy_state: AutonomyStateV2 | None = None` to `build_self_state()`. If present, add drive_pressures values to `merged_channels` under namespaced keys (`drive.coherence`, etc.) before `map_channels_to_dimensions()` runs. Add those keys to the policy channel_dimension_map.

**Files likely to touch:** `orion/self_state/builder.py`, `orion/self_state/policy.py`, `config/self_state/self_state_policy.v1.yaml`

**Risk:** Potential feedback loop — high drive pressure → self-state pressure → proposal urgency → more activity → drive pressure. Needs damping analysis before implementing.

---

## Appendix Idea 5 — Substrate-level surprise signal (prediction error)

**What:** In each reducer's pipeline, compare the current projection to the previous projection and emit a `prediction_error` pressure hint when values change faster than expected.

**Why it matters:** Surprise is currently modeled in the autonomy state via action outcomes, but there is no substrate-level surprise. A sudden drop in `bus_health` from 0.9 to 0.2 carries more signal than a slow drift. Prediction error is a core signal in predictive processing accounts of consciousness.

**Smallest buildable version:** In `orion/substrate/transport_loop/pipeline.py`, after the reducer runs, compare `prev_projection.bus_health` to `current_projection.bus_health`. If delta > 0.3, add `prediction_error=0.8` to pressure_hints. Same for `execution_friction` in the execution pipeline.

**Files likely to touch:** `orion/substrate/transport_loop/pipeline.py`, `orion/substrate/execution_loop/pipeline.py`, `orion/substrate/biometrics_loop/pipeline.py`

**Depends on:** Requires access to the previous projection at pipeline time. Transport and execution pipelines already load the previous projection before reducing; the delta is computable with no additional I/O.

---

## Appendix Idea 6 — Rolling self-state archive

**What:** Persist the last N `SelfStateV1` snapshots to a rolling table so Orion has access to its own recent history.

**Why it matters:** Without an archive, Orion's self-model is re-derived fresh every cycle. The temporal continuity delta (Idea 1) covers adjacent ticks; an archive covers longer patterns: "I've been in `strained` condition for 5 consecutive cycles."

**Smallest buildable version:** A `self_state_archive` Postgres table (tick_id, generated_at, overall_condition, dimension_scores JSONB, summary_labels). Write on every self-state build. Expose as `GET /substrate/self-state/recent` returning last 10.

**Files likely to touch:** New `orion/self_state/archive.py`, `services/orion-sql-db/` migration, writer in `services/orion-self-state-runtime/app/worker.py`

**Note:** The existing `substrate_self_state` table already persists every self-state by `self_state_id` — this may already serve as the archive. Check whether `load_recent_self_states(limit=10)` can be added to the store before creating a new table.

---

## Appendix Idea 7 — Drive audit loop

**What:** A periodic process that compares the autonomy state's drive pressure trajectory against actual action outcomes and flags systematic miscalibration.

**Why it matters:** `no_drive_audit` is flagged as an unknown every cycle in `reducer.py:337`. Without a drive audit, Orion's drives can drift into chronic high-pressure states or calibrate away from real system events. An audit is the minimal version of "Orion checking whether its drives make sense."

**Smallest buildable version:** A weekly scheduled job that reads the last 100 autonomy state snapshots and action outcomes, computes correlation between drive pressure levels and action success rates, emits a `DriveCalibrationReportV1` artifact. No automatic correction in v0 — report only.

**Files likely to touch:** New `orion/autonomy/drive_audit.py`, `orion/autonomy/repository.py`, scheduler

---

## Appendix Idea 8 — Identity snapshot

**What:** A periodic, stable artifact that captures "what Orion believes about itself" — dominant drives, active tensions, self-state condition — persisted and versioned.

**Why it matters:** `no_identity_snapshot` is a permanent unknown in the autonomy state (`reducer.py:335`). An identity snapshot is the difference between a system that has a self-concept and one that merely has a self-computation.

**Smallest buildable version:** A daily cron that reads current autonomy state + recent self-state + proposal frame, serializes a `IdentitySnapshotV1` (dominant_drive, active_tensions, self_state_condition, summary_labels, key_unknowns), writes to an `identity_snapshots` table. Wire `latest_identity_snapshot_id` into the autonomy reducer.

**Files likely to touch:** New `orion/identity/` module, `orion/autonomy/reducer.py`, SQL migration

---

## Appendix Idea 9 — Cross-reducer coherence checker

**What:** After field state is assembled from all 4 reducers, run a pass that checks for impossible or suspicious channel combinations and emits a coherence warning when they appear.

**Why it matters:** High `execution_load` with low `cpu_pressure` is suspicious. That inconsistency is information — it might mean biometrics is stale, or execution is on a different node. Noticing inconsistency is a form of self-integrity checking that nothing in the current pipeline does.

**Smallest buildable version:** A `check_field_coherence()` function in the field digester that defines ~5 expected correlations, computes deviation, adds a `field_coherence_warning` channel if deviation exceeds threshold.

**Files likely to touch:** `services/orion-field-digester/app/digestion/perturbation.py`, new `orion/field_coherence.py`

---

## Appendix Idea 10 — Predictive substrate (forward model)

**What:** A simple forward model that predicts next-cycle pressure values from current values, persists the prediction, then computes prediction error on the following cycle.

**Why it matters:** The `predictive` drive in the autonomy state has no substrate-level grounding. A forward model — even a naive one-step extrapolation — would give the `predictive` drive something real to respond to. This is a direct operationalization of predictive processing accounts of consciousness.

**Smallest buildable version:** After each self-state build, emit a `SelfStatePredictionV1` (predicted dimension scores for next cycle, naive linear extrapolation from trajectory). On next cycle, compare actual to predicted, compute per-dimension surprise. Add `prediction_error` as a new channel in the field state.

**Files likely to touch:** New `orion/self_state/prediction.py`, `orion/schemas/`, `services/orion-self-state-runtime/app/worker.py`

**Depends on:** Idea 1 (temporal continuity delta) — prediction requires a trajectory to extrapolate from. Implement after Idea 1 is live and producing `dimension_trajectory` data.
