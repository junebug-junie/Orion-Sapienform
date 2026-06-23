# Self-State: Temporal Continuity Delta + Live Dimensions v1

**Date:** 2026-06-20
**Status:** Implemented — 2026-06-20
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

*Appendix ideas 3–10 moved to `reviews/pending/2026-06-20-brainstorming-session-1-appendix.md`.*

