# PR: Field Lattice — Capability→Capability Edges

**Branch:** `feat/field-lattice-cap-cap-edges`  
**Base:** `fix/field-saturation-execution-replace-mode`  
**Head:** `feat/field-lattice-cap-cap-edges`

## Summary

Wires the first cross-reducer interaction effects into the field lattice. Previously the lattice was a flat fan-in: each reducer wrote to its own node channels, all fanned out independently to capabilities, and there was no way for one capability to influence another. This PR adds capability→capability edges so transport degradation can now bleed into orchestration — independently of what the execution reducer is doing.

Also fixes a dark channel: `reasoning_load` was computed by the execution reducer and written to `node:athena` but had no edge mapping it to any capability, so it contributed nothing to the field.

## What changed

### `orion/schemas/field_state.py`

Added `capability_capability` to the `edge_type` Literal on `FieldEdgeV1`. Required for the YAML loader to accept the new edge type without a validation error.

### `services/orion-field-digester/app/digestion/diffusion.py`

`apply_diffusion` previously hardcoded source lookup to `state.node_vectors`. A capability source would resolve to `{}` and silently no-op. Changed to:

```python
src = state.node_vectors.get(edge.source_id) or state.capability_vectors.get(edge.source_id, {})
```

This is the only code change required for cap→cap to work. The rest is YAML.

### `config/field/orion_field_topology.v1.yaml`

**Dark channel fix:**

```yaml
# node:athena → capability:orchestration now includes:
reasoning_load: reasoning_pressure
```

`reasoning_load` was listed in `node_channels` and written by the execution reducer but had no downstream edge. It now flows to `orchestration.reasoning_pressure`.

**New capability→capability edges:**

```yaml
- source_id: capability:transport
  target_id: capability:orchestration
  edge_type: capability_capability
  weight: 0.70
  channel_map:
    transport_pressure: transport_pressure
    contract_pressure: reliability_pressure

- source_id: capability:llm_inference
  target_id: capability:orchestration
  edge_type: capability_capability
  weight: 0.60
  channel_map:
    pressure: pressure
```

Edge count: 7 → 9.

**Signal path unlocked:**

```
transport bus reducer
  → node:athena.catalog_drift_pressure
    → (diffusion, w=0.85) capability:transport.contract_pressure
      → (diffusion, w=0.70) capability:orchestration.reliability_pressure
```

Transport degradation and contract drift now raise orchestration's reliability pressure independently of what the execution reducer reports. Two reducers can now interact through the field.

## What this does NOT change

- No changes to decay rates, diffusion rates, or suppression logic
- No new reducers or grammar extractors
- No changes to how execution or biometrics perturbations are written
- The biometrics backlog (~9,800 events, 6.5hr lag) is a separate concern

## Tests

10/10 passing. New tests:

- `test_capability_to_capability_diffusion` — verifies a cap→cap edge propagates intensity correctly
- `test_reasoning_load_diffuses_to_orchestration` — verifies `reasoning_load` on `node:athena` reaches `orchestration.reasoning_pressure` through the full lattice

## Live field state after deploy

```
edge_count: 9  (was 7)
capability:orchestration.reasoning_pressure: present, draining  (was dark)
capability:orchestration.transport_pressure: 0.0 → rises on next transport event
capability:transport → capability:orchestration: active cross-reducer path
```

## Acceptance checks

- [ ] `edge_count = 9` in `substrate_field_state`
- [ ] `capability:orchestration.reasoning_pressure` present and nonzero after an execution event
- [ ] After a transport degradation event, `orchestration.reliability_pressure` rises without a corresponding execution event
- [ ] All 10 field tests pass
