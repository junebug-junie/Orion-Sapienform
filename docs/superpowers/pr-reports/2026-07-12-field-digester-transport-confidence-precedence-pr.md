## Summary

- Fixed a live-but-masked bug in `apply_diffusion()`: `capability:transport`'s direct diffusion edges into `confidence` (from `delivery_confidence`) and `available_capacity` (from `bus_health`) were unconditionally overwritten by a pressure-derived formula whenever `pressure` was also a target for that capability, silently making the direct edges pointless.
- This was flagged as a known, unfixed quirk in the previous diffusion-saturation-fix PR (`ed452331`) and is resolved here as its own scoped follow-up.
- Gated the derived-formula fallback on `best_source` (a real `>0` contribution landing THIS tick for that specific channel) rather than static channel-map membership — this matters because a first draft of the fix (gating on the static `channels` set) would have hard-floored `confidence`/`available_capacity` at `0.0` on any tick where the direct edge's source happened to have no value for that field, which is worse than the original bug. Caught and fixed via code review before commit.
- Two new regression tests added.

## Outcome moved

`capability:transport`'s `confidence`/`available_capacity` now genuinely reflect `delivery_confidence`/`bus_health` diffusion when real, and fall back gracefully to the pressure-derived formula only when nothing real landed that tick — instead of one formula always winning regardless of what actually happened.

## Current architecture

`services/orion-field-digester/app/digestion/diffusion.py`'s `apply_diffusion()` (rewritten 2026-07-12 in the earlier saturation-fix PR to be memoryless per diffusion-target channel) recomputes every capability channel each tick from `state.edges`. A post-loop block derives `confidence`/`available_capacity` from `pressure` whenever `pressure` is a target for that capability — originally with no awareness that some capabilities also receive those two channels directly from other edges.

## Architecture touched

`orion-field-digester` only — no schema, bus, or cross-service contract changes.

## Files changed

- `services/orion-field-digester/app/digestion/diffusion.py`: gate the pressure-derived `confidence`/`available_capacity` formula on `best_source` (per-tick real contribution) instead of unconditionally overwriting, and instead of the static-membership check that would have hard-floored the channels to `0.0` on a temporarily-missing direct contribution.
- `services/orion-field-digester/tests/test_diffusion_provenance.py`: added `test_direct_confidence_and_capacity_diffusion_beats_pressure_derived_formula` (direct diffusion wins when it fires) and `test_derived_formula_still_falls_back_when_direct_edge_contributes_nothing_this_tick` (derived formula still runs as a safety net when the direct edge contributes nothing that tick).

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. :services/orion-field-digester pytest services/orion-field-digester/tests -q
22 passed

PYTHONPATH=. pytest tests/test_self_state_runtime_store.py tests/test_self_state_deviation.py \
  tests/test_self_state_transport_dimension.py tests/test_self_state_builder.py \
  tests/test_self_state_reliability_decontamination.py tests/test_self_state_prediction.py \
  tests/test_self_state_policy_loader.py tests/test_self_state_builder_hardening.py \
  tests/test_self_state_scoring.py tests/test_self_state_schemas.py tests/test_field_state_schemas.py -q
56 passed
```

## Evals run

None applicable — this is a deterministic formula-precedence fix, covered fully by unit tests.

## Docker/build/smoke checks

Not run for this branch — not deployed. (The prior saturation fix on this same file IS live on Athena; this follow-up fix is a separate, additive change not yet built/deployed.)

## Review findings fixed

- Finding: gating the derived-formula fallback on static `channels` membership (every configured target for a capability, whether or not it fired this tick) instead of per-tick real contribution would hard-floor `confidence`/`available_capacity` at `0.0` whenever `delivery_confidence`/`bus_health` are temporarily absent from the source data — worse than the pre-existing bug.
  - Fix: gate on `(target_id, channel) not in best_source` instead, which only tracks real (`>0`) contributions from this specific tick.
  - Evidence: new test `test_derived_formula_still_falls_back_when_direct_edge_contributes_nothing_this_tick` reproduces the missing-source-field scenario and asserts the derived formula still fires; independently re-verified via a second review pass confirming no other capability in `config/field/orion_field_topology.v1.yaml` has any edge feeding `confidence`/`available_capacity` besides `capability:transport`, so the fix cannot wrongly suppress the safety net elsewhere.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml \
  up -d --build orion-athena-field-digester
```

Not run — deliberately left for Juniper to trigger, matching the standing deploy-authorization pattern from this session.

## Risks / concerns

- Severity: low
- Concern: this changes `capability:transport`'s `confidence`/`available_capacity` values whenever `transport_pressure` is nonzero AND `delivery_confidence`/`bus_health` diverge from what the pressure-derived formula would produce. Currently masked in live traffic (`transport_pressure` has been idle), so the behavioral change will be invisible until real transport pressure occurs.
- Mitigation: none needed beyond the tests — this is strictly a correctness fix (direct sensor data should win over a derived proxy), not a new behavior needing a flag.

## PR link

Branch pushed: `fix/field-digester-transport-confidence-precedence`
