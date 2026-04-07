# Phase 15 — Durable calibration profile state in SQL

## Objective

Make calibration profile **state** (staged/active/rolled-back), not only audit events, durably stored in SQL and used for runtime resolution.

## State vs audit split

- **State table** (`calibration_profiles`): canonical current profile definitions and lifecycle state.
- **Audit table** (`calibration_profile_audit`): immutable history of stage/activate/rollback events.

State answers “what is active now?”. Audit answers “what happened?”.

## Implementation summary

- Added `SqlCalibrationProfileStore` in `orion/reasoning/calibration_profiles.py` with explicit manual lifecycle methods:
  - `apply(stage|activate)`
  - `rollback(to_previous|to_baseline)`
  - `active_profile()`
  - `list_profiles()`
  - `resolve(...)`
- Added durable table initialization for `calibration_profiles` and persisted lifecycle transitions.
- Kept manual-first semantics:
  - stage is explicit
  - activate is explicit
  - rollback is explicit
  - baseline fallback remains explicit
- Runtime now defaults to SQL-backed profile store (falling back to in-memory only on initialization failure), making active-profile resolution restart-safe.
- Added SQL model `CalibrationProfileStateSQL` for writer-side schema visibility.

## Safety posture

- No automatic recommendation adoption was added.
- Runtime remains baseline-safe if no active SQL profile exists.
- Runtime path stays conservative; failures in state store initialization degrade to in-memory fallback.

## Follow-up

A future phase can persist emitted calibration audit events transactionally alongside state transitions from a shared repository boundary to strengthen cross-service exactly-once lineage guarantees.
