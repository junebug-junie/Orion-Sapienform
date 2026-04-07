# Phase 12 — Manual Calibration Profile Adoption and Staged Runtime Rollout Controls

## Objective

Phase 12 adds a bounded, operator-controlled path to adopt Phase 11 calibration recommendations into runtime behavior **manually**.

This phase intentionally does **not** auto-apply calibration output, does not broaden live autonomy, and keeps rollback first-class.

## Design

### 1) Typed contracts

Added canonical profile/adoption contracts in `orion/core/schemas/calibration_adoption.py`:

- `CalibrationRolloutScopeV1`
- `CalibrationProfileV1`
- `CalibrationAdoptionRequestV1` / `CalibrationAdoptionResultV1`
- `CalibrationRollbackRequestV1` / `CalibrationRollbackResultV1`
- `CalibrationProfileAuditV1`
- `CalibrationProfileResolutionV1`

These model:
- profile identity/version/provenance
- rollout scope (surface/workflow/mentor posture/operator-only/canary %)
- explicit manual stage/activate operations
- explicit rollback operations
- audit trail events
- deterministic runtime resolution result

### 2) Manual adoption seam

`orion/reasoning/calibration_profiles.py` introduces `InMemoryCalibrationProfileStore` with explicit operations:
- `apply(stage|activate)`
- `rollback(to_previous|to_baseline)`
- `resolve(context)`
- `list_profiles(...)`
- `list_audit(...)`

Recommendation-to-live remains manual and explicit.

### 3) Runtime integration (bounded)

`services/orion-cortex-exec/app/endogenous_runtime.py` now:
- hosts a calibration profile store inside runtime service
- resolves active profile per invocation using deterministic scoped rules
- applies supported overrides into `TriggerPolicy` only when an adopted profile resolves
- records calibration mode/reason/profile-id into audit actions for observability
- exposes operator helpers for:
  - stage/activate (`apply_calibration_adoption`)
  - rollback (`rollback_calibration_adoption`)
  - inspect profiles/audit
  - inspect current resolution (`resolve_runtime_calibration_profile`)

### 4) Safety posture

- No automatic adoption from Phase 11 outputs.
- No hidden/implicit profile activation.
- Baseline (no active profile) remains valid and safe.
- Rollback is explicit and auditable.
- Runtime surface scope remains unchanged.

## Manual operator flow

1. Generate recommendations via Phase 11 offline evaluation.
2. Construct profile with provenance and bounded scope.
3. Stage profile (`action=stage`).
4. Explicitly activate profile (`action=activate`).
5. Observe runtime behavior and audit events.
6. Roll back to previous or baseline if needed.

## Future path (out of scope)

A later phase may persist profiles durably beyond process memory and add explicit before/after metric compare artifacts, but this phase keeps implementation low-blast-radius and manual-first.
