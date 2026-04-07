# Phase 16 — Operator/debug inspection surface for endogenous runtime and calibration state

## Objective

Provide one bounded, read-only operator/debug surface to inspect durable endogenous runtime records and calibration profile state/history with explicit source metadata.

## What was added

- Added `inspect_endogenous_operator_debug_surface(...)` in cortex-exec as a unified inspection entry point.
- Added `inspect_calibration_profile_state_with_source(...)` for active/staged/profile-by-id state visibility.
- Surface includes four bounded sections:
  - `runtime_records` (durable runtime execution reads)
  - `calibration_state` (active/staged/current profile state)
  - `calibration_audit` (lifecycle history)
  - `comparisons` (baseline-vs-active and previous-vs-current advisory readouts)

## Source and failure semantics

- Runtime and audit reads continue SQL-first through existing SQL reader.
- Calibration profile state source is explicit:
  - `sql` when SQL-backed profile store is active
  - `fallback_local:memory` when in-memory fallback store is active
- Unified operator surface is safe-failure by design:
  - returns bounded fallback payload with `fallback_used=true` and error metadata when inspection fails
  - does not mutate stage/activate/rollback state
  - does not alter primary runtime invocation behavior

## Safety posture

- Read-only/operator-focused; no hidden control actions.
- Manual-first calibration lifecycle remains unchanged (`stage`, `activate`, `rollback`).
- No GraphDB operational history dependency was introduced.
- Comparison outputs remain advisory summaries over operational records.

## Operator questions answered by this surface

- Which calibration profile is active now?
- Which profiles are staged?
- What lifecycle history (including rollback linkage) exists?
- What recent runtime workflows ran, with bounded filters?
- What is current baseline-vs-active and previous-vs-current posture?
- Which source produced each read path (`sql` vs fallback)?
