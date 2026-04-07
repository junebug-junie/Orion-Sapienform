# Phase 13b — SQL-writer durability for endogenous runtime + calibration audit

## Objective

Move endogenous runtime operational durability from local-only JSONL toward bus-driven SQL persistence using `orion-sql-writer`, while preserving JSONL/in-memory fallback for dev or publish-failure conditions.

## Storage split

- **GraphDB** remains the durable semantic/graph system (concepts, relations, promoted reasoning artifacts).
- **SQL** becomes the durable operational history for endogenous runtime execution/audit and calibration lifecycle audit.
- **JSONL/memory** remains fallback/dev-only local persistence.

## Bus contracts and channels

This phase uses shared canonical schema payloads on explicit channels:

- `orion:endogenous:runtime:record` → `EndogenousRuntimeExecutionRecordV1` (`endogenous.runtime.record.v1`)
- `orion:endogenous:runtime:audit` → `EndogenousRuntimeAuditV1` (`endogenous.runtime.audit.v1`)
- `orion:calibration:profile:audit` → `CalibrationProfileAuditV1` (`calibration.profile.audit.v1`)

All schema kinds are wired through shared exports and registry.

## Runtime publishing behavior

`services/orion-cortex-exec/app/endogenous_runtime.py` now:

- publishes runtime record + runtime audit events onto bus when bus is available
- supports `bus_sql` backend posture where bus publish is the primary durability path
- preserves non-blocking behavior (publish failures do not break main runtime path)
- falls back to local store on publish failure in `bus_sql` mode
- emits calibration profile lifecycle audit events during stage/activate/rollback operations

## SQL-writer integration

`orion-sql-writer` now subscribes/routes and persists the new operational events.

Added SQL models:

- `endogenous_runtime_records`
- `endogenous_runtime_audit`
- `calibration_profile_audit`

Each table stores query-friendly columns plus raw canonical payload JSON.

## Safety posture

- Runtime/chat path remains non-blocking on publish/persist issues.
- No semantic state moved from GraphDB to SQL.
- No runtime autonomy broadening introduced by this phase.

## Follow-up

A later phase can add SQL-backed read APIs and profile state durability migration from in-memory store, now that canonical write-path durability is wired.
