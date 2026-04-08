# Unified Cognitive Substrate — Phase 19: SQL-Backed Policy Control-Plane Persistence

## Why this phase exists

Phase 18 hardened policy lifecycle durability with file-backed JSON, but that storage was not ideal for operational multi-process/control-plane posture.

Phase 19 moves substrate policy profile lifecycle durability to SQL-backed storage while preserving manual lifecycle semantics and bounded runtime behavior.

## Preserved lifecycle semantics

Phase 19 intentionally preserves Phase 17 behavior:

- stage is explicit,
- activate is explicit,
- rollback is explicit,
- baseline remains valid,
- no auto-adoption.

Store backend changed; lifecycle semantics did not.

## SQL-backed policy persistence

`SubstratePolicyProfileStore` now supports SQL-backed durability through `sql_db_path` and keeps optional legacy JSON support only for migration/fallback.

Persisted SQL control-plane entities (logical concerns):

- profile state (`substrate_policy_profile`),
- active scope bindings (`substrate_policy_active_scope`),
- audit history (`substrate_policy_audit`).

Runtime state reconstruction on startup restores active/staged/rolled-back profiles deterministically.

## Runtime resolution continuity

Runtime/scheduler policy resolution still matches by:

- invocation surface,
- target zone,
- operator mode.

No staged profile is implicitly activated after restart.

## Operator integration

Standalone substrate calibration/operator surfaces now read policy inspection state from the policy store and expose:

- active profile,
- staged profiles,
- rolled-back profiles,
- recent audit events,
- advisory recommendations (telemetry-driven, still manual adoption).

## GraphDB vs SQL split (preserved)

- GraphDB: semantic/cognitive substrate graph reads.
- SQL-backed control plane: policy lifecycle state, review queue/runtime telemetry/calibration operational records.

Policy lifecycle storage is not moved into GraphDB.

## Migration posture

The store supports optional legacy JSON migration into SQL-backed persistence on first load when configured.

This avoids silent state loss during transition.

## Follow-on work

- PostgreSQL-backed centralized policy store for multi-instance deployments,
- stronger transactional concurrency controls for highly parallel writers,
- richer operator profile diff tooling.
