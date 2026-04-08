# Phase 14 — SQL-backed read path for endogenous runtime and calibration audit

## Objective

Phase 14 moves operator/debug reads for endogenous runtime history and calibration lifecycle audit to SQL-first retrieval, completing the Phase 13b durability loop.

## Read posture

- **Primary**: SQL (`endogenous_runtime_records`, `endogenous_runtime_audit`, `calibration_profile_audit`)
- **Fallback**: local in-memory/JSONL runtime store and in-memory calibration audit cache
- **Transparency**: read helpers return source metadata (`sql` vs `fallback_local:*`) and fallback flags.

## Implemented surfaces

In `services/orion-cortex-exec/app/endogenous_runtime.py`:

- Runtime records:
  - `inspect_recent_records_with_source(...)` (SQL-first)
  - `inspect_recent_records(...)` (backward-compatible list facade)
  - Expanded filters include audit status, anchor scope, execution success, calibration profile id, correlation id, request id.
- Calibration audit:
  - `inspect_calibration_audit_with_source(...)` (SQL-first)
  - `inspect_calibration_audit(...)` (backward-compatible list facade)
- Comparison helper:
  - `compare_runtime_profile_outcomes(...)` / module wrapper for before/after profile outcome summaries using SQL-backed record inputs as primary source.

## SQL reader helper

Added `EndogenousRuntimeSqlReader` (`app/endogenous_runtime_sql_reader.py`) with bounded deterministic queries:

- `runtime_records(...)`
- `calibration_audit(...)`

Both return a typed `SqlReadResult` carrying:
- source
- rows
- filter metadata
- optional error

## Config

Added explicit SQL read config in cortex-exec settings/env/compose:

- `ENDOGENOUS_RUNTIME_SQL_READ_ENABLED`
- `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL`

## Safety

- SQL read failures never break runtime execution path.
- Fallback mode is explicit in metadata and bounded by limits.
- GraphDB/SQL responsibility split is unchanged (operational reads remain SQL).
