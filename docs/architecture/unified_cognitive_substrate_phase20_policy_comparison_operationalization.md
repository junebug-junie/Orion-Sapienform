# Unified Cognitive Substrate — Phase 20: Policy Comparison Operationalization

## Objective

Phase 20b makes policy comparison operational by wiring advisory effectiveness reads to SQL-backed policy and telemetry state, exposing bounded operator-facing comparison output.

## What ships in Phase 20b

- SQL-backed policy comparison plumbing:
  - policy pair resolution reads active/previous/selected profiles from SQL-backed `SubstratePolicyProfileStore`
  - telemetry reads come from SQL-backed `GraphReviewTelemetryRecorder`
- Deterministic pair resolution modes:
  - `baseline_vs_active`
  - `previous_vs_current`
  - `selected_pair`
- Bounded metric aggregation with deterministic deltas/verdict/confidence:
  - execution_count / execution_rate
  - noop/suppressed/terminated/failed rates
  - avg_cycles_to_resolution
  - frontier_followup_rate
  - operator_only_rate
  - strict_zone_surface_rate
  - queue_revisit_rate
- Operator-facing surface:
  - `/api/substrate/policy-comparison` returns bounded, read-only advisory payloads
  - standalone substrate inspector renders comparison section

## Safety / semantics

- Advisory-only behavior: comparison never activates/rolls back/mutates policy.
- Missing profile pair members fail safely with bounded operator-readable errors.
- Insufficient-data posture is explicit and returned as verdict `insufficient_data` with low confidence.
- GraphDB semantic reads are not introduced into comparison logic; SQL-backed control-plane + runtime telemetry remain the source for this phase.
