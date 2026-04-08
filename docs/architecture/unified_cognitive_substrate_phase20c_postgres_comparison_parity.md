# Unified Cognitive Substrate — Phase 20c: Postgres Comparison/Control-Plane Parity

## Why this phase exists

Phase 20b made policy comparison operational, but retained sqlite-style local SQL seams in parts of the control-plane read path.

Phase 20c aligns runtime architecture with intended durability boundaries:

- **GraphDB**: semantic/cognitive substrate truth
- **Postgres**: operational/control-plane truth (policy state, audit history, runtime review telemetry, and comparison reads)

## What changed

- `SubstratePolicyProfileStore` now supports Postgres-backed persistence/read paths and marks source posture explicitly (`postgres`, `sqlite`, `fallback`, `memory`).
- `GraphReviewTelemetryRecorder` now supports Postgres-backed persistence/read paths and exposes explicit source/degraded/error posture.
- Hub substrate comparison/calibration/telemetry payload source metadata now reflects actual operational source posture and no longer implies generic SQL truth when degraded.
- Existing comparison contracts and endpoint semantics were preserved (`baseline_vs_active`, `previous_vs_current`, `selected_pair`; advisory-only verdict/confidence/insufficient-data semantics).

## Fallback posture

- If Postgres is configured and unavailable, control-plane readers enter explicit `fallback` posture with surfaced errors.
- sqlite/local paths remain bounded dev/test fallback only.
- No silent fallback is presented as durable operational truth.

## Safety and boundary guarantees

- Advisory-only behavior is preserved: comparison never mutates policy activation/rollback state.
- GraphDB is still not used for control-plane policy/telemetry comparison state.
- Operator endpoint/UI semantics remain stable and bounded while source honesty improves.

## Follow-on phases

- Optional tightening: move queue/execution inspection helpers fully to shared Postgres repositories for complete control-plane read unification.
- Optional migrations: one-way bootstrap tooling for sqlite historical seeds into Postgres in environments transitioning from local seams.
