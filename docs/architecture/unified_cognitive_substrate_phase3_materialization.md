# Unified Cognitive Substrate — Phase 3 (Materialization and identity reconciliation)

## Why this phase exists

Phase 1 defined ontology/contracts and Phase 2 produced adapter graph records, but records were still local payloads. Phase 3 introduces deterministic identity-aware materialization so Orion can maintain a unified persistent substrate graph state across repeated inputs.

## Record-local vs materialized graph

- **Record-local (`SubstrateGraphRecordV1`)**: immutable adapter output snapshots.
- **Materialized graph state**: reconciled nodes/edges with stable identity indexes and merge behavior.

Phase 3 adds a bounded in-memory materialized graph store and reconciliation engine; it does not force immediate system-wide read cutover.

## Identity reconciliation posture (conservative)

- **Concept**: reconcile by stable concept id metadata, then normalized label fallback.
- **Drive**: reconcile by `(anchor_scope, subject_ref, drive_kind)`.
- **Goal**: reconcile only by explicit proposal signature lineage.
- **Evidence**: reconcile by explicit evidence content reference/type.
- **Tension/Contradiction**: reconcile only when explicit source identity exists (artifact/delta ids), otherwise avoid over-merging.
- **StateSnapshot/Hypothesis**: default to distinct unless explicit identity semantics are provided.

This intentionally prefers under-merging over false merges.

## Edge reconciliation posture

- Edges reconcile by canonical `(source, predicate, target)` identity key after node canonicalization.
- Repeated edges merge deterministically while preserving provenance/evidence lineage and confidence/salience envelopes.
- Duplicate edge spam is avoided via edge identity index.

## Provenance preservation

Materialization accumulates lineage in node/edge metadata and preserves provenance evidence refs over merges. Temporal windows and confidence/salience are merged conservatively.

## Phase boundary

- No activation/pressure propagation or decay dynamics yet.
- No frontier/teacher mutation paths yet.
- No GraphDB/SQL role collapse.
- No broad runtime/workflow rewrites.

## What later phases can build on

- durable graph-backed stores beyond bounded in-memory materialization,
- graph cognition read paths over reconciled substrate state,
- controlled dynamics layers (activation/decay/pressure) on top of stable identities,
- future mutation sources (frontier/teacher) targeting canonical substrate identity rules.
