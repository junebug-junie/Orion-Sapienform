# Reasoning Canonicalization Phase 5

Phase 5 reduces the two major remaining drifts by canonicalizing concept and spark source contracts.

## What was fixed at source vs translated

### Concept domain

- **Canonicalized now:** Introduced `ConceptV1` as a first-class reasoning artifact payload.
- **Adapter update:** concept induction now emits canonical `ConceptV1` artifacts and can optionally emit legacy `ClaimV1` concept translations for bounded compatibility (`include_legacy_claims=True`).
- **Promotion update:** canonical concept promotion now uses concept-specific thresholds (confidence/salience/evidence) instead of blanket translation suppression.
- **Summary update:** compiler can include canonical concepts via `active_concepts` while legacy translated concept-claims remain conservative.

### Spark domain

- **Canonicalized now:** Introduced `SparkSourceSnapshotV1` as canonical spark source seam.
- **Adapter update:** legacy spark snapshot shapes normalize into `SparkSourceSnapshotV1`; reasoning artifacts are derived from this canonical source snapshot.
- **Compatibility:** direct legacy and embedded spark forms remain accepted through normalization helpers.
- **Policy posture unchanged:** spark remains temporal/provisional; no durable identity hardening.

## Bounded compatibility posture

- Legacy concept-claim translations are still readable and optionally emitted for transition safety.
- Legacy spark source shapes are still accepted via deterministic normalization seams.
- Canonical forms are now preferred by adapters, promotion semantics, and summary compilation.

## Registry and contracts

New canonical contracts were registered and exported:
- `ConceptV1`
- `SparkSourceSnapshotV1`
- `ReasoningConceptDigestV1`

## Why this reduces drift

- Concept semantics are no longer forced through generic claim fields for the main path.
- Spark source ambiguity is reduced to one canonical source model with bounded adapters.
- Phase 2–4 layers now consume canonical concept/spark semantics while preserving fallback compatibility.
