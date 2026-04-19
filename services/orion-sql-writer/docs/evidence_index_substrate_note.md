# Evidence index substrate (slice note)

This slice introduces a **minimal evidence index substrate** inside `orion-sql-writer` as a repo-native module, rather than a brand-new standalone service process.

## Why module-first in this slice

- We already have a validated runtime write path in sql-writer.
- The smallest safe migration path is to persist normalized evidence units in the same transaction rail where journal/collapse writes already succeed.
- This keeps current journal behavior intact while introducing a clean adapter seam that can be lifted into `orion-evidence-index` later with low risk.

## Core vs adapter split

Core substrate (service-local):
- `evidence_units` storage model/table
- filter/query/search primitives
- context expansion (parent/children/siblings)

Adapters (artifact-specific logic):
- `JournalEvidenceAdapter`
- `CollapseMirrorEvidenceAdapter`
- `MarkdownSpecEvidenceAdapter` (hierarchical doc/section/leaf units)

The worker calls `build_evidence_units(kind, payload)` and persists resulting normalized `EvidenceUnitV1` rows.

## Journal migration seam

Journal-specific `journal_entry_index` remains intact for compatibility.
In parallel, journal writes now also emit normalized `evidence_units` rows through the journal adapter.
This gives a no-downtime migration path for retrieval consumers.

## Follow-up to become standalone `orion-evidence-index`

1. Move `evidence_units` model/repository + adapter ingestion dispatch into a dedicated service package.
2. Route `orion:evidence:index:upsert` exclusively to that service.
3. Keep adapter contracts and schemas unchanged so producers do not churn.
