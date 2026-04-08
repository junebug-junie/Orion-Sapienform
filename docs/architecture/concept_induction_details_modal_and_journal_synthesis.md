# Concept Induction Details Modal + Bounded Journal Synthesis

## Why add a Concept Induction details modal

`concept_induction_pass` previously surfaced only a compact summary string. That kept chat readable but hid the typed artifact structure now available in Spark concept-profile storage (profiles, concepts, clusters, state estimate, and repository trace context).

The details modal keeps the main chat bubble compact while exposing inspectable typed payloads for operator review.

## What payloads are exposed

The workflow metadata now includes `concept_induction_details` with bounded structure:

- `profiles[]`
  - subject, profile id, revision, created/window timestamps, concept/cluster counts
  - bounded concept rows (ids, labels, salience/confidence/evidence_count)
  - bounded cluster rows (ids, labels, representative labels, cohesion)
  - state estimate summary and compact provenance hints
- `trace`
  - repository resolution (requested/resolved backend, fallback policy/use, source path)
  - bounded lookup artifact rows used for review

This is intentionally typed and bounded so the UI can render deterministic sections without surfacing freeform internal reasoning.

## What “Synthesize to Journal” does

From the Concept Induction details modal, the user can trigger `Synthesize to Journal`.

The action sends an explicit workflow request override to run `concept_induction_pass` with `action=synthesize_to_journal` and the reviewed bounded details payload.

Server-side behavior:

1. Build bounded synthesis text from reviewed profile artifacts.
2. Reuse existing append-only journal write boundary (`journal.entry.write.v1`).
3. Persist journal entry through the normal journal channel.
4. Return workflow metadata with synthesis status and provenance.

## Why synthesis is bounded

The synthesis is deterministic over typed profile artifacts and repository trace context. It does **not** invoke an unbounded freeform reasoning dump and does not create a second journal persistence path.

## Provenance attachment

When synthesis is persisted, workflow metadata includes:

- source workflow id (`concept_induction_pass`)
- reviewed subjects and reviewed profile id/revision pairs
- timestamp and resolved backend
- persisted journal entry metadata

The journal entry `source_ref` is anchored to `concept_induction_pass` and reviewed profile refs for traceability.
