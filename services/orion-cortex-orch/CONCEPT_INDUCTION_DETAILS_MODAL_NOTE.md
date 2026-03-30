# Concept Induction Details Modal + Journal Synthesis Note

## Why this modal exists

`concept_induction_pass` previously surfaced only a compact workflow summary in chat. That summary remains intentionally compact, but operators also need an inspectable view of **what was reviewed** (profile revisions, concepts/clusters, state estimate, backend resolution, and typed artifacts) without exposing hidden reasoning traces.

## What is exposed

The workflow now emits a bounded `concept_induction_details` payload under workflow metadata.

The payload includes:
- profile-level review fields (`subject`, `profile_id`, `revision`, timestamps/window, concept/cluster counts)
- bounded concept and cluster slices for each reviewed profile
- state-estimate summary per profile
- repository/backend resolution metadata (`requested_backend`, `resolved_backend`, fallback info)
- bounded typed artifact rows used in the review (e.g., profile lookup rows)

## Why this is bounded (and not hidden reasoning)

This surface is deliberately typed and size-clamped. It is not an agent-thought stream and does not expose freeform chain-of-thought. It only exposes workflow artifacts required to understand what the pass reviewed and how data was sourced.

## What `Synthesize to Journal` does

The Hub modal includes a deterministic `Synthesize to Journal` action for `concept_induction_pass`.

That action sends a `workflow_request_override` with:
- `workflow_id = concept_induction_pass`
- `action = synthesize_to_journal`
- the reviewed `concept_induction_details`

Orch then performs a bounded synthesis over reviewed subjects (orion/juniper/relationship), notable concepts/clusters/state-estimate patterns, and repository resolution context.

## Journal persistence + provenance

Synthesis persistence reuses the existing append-only journal boundary (`journal.entry.write.v1`) and does not introduce a second path.

Workflow metadata includes synthesis provenance:
- `source_workflow_id = concept_induction_pass`
- reviewed subjects
- reviewed profile ids/revisions
- repository backend used
- synthesis timestamp
