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

Orch now routes this action through a dedicated **brain-lane LLM verb** (`concept_induction_journal_synthesize`) using a payload-only grounding contract:
- reviewed subjects
- reviewed profile slices (concepts, clusters, state estimates)
- repository trace/back-end resolution metadata
- explicit provenance metadata

The synthesis request is deterministic:
- no prompt matching heuristics
- no agent mode
- no open chat-history context
- recall disabled

## Journal persistence + provenance

Synthesis persistence reuses the existing append-only journal boundary (`journal.entry.write.v1`) and does not introduce a second path.

Workflow metadata includes synthesis provenance:
- `source_workflow_id = concept_induction_pass`
- reviewed subjects
- reviewed profile ids/revisions
- repository backend used
- synthesis timestamp
- `synthesis_mode = brain_grounded`
- `synthesis_prompt_version = concept_induction_journal_grounded.v1`

## Anti-fabrication safeguards

Grounding protections now include:
- Prompt-level hard rules: only use provided payload, never invent missing details.
- Required missing-data language: `"not available in reviewed profile data"` when evidence is absent.
- Payload-only request construction in Orch (bounded metadata contract).
- Lightweight post-check for explicit unsupported concept/cluster references before persistence.

This differs from agent traces or open-ended reasoning: the LLM is used only for **faithful interpretation** of supplied artifacts, not discovery of new facts.
