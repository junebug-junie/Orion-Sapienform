# Mentor Gateway Phase 6

Phase 6 adds a bounded external critique loop that is advisory-only and routed through the existing reasoning substrate.

## What was implemented

- Typed mentor contracts in `orion/core/schemas/mentor.py`:
  - `MentorRequestV1`
  - `MentorResponseV1`
  - `MentorProposalItemV1`
  - `MentorGatewayResultV1`
  - bounded task taxonomy (`ontology_cleanup`, `contradiction_review`, `concept_refinement`, `autonomy_review`, `missing_evidence_scan`, `goal_critique`, `verb_eval_review`)
- Bounded context packer in `orion/reasoning/mentor_context.py`
- Deterministic mentor response mapper in `orion/reasoning/mentor_mapper.py`
- Provider-agnostic mentor gateway seam in `orion/reasoning/mentor_gateway.py` with safe stub default provider

## Advisory-only enforcement

- Mentor responses map only to `MentorProposalV1` artifacts with:
  - `status="proposed"`
  - `authority="mentor_inferred"`
  - provider/model/task provenance
- Materialization flows through existing `ReasoningMaterializer` write path.
- Promotion policy blocks direct canonicalization from mentor proposals and escalates/blocks contradiction-heavy mentor proposals.

## Bounded context posture

Mentor requests use bounded, inspectable context slices:
- selected artifact ids
- selected evidence refs
- deterministic packet of summary fields

No broad memory dump is passed to mentor providers in this phase.

## Observability

Gateway emits deterministic audit fields and logs for:
- request id
- task type
- provider/model
- context artifact counts
- proposal counts
- materialization counts
- failure reason on safe failure path

## Phase posture

This is an infrastructure phase:
- no broad ambient mentor invocation from chat turns
- no direct canonical selfhood/autonomy/relationship mutation
- no bypass of promotion/HITL
