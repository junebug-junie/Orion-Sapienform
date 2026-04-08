# Reasoning Promotion Phase 3

Phase 3 adds deterministic policy governance on top of the Phase 1 schemas and Phase 2 materialization seam.

## What was added

- `orion/core/schemas/reasoning_policy.py`
  - `PromotionEvaluationRequestV1`
  - `PromotionEvaluationItemV1`
  - `PromotionEvaluationResultV1`
  - contradiction finding contracts
  - lifecycle evaluation request/result contracts
- `orion/reasoning/promotion.py`
  - deterministic transition evaluator
  - contradiction-aware gating
  - HITL escalation policy
  - structured `PromotionDecisionV1` production
- `orion/reasoning/lifecycle.py`
  - deterministic dynamic entity/domain lifecycle governance (`emerge/strengthen/dormant/decay/retire/revive`)
- repository seam upgrades in `orion/reasoning/repository.py`
  - get by id
  - status updates
  - subject-ref listing
  - contradiction queries

## Deterministic policy posture

- Explicit transition matrix only allows:
  - `proposed -> provisional`
  - `provisional -> canonical`
  - `canonical -> deprecated`
  - `* -> rejected`
- Contradictions are first-class blockers:
  - unresolved medium+ blocks canonical promotion
  - unresolved high/critical blocks provisional promotion
- HITL escalation triggers include:
  - canonical promotions in anchor scopes `orion`, `juniper`, `relationship`
  - autonomy goal proposal canonicalization
  - high-risk mentor proposals
  - provenance weakness/contradiction pressure where applicable

## Drift handling (explicit)

1. **Concept induction translation drift**
   - still translation-for-now from concept outputs to claim/relation forms.
   - policy blocks canonicalization for `claim_kind in {"concept_item", "concept_delta"}` by default.

2. **Spark source fragmentation drift**
   - remains translation-for-now.
   - policy treats spark artifacts as valid for reasoning pressure but blocks canonical identity promotion.

## Phase 4 handoff

Phase 4 turn-state compilation can consume:
- promoted artifacts (`PromotionEvaluationResultV1` + `PromotionDecisionV1`)
- unresolved contradiction findings
- lifecycle outputs for active/dormant entity selection

without requiring broad producer cutover.
