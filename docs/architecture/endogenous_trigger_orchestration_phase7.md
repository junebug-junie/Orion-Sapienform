# Phase 7: Endogenous Trigger Orchestration and Bounded Self-Revision

## Goals

Phase 7 introduces a deterministic orchestration layer that decides whether Orion should run bounded self-revision workflows, suppress execution, or no-op.

This layer converts internal pressure into typed, inspectable decisions and plans without introducing recursive agentic behavior.

## New canonical contracts

`orion/core/schemas/endogenous.py` defines:

- `EndogenousTriggerRequestV1`: bounded trigger input envelope (scope, subject refs, contradiction pressure, spark/autonomy pressure, concept quality signals, mentor gap counters, lifecycle state, recent history, policy metadata).
- `EndogenousTriggerSignalV1`: deterministic normalized pressure outputs.
- `EndogenousTriggerDecisionV1`: trigger outcome (`trigger | suppress | defer | coalesce | noop`), selected workflow, reasons, alternatives, cooldown/debounce state, and debug payload.
- `EndogenousWorkflowPlanV1` + `EndogenousWorkflowActionV1`: bounded ordered action list.
- `EndogenousWorkflowExecutionResultV1`: orchestration result, audit events, mentor invocation state, and produced artifacts.

## Deterministic trigger evaluation

`orion/reasoning/triggers.py` adds `EndogenousTriggerEvaluator`:

- combines contradiction, concept, autonomy, mentor, and reflective pressure deterministically;
- prioritizes workflow taxonomy:
  - contradiction_review
  - concept_refinement
  - autonomy_review
  - mentor_critique
  - reflective_journal
  - no_action
- supports lifecycle suppression for dormant/retired dynamic entities;
- emits alternatives-not-chosen and deterministic cause signatures for auditability.

## Cooldown, debounce, and anti-loop controls

`orion/reasoning/trigger_history.py` adds in-memory history/cooldown helpers:

- per-workflow cooldown checks
- per-subject cooldown checks
- contradiction signature debounce/coalescing
- deterministic recent-history queries

Repeated unchanged contradiction signatures are coalesced instead of re-triggering plans.

## Bounded planning and orchestration seam

`orion/reasoning/workflows.py` adds:

- `EndogenousWorkflowPlanner`: maps decision -> bounded action sequence with max action cap.
- `EndogenousWorkflowOrchestrator`: evaluate + plan + optional bounded mentor execution.

Planner action families include context slicing, contradiction/concept/autonomy review, mentor gateway invocation, advisory materialization, promotion gate checks, reflective journaling, audit tracing, and explicit stop.

## Mentor and promotion safety posture

Mentor workflow remains bounded:

- uses only Phase 6 `MentorGateway`
- constrained task type selection
- bounded context packet
- advisory artifact materialization only
- explicit promotion gate action in plan (`promotion_gate_check`) to preserve Phase 3 safeguards

## Runtime adoption guidance

Phase 7 does **not** force broad runtime activation.

Future runtime surfaces should call `EndogenousWorkflowOrchestrator.orchestrate(...)` in narrowly-scoped, policy-governed entrypoints and keep `execute_actions=False` by default unless a controlled execution surface is desired.

This preserves deterministic testability and prevents trigger storms while enabling incremental adoption.
