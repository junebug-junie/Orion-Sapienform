# Runtime Introspection Interrogation - End-to-End Evidence-First Design

**Date:** 2026-05-09  
**Status:** Approved for implementation  
**Primary Goal:** Operator-grade runtime introspection on Hub agent lane with strict anti-fabrication guarantees

---

## Problem

Orion can produce plausible but incorrect explanations when asked to explain why it answered a certain way, especially when those questions are routed through conversational lanes (`chat_general` or `chat_quick`) instead of a runtime-evidence lane.

Current shortcomings:
- Runtime introspection questions are not consistently routed to capability-backed runtime analysis.
- Existing runtime verbs/skills are oriented toward system health/housekeeping, not full response-pipeline forensics.
- There is no unified evidence contract for hop-by-hop diagnosis (Hub -> Gateway -> Orch -> Exec -> LLM plus storage and trace rails).
- Fallback behavior is not consistently surfaced as degraded certainty to operators.
- Claims can be produced without strict claim-to-evidence enforcement.

---

## Goal

Deliver a phased but complete introspection system that:

1. Routes introspection/interrogation intents to a dedicated agent/runtime interrogation path.
2. Performs multi-hop evidence collection across the full response pipeline.
3. Emits dual output:
   - strict machine-readable evidence contract
   - human-readable report preserving the same rigor
4. Hard-blocks unsupported runtime claims (anti-bullshit guardrail).
5. Supports optional cross-correlation interrogation with explicit authorization and gating.
6. Always includes:
   - Root Cause Summary
   - Recommended Next Steps
   - Summary

---

## Non-Goals

- Replacing normal conversational lanes for non-introspection queries.
- Replacing existing operational skills (they are inputs/capabilities, not removed).
- Building a new standalone UI surface first (output can initially ride existing response surfaces).
- Relaxing existing privacy/security constraints to increase observability.

---

## Selected Approach

Adopt **Approach B: Layered Interrogation Graph**.

Why:
- Supports multi-hop evidence gathering and partial degradation.
- Enables phase rollout with stable contract semantics.
- Separates strict evidence assembly from readable narration.
- Best fit for operator debugging and forensic analysis.

Rejected alternatives:
- Monolithic mega-verb: too brittle and hard to maintain.
- Light adapter over existing debug payloads: too inconsistent for strict truth enforcement.

---

## Routing Policy

### Introspection intent handling

When a query is classified as runtime introspection/interrogation:
- route to agent-runtime interrogation verb path (not `chat_quick`/`chat_general` direct reasoning lane)
- do not rely on static conversational prompt synthesis for root-cause claims

### Fallback policy

Fallback is allowed but explicit:
- if interrogation path cannot fully execute, return best-effort result
- include visible fallback callout and degraded confidence
- never fabricate missing evidence

---

## End-to-End Hop Topology (Required Coverage)

The interrogation orchestrator must support evidence probes across:

1. Hub ingress and request shaping
2. Cortex Gateway normalization and relay
3. Cortex Orch intake, route handling, mind invocation and handoff merge
4. Cortex Exec step execution path (including stance shortcut and fallback behavior)
5. LLM Gateway invocation and response envelope behavior
6. State-service dependencies (`orion:state:request/reply`)
7. Verb runtime rails (`orion:verb:request/result`)
8. Planner/AgentChain capability bridge nested rails (`orion:cortex:request/result` from capability execution)
9. Mind artifact persistence rail (`orion:mind:artifact` -> SQL writer -> `mind_runs`)
10. Chat history turn envelopes and trace payloads
11. Signals trace cache / OTEL trace linking where available

Cross-cutting cognitive/runtime surfaces to interrogate (phased execution, full inventory retained):
- thinking traces and reasoning availability
- recall directives and payload lineage
- mind run/handoff/stance skip path
- memory graph-related context
- concept induction/contextual profile signals
- autonomy and route/debug payloads
- presence and situation context propagation
- provenance and hop-level correlation continuity

---

## New Orchestration Verb

Working verb id: `interrogate_runtime_state`

Execution posture:
- capability-backed, agent-runtime route
- deterministic evidence probes first
- bounded fan-out for deeper probes

Expected behavior:
- gather normalized evidence records
- compute hop status and contradictions
- rank candidate root causes with confidence
- emit strict contract and readable report

---

## Evidence Contract (Strict Layer)

The strict layer is authoritative and machine-verifiable.

```yaml
query_scope:
  mode: current_turn | target_correlation | prior_window | explicit_set
  session_scope_required: true
  as_of_utc: "<iso8601>"

correlation_resolution:
  target_correlation_id: "<optional>"
  requested_prior_turns: <int>
  requested_correlation_ids: ["..."]
  resolved_correlation_ids: ["..."]
  resolution_strategy: explicit_set | target_plus_prior | current_only
  authorization_scope: same_session | cross_scope
  authorization_decision: allow | deny
  authorization_reason: "<string>"

hop_status:
  hub: ok | degraded | fail | unknown
  cortex_gateway: ok | degraded | fail | unknown
  cortex_orch: ok | degraded | fail | unknown
  cortex_exec: ok | degraded | fail | unknown
  llm_gateway: ok | degraded | fail | unknown
  state_service: ok | degraded | fail | unknown
  mind_artifact_path: ok | degraded | fail | unknown

trace_context:
  trace_ids: ["<otel_trace_id>"]
  channel_refs: ["request/reply channels touched"]
  hop_event_refs: ["service:event ids"]

evidence:
  - evidence_id: "<id>"
    evidence_type: artifact_row | execution_step | hop_event | trace_span | derived_inference
    source: "<service/component>"
    source_id: "<row/event/step id>"
    correlation_id: "<id>"
    observed_at_utc: "<iso8601>"
    freshness_ms: <int|null>
    staleness_class: fresh | stale | unknown
    confidence_base: <0..1>
    redacted: true | false

claims:
  - claim_id: "<id>"
    statement: "<runtime claim>"
    supported: true | false | partial
    confidence: <0..1>
    evidence_refs: ["evidence_id"]
    conflict_set: ["claim_id"]
    selected_authority: artifact_row | execution_step | hop_event | trace_span | derived_inference
    selection_reason: "<string>"

fallback:
  fallback_level: none | partial | best_effort | insufficient_evidence
  fallback_used: true | false
  fallback_reason: "<string|null>"
  missing_primitives: ["<primitive id>"]
  degraded_hops: ["<hop id>"]
  budget_clipped: true | false

root_cause_summary:
  first_failing_boundary: "<hop pair or component>"
  rankings:
    - cause_id: "<id>"
      description: "<root cause hypothesis>"
      score: <float>
      confidence: <0..1>
      evidence_refs: ["evidence_id"]
      contradiction_count: <int>
      freshness_penalty: <float>

recommended_next_steps:
  - step_id: "<id>"
    action_type: config_change | routing_policy_change | primitive_missing | service_health | data_lag | prompt_contract_update
    description: "<what to do>"
    expected_outcome: "<why this resolves issue>"
    evidence_refs: ["evidence_id"]
    priority: high | medium | low

summary:
  headline: "<one-line technical conclusion>"
  certainty: high | medium | low
```

---

## Human Report (Readable Layer)

Readable report must be fully derived from the strict layer. No additional unsupported claims.

Sections:
1. What happened
2. Root Cause Summary
3. Recommended Next Steps
4. Summary
5. Known Unknowns / Degraded Signals

If fallback used:
- prepend explicit banner: `Fallback mode active: best-effort analysis with reduced certainty.`

---

## Anti-Bullshit Guardrail

Hard enforcement before output emission:

1. Every runtime declarative claim must reference evidence ids.
2. Unsupported hard claims block publish (fail-closed to `insufficient_evidence` envelope).
3. Soft inferences without evidence are stripped or downgraded with explicit uncertainty.
4. Missing primitives must be listed in output, never silently ignored.

Validator outputs:
- `unsupported_claim_count`
- `blocked_claim_count`
- `downgraded_claim_count`
- `guardrail_outcome: pass | degraded | blocked`

---

## Authorization, Privacy, and Redaction

### Scope policy

- Default: same-session introspection only.
- Cross-session/cross-user introspection requires explicit operator capability (`introspection.cross_scope`).
- Denied requests return explicit policy decision and no data leakage.

### Redaction policy

Redact sensitive values from evidence/report:
- secrets, tokens, credentials, auth headers, sensitive raw payloads

Output fields:
- `redaction_applied`
- `redaction_rules_triggered[]`

---

## Correlation Traversal and Gating

Inputs:
- `target_correlation_id` (optional)
- `prior_turns` (optional bounded integer)
- `correlation_ids[]` (optional explicit set)
- `session_scope_required` (default true)

Resolution order:
1. use explicit `correlation_ids[]` if provided and authorized
2. else use `target_correlation_id` (+ prior expansion if requested)
3. else use current turn correlation only

Required outputs:
- `resolved_correlation_ids`
- `correlation_resolution_strategy`
- `join_confidence` per hop/join

Hard caps:
- max correlations per request
- max prior turn expansion
- max per-hop fan-out

---

## Phased Delivery Plan (Full Inventory Preserved)

### Phase 1 - Core truth path

Ship minimum reliable forensic path:
- routing/hop continuity
- mind/stance skip/fallback correctness
- step-level execution evidence
- strict contract + human report + guardrail

### Phase 2 - Extended cognition/runtime surfaces

Add deep probes for:
- recall lineage
- memory graph context
- concept induction/context profile
- autonomy payload exports
- presence/situation propagation

### Phase 3 - Multi-correlation comparative diagnostics

Add:
- differential analysis across turns
- contradiction-aware root-cause ranking
- stronger recommendation sequencing

---

## Cost and Performance Budgets

Per request constraints:
- `max_total_latency_ms`
- `max_probe_fanout`
- `max_store_reads_per_probe_family`
- `max_correlation_targets`

Behavior on budget exhaustion:
- return partial contract with `budget_clipped=true`
- preserve strict uncertainty and fallback semantics

---

## Testing Strategy

### Unit tests

- correlation resolution and authorization gating
- evidence-type normalization and precedence
- claim-evidence validator fail-closed behavior
- fallback-level transitions and missing primitive reporting
- redaction correctness

### Integration tests

- end-to-end single-turn interrogation
- target correlation + prior-turn expansion
- explicit multi-correlation interrogation
- routing failure localization by boundary
- mind/stance skip verification path

### Golden-path scenarios

1. Valid introspection with complete evidence and no fallback
2. Missing hop data with explicit degraded output
3. Contradictory evidence with deterministic precedence outcome
4. Guardrail block on unsupported claims
5. Cross-scope denial with policy-explicit response

---

## Rollout and Safety

1. Feature flag on Hub agent route.
2. Shadow mode (compute contract without default user surfacing).
3. Compare against operator-labeled incident diagnoses.
4. Enable default for introspection intents after precision/coverage SLO is met.

Suggested readiness gates:
- high supported-claim ratio
- low unsupported-claim block rate
- stable latency within introspection budget
- strong operator agreement on root cause ranking

---

## Open Questions and Resolutions

Resolved in this design:
- strict-vs-readable output: both, strict first
- fallback allowed: yes, explicit and non-speculative
- full stack coverage: yes, phased but complete inventory retained
- correlation traversal: optional and gated with authorization
- mandatory sections: root cause, next steps, summary always present

Deferred (explicitly):
- standalone dedicated Hub UI for interrogation artifact browsing
- cross-org policy federation for multi-tenant introspection controls

---

## Summary

This design introduces a dedicated agent-route runtime interrogation capability that can explain "why Orion said what it said" using multi-hop evidence across Hub, Gateway, Orch, Exec, LLM, and persistence/trace rails. It enforces an evidence-first contract, explicit fallback semantics, correlation-aware traversal, and hard anti-fabrication guardrails while still providing a human-readable report operators can quickly act on.

