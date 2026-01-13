# Architecture Review — Audit 001 (Postfix)

## Fixes applied
- Replaced equilibrium-service verb request emission with a domain event and routed the corresponding verb request through cortex-orch.
- Removed cortex-exec verb-request publishing; collapse log requests now emit collapse intake events.
- Reconciled catalog entries for previously unknown channels (conversation, exec LLM/recall, spark candidate, vision edge).
- Restored config lineage for bus-mirror, bus-tap, chat-memory, and hub (added missing env/requirements/settings).
- Re-ran platform audits and stored post-fix artifacts in `codex_reviews/audit_001/reports_postfix/`.

## Remaining debt
- Placeholder channels / pattern subscriptions still appear in channel drift (env-derived channels and `orion:*` pattern).
- Schema drift persists: multiple BaseEnvelope emissions still lack schema_id enforcement.
- Config lineage still reports missing compose stanzas for several services (out of scope for this phase).

## Verdict
**PASS-WITH-DEBT**

Spine violations are cleared. Remaining drift items are non-spine debt that should be addressed in subsequent phases.
