# Reasoning Summary Compiler Phase 4

Phase 4 introduces a deterministic compiler that converts promoted reasoning artifacts into bounded turn-time summaries and integrates them into chat-stance input building with fallback safety.

## Added components

- `orion/core/schemas/reasoning_summary.py`
  - `ReasoningSummaryRequestV1`
  - `ReasoningSummaryV1`
  - typed summary digests for claims/autonomy/spark/debug
- `orion/reasoning/summary.py`
  - `ReasoningSummaryCompiler`
  - contradiction-aware filtering
  - lifecycle-aware dynamic subject selection
  - drift-aware suppression for concept-translation claims
  - spark as temporal posture only
- `services/orion-cortex-exec/app/chat_stance.py`
  - bounded summary compilation seam (`_compile_reasoning_summary`)
  - summary injected into `chat_stance_inputs["reasoning_summary"]`
  - context debug flags:
    - `chat_reasoning_summary`
    - `chat_reasoning_debug`
    - `chat_reasoning_summary_used`

## Conservative compilation posture

- Preferred inclusion: canonical and strong provisional claims.
- Excluded: rejected/deprecated claims, unresolved contradicted artifacts, high-risk mentor proposals.
- Drift protection:
  - concept-translated claims (`concept_item`, `concept_delta`) are suppressed from summary claim output.
  - spark snapshots are compiled only into temporal spark posture fields.
- Lifecycle awareness:
  - active dynamic subject refs are selected only when lifecycle does not resolve to `dormant`/`decay`/`retire`.

## Chat stance integration behavior

- If reasoning summary compile succeeds and is strong, it is used as additional structured context.
- If compile is empty/weak/errors, fallback remains intact and stance behavior stays stable.
- Observability is explicit via deterministic debug counters and booleans.

## Drift status after Phase 4

- Concept translation drift: **translation-for-now** with explicit compiler suppression of over-strong usage.
- Spark source fragmentation drift: **translation-for-now**; summary treats spark as temporal state only.

## Phase 5+ extension path

Later phases can reuse `ReasoningSummaryV1` across additional runtime surfaces (routing/planning/stance variants)
without changing producer contracts, by extending compiler inputs and policy thresholds rather than replacing the seam.
