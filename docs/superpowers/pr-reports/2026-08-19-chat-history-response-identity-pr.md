# PR report: chat_history_log response_identity

**PR:** https://github.com/junebug-junie/Orion-Sapienform/pull/1742
**Branch:** `feat/chat-history-response-identity`
**Status:** DONE

## Summary

- `chat_history_log` gets a new additive `response_identity` column: the identity that produced `response` on an assistant-authored row (real served model-card name when known, e.g. `qwen-36-instruct`, else a human-readable responder name like `Claude`/`Orion`). `user_id` keeps its existing meaning (prompt-side/human identity) untouched -- no breaking change for any of its ~50 existing readers (per the 2026-08-19 AI Town table-split Phase 2 consumer audit).
- Fixed the actual drop bug: `orion-sql-writer`'s `_ensure_chat_history_from_message` was receiving `model`/`speaker` on every assistant-authored message already, but silently discarded both before merging into the row.
- Root-caused and fixed the deeper "model name never came through a few PRs" gap: `orion-cortex-exec` only ever computed the real served model name inside a reasoning-trace object that's empty on ~all real turns. Added `_last_model_used()`, independent of reasoning content, and wired it into the metadata key Hub was already (uselessly) reading.
- Wired `response_identity` through all 3 real turn-publish call sites (WS normal lane, WS agent-claude lane, HTTP lane) -- 2 of these were caught missing it by code review and fixed.
- "Ask Claude" (`room_claude_relay.py`) already used the response lane correctly (`role="assistant"` -> `response` column, not `prompt`) -- it only needed the merge-drop fix, no call-site change.
- Endogenous outreach: honestly documented as a partial fix -- it now runs through a different pipeline (harness finalize, from a very recently merged PR) with no model-tracking field at all; `response_identity` goes from blank to `"Orion"` (real improvement), full model-name support is flagged as a follow-up.

## Outcome moved

`chat_history_log.response_identity` (the "username" field Juniper was looking at) goes from always-NULL on every assistant-authored row to correctly populated for: normal Hub chat turns (real served model name), the agent-claude lane (fcc model label), the HTTP chat lane (real served model name), and Ask-Claude/room-companion turns (real Claude model name, e.g. `claude-sonnet-5`). Endogenous outreach goes from blank to `"Orion"`.

## Current architecture

`chat_history_log` is written two ways: a turn-level row (`ChatHistoryTurnV1` -> direct upsert, `user_id` set from the inbound human message) and per-role messages (`ChatHistoryMessageV1`, fill-only merge via `_ensure_chat_history_from_message`). The message schema already carried `model`/`speaker`, but the merge function only ever forwarded `prompt`/`response`/`session_id`/`memory_status`/`memory_tier`/`client_meta` -- never `model`/`speaker`. Separately, `orion-cortex-exec`'s `metadata["model"]` (already read by 4 Hub call sites) was never populated by anything -- the only place a served model name landed was `MetacognitiveTraceV1.model`, gated on non-empty `reasoning_content`, which reads "unknown" on essentially every real turn (confirmed by a prior graphify-recorded investigation).

## Architecture touched

- `orion/schemas/chat_history.py` -- `ChatHistoryTurnV1.response_identity`
- `orion-sql-writer`: `ChatHistoryLogSQL`/`AitownChatHistoryLogSQL` models, boot-time DDL, `_ensure_chat_history_from_message`
- `orion-cortex-exec`: `_last_model_used()` in `router.py`, wired into `PlanExecutionResult.metadata["model"]`
- `orion-hub`: `chat_history.py` (`build_chat_turn_envelope`), `websocket_handler.py` (2 call sites), `api_routes.py` (1 call site), `endogenous_outreach.py` (documentation only)

## Files changed

- `orion/schemas/chat_history.py`: add `response_identity` field to `ChatHistoryTurnV1`
- `services/orion-sql-writer/app/models/chat_history_log.py`: add `response_identity` column
- `services/orion-sql-writer/app/models/aitown_chat_history_log.py`: mirror column (schema-parity gate)
- `services/orion-sql-writer/app/main.py`: boot-time `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for both tables
- `services/orion-sql-writer/app/worker.py`: stop dropping `model`/`speaker` in the message-path merge
- `services/orion-sql-db/manual_migration_chat_history_response_identity_v1.sql`: documented migration (boot DDL is authoritative, this is for parity/manual apply)
- `services/orion-cortex-exec/app/router.py`: `_last_model_used()`, populate `metadata["model"]`
- `services/orion-hub/scripts/chat_history.py`: `build_chat_turn_envelope(response_identity=...)`
- `services/orion-hub/scripts/websocket_handler.py`: wire `response_identity` at both WS call sites (normal + agent-claude lane, with `fcc_model_label` fallback)
- `services/orion-hub/scripts/api_routes.py`: wire `response_identity` at the HTTP call site
- `services/orion-hub/scripts/endogenous_outreach.py`: comment documenting the known harness-pipeline model-tracking gap
- Tests: `test_chat_history_response_identity_merge.py`, `test_router_response_model_export.py`, additions to `test_chat_turn_spark_meta_turn_effect.py`

## Schema / bus / API changes

- Added: `chat_history_log.response_identity` (TEXT, nullable), `aitown_chat_history_log.response_identity` (mirror), `ChatHistoryTurnV1.response_identity` (Optional[str])
- Removed: none
- Renamed: none
- Behavior changed: `orion-cortex-exec`'s `PlanExecutionResult.metadata`/`CortexClientResult.metadata` now includes a `model` key when an `LLMGatewayService` step ran -- additive, no existing consumer breaks (generic `Dict[str, Any]`)
- Compatibility notes: fully additive; `user_id` semantics unchanged

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no (no env keys touched)
- local `.env` synced: n/a
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-sql-writer: 37 passed
  (test_chat_history_response_identity_merge.py, test_aitown_chat_history_dual_write.py, test_chat_history_turn_coalesce.py)
services/orion-cortex-exec: 27 passed
  (test_router_response_model_export.py, test_router_autonomy_payload_export.py)
services/orion-hub: 133 passed
  (test_chat_history_no_raw_publish.py, test_agent_claude_chat_history.py, test_endogenous_outreach.py,
   test_chat_history_rehydrate.py, test_chat_turn_spark_meta_turn_effect.py)

Full-service-suite baselines compared against unmodified main (worktree vs primary checkout, same commit):
- sql-writer: 7 pre-existing failures, identical set on both branches (notify_attention_ack/escalate,
  biometrics_summary_sql_shape, journal_entry_payload_boundary -- unrelated to this change)
- cortex-exec: 13 pre-existing collection errors, identical on both branches (full-suite collection
  conflicts unrelated to this change)
- hub: full suite has pre-existing order-dependent flakiness present on main too (confirmed via isolated
  re-runs of every differing test name -- e.g. test_llm_route_selector.py, test_workflow_schedule_runtime_paths.py
  fail identically on main when run in isolation). No failure traced to this branch's diff.
```

## Evals run

No dedicated eval harness exists for these seams (schema/plumbing fix, not a cognition/quality surface). Not applicable.

## Docker/build/smoke checks

Not run -- Docker was not exercised in this environment for this change. The boot-time DDL
(`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`) is the same self-healing convention every prior
`chat_history_log` column addition (`memory_status`, `thought_process`, `llm_uncertainty_*`, etc.) already
used. Verify on deploy via `\d chat_history_log` showing `response_identity`, and check a live row after a
real turn.

## Review findings fixed

- Finding: HTTP `/api/chat` path's `build_chat_turn_envelope` call never passed `response_identity`, despite `spark_meta["model"]` already being available there -- every chat_history_log turn row written via HTTP would have shipped with `response_identity=NULL`.
  - Fix: `services/orion-hub/scripts/api_routes.py:3289` now passes `response_identity=spark_meta.get("model")`.
  - Evidence: `test_chat_turn_spark_meta_turn_effect.py`'s new `response_identity` round-trip tests; call-site trace confirmed by the review subagent against the live code.
- Finding: `websocket_handler.py`'s agent-claude lane sets `gateway_meta = route_debug.get("agent_claude")`, whose payload only ever has `fcc_model_label`, never `model` -- so `response_identity` was always NULL for every agent-claude-lane turn despite the sibling `build_chat_history_envelope` call a few lines below already applying that exact fallback for the *message*-level write.
  - Fix: added `... or (gateway_meta or {}).get("fcc_model_label")` at the turn-publish call site, matching the existing pattern.
  - Evidence: the review's second pass (cleanup finder) independently confirmed the same gap and noted it had been silently masked by the message-level write's fill-only merge -- the turn-path fix removes that fragile masking dependency, not just papers over the symptom.
- Finding (doc-accuracy nit): `_last_model_used`'s docstring and its test file referenced the wrong function name (`_collect_metacognitive_traces` instead of the real `_collect_metacog_traces`).
  - Fix: corrected both references.
  - Evidence: `grep -n "_collect_metacognitive"` returns no hits after the fix.
- Not fixed (non-material, explicitly deferred): `_last_model_used` duplicates the `payload.get("model_used") or payload.get("model")` extraction already inline at router.py:757, and re-walks `step_results` a second time. Both flagged by the reviewer as low/negligible severity with no runtime effect; left as-is per CLAUDE.md's "don't build cathedrals" (a shared-helper refactor for two call sites is not worth the seam right now).

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `endogenous_outreach.py` still cannot report the real served model name (its pipeline, `execute_unified_turn`/`HarnessRunV1`, has no such field at all) -- `response_identity` will show `"Orion"` there, not a model-card name, until a follow-up adds model tracking to the harness-finalize pipeline.
- Mitigation: documented explicitly in code (`endogenous_outreach.py`'s `_publish_history`) and in this report rather than silently left broken; not a regression -- was blank before, is `"Orion"` now.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1742
