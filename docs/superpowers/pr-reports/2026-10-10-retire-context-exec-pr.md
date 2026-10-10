# Retire orion-context-exec completely

## Summary

- orion-context-exec was a finished experiment with no production use, and this PR deletes it completely. Nothing was running it, nothing listened on :8096, and no Postgres tables belong to it. Yet three other services were still wired to it.
- **cortex-exec** used to send every depth-2 ("agent") turn to a bus channel that nothing listened on. Each one waited 60 seconds, then answered "Insufficient grounding". It now fails straight away and says plainly that no depth-2 agent runtime exists. There is no fallback to context-exec.
- **cortex-orch** no longer bumps prompts containing words like "impact", "replace" or "where did ... come from" up to depth 2, and no longer tags them for context-exec.
- **Hub** loses the context-exec Agent lane, the live agent_step relay, the "Pending Decisions" proposal-review panel (routes, client, JS and template slots), and every setting that fed them. Agent and Orion mode still go through FCC, unchanged.
- **self-experiments** loses its dispatch and retry routes. context-exec was the only place they could send work, and per memory nothing ever called them (0 dispatches out of 27 rows).
- **Shared code** loses the context-exec bus channels, its request/run/artifact/proposal schemas and registry entries, and the five `context_exec_*` cognition verbs. The metric lock is re-locked.

## Outcome moved

- **Failure mode removed:** a depth-2 cortex turn no longer holds a supervisor for 60 seconds waiting on a reply that can never come.
- **Honest answer:** the turn now returns this text immediately: "No depth-2 agent runtime is available: orion-context-exec was retired (2026-10-10) and the planner-react/agent-chain organs were removed before it."
- **About 27k lines deleted.** 134 files were removed, including the whole service.

## Current architecture

### Before this patch

- **cortex-exec Supervisor.**
  - With `CONTEXT_EXEC_ENABLED=true` (set by the compose fallback and live on all four cortex-exec containers), any mode=agent or mode=council turn was sent through `_context_exec_escalation`.
  - That function did `ContextExecClient.run`, which made an RPC call to `orion:exec:request:ContextExecService`. Nothing consumes that channel, so after a 60s timeout it returned "Insufficient grounding: context-exec failed before acquiring evidence."
  - `CONTEXT_EXEC_LEGACY_FALLBACK` was set in compose but **never read by any code**. There was no "legacy path" behind it: the planner/agent-chain runtime it would have fallen back to had already been removed.
- **cortex-orch DecisionRouter.** `_context_exec_mode_for_request` matched keywords in the prompt, forced the turn to depth 2, and tagged it `agent_runtime_engine=context_exec`.
- **Hub.**
  - The `should_use_context_exec_agent_lane` branch existed in both `api_routes.handle_chat_request` and `websocket_handler`, gated off by `HUB_AGENT_CONTEXT_EXEC_ENABLED=false`.
  - `AgentStepRelay` was subscribed to `orion:context_exec:event`. It was the only live subscriber on that channel and nothing published to it.
  - `/api/proposal-review/*` proxied to :8096, gated off by `HUB_PROPOSAL_REVIEW_ENABLED=false`.
- **self-experiments.** `POST /v1/experiments/{id}/dispatch` compiled a `ContextExecRequestV1` and sent it over the bus. Live had `SELF_EXPERIMENTS_DISPATCH_ENABLED=true`.

### Live checks before the patch

- Postgres: no context-exec or proposal-ledger tables. The only proposal tables are the unrelated `substrate_*` ones.
- `cognition_traces`: 0 rows with mode agent/council and 0 rows with a ContextExecService step in the last 30–60 days.
- Redis: `PUBSUB NUMSUB orion:exec:request:ContextExecService` = 0, and `orion:context_exec:event` = 1 (Hub's relay).
- Bus captures: two PSUBSCRIBE captures on the context-exec channel families recorded 0 messages.

## Architecture touched

| Area | Change |
| --- | --- |
| `services/orion-context-exec/` | deleted |
| cortex-exec | `ContextExecClient`, settings, env, compose and README rows removed. `_context_exec_escalation` replaced by `_agent_runtime_unavailable`. The drift guardrail no longer rewrites that stub text, but only while it is still the answer. |
| cortex-orch | keyword promotion and tagging removed. The agent plan step is now `agent_runtime` with `services=[]`. The answer-depth reader learned the `AgentRuntime` key. |
| hub | agent lane, step relay, proposal review and the curiosity hint prepend (which only fed that lane) removed, along with 10 settings/env/compose keys and the template slots |
| self-experiments | dispatch and retry routes, the context-exec client, compile/parse/profile code and 6 settings removed |
| shared | `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `orion/schemas/context_exec.py` (trimmed), `proposal_ledger.py` and `proposal_lifecycle.py` (deleted), `orion/normalizers/agent_trace.py` (learned `AgentRuntime`), `orion/hub/chat_route.py` |

## Files changed

### Deleted

- The whole `services/orion-context-exec/` tree.
- Scripts:
  - all of `scripts/context_exec_*`
  - `self_experiment_context_exec_smoke.sh`, `proposal_review_api_smoke.sh`, `denver_memory_correction_vertical_smoke.sh`
  - `orion_proposal_cli.py`
  - `repl/orion_fresh_main_smoke.sh` (the context-exec proposal smoke ladder)
  - `run_answer_depth_live_proof.py` (its pass condition required context-exec, planner or agent-chain bus hops, so it could never pass again)
- `orion/cognition/verbs/context_exec_*.yaml` (5 files).
- `orion/schemas/proposal_ledger.py` and `orion/schemas/proposal_lifecycle.py`.
- Hub:
  - `scripts/context_exec_agent_bridge.py`, `context_exec_client.py`, `agent_step_relay.py`
  - `proposal_review_client.py`, `proposal_review_routes.py`, `verify_agent_repl_stream_live.py`
  - `static/js/proposal-review-ui.js`
- Operational docs: `docs/context-exec-beta-runbook.md`, `docs/architecture/context_exec_rlm.md`, `docs/proposal-review-api.md`, `docs/postflight/context_exec_adoption_manifest.md`. Historical PR reports and specs are kept.
- Tests for all of the above.

### Changed

- `orion/schemas/context_exec.py`: keeps only `ContextExecPermissionV1`.
- `services/orion-cortex-exec/app/supervisor.py`, `clients.py`, `executor.py`, `service_registry.py`, `settings.py`.
- `services/orion-cortex-orch/app/decision_router.py`, `orchestrator.py`, `main.py`.
- Hub:
  - `services/orion-hub/scripts/api_routes.py`, `websocket_handler.py`, `main.py`, `curiosity_hint.py`
  - `app/settings.py`
  - `static/js/app.js` (dropped the `agent_step` frame handler and the context-exec `operator_summary` rendering), `static/js/agent-trace.js` (dropped the live-step helpers that only `agent_step` fed)
  - `templates/index.html`
- `services/orion-self-experiments/app/main.py`, `experiment_registry.py`, `settings.py`.
- `scripts/report_dead_env_keys.py`: every retired key is now `KNOWN_DEAD`.
- `scripts/sync_local_env_from_example.py`, `check_settings_defaults.py`, `check_chat_route_poachers.py`, `orion/schema_skew_discovery.py`: their context-exec entries were removed. The schema-skew gate itself flagged its entry as stale.
- `config/metrics/metric_definitions.lock.json`: re-locked.

## Kept because shared

These pieces are kept on purpose because something else still uses them.

| Kept | Caller that keeps it |
| --- | --- |
| `ContextExecPermissionV1` in `orion/schemas/context_exec.py`, plus its registry entry | `HarnessRunRequestV1.permissions` (`orion/schemas/harness_finalize.py`) and `orion/hub/turn_orchestrator.py` on every unified turn. Renaming it is a separate contract migration. |
| The `"ContextExecService"` step-result key | the live bound-capability step (`bound_capability_exec.py`), the autonomy-goal steps in `supervisor.py`, cortex-orch `main.py`'s answer-depth reader, and `orion/normalizers/agent_trace.py`. It is now only a payload key and no longer means a call to context-exec. |
| The `context_exec` / `ContextExecService` taxonomy in `orion/normalizers/agent_trace.py` | same reason as the row above |
| `context_exec_*` fields on `SelfExperimentRecordV1`, `SelfExperimentSpecV1` and `SelfExperimentCreateRequestV1`, plus the SQLite columns | stored rows and Hub list readers validate under `extra="forbid"`. They are annotated as ignored, and removing them is a follow-up. |
| `curiosity_hint._fetch_fresh_candidates` / `usable_candidates` | `endogenous_outreach.py` |
| `.agent-live-trace` CSS | `agent-claude-trace.js` |

## Schema / bus / API changes

**Removed bus channels:**

- `orion:exec:request:ContextExecService`
- `orion:exec:result:ContextExecService`
- `orion:exec:result:ContextExecService:*`
- `orion:self_experiments:context_exec:reply:*`
- `orion:context_exec:event`
- `orion-context-exec` as a producer of `orion:exec:request:RecallService`

**Removed schemas:**

- `ContextExecRequestV1`, `ContextExecRunV1`, `ContextExecOperatorSummaryV1`, `ContextExecSafetySummaryV1`, `ContextExecBudgetV1`, `ContextExecFindingV1`, `ContextExecVerbStepV1`
- `BeliefProvenanceReportV1`, `TraceAutopsyReportV1`, `RepoImpactAnalysisReportV1`, `InvestigationReportV2`, `InvestigationSectionV2`, `EvidenceBundle`, `SourceResult`
- `PatchProposalV1`, `MemoryCorrectionProposalV1`, `ProposalEnvelopeV1`
- `ProposalLedgerRecordV1`, `ProposalTriageDecisionV1`, `ProposalReviewDecisionV1`, `ProposalExecutionEligibilityV1`, `ProposalExecutionReceiptV1`
- `SelfExperimentDispatchRequestV1`, `SelfExperimentDispatchResponseV1`

**Removed APIs:**

- Hub `/api/proposal-review/*`
- self-experiments `POST /v1/experiments/{id}/dispatch` and `/retry`
- the `dispatch_enabled` field in the self-experiments `/health` response

**Behavior changes:**

- cortex-exec mode=agent now fails fast with an `agent_runtime_unavailable` step (keyed `AgentRuntime`) instead of a 60-second wait.
- **mode=council now reaches the agent-council checkpoint after that stub.** This is the code-default path. Before, with `CONTEXT_EXEC_ENABLED=true`, a council turn stopped at the dead context-exec call. Council is not offered in the Hub mode dropdown; only the auto-router at depth 3 or an explicit `council_runtime` verb can reach it.
- Hub WebSocket payloads no longer carry `context_exec_lane`. No JS read it.
- The chat route tag `context_exec_agent` no longer exists.

**Compatibility:** nothing produced on or consumed from the removed channels (0 subscribers, 0 messages captured).

## Env/config changes

**Removed keys:**

- hub: `HUB_PROPOSAL_REVIEW_ENABLED`, `HUB_PROPOSAL_REVIEW_API_URL`, `HUB_PROPOSAL_REVIEW_TIMEOUT_SEC`, `HUB_AGENT_CONTEXT_EXEC_ENABLED`, `HUB_CONTEXT_EXEC_API_URL`, `HUB_CONTEXT_EXEC_TIMEOUT_SEC`, `HUB_CONTEXT_EXEC_EVENT_CHANNEL`, `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED`, `HUB_AGENT_REPL_ENABLED`, `HUB_AGENT_CURIOSITY_HINT_ENABLED`
- cortex-exec: `CHANNEL_CONTEXT_EXEC_INTAKE`, `CHANNEL_CONTEXT_EXEC_REPLY_PREFIX`, `CONTEXT_EXEC_ENABLED`, `CONTEXT_EXEC_TIMEOUT_SEC`, `CONTEXT_EXEC_DEPTH2_DEFAULT`, `CONTEXT_EXEC_LEGACY_FALLBACK` (that last one was compose-only)
- self-experiments: `SELF_EXPERIMENTS_DISPATCH_ENABLED`, `SELF_EXPERIMENTS_CONTEXT_EXEC_DISPATCH_TRANSPORT`, `SELF_EXPERIMENTS_CONTEXT_EXEC_URL`, `SELF_EXPERIMENTS_CONTEXT_EXEC_REQUEST_CHANNEL`, `SELF_EXPERIMENTS_CONTEXT_EXEC_TIMEOUT_SECONDS`, `SELF_EXPERIMENTS_MAX_DISPATCH_ATTEMPTS`

**`.env_example` updated:** yes, for all three services.

**Local `.env` synced:** yes. `sync_local_env_from_example.py` was run from the branch, against the primary checkout's `.env`. It added nothing, because this PR only removes keys.

**Live `.env` lines removed by hand** in the primary checkout. The sync script only adds keys, so these had to be deleted directly. A backup of each file is at `<file>.bak.20261010T225625Z`, which is gitignored.

- `services/orion-hub/.env`, 10 lines: 42–44 (`HUB_PROPOSAL_REVIEW_*`), 361–364 (`HUB_AGENT_CONTEXT_EXEC_ENABLED`, `HUB_CONTEXT_EXEC_API_URL`, `HUB_CONTEXT_EXEC_TIMEOUT_SEC`, `HUB_AGENT_REPL_ENABLED`), 376–377 (`HUB_CONTEXT_EXEC_EVENT_CHANNEL`, `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED`), 395 (`HUB_AGENT_CURIOSITY_HINT_ENABLED`)
- `services/orion-cortex-exec/.env`, lines 95–100 (6 keys)
- `services/orion-self-experiments/.env`, lines 17–22 (6 keys)

**Watch for this until the PR merges:** running main's own sync script from the primary checkout before this merges would put those keys back, because main's `.env_example` still has them. The containers would ignore them, so it is harmless, and `report_dead_env_keys.py --apply` lists them as KNOWN_DEAD.

**Skipped keys needing operator action:** none.

## Tests run

All runs used `/mnt/scripts/Orion-Sapienform/.venv/bin/python` with `PYTHONWARNINGS=ignore`. Baselines came from a clean detached worktree at main `74c105131`.

| Suite | Baseline (main) | Branch | Notes |
| --- | --- | --- | --- |
| cortex-exec, run per file (whole-suite collection collides on "Verb already registered") | 1342 passed / 10 failing | 1346 passed / same 10 failing | 0 new failures; adds `test_agent_runtime_retired.py` (5) |
| cortex-orch, run per file | 250 passed / 3 failing | 245 passed / same 3 failing | 0 new failures (the 2 deleted context-exec router test files account for the drop) |
| hub, `pytest services/orion-hub/tests` | 36 failed / 3379 passed | 33 failed / 3330 passed | 0 new failures. The 3 that dropped out were failing context-exec bridge tests, now deleted. |
| harness-governor | 77 passed | 77 passed | |
| shared: `orion/schemas/tests orion/harness/tests orion/bus/tests orion/hub tests` (`--continue-on-collection-errors`) | 79 failed / 61 errors / 5188 passed | 75 failed / 42 errors / 5160 passed | 0 failures or errors appear only on the branch; the passed count drops because tests for deleted code were removed |
| self-experiments | — | 17 passed | |
| `tests/test_report_dead_env_keys.py`, `tests/test_agent_trace_js.py`, `tests/scripts/test_schema_skew_discovery.py` | — | pass | |

## Evals run

```text
No eval harness covers the retired paths. The context-exec RLM eval itself was deleted with the service.
```

## Docker/build/smoke checks

```text
docker compose config --quiet  (hub, cortex-exec, self-experiments)  -> ok
scripts/check_definition_drift.py --gate      -> PASS (after --update re-lock)
scripts/check_bus_reply_channels.py           -> 19 resolved, 0 uncovered
scripts/check_single_consumer_channels.py     -> OK, 48 channels
scripts/check_chat_route_poachers.py          -> PASS
scripts/check_env_template_parity.py          -> PASS
scripts/check_settings_defaults.py --example-drift orion-hub -> OK
scripts/check_service_hostname_refs.py, check_env_key_single_source.py,
  check_system_health_producers.py, check_scripts_dir_no_stdlib_shadow.py -> OK
```

The live runtime check is UNVERIFIED until the rebuild below. After the rebuild:

```bash
docker exec orion-athena-cortex-exec-background python -c "import app.clients as c; print('ContextExecClient' in dir(c))"   # expect False
docker exec orion-athena-cortex-exec-background env | grep -c CONTEXT_EXEC          # expect 0
timeout 600 redis-cli -u redis://100.92.216.81:6379/0 PSUBSCRIBE 'orion:exec:request:ContextExecService*' 'orion:context_exec:*'   # expect no pmessage lines
redis-cli -u redis://100.92.216.81:6379/0 PUBSUB NUMSUB orion:context_exec:event      # expect 0 once Hub is rebuilt
```

## Review findings fixed

- **Finding:** the drift-guardrail bypass was set whenever an `agent_runtime_unavailable` step existed. A council reply that later replaced the stub text would therefore skip the drift check.
  - **Fix:** the bypass now applies only while the final text is still exactly the stub.
  - **Evidence:** `test_council_answer_does_not_inherit_the_stub_drift_bypass` and `test_council_mode_still_reaches_council_after_the_stub`.
- **Finding:** the stub's new `AgentRuntime` result key was unknown to `agent_trace` and to orch's answer-depth reader, so it showed up as an unknown step.
  - **Fix:** taught both about the key.
  - **Evidence:** `test_agent_trace_renders_agent_runtime_unavailable_as_failed_delegate`.
- **Finding:** `run_answer_depth_live_proof.py` could never pass, since it needed context-exec, planner or agent-chain bus hops.
  - **Fix:** deleted it and its two tests.
- **Finding:** some context-exec dispatch residue was left: the self-experiments conftest env default and a heartbeat test docstring.
  - **Fix:** removed both.
- **Finding:** the kept `requested_context_exec_mode` and `context_exec_*` schema fields are now silently ignored.
  - **Fix:** annotated them as ignored and legacy, with a follow-up to remove them.
- **Finding:** unused imports in the new test.
  - **Fix:** removed.
- **Not changed:** the reviewer also suggested asserting the routing depth in the router test (a nit). It was left as is, because other heuristics legitimately set depth 2 for some of those phrasings. The test asserts on the context-exec tags and reason instead.

## Restart required

Run from the primary checkout once this is merged and pulled onto main:

```bash
cd /mnt/scripts/Orion-Sapienform && docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
cd /mnt/scripts/Orion-Sapienform && docker compose --env-file .env --env-file services/orion-cortex-orch/.env -f services/orion-cortex-orch/docker-compose.yml up -d --build
cd /mnt/scripts/Orion-Sapienform && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
cd /mnt/scripts/Orion-Sapienform && docker compose --env-file .env --env-file services/orion-self-experiments/.env -f services/orion-self-experiments/docker-compose.yml up -d --build
```

Optional operator cleanup. These steps are destructive and need Juniper's approval:

- `docker rmi orion-context-exec-context-exec:latest` removes the stale 262MB image.
- After pulling, the primary checkout still holds the gitignored `services/orion-context-exec/.env` and `__pycache__/` leftovers. Remove them by hand.
- `/mnt/rlm-nvme/context-exec` holds about 1.8MB of old run, ledger and artifact files. Archive or remove them by hand.

## Risks / concerns

- **Medium: council mode behaves differently.** mode=council turns now reach orion-agent-council after the stub, where before they stopped at the dead context-exec call. This is the code-default behavior and is not reachable from the Hub dropdown. Mitigation: a test covers it, and setting the auto-router's depth 3 off avoids it.
- **Low: depth-2 still has no runtime.** The auto-router can still choose depth 2 (for example `output_mode_tool_lane` or engineering heuristics), and those turns now fail fast with an honest message. Follow-up: decide whether the router should stop choosing depth 2 at all.
- **Low: some old names remain.**
  - `ContextExecPermissionV1` and the `"ContextExecService"` result key keep their context-exec names on live paths. Follow-up: a contract migration to rename them.
  - The self-experiments legacy schema fields are kept for rollout compatibility. Follow-up: remove them.
- **Low: database.** There are no Postgres tables to retire, so there is no destructive SQL file.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2603
