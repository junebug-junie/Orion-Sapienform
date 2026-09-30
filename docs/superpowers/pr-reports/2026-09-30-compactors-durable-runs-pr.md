# fix(compactors): run the LLM digest as an admitted `compactor.digest` durable run

## Summary

- The two daily compactors (GitHub merged-PR digest at 06:10, Hub chat digest at 06:00) no longer make their LLM calls inside cortex-orch. Each chunk digest and the merge call now runs as one step of a durable run in orion-durable-runs that holds a GPU pool hold. When the pool is busy at 06:00, the run waits for its turn instead of failing.
- cortex-orch still does the fetching and chunking, then submits the run and answers the scheduler right away with "accepted". When the run finishes its LLM calls it calls cortex-orch back, and orch writes the memory card and journal entry exactly as before.
- Re-submitting the same day finds the existing run instead of starting a second one, because the run id is built from the workflow, the day, the repo, and a hash of the input. If the earlier run failed, the next attempt gets a fresh generation id (`:g2`, `:g3`, ...).
- Removed: the in-process "try the `agent` route twice" retry, the 50-minute whole-pass budget, and the 3600 s scheduler wait. The scheduler's wait for compactors is now 600 s, which only has to cover the fetch.
- orion-actions now listens to durable run state and marks the scheduled compactor run completed or failed when the durable run actually ends. It no longer blocks its one-at-a-time scheduler loop for up to an hour.

Merge order: #2422 (full-window compactors) is already merged into main, so this PR targets main directly. It overlaps with #2419 (world-pulse journal as a `journal.compose` durable run), covered under Risks.

## Outcome moved

- **Failure mode removed:** at 06:00 the digest calls took one-inference gateway leases and failed with `gpu_pool_unavailable:deadline` while the pool was busy. Now those calls ride the run's own hold. Waiting for the hold is never counted as an attempt (`admission_runtime.py` `register`, pool-refusal handling around L258-267).
- **Restart safety:** each finished chunk digest is saved before the next call starts, so a durable-runs restart mid-day resumes at the first chunk not yet digested. A failed finalize (card + journal write) is retried without redoing any LLM call.
- **Scheduler:** a compactor dispatch no longer holds the scheduler loop for up to 3600 s. Schedule health stays accurate: the schedule run stays "in flight" until the durable run's terminal state row settles it through the normal success and failure paths (retry budget, attention signal, next occurrence).

## Current architecture

Before this patch (PR #2422): orion-actions called cortex-orch synchronously and waited up to 3600 s. Inside that wait, `_execute_*_compactor_pass` fetched the day, split it into chunks, and called `_run_compactor_digest`, which ran every chunk digest plus the merge in-process through `call_verb_runtime`, trying each call over `DIGEST_LLM_ROUTES=("agent","agent")` within a 3000 s pass budget. Each call took its own one-inference lease through llm-gateway. When the pool was busy those leases were refused and the whole day failed.

## Architecture touched

- **New durable workflow `compactor.digest`** in orion-durable-runs: `resource_request -> resource_wait -> digest (loops once per LLM call) -> finalize -> finish`. `WORK_NODES["compactor.digest"] = {"digest"}`.
- **Shared step machine** `orion/cognition/compactor/map_reduce.py`: which call comes next, merge fallbacks, and assembling the final digest. It is pure code, moved out of orch so the durable graph can drive it.
- **cortex-orch** `workflow_runtime.py`: each pass is now split into three parts. Prepare (fetch, chunk, build a small finalize context), submit (`dispatch_durable_run`, which returns `accepted`), and finalize (runs when `workflow_request.durable_digest` is present). A quiet day finalizes inline with no durable run. `execute_chat_workflow` skips the notification when the result is `accepted`; the finalize call sends it later with the real result.
- **orion-actions:** `accepted_durable_run` / `mark_awaiting_durable` / `settle_durable_run`. The reaper waits for the durable run's own deadline plus 15 minutes instead of the 300 s claim TTL. The service subscribes to `orion:durable:run:state`.

## Files changed

- `orion/schemas/compactor_digest_run.py`: new contract. `CompactorDigestRunBriefV1` (the chunk inputs plus an opaque finalize context; reserved `chat`/`harness` routes are refused) and `CompactorDigestResultV1` (what the run sends back to orch).
- `orion/schemas/durable_run.py`: adds `compactor.digest` to `DurableWorkflowV1` and the brief union; admission is required.
- `orion/schemas/registry.py`: registers both new models.
- `orion/cognition/compactor/map_reduce.py` (new) + `constants.py`: the step machine, the request builder (carries `options.gpu_lease`), and the result parser. `DIGEST_LLM_ROUTES` is replaced by `DIGEST_LLM_ROUTE`. `COMPACTOR_DIGEST_TOTAL_BUDGET_SEC` and `DIGEST_MIN_CALL_SEC` are removed. Added: deadline, finalize timeout, and generation cap.
- `services/orion-durable-runs/app/compactor_digest_graph.py` (new), `admission_runtime.py` (graph, WORK_NODES, finish detail), `runner.py` (`_cortex_orch_rpc`).
- `services/orion-cortex-orch/app/workflow_runtime.py`: prepare / submit / finalize; the in-process digest machinery is deleted.
- `services/orion-actions/app/main.py`, `workflow_schedule_store.py`, `settings.py`, `.env_example`: settle schedule runs from durable state; wait lowered from 3600 to 600.
- `orion/bus/channels.yaml`: orion-durable-runs is now listed as a producer on `orion:cortex:request` (reflect already sent there without being declared) and a consumer of `orion:cortex:result*`; orion-actions is a consumer of `orion:durable:run:state`.
- `config/metrics/metric_definitions.lock.json`: re-locked for those three routing changes.
- `.github/workflows/orion-durable-runs-tests.yml`: now triggers on the new schema file and `orion/cognition/compactor/**`.
- Tests: `services/orion-durable-runs/tests/test_compactor_digest_graph.py`, `orion/cognition/compactor/tests/test_map_reduce.py`, `services/orion-cortex-orch/tests/test_compactor_durable_submit.py`, `services/orion-cortex-orch/tests/compactor_durable_sim.py` + a conftest fixture, `services/orion-actions/tests/test_workflow_durable_settle.py`, plus updates to `test_compactor_full_window.py`, `test_workflow_lane.py`, `test_workflow_dispatch_timeout.py`.
- READMEs: orion-durable-runs, cortex-orch, orion-actions, `orion/cognition/compactor/README.md`.

## Schema / bus / API changes

- Added: `DurableWorkflowV1` value `compactor.digest`; `CompactorDigestRunBriefV1`, `CompactorDigestResultV1`; `workflow_request.durable_digest` (the finalize call into cortex-orch); `metadata.workflow.durable_run` on a compactor dispatch reply (`run_id`, `status`, `generation`, `deadline_at`, `chunk_count`); result metadata `digest_gpu_roles` and `durable_run_id`.
- Removed: nothing on the wire.
- Renamed: none.
- Behavior changed: a compactor dispatch replies `status="accepted"` (not yet finished) unless the day is quiet or was already finalized. A failed LLM phase no longer triggers orch's per-run failure notification; it reaches Juniper through the schedule attention signal once the retry budget is spent (see Risks).
- Compatibility notes: these are additions to `extra="forbid"` / `Literal` contracts, so the deploy order below matters.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Value / meaning changed: `ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS` 3600 -> 600 (it now covers only the fetch and receipts).
- `.env_example` updated: `services/orion-actions/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran it. The sync does not overwrite existing values, so `services/orion-actions/.env` was set to 600 by hand; verified at line 168.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs tests + orion/cognition/compactor: 233 passed, 68 skipped (Postgres-backed tests skip locally; CI runs them)
services/orion-cortex-orch:  pytest tests -> no new failures vs origin/main baseline (34 pre-existing in-suite failures on main, unchanged)
services/orion-actions: 210 passed (origin/main baseline also clean)
orion/cognition/compactor + github_compactor + chat_history_compactor + tests/test_check_chat_route_poachers.py + memory-card tests -> passed
Static gates (derived from .github/workflows/orion-static-gates.yml): env-sync tests, schema registry, schema skew, grammar catalog, metric lineage, definition drift (after re-lock), inner-state, hostname refs, journal dispatch registry, chat route poachers, async routes, system_health producers, stdlib shadow -> all PASS
```

## Evals run

```text
services/orion-cortex-orch/evals -> 5 passed (chat compactor digest eval, unchanged)
services/orion-durable-runs/evals/hold_fairness.py, deploy_order_skew.py -> need Postgres; run in CI
```

The compactor path has no dedicated eval. The durable graph tests replay the 06:00 failure mode (pool busy -> checkpointed wait, zero attempts spent) and a 40-chunk heavy day.

## Docker/build/smoke checks

```text
Not run: this task said do not deploy. No compose or Dockerfile changes (durable-runs already copies orion/ into its image).
```

## Review findings fixed

A code-review subagent reviewed commit a12a805c4 and found no blockers and six should-fix items. All six are fixed in the follow-up commit, along with three of the nits.

- Finding: after the deadline, a re-dispatch raised `compactor_window_deadline_passed` before submitting, so a completed run whose terminal row orion-actions missed could never be found. That broke the recovery path.
  - Fix: `_submit_compactor_digest_run` now always submits the identical request. It refuses only when the receipt shows a fresh row. The deadline is derived from the window and never raises on its own.
  - Evidence: `test_past_deadline_redispatch_still_finds_a_completed_run_but_refuses_a_fresh_one`.
- Finding: a terminal row that arrived before the scheduler recorded the accepted reply was lost, leaving the schedule run silent for about 18h.
  - Fix: the store keeps a bounded map of terminal rows nobody was waiting on. `mark_awaiting_durable` settles from it at once.
  - Evidence: `test_terminal_row_before_the_accepted_reply_settles_on_mark`.
- Finding: a failed finalize sent a failure notice on every driver retry (up to 10).
  - Fix: when the request carries `durable_digest`, `execute_chat_workflow` re-raises without notifying. The terminal failure is reported once, by orion-actions.
  - Evidence: `test_failed_finalize_does_not_notify_per_retry`.
- Finding: `notify_on=failure` no longer fired when the digest itself failed.
  - Fix: orion-actions sends one `orion.workflow.failed` notice per failed/cancelled durable run, deduped on the run id and following the schedule's `notify_on`. This is `durable_failure_notify_request` and `_report_durable_settlements`.
  - Evidence: `test_failure_notifies_once_per_durable_run_per_policy`.
- Finding: the orch test simulator used a hand-copied finalize request that had already drifted from the real one.
  - Fix: `finalize_request_payload` moved to `orion/cognition/compactor/map_reduce.py`; the durable graph and the simulator both import it.
  - Evidence: orch suite shows no new failures against the origin/main baseline.
  - Not done: a front-door test that sends a digest payload through orch's `main.py` handler.
- Finding: the deploy order left out sql-writer and Hub, which validate `DurableRunStateV1.workflow` as a Literal.
  - Fix: the Restart section, the contract docstring, and the durable-runs README now list them first.
- Nit 8 (finalize ignored pause/cancel): finalize now calls `admission.guard`. A pause or cancel stops it before the card and journal are written. A passed deadline does not, so a finished day's digest is not discarded (also addresses nit 13). Evidence: `test_finalize_honours_pause_but_not_a_passed_deadline`.
- Nit 9: `_cancel_harness` returns early for `compactor.digest`, which has no harness turn.
- Nit 12 (an already-finalized re-dispatch notified success twice): `execute_chat_workflow` skips the notice whenever the result carries `metadata.workflow.durable_run`. Evidence: `test_already_finalized_redispatch_does_not_notify_twice`.
- Not fixed, documented under Risks: nit 7 (no backoff between failed digest attempts), nit 10 (the execute timeout and the RPC timeout are equal), nit 11 (a PR edited mid-flight can start a second run for the same day).

## Restart required

Deploy in this order. The new workflow value and brief are additions to `extra="forbid"`/`Literal` models, so each producer must go out after its consumers:

```bash
# 1. state-row consumers (an old sql-writer drops compactor.digest rows; an old Hub ignores them)
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# 2. the runner (must accept compactor.digest before anything submits one)
scripts/safe_docker_build.sh orion-durable-runs up -d --build
# 3. the submitter + finalizer
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
# 4. last: the scheduler (its 600 s wait is too short for an OLD in-process orch)
scripts/safe_docker_build.sh orion-actions up -d --build
```

No migration. `CORTEX_DURABLE_ADMISSION_ENABLED=true` must stay set in cortex-orch (it already is locally).

## Risks / concerns

- Severity: medium. Concern: overlap with #2419 (`journal.compose` durable workflow, branch `fix/world-pulse-journal-retry`, not yet pushed with the durable version when this was written). Both add a `DurableWorkflowV1` value, a brief union member, a validator clause, admission_runtime graph/WORK_NODES/finish-detail entries, registry entries, and a runner method. Mitigation: the conflicts are purely additive and adjacent; resolve at merge by keeping both. There is no shared "LLM call with lease" helper to reuse yet: `journal.compose` is a single call that calls cortex with a verb, while this workflow is a multi-call loop. Both use the same pattern (`options.gpu_lease` on a cortex-orch RPC under `AdmissionRuntime.execute`).
- Severity: low. Concern: when the LLM phase fails, the failure notice now comes from orion-actions (one per durable run, deduped on its id) instead of from orch inside the RPC. Fetch failures still notify from orch as before.
- Severity: low. Concern: a failed digest call goes straight back to wait for the hold, with no backoff (review nit 7). An orch that fails fast can burn all 3 attempts in seconds; the next generation then re-digests from scratch. Mitigation: bounded by `COMPACTOR_MAX_RUN_GENERATIONS` and the scheduler's own backoff. Follow-up: a `retry_wait` interrupt like curiosity's `harness_turn`.
- Severity: low. Concern: the execute timeout and the RPC timeout are both `brief.timeout_sec`=660 s (nit 10). If execute's timeout wins, an in-flight call can outlive the released hold. The verb's own 600 s timeout makes this unlikely.
- Severity: low. Concern: if a PR in the window is edited between a timed-out dispatch and its retry, the input hash changes and a second run starts for the same day (nit 11). Card and journal writes are upserts; the cost is duplicated GPU work.
- Severity: low. Concern: if orion-actions is down when the terminal state row is published, the schedule run is failed as `durable_run_completion_unobserved` after deadline + 15 min. Mitigation: the retry re-dispatches the same window, finds the completed run by its id, and reports it as already finalized without re-running anything.
- Severity: low. Concern: a window whose deadline (window end + 24 h) has already passed is refused (`compactor_window_deadline_passed`) rather than admitted only to fail at once. So a scheduled day missed for more than about 18 h is recorded as failed, not silently backfilled.

## Live verification (after deploy)

1. `orion:durable:run:state` / `substrate_durable_run_state`: a `compactor.digest` run for yesterday reaches `completed` with a `detail.journal_entry_id`.
2. `durable_resource_events` for that run: `run.waiting_resource` while the pool is busy at 06:00, then `run.resource_granted`. No `gpu_pool_unavailable:deadline` failures on the compactors.
3. The journal entry for the day exists with its full body (`journal_body_chars` in the finalize result matches the stored body length).
4. orion-actions log `scheduled_workflow_durable_settled ... status=completed`, and the schedule's `last_result_status=completed`.
5. The scheduler loop is not blocked: other schedules due between 06:00 and 06:15 dispatch on time.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2431

🤖 Generated with [Claude Code](https://claude.com/claude-code)
