## Summary

- The daily world-news journal (and its email) is now written by a **durable run that holds a GPU slot from the GPU pool**, instead of one direct call from orion-actions. Before, a busy fast GPU lane at 06:00 local (`gpu_pool_unavailable:deadline`) failed the write, nothing retried it, and the email stopped after 2026-09-24.
- New generic durable workflow `journal.compose`, reusable for the upcoming daily letter. It asks the pool for a hold, waits for it (the wait is saved and never counts as an attempt), composes with the existing `journal.compose` cortex verb attached to the hold, then publishes the journal write itself.
- orion-actions only *submits* the run, through cortex-orch's existing durable entry point. The run id is `world-pulse-journal:<world-pulse run_id>` and the entry id and correlation id are derived from it, so a redelivered run result resubmits the identical request and durable-runs answers with the existing run. The run has a deadline of the next local midnight, so it can never use up tomorrow's one world-pulse email.
- The previous design is removed entirely: the in-process retry queue (`pending_journals.json`), the scheduler drain, the two `ACTIONS_WORLD_PULSE_JOURNAL_RETRY_*` env keys and the env-sync prefix. No retry state lives outside durable runs.

## Outcome moved

Failure mode: a congested GPU pool at 06:00 permanently lost that day's world-news journal. Now the pool's own queue makes the run wait until 00:00 local, and a restart resumes it from its checkpoint.

## Current architecture

`orion:world_pulse:run:result` -> orion-actions `_dispatch_journal` -> `_run_journal` (one direct cortex RPC on `quick_background`) -> `journal.entry.write.v1` -> sql-writer -> `journal.created` -> orion-actions post-persist email (one per day for `world_pulse_digest`). A compose failure was logged and dropped.

## Architecture touched

- **Contract:** `orion/schemas/journal_compose_run.py` is new. `durable_run.py` gets the Literal entry, the brief union member and a validator that makes admission required.
- **orion-durable-runs:** new admitted graph `resource_request -> resource_wait -> compose -> (retry_wait) -> publish -> finish`, registered in `AdmissionRuntime` (graphs, finish detail, `WORK_NODES={"compose"}`). The runner gains `_compose_journal`, which sends the verb with `options.gpu_lease`.
- **orion-actions:** the world-pulse handler submits the durable run. `_run_journal`/`_dispatch_journal` no longer carry the world-pulse-only argument. Other triggers are unchanged.
- **orion/journaler:** the curiosity-section merge is split into `world_pulse_curiosity_appendix` and `append_unless_present`, so the brief carries plain text instead of the whole world-pulse result.

## Files changed

- `orion/schemas/journal_compose_run.py`: new contract (`JournalComposeRunBriefV1`, `JOURNAL_COMPOSE_WORKFLOW`).
- `orion/schemas/durable_run.py`: workflow Literal, brief union, admission-required validator, deploy-order note.
- `orion/schemas/registry.py`: registers `JournalComposeRunBriefV1`.
- `orion/journaler/worker.py`, `orion/journaler/__init__.py`: appendix helpers. The existing merge function now uses them, with the same behavior.
- `services/orion-durable-runs/app/journal_compose_graph.py`: the new graph.
- `services/orion-durable-runs/app/admission_runtime.py`: registers the workflow.
- `services/orion-durable-runs/app/runner.py`: `_compose_journal`, `_publish_journal_write`.
- `services/orion-durable-runs/tests/test_journal_compose_graph.py`: graph, contract and runner tests.
- `services/orion-durable-runs/README.md`: workflow section and deploy order.
- `services/orion-actions/app/world_pulse_journal.py`: request builder, cortex durable submit with a receipt check, handler.
- `services/orion-actions/app/main.py`: wiring; in-process world-pulse compose removed.
- `services/orion-actions/tests/test_world_pulse_reflective_journal_handler.py`, `test_handle_envelope_world_pulse_journal.py`: submission tests.
- `services/orion-actions/README.md`: describes the new path.

## Schema / bus / API changes

- **Added:** `DurableWorkflowV1` value `journal.compose`; `JournalComposeRunBriefV1` in the `DurableRunRequestV1.brief` union. A `journal.compose` request must carry `admission`.
- **Removed:** none on the bus.
- **Renamed:** none.
- **Behavior changed:** `world_pulse_digest` journal writes are now published by orion-durable-runs (same channel `orion:journal:write`, same `trigger_kind`, fixed entry id). New durable state rows use `workflow=journal.compose`.
- **Compatibility / deploy order:** these are additive fields on `extra="forbid"`/Literal models.
  1. Deploy **orion-durable-runs** first.
  2. Then **orion-cortex-orch** (it validates the request) and **orion-sql-writer** (it validates `DurableRunStateV1`).
  3. Then **orion-actions**.

  If orion-actions goes first, cortex-orch rejects the submit. That is audited as `durable_submit_failed`, and the day's journal is lost until the others are deployed.

## Env/config changes

- Added keys: none.
- Removed keys: none relative to main. The earlier revision's `ACTIONS_WORLD_PULSE_JOURNAL_RETRY_*` keys were never merged. I removed them from the local `services/orion-actions/.env` that my earlier sync had added them to.
- `.env_example` updated: no.
- Local `.env` synced: nothing to sync. I checked that the two stale keys are gone from `/mnt/scripts/Orion-Sapienform/services/orion-actions/.env`.
- Skipped keys: none.
- Route: the hold's route is `ACTIONS_JOURNAL_LLM_ROUTE` (live: `quick_background`). In `gpu_pool.yaml` that is class `fast` with `on_unavailable: wait`, background priority. The chat-route-poacher gate passes.

## Tests run

```text
pytest services/orion-durable-runs/tests                -> 205 passed, 68 skipped (skips = Postgres-backed, no DB here)
pytest services/orion-durable-runs/tests/test_journal_compose_graph.py -> 11 passed
pytest services/orion-actions/tests                     -> 210 passed
pytest services/orion-cortex-exec durable/lease/reflect tests -> 29 passed
pytest services/orion-cortex-orch -k durable            -> 14 passed
pytest services/orion-hub durable_run tests             -> 245 passed
pytest orion/world_pulse_read/tests/test_durable.py tests/test_curiosity_urgent_schema.py orion/gpu_pool/tests/test_client_attach.py tests/test_agent_trace_schema_registry.py -> 52 passed
pytest tests/test_journaler_worker.py tests/test_journal_e2e.py tests/test_world_pulse_reflective_journal.py -> 30 passed, 1 failed
   (test_draft_from_cortex_result_raises_structured_parse_error fails identically on main: pre-existing)
Static gates: metric lineage, definition drift, env parity, journal dispatch registry, chat route poachers,
async routes, bus reply channels, hostname refs, sentience instruments, system health producers,
inner state registry -> all PASS
```

What the tests cover:
- **Waiting and restart:** a busy pool is a saved wait and costs no attempt, including across a restart. A lost hold is not an attempt.
- **Failed composes:** a failed compose backs off in `retry_wait` (30 s times 2^n) and then succeeds. After the minimum number of attempts it fails without publishing.
- **Publish:** a publish failure resumes at publish without recomposing, and sends a byte-identical write.
- **Runner:** it sends `journal.compose` with `gpu_lease` and keeps the route. It raises on a failed or empty reply. It appends the pre-rendered curiosity section only when the body lacks it.
- **Contract:** admission is required, and brief and workflow must agree.
- **orion-actions submission:**
  - The request is deterministic: two deliveries at different times give requests equal on everything durable-runs compares.
  - The deadline is midnight in Denver.
  - Submit is retried, then audited as failed.
  - Ineligible results are not submitted.
  - The cortex receipt is checked for the right run id and workflow.
  - The in-process world-pulse compose is gone.

## Evals run

```text
None: this is deterministic orchestration, covered by gate tests. The Postgres-backed durable-runs tests skip here and run in CI (orion-durable-runs-tests.yml).
```

## Docker/build/smoke checks

```text
Not run: deploy/restart was out of scope. UNVERIFIED live.
```

## Review findings fixed

- **Finding (MUST):** the brief carried the whole world-pulse result (extra="forbid" nested models), and durable-runs validated it again only *after* the GPU call. A schema skew would burn every attempt and lose the day. It was also bulky.
  - Fix: the brief carries only the pre-rendered curiosity section plus "already present" markers (`body_appendix`, `body_appendix_markers`). No foreign schema is validated in the runner.
  - Evidence: `test_curiosity_followups_travel_as_prerendered_text`, `test_runner_appends_the_prerendered_appendix_unless_already_in_the_body`.
- **Finding (SHOULD):** three compose attempts with no backoff could burn the day in seconds.
  - Fix: a `retry_wait` backoff node (resumed by the driver's existing `retry_at` gate) and at least 6 attempts. The deadline stays the real bound.
  - Evidence: `test_compose_failure_backs_off_in_retry_wait_then_succeeds`, `test_compose_fails_after_min_attempts_without_publishing`.
- **Finding (SHOULD):** the deploy order missed orion-sql-writer, which validates `DurableRunStateV1`.
  - Fix: added to the schema comment, the READMEs and this report.
- **Finding (SHOULD):** the replay-safety explanation was wrong. sql-writer's journal table is insert-only, not an upsert.
  - Fix: docstrings and README reworded. The duplicate is dropped and `journal.created` is not re-emitted.
- **Finding (SHOULD):** the redelivery test compared only ids.
  - Fix: it now asserts full equality of what the store compares.
  - Evidence: `test_redelivery_resubmits_an_identical_request`.
- **Finding (SHOULD):** there was no test of the curiosity merge in the runner. Fixed (see the MUST evidence).
- **Finding (SHOULD):** the PR report was missing. This file restores it.
- **Finding (SHOULD):** a failed submit is lost.
  - Fix: submit tries now span about 6.5 minutes (0/10/30/90/270 s). The residual loss is documented in the docstring and README and audited at ERROR.
- **NITs fixed:**
  - `build_write` reuses `orion.journaler.build_write_payload`.
  - Removed the unused `JOURNAL_COMPOSE_NODES`.
  - `pytest.raises` replaces a bare try/except.
  - Building the request is guarded, so a failure still writes an audit row.
- **NITs not fixed:**
  - Source-inspection tests are kept as cheap registration guards.
  - There is no cancel check between compose and publish (same as the reading graph).
  - A settings change between two deliveries of the same run makes the store answer with a conflict, which is audited as a failed submit. The run itself still exists.

## Restart required

Do not deploy yet. When approved, run from the worktree, in this order:

```bash
cd /mnt/scripts/Orion-Sapienform-world-pulse-journal-retry
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
```

## Risks / concerns

- **Medium, UNVERIFIED live:** no `journal.compose` run has executed. After deploy, the next 06:00 run should show:
  - an orion-actions audit row with `status=submitted` for `world-pulse-journal:<id>`;
  - durable events `run.waiting_resource`, `run.resource_granted`, `run.started`;
  - a `journal_compose_drafted` log line in orion-durable-runs;
  - one `journal_entry_index` row with `trigger_kind=world_pulse_digest` for that run's `source_ref`;
  - one email.
- **Low:** a submit failure that lasts longer than about 6.5 minutes loses that day's journal. It is audited, not silent.
- **Low:** the run's `created_at` is the compose time, not the world-pulse time. Harmless.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2419

🤖 Generated with [Claude Code](https://claude.com/claude-code)
