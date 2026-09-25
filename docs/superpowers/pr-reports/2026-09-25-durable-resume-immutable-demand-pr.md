# durable-runs: a resumed run registers its accepted demand, and a failing resume can no longer spin forever

Branch: `fix/durable-resume-immutable-demand`

## Summary

- One curiosity run (`54537b5b5ccc`) failed to resume about every 7 seconds for three days: 53,242 `run.checkpoint_resume_failed` events and no terminal reason.
- Cause: the run's saved graph progress holds its original resource request (`alternatives=["agent-burst"]`). The run's database rows had been widened out-of-band to `["agent-burst","chat-burst"]`. After a restart, the run re-submitted the old copy, the store rejected it as a changed request ("run demand is immutable"), and the loop retried with no limit.
- Fix 1: a resumed run now re-submits the request stored in the database row, not the copy in its checkpoint. The store also treats two requests that mean the same thing as the same.
- Fix 2: a resume that keeps failing now gives up. After 10 failures with no real progress, spread over at least 10 minutes, the run is marked `failed` with a readable reason, its lease is released and its demand withdrawn.

## Outcome moved

- The live run's re-submission now matches its stored demand. After deploy it goes back to waiting for capacity instead of failing.
- Any future resume failure ends in a terminal `run.failed` event with the error text in it, instead of looping forever.

## Current architecture

- `AdmissionRuntime.register(state)` sent `state["admission"]`, the checkpoint's snapshot from acceptance, which is never refreshed.
- `PostgresAdmissionStore.register_demand` compared the stored JSON to the new JSON byte-for-byte.
- `_drive` caught every other exception with `logger.exception` plus one event, and the reconcile loop re-drove the run on the next tick with no bound.
- The `resource_wait` re-register on lease expiry sat outside that try block altogether.

## Root cause (live evidence, 2026-09-25)

- `durable_admission_runs.request.admission.alternatives` and `durable_resource_demands.requirement.alternatives` for `54537b5b5ccc` are both `["agent-burst","chat-burst"]`.
- The checkpoint blob for channel `admission` (version 1, written 2026-09-21 13:34), decoded with ormsgpack, is `["agent-burst"]`.
- Every run queued around the chat-burst rollout (2026-09-22 ~02:xx) shows the widened list, while runs completed before it show `["agent-burst"]`.
- No code in the repo rewrites these rows, so the widening was a manual out-of-band edit. Who made it is **UNVERIFIED**.
- The other widened runs drained because they were granted while waiting and never re-registered.
- `54537b5b5ccc` was granted `chat-burst` (lease gen 151, 02:44:42) and then fenced by `worker_recovery` at 02:47:31. The recovery routed it through `retry_wait` -> `resource_request`, which re-registered the stale copy. The first `run.checkpoint_resume_failed` event is at 02:47:32.

## Architecture touched

- orion-durable-runs admission runtime and the shared admission store.
- No bus, channel or schema-registry change. `ResourceEventV1.detail` is a free dict, and the new `checkpoint_id` and `error` keys are additive.

## Files changed

- `orion/durable_admission/store.py`: meaning-based demand comparison, where a stored demand that is invalid under the contract counts as a conflict. New `resume_failures_since_progress()`.
- `services/orion-durable-runs/app/admission_runtime.py`: `register` uses the request row. The resume block, including the lease-expiry re-register, is inside the bounded handler. New `_resume_failed` gives up on the run.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: two new keys.
- `services/orion-durable-runs/README.md`: documents the behaviour and known gaps.
- `services/orion-durable-runs/tests/test_resume_demand_conflict_postgres.py`: 7 new real-Postgres tests.

## Schema / bus / API changes

- Added: `run.checkpoint_resume_failed.detail.checkpoint_id` and `.error`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - A run can now end `failed` with `error` starting `checkpoint_resume_failed:`.
  - Re-registration accepts a stored demand that is equal by meaning.
- Compatibility notes: failure events written before this change carry no `checkpoint_id` and are not counted, so the live run is not failed on the spot at deploy.

## Env/config changes

- Added keys:
  - `DURABLE_RUNS_RESUME_MAX_FAILURES=10`
  - `DURABLE_RUNS_RESUME_MIN_FAILURE_SPAN_SEC=600`
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys were added to `services/orion-durable-runs/.env` in the primary checkout.
- Skipped keys requiring operator action: none.

## Tests run

```text
# old code (origin/main) + new test file: 7 failed. The first reproduces the live
# traceback: SubmissionConflict: run demand is immutable.
ORION_ADMISSION_TEST_DSN=<throwaway postgres:16 on 127.0.0.1:55480> pytest services/orion-durable-runs/tests -q
142 passed
python scripts/check_env_template_parity.py -> PASS (90 services)
```

## Evals run

```text
evals/admission_fairness.py rc=0
evals/elastic_fairness.py   rc=0
evals/gateway_capacity.py   rc=0
```

## Docker/build/smoke checks

```text
Not deployed (by instruction). Live runtime effect UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: a count alone could fail healthy runs during a brief infrastructure squeeze. The loop wakes early on other runs' activity, so 10 failures can arrive within seconds.
  - Fix: a run is abandoned only when both the count limit and `DURABLE_RUNS_RESUME_MIN_FAILURE_SPAN_SEC` (600s) are met. The settings comment was corrected.
  - Evidence: `test_failures_must_span_the_minimum_time_before_the_run_is_failed`.
- Finding: counting per checkpoint could never stop a grant -> fail -> worker_recovery cycle, because each cycle writes a new checkpoint.
  - Fix: the count now runs since the last real node progress. Completions of `resource_request`, `resource_wait` and `retry_wait` do not reset it.
  - Evidence: `test_wait_machinery_rewrites_do_not_reset_the_count_but_real_progress_does`.
- Finding: a projection failure on a finished graph could be relabelled `failed`.
  - Fix: with no `next` node, `_resume_failed` records the event and returns without rewriting.
  - Evidence: `test_projection_failure_on_a_finished_graph_is_never_relabelled_failed`.
- Finding: the meaning-based comparison could raise `ValidationError` instead of `SubmissionConflict`.
  - Fix: it is caught and turned into a conflict.
- Finding: the lease-expiry re-register path moved into the try block but had no test.
  - Fix: `test_lease_expiry_reregistration_failure_is_counted_not_escaped`.
- Finding (mine): the give-up path must still mark the database row terminal if the graph update throws.
  - Fix: the graph update is wrapped in try/except, and `_terminal` always runs.
- Not fixed (documented): the checkpoint's `deadline_at` is not re-synced from the row. It is harmless today because only `alternatives` was ever rewritten; it is noted in the README and a code comment.
- Not fixed (noted): the count query reads every event row for the run on the failure path only. That is acceptable.

## One-off data repair

No SQL is needed. After deploy, `54537b5b5ccc` re-submits its stored demand, which matches the stored demand row. It resumes into `resource_wait` and waits for capacity like any queued run.

If the orchestrator prefers to drop this four-day-old run instead of running it:

```bash
curl -fsS -X POST http://localhost:8124/runs/54537b5b5ccc/cancel
```

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build   # from a worktree on merged main
```

## Risks / concerns

- Severity: low.
  - Concern: a run waiting days for capacity can accumulate 10 scattered infrastructure failures, more than 10 minutes apart, and be failed.
  - Mitigation: raise `DURABLE_RUNS_RESUME_MAX_FAILURES`. The reason is visible in `run.failed`.
- Severity: low.
  - Concern: the finished-graph projection, cancel, deadline and worker_recovery blocks still run outside this bound. This is pre-existing.
  - Mitigation: documented in the README.
- Severity: note.
  - Concern: who edited the rows out-of-band is UNVERIFIED.
  - Mitigation: this fix makes the service resilient to that edit.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
