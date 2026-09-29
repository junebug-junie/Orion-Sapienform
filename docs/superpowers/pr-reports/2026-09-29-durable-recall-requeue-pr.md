# fix(gpu-pool, durable-runs): a recall past its grace re-queues the hold, never fails the run

## Summary

- When the GPU pool takes a seat back from a durable run (gpu2's one-hour seat limit, Juniper's chat
  reclaiming lent gpu0, an unlend, a drain) and the run is still mid-step after the 10-minute grace,
  the run now stops that step, waits for the same hold to come back, and replays the step. It no
  longer burns a retry, and it no longer fails.
- Pool side (`orion/gpu_pool/lease_graph.py`): an aborted retryable **hold** goes straight back in
  line in its original place without spending one of the pool's three attempts -- the same path the
  urgent-preempt feature (#2385) already used. Before, the third recall dead-lettered the hold, which
  the pool reports as `unavailable:recall_grace_exceeded`. Request leases and lost heartbeats are
  unchanged (they still spend a pool attempt).
- Durable-runs side: every admitted graph (curiosity, self-sense, reflect; reading and reverie were
  already right) treats `HoldLost` like `HoldPreempted`: release with `keep_requeued`, wait, replay,
  attempt untouched. `_beat` now classifies the pool's answer (`_taken_back`): re-queued or already
  re-granted at a newer generation -> wait for the same hold; ended with a non-terminal reason -> ask
  afresh; `deadline` -> `WorkflowDeadline`; other run-terminal refusal -> normal failure path.
- A turn that fails on its own before the heartbeat sees the take-back (its next LLM call cannot
  attach) is checked against the pool once, for any take-back, not only urgent
  (`AdmissionDeps.preempted` -> `requeued`, `replay_if_preempted` -> `replay_if_requeued`).
- `release(keep_requeued=True)` also keeps a hold the pool already re-granted at a newer generation,
  but only on a take-back (it used to end it and lose the grant).
- New bound so a step that never fits cannot replay forever: `hold_takebacks` counts take-backs
  per run; past `DURABLE_RUNS_HOLD_MAX_TAKEBACKS` (default 12, 0 = unbounded) the run fails with
  `hold_takeback_limit:<n>` and hands the seat back.

## Outcome moved

Live, 2026-09-26..28 (`durable_resource_events`, `event='run.failed'`): 19 of 39 failed runs were
the pool taking a seat back, not the run failing:

| live error | count | cause |
|---|---|---|
| `HoldLost: gpu_hold_lost:queued:recall_grace_exceeded` | 10 | recall past grace, re-queued by the pool; curiosity spent 1 of 3 attempts per recall, self-sense failed on the first |
| `HoldLost: gpu_hold_lost:queued` | 7 | same, answered by the pre-#2385 pool (its `queued` reply had no reason) -- confirmed in `gpu_pool_events`: `recalled max_hold/owner_waiting` -> `aborted recall_grace_exceeded` -> `queued retry_due` |
| `HoldLost: gpu_hold_lost:unavailable:recall_grace_exceeded` | 2 | third abort: the pool had spent its own 3 attempts and dead-lettered the hold |

All three now leave the run waiting and then completing (regression tests below). Over 7 days the
pool recalled durable holds 113 times (`agent-gpu2`: 26 `max_hold` + 4 `draining`, 24 aborted;
`chat`: 82 `owner_waiting` + 1 `card_unlent`, 43 aborted) -- chat reclaim is the larger source, not
only gpu2.

## Current architecture

Pool: recall -> `recalling` for `hold_clawback_grace_sec` (600 s) -> scheduler `Abort` ->
lease graph: urgent_preempt re-queued in place (U2), every other abort -> `retry_wait`, attempt+1,
dead_letter at `retry.max_attempts` (3), reported as `unavailable`. Durable-runs: `execute` beats
the hold every few seconds; a non-granted answer raised `HoldLost`; only `HoldPreempted` (urgent) was
exempt from the attempt budget; self-sense's catch-all failed the run outright.

## Architecture touched

- `orion/gpu_pool/lease_graph.py` (pool lease state machine), `scheduler.py` (docstring U2).
- `services/orion-durable-runs/app/`: `admission_runtime.py` (`_beat`, `_taken_back`, `_settled`,
  `_requeued_reply`, `requeued`, `release`, `_beat_outreach`), `admitted_graph.py` (`HoldLost`
  semantics + `release_reason`, `AdmissionDeps.requeued`, `replay_if_requeued`, curiosity
  `harness_turn`), `admitted_self_sense_graph.py`, `admitted_reflect_graph.py`, `reading_graph.py`,
  `reverie_visual_graph.py`.

## Files changed

- `orion/gpu_pool/lease_graph.py`: abort of a retryable hold re-queues in place, no attempt.
- `orion/gpu_pool/scheduler.py`: U2 docstring.
- `orion/gpu_pool/tests/test_lease_graph.py`: 4-cycle recall/abort per recall reason never dead-letters; request-lease abort and hold heartbeat loss still spend an attempt.
- `services/orion-gpu-pool/tests/test_holds_and_actuation.py`: max_hold recall on the loaded gpu2 seat re-queues then re-grants on `agent` (gen 2, attempt 1); three owner recalls never dead-letter; existing recall test asserts no `retried`/`dead_lettered`.
- `services/orion-durable-runs/app/*`: as above.
- `services/orion-durable-runs/tests/test_recall_requeue_postgres.py` (new, real Postgres + real in-process pool): the three live failure reasons.
- `services/orion-durable-runs/tests/test_urgent_preemption.py`, `test_admission_runtime_postgres.py`: updated for the decision change (a lost hold no longer spends an attempt / backs off).
- `services/orion-durable-runs/README.md`, `services/orion-gpu-pool/README.md`: behaviour documented.

## Schema / bus / API changes

- Added: none. Removed: none.
- Renamed: `AdmissionDeps.preempted` -> `AdmissionDeps.requeued` (now returns the release reason or
  None), `replay_if_preempted` -> `replay_if_requeued` (in-service Python only; no callers outside
  `services/orion-durable-runs`).
- Behavior changed: pool `status`/`heartbeat` for a hold aborted after a recall now answers `queued`
  (reason `recall_grace_exceeded`) immediately instead of passing through `retry_wait`; the lease
  keeps `attempt`. The pool no longer emits `retried`/`dead_lettered` for such aborts, only `aborted`
  (same as the urgent path). Durable-runs records `resource.lease_expired` (reason
  `recall_grace_exceeded`) per take-back, and `last_error` carries the `HoldLost` text while waiting.
- Compatibility: either side can deploy first. New durable-runs + old pool: the old pool's
  `unavailable:recall_grace_exceeded` is non-terminal -> the run asks afresh (tested). New pool + old
  durable-runs: recalls stop dead-lettering, but old durable-runs still spends attempts.

## Env/config changes

- Added keys: `DURABLE_RUNS_HOLD_MAX_TAKEBACKS` (default 12; 0 = unbounded) in
  `services/orion-durable-runs/.env_example`, `settings.py` (`hold_max_takebacks`),
  `docker-compose.yml`, README.
- Removed / renamed keys: none.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
  (`orion-durable-runs: +DURABLE_RUNS_HOLD_MAX_TAKEBACKS='12'`). Skipped keys: none.
- `scripts/check_env_template_parity.py`: PASS.
- `config/gpu_pool.yaml` unchanged (`max_hold_sec: 3600` left for Juniper -- see below).

## Tests run

```text
services/orion-durable-runs: pytest tests (ORION_ADMISSION_TEST_DSN on throwaway postgres:16 :55491)  252 passed
orion/gpu_pool/tests                                                                                246 passed
services/orion-gpu-pool: pytest tests (GPU_POOL_TEST_POSTGRES_URI on the same throwaway)             97 passed
scripts/check_gpu_pool_config.py                                                                     ok
test_recall_requeue_postgres.py x60 sequential loops (before the limit test was added)               60/60 green
(an earlier ~4% flake was the test's own clock stepping expiring the chat owner's 30 s lease;
 fixed by stepping 20 s. Parallel runs of these Postgres files collide on database-wide advisory
 locks -- existing suite too -- so run them one at a time.)
```

Fail-on-main check (new tests copied onto an `origin/main` export):

```text
test_recall_requeue_postgres.py   4 failed  -- self-sense run.failed with the live strings
  'HoldLost: gpu_hold_lost:queued', 'HoldLost: gpu_hold_lost:unavailable:recall_grace_exceeded',
  and 'HoldLost: gpu_hold_lost:queued:retry_due' (the recall_grace_exceeded case: on main the pool's
  5 s retry_wait had already re-queued under the fake clock); curiosity spent attempt 1 on recall 1.
test_lease_graph.py               4 failed (every recall reason dead-letters by cycle 3)
test_holds_and_actuation.py       3 failed (max_hold -> retry_wait attempt 2; 3 owner recalls dead-letter)
```

## Evals run

```text
services/orion-durable-runs/evals/hold_fairness.py        failed_checks: []
services/orion-durable-runs/evals/deploy_order_skew.py    failed_checks: []
services/orion-durable-runs/evals/gateway_capacity.py     ran clean
services/orion-gpu-pool/evals/run_pool_day_eval.py        VERDICT: PASS (includes the max_hold recall + lent-gpu0 recall paths)
```

## Docker/build/smoke checks

```text
Not deployed (task: do not deploy). No image/compose/env change.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`.

- Finding (MUST): no bound left on take-backs -- a run whose step never fits could replay forever
  (both 3-attempt caps are gone for take-backs; most runs have no `deadline_at`).
  - Fix: `hold_takebacks` counter + `DURABLE_RUNS_HOLD_MAX_TAKEBACKS` (default 12) in every graph
    that catches `HoldLost` through the shared `taken_back()` helper (curiosity, self-sense, reflect,
    reading, including the result-shaped `replay_if_requeued` path); at the limit the run fails
    `hold_takeback_limit:<n>` and releases the seat.
  - Evidence: `test_the_take_back_limit_fails_the_run_instead_of_replaying_forever` (graph) and
    `test_a_step_that_never_fits_fails_at_the_take_back_limit_instead_of_replaying_forever` (real
    Postgres + pool: 2nd recall past grace with limit 1 -> `failed`, lease `released`).
- Finding (SHOULD): `release(keep_requeued)` kept a hold re-granted at a newer generation even on
  the failed-attempt path, which then backs off 30-300 s with nothing heartbeating it.
  - Fix: kept only when the release reason is a take-back (`TAKEBACK_RELEASE_REASONS`).
  - Evidence: `test_release_keeps_a_regranted_hold_only_on_a_take_back[attempt_failed]` hands it back.
- Finding (SHOULD): new classification branches untested.
  - Fix: `test_taken_back_classifies_every_mid_node_pool_answer`,
    `test_a_work_failure_after_a_non_urgent_requeue_is_a_take_back_not_an_attempt`, release test above.
- Finding (SHOULD): reading turn now got `WorkflowDeadline` from `_beat` and would end as
  `turn_exception:WorkflowDeadline`.
  - Fix: explicit `except WorkflowDeadline` -> `failed`, `last_error="workflow_deadline"` (as reverie).
- Finding (SHOULD): pool spec diagram still showed `aborted -> retry_wait`.
  - Fix: `docs/superpowers/specs/2026-09-24-gpu-pool-design.md` lifecycle updated.
- Nits fixed: stale "lost hold: bounded re-try" comment in reflect; `_beat_outreach` catches
  `(HoldLost, WorkflowDeadline, RuntimeError)` and records the class name as `pool_status`;
  lease-graph variable renamed `preempted` -> `requeue_in_place`.
- Not changed (UNVERIFIED): a waiting run now carries `last_error` = the `HoldLost` text while its
  status is `retrying`/`waiting_resource`. The failed-attempt path already did the same before this
  PR; I did not check how the Hub run panel renders it.

## gpu2 seat limit vs real run lengths (read-only data, 7 days; value NOT changed)

How long one work step (`run.started` -> first release/expiry/next node) takes, by where it ran:

| lane | workflow | n | median | p90 | max | >= 60 min |
|---|---|---|---|---|---|---|
| agent (gpu1, no limit) | curiosity.investigate | 31 | 82 min | 147 min | 258 min | 20 |
| agent (gpu1, no limit) | self_sense_eval | 10 | 59 min | 110 min | 147 min | 5 |
| agent (gpu1, no limit) | reading.turn | 31 | 20 min | 58 min | 97 min | 1 |
| agent-gpu2 | curiosity.investigate | 27 | 50 min | 70 min | 70 min | 12 (14 lost) |
| agent-gpu2 | self_sense_eval | 6 | 33 min | 57 min | 70 min | 1 |
| agent-gpu2 | reading.turn | 16 | 19 min | 58 min | 60 min | 0 (4 lost) |

gpu2's limit is counted from when the *seat* was loaded, not from when the run got it: across 26
`max_hold` recalls, a run had held gpu2 for a median 51 min (min 2, max 60) before the recall, then
10 min of grace. On gpu2 nothing runs past ~70 min -- the cap is visible in the data -- while the
same curiosity work, unconstrained on gpu1, has a median of 82 min.

What this means after this PR: a long curiosity step on gpu2 no longer fails the run, but it still
gets cut at ~1 h and replayed **from the start of the step** on the next grant. So most curiosity
steps placed on gpu2 throw away up to an hour of gpu2 work before finishing elsewhere.

Recommendation (Juniper decides; `config/gpu_pool.yaml` untouched): either raise
`agent-gpu2.max_hold_sec` to cover a real curiosity step (p90 on gpu1 is 147 min, so ~9000 s), or
keep diffusion's hour and stop placing `curiosity.investigate`/`self_sense_eval` holds on gpu2 (they
rarely fit), leaving gpu2 for `reading.turn` (p90 58 min fits). A third option, measuring the
limit from the hold's grant instead of the seat load, only helps runs that land on a freshly loaded
seat. The cheapest honest first step is the placement one: it wastes no gpu2 hours and keeps
diffusion's reclaim promise.

## Restart required

Not deployed. When Juniper deploys (either order is safe):

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a step that keeps landing on seats it cannot finish on replays (and
  wastes GPU time) up to 12 times before failing with `hold_takeback_limit`. Mitigation: the hold
  keeps its original place and `agent` (no limit) is first in the role order; the limit is an env
  knob. The real fix for gpu2 is the seat-limit decision below.
- Severity: low. Concern: reverie.visual's `HoldLost` path is not counted (it always has a deadline,
  so it was and stays bounded by that).
- Severity: low. Concern: a hold heartbeat that keeps getting lost (durable-runs restarting) no
  longer spends a durable attempt; the pool still dead-letters it after 3 expiries, and durable-runs
  then asks afresh. Same as what `resource_wait` already did before a step.
- Severity: low. Concern: the pool re-queues immediately instead of after a 5 s retry delay, so a
  hold can be re-granted before durable-runs' next beat; durable-runs now treats "granted at a newer
  generation" as a take-back and keeps the new grant.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2402

🤖 Generated with [Claude Code](https://claude.com/claude-code)
