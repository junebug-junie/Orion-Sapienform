# GPU pool stage 4.5 — durable runs run on pool holds; the durable broker and gpu2 decider are gone

> **Stacked PR.** Base: `feat/gpu-pool-stage4-3-holds` (#2352). This branch ALSO contains stage 4.4
> (`feat/gpu-pool-stage4-4-consumers`, #2351) merged in, because 4.5 needs both 4.3's hold client and
> 4.4's consumers. Merge only after 4.1 (#2349), 4.2 (#2350), 4.3 (#2352) and 4.4 (#2351). The diff
> against the base shows 4.4's changes too; 4.5's own changes are the commits after the merge commit.

## Summary

- **A durable run now asks the GPU pool for one hold for its whole life** and waits in the pool's
  queue; the pool is the only scheduler. It is woken by the pool's own `granted` event for its
  holder (fallback: one `status` read per run per `DURABLE_RUNS_HOLD_STATUS_POLL_SEC`, 60 s — no
  polling storm).
- **Every LLM call a run makes carries the hold's ref**, so the gateway attaches it to the hold
  instead of queueing it behind the run itself. That now includes the reflect run's call: an
  admitted `self_study.reflect` run used to fall back to the *curiosity* graph and its LLM call
  carried no lease at all; it now has its own admitted graph and sends `options.gpu_lease`.
- **The hold's role is never a route.** A run placed on `agent-gpu2` still names the `agent` route;
  `assigned_lane` is no longer sent.
- **Heartbeat, recall, restart, Door-A.** A working run heartbeats its hold; a recalled hold is let
  go at the next node boundary (inside the 600 s grace); a lost hold stops the turn and the run
  waits for the same lease id; a restarted durable-runs fences the old turn (harness cancel + a new
  turn identity) and replays under the same hold; a Door-A hold is kept and heartbeated until Hub
  releases it, and a restarted process adopts it.
- **Kill means kill:** the durable broker, lane policy/widening, gpu2 elastic decider,
  `elastic_runtime.py`, `http_hops.py`, `/leases/validate`, `/admission`, `/elastic/*` are deleted,
  with no fallback. `capacity.py` and `/capacity` stay (world-model / visual-chain permits, stage 5),
  now without the frozen-demand drain reservation.
- **Cutover runbook** with exact commands, checks and rollback:
  `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md`.

## Outcome moved

- The failure the whole stage exists for — "the run's own call queues behind its own hold" — is
  proven absent end to end on the real pool runtime: in the acceptance test every model call of a
  curiosity turn (stance, FCC over HTTP, finalize, repair) is an `attach` under the run's hold, on
  the home card and on gpu2 loaded by the pool. Dropping the ref fails it (mutation-checked).
- gpu2 is decided by one scheduler: the acceptance and eval show the pool loading gpu2 after a hold
  waits past 1200 s, granting the oldest waiting run there first, and unloading when idle.
- A hazard the spec named (reflect carries no lease) is closed, and a bug it did not name (admitted
  reflect runs driven through the curiosity graph) is fixed.

## Current architecture (before this PR)

- `resource_request` registered a row in `durable_resource_demands`; the broker (`broker.py` +
  `policy.py`) granted exclusive rows in `durable_resource_leases`; `elastic_runtime.py` decided
  and actuated gpu2 over HTTP to circe's controller. The durable lease rode every call as
  `X-Orion-Resource-Lease` and the gateway ALSO took its own pool lease per call (scheduled twice).
- The pool (stages 1–4.3) could not see any of that queue.

## Architecture touched

- `services/orion-durable-runs` only, plus `orion/durable_admission` (the broker's shared package).
- Contracts consumed (no new ones): `GpuLeaseRequestV1` acquire/status/heartbeat/release (4.1/4.3),
  `GpuLeaseRefV1` on `CuriosityTurnRequestV1.gpu_lease` and bus `options.gpu_lease` (4.4),
  `orion:gpu_pool:event` (durable-runs already listed as a consumer in `channels.yaml`).
- Lifecycle events for Hub's run views keep their names; `run.lane_assigned`'s `lane` is the hold's
  role. No longer emitted: `run.lane_swap_suppressed`, `run.resource_eligibility_expanded`,
  `resource.elastic_*`.

## Files changed (4.5 commits)

- `services/orion-durable-runs/app/admission_runtime.py`: rewritten on pool holds (register/lease/
  execute/guard/release, `_recover`, `_hold_ready`, `on_pool_event`, Door-A upkeep, `release_outreach`).
- `app/pool_hold.py` (new): the pool verbs, route → class/priority/min ctx from `config/gpu_pool.yaml`.
- `app/admitted_graph.py`: shared `resource_nodes`; tail guard never waits; Door-A keeps the hold.
- `app/admitted_self_sense_graph.py`, `app/admitted_reflect_graph.py` (new), `app/reflect_graph.py`,
  `app/runner.py` (`_call_reflect_llm(gpu_lease=)`), `app/graph.py` / `app/self_sense_graph.py`
  (turn carries `gpu_lease`, `turn_fence`, finish detail `gpu_lease`).
- `app/main.py`: pool events wake runs; deleted endpoints; `release-outreach-lease` → pool release;
  permits built with `reserve_waiting=False`. `app/settings.py`: removed broker/elastic keys, added two.
- Deleted: `orion/durable_admission/{broker,policy,elastic}.py`, `app/elastic_runtime.py`,
  `app/http_hops.py`. `orion/durable_admission/store.py` trimmed (no demand/lease writes).
- `Dockerfile` (copies `config/gpu_pool.yaml`), `docker-compose.yml`, `.env_example`, `README.md`.
- Tests: `tests/pool_fixture.py` (the real pool runtime in process), rewritten
  `test_admission_runtime_postgres.py`, `test_admitted_graph.py`, `test_durable_acceptance.py`,
  `test_held_auxiliary_chain.py`, `test_resume_failure_bound_postgres.py`, `test_capacity_postgres.py`,
  `test_admission_review_regressions.py`, `test_finish_timing.py`, `test_attempt_correlation.py`,
  `test_runner_turn_contract.py`, `test_admitted_self_sense_graph.py`, `test_rpc_health_publisher.py`;
  `acceptance_turn.py` gains a real-pool mode. Deleted with the code they tested: `test_admission_policy`,
  `test_elastic_{api,policy,postgres}`, `test_running_recovery_postgres` (its restart claim is now
  `test_restart_mid_turn_fences_the_old_turn_and_replays_under_the_same_hold`).
  `test_cabinet_endpoint_contract.py` moved to `services/orion-gpu-pool/tests` (the pool owns the
  thermal guard now).
- Evals: `evals/hold_fairness.py` (new; replaces `admission_fairness.py` and `elastic_fairness.py`),
  `evals/gateway_capacity.py` rewritten without the broker.
- `.github/workflows/orion-durable-runs-tests.yml`: new eval step, path triggers for the pool runtime.
- `services/orion-gpu-pool/app/guards.py`: comment only.
- `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md` (new).

## Schema / bus / API changes

- Added: none (consumes 4.1/4.3/4.4 contracts).
- Removed (HTTP): `POST /leases/validate`, `GET /admission`, `GET /elastic/status`, `POST /elastic/target`.
- Behavior changed: `POST /runs/{id}/release-outreach-lease` releases the pool hold;
  `GET /runs/{id}` returns `hold`, the granted `lease` (a hold ref) and live `pool` status instead of
  the broker decision; finish detail carries `gpu_lease` instead of `resource_lease`.
- Compatibility: `ResourceRequirementV1.{alternatives, allow_elastic_activation, pinned_lane,
  operator_override}` are accepted and ignored (also in the duplicate-receipt comparison). A legacy
  durable lease in an old checkpoint is dropped and the run asks the pool. The gpu2 controller must
  already be on `GPU2_AUTHORITY=pool` before this deploys (it no longer answers `/elastic/status`).

## Env/config changes

- Added: `DURABLE_RUNS_HOLD_STATUS_POLL_SEC=60.0`, `DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC=1800.0`.
- Removed: `DURABLE_RUNS_ELASTIC_*` (15 keys), `DURABLE_RUNS_ADMISSION_SHADOW`, `DURABLE_RUNS_WIDENING_*`
  (3), `DURABLE_RUNS_LANE_POLICY_JSON`, `DURABLE_RUNS_GATEWAY_URL`. (The live `.env` keeps them until
  runbook step 2 has used `DURABLE_RUNS_ELASTIC_SHADOW`; the 4.5 build ignores them.)
- Compose now also exposes `CORTEX_REQUEST_CHANNEL` and `DURABLE_RUNS_REFLECT_LLM_CALL_TIMEOUT_SEC`
  (pre-existing parity gap).
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py
  orion-durable-runs`: yes — `+DURABLE_RUNS_HOLD_STATUS_POLL_SEC`, `+DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC`
  landed in `/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env`. Diverged (unchanged,
  intentional host value): `DURABLE_RUNS_GRAPH_HOST`.
- Skipped keys requiring operator action: none. The runbook's step-2/3/4 flips
  (`DURABLE_RUNS_ELASTIC_SHADOW`, circe `GPU2_AUTHORITY`, `GPU_POOL_ACTUATE_ROLES`) are cutover
  actions, not template changes.

## Tests run

```text
services/orion-durable-runs (throwaway postgres:16 on :55491, ORION_ADMISSION_TEST_DSN):
  pytest tests -q                                  -> 117 passed (incl. 6 hold-leak-path tests added after review)
services/orion-gpu-pool: pytest tests -q            -> 84 passed, 7 skipped
orion/gpu_pool/tests -q                             -> 174 passed
services/orion-hub: test_curiosity_gpu_lease + test_curiosity_routes_runs -> 29 passed
tests/test_curiosity_run_story.py                   -> 48 passed
static gates (orion-static-gates.yml list): env-sync/env-parity/schema-registry/topology/grammar/
  substrate pytest (113 passed), check_metric_lineage --gate, check_definition_drift --gate,
  check_inner_state_registry, check_service_hostname_refs, compose mounts, journal registry,
  sentience instruments --static-only, system_health producers, control surface parity,
  async routes, chat route poachers, check_gpu_pool_config, check_env_template_parity,
  check_service_env_compose_parity orion-durable-runs   -> all PASS
```

Mutation checks (each must fail a test; each did):

- drop `gpu_lease` from the curiosity turn → 4 acceptance cases fail (the gateway's plain acquire
  queues behind the run's own hold);
- drop the hold from the reflect call → the reflect test fails;
- no harness cancel when a restarted driver fences a turn → the restart test fails;
- ignore a lost hold on heartbeat → the lost-heartbeat test fails;
- write the hold role into `assigned_lane` → `test_turn_carries_the_hold_ref_and_never_the_pool_role_as_a_route` fails.

## Evals run

```text
python services/orion-durable-runs/evals/hold_fairness.py   -> PASS
  home card: 20 runs, FIFO grant order, one hold per run, every mid-run system agent call granted
  on the card between the run's calls; gpu2: no load before 1200 s, load -> 3 runs FIFO on
  agent-gpu2 -> idle unload -> diffusion restored, no failed actuation
python services/orion-durable-runs/evals/gateway_capacity.py -> PASS
  40 contended rounds, 2 gateways, one winner each; frozen demand would have reserved under the
  pre-4.5 wiring and does not now; an active legacy lease still fences its backend
```

## Docker/build/smoke checks

```text
Not built or deployed (task: do not deploy). Import check of app.main/app.admission_runtime with
the service requirements passes. The Dockerfile now copies config/gpu_pool.yaml (pool_hold.py
reads routes and hold_lease_ttl_sec from it at startup).
```

## Review findings fixed

Code-review subagent (read-only) on the 4.5 commits: 16 findings (0 blocker, 12 should, 4 nit).
The top-priority checks came back clean: the hold ref rides every admitted LLM path, the pool role
never reaches a route label, and every returned state key is declared.

- Finding: a failed release RPC leaked the hold (a retryable hold is re-granted to nobody until
  dead-lettered) and the run could queue behind its own leak.
  - Fix: `_end_hold` records failures in `_pending_release`; every reconcile retries until the pool
    confirms.
  - Evidence: `test_a_failed_release_rpc_is_retried_until_the_pool_confirms`; mutation (no record) fails it.
- Finding: a Door-A hold whose release failed was never retried; adoption ran only once at boot.
  - Fix: covered by the retry above; adoption now re-runs every `DURABLE_RUNS_HOLD_STATUS_POLL_SEC`
    and skips holds pending release.
  - Evidence: `test_door_a_keeps_heartbeating_the_hold_until_hub_releases_it_and_a_restart_adopts_it`.
- Finding: a Door-A hold was kept for a run that did not end completed (a cancel raced the terminal).
  - Fix: `_terminal` ends the outreach hold when the projected status is not `completed`.
  - Evidence: `test_a_door_a_hold_is_ended_when_the_run_does_not_end_completed`.
- Finding: a hold dead-lettered by expiries during a durable-runs outage failed the run.
  - Fix: only door refusals, `deadline` and `backlog_max_age` fail the run; any other unavailable
    ends the hold and the run asks again under a new request id.
  - Evidence: `test_a_hold_dead_lettered_during_an_outage_is_replaced_not_fatal`; mutation fails it.
- Finding: a cancel during an in-flight acquire left the landed hold.
  - Fix: cancel with no checkpointed request id probes `<run_id>:<seq+1>` and ends it.
  - Evidence: `test_cancel_during_an_in_flight_acquire_ends_the_hold_that_landed`; mutation fails it.
- Finding: a turn could start on a hold already being recalled, and burn an attempt at the grace abort.
  - Fix: `execute` releases it and raises `HoldRecalled`; all three graphs requeue without counting
    an attempt.
  - Evidence: `test_work_never_starts_on_a_hold_already_being_recalled`.
- Finding: paused runs were re-driven every tick, with one pool RPC each, forever.
  - Fix: reconcile skips paused runs; the pausing `control()` already released the hold.
  - Evidence: `test_a_paused_run_is_not_re_driven_every_tick`.
- Finding: runbook step 1 does not converge, because the old broker keeps granting.
  - Fix: step 1a pauses every live run without an active lease, and step 2 also sets
    `DURABLE_RUNS_ADMISSION_SHADOW=true`. The count is re-checked before steps 2, 4 and 5, START is
    taken after it reaches 0, and step 5 resumes the paused runs explicitly.
- Finding: the runbook edited the primary `.env` but deployed from a worktree that reads its own.
  - Fix: link the env files into `$WT`, and check each flip inside the container.
- Finding: the step-6 rollback guard could not work (`COMMIT` always ran), and "only step 6
  writes" was false.
  - Fix: the UPDATE is limited to the snapshot's ids and aborts on a count mismatch (division by
    zero). Every production-changing step is marked **[GO]**.
- Finding: rollback ordering let two schedulers overlap, and ran on a bare interpreter.
  - Fix: stop 4.5, cancel holds, then start the old image; use the repo venv.
  - Remaining: the old build clearing 4.5 checkpoints is marked UNVERIFIED, and reflect runs are
    cancelled first.
- Finding (nit): self-sense `publish` did not persist a hold released at recall.
  - Fix: done.
- Finding (nit): `_hints`/`_checked` never pruned.
  - Fix: pruned at `_terminal`.
- Not fixed, accepted:
  - (nit) a replayed reflect call after a restart is not fenced; the old cortex RPC cannot be
    cancelled, and the duplicate call attaches to the same hold.
  - (nit) a turn that finishes after its hold was lost between beats is accepted; the pool refuses
    stale-generation attaches, so its calls ran under a valid generation.
- Partly covered (finding 12): `runner._call_reflect_llm` building `options.gpu_lease` IS tested
  (`test_reflect_llm_call_sends_the_hold_ref_in_options_and_keeps_its_route`). cortex-orch passing
  `options` through to cortex-exec is not exercised end to end here (UNVERIFIED; it is the same
  path `llm_route` already rides).

## Restart required

Follow `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md` — in order: step 1 wait for zero active
legacy leases; step 2 `DURABLE_RUNS_ELASTIC_SHADOW=true` + restart (current build); step 3 circe
`GPU2_AUTHORITY=pool` + restart; step 4 `GPU_POOL_ACTUATE_ROLES=agent-gpu2` + pool restart; step 5
deploy this build:

```bash
cd <worktree at merged main>
python3 scripts/sync_local_env_from_example.py orion-durable-runs
scripts/safe_docker_build.sh orion-durable-runs up -d --build durable-runs
```

Step 6 (withdraw the frozen pending demands) is a production write: snapshot, then only with
Juniper's go.

## Risks / concerns

- Severity: medium.
  - Concern: the 4.3 scheduler never loads gpu2 for a hold that carries `min_ctx_tokens` (its
    `fits` needs the seat's LIVE `ctx_per_slot`, which an unloaded seat has none of) and emits
    nothing saying so. Confirmed on the real runtime: `min_ctx_tokens=0` loads, `32768` stays silent.
  - Mitigation: no live producer sets `requirements.minimum_context_tokens` today, so 4.5 passes 0
    in practice. Fix belongs in the pool (use the role's last-seen/announced ctx); the gpu2
    acceptance variant documents it.
- Severity: medium.
  - Concern: a retry after a failed attempt releases the hold and takes a new one, which may land
    on a different role (e.g. lent gpu0's 35B instead of the 27B). The old broker pinned a retry
    to its first lane.
  - Mitigation: a retry re-runs the whole turn from scratch; a hold the pool re-queued (lost
    heartbeat, recall past grace) is kept with its place. Pinning is a pool feature if wanted.
- Severity: medium.
  - Concern: the acceptance check 2 SQL joins request leases to durable turns via
    `turn_correlation_id`; that the gateway stamps the harness turn id on every call is UNVERIFIED
    live.
  - Mitigation: the runbook pairs it with a positive children-count query and says to treat check 2
    as UNVERIFIED if that is zero.
- Severity: low.
  - Concern: a durable-runs restart during Door-A composition adopts the hold only from the event
    history; if the restart outlasts the 90 s TTL the pool re-queues it and the adopted heartbeat
    then ends it (Hub's composition is refused by its own status check).
  - Mitigation: Hub records an outreach skip; nothing is stranded.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2356

🤖 Generated with [Claude Code](https://claude.com/claude-code)
