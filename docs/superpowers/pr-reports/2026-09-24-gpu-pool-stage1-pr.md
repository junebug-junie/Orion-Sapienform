## Summary

- **One service decides GPU placement now: `orion-gpu-pool`.** Every GPU user asks it for a lease and gets a grant, a place in line, a durable backlog entry (opt-in), or a typed "unavailable" with a reason. Stage 1 runs in **observe mode**: it discovers, answers and publishes, but nothing depends on it yet.
- **The rules are one YAML file, `config/gpu_pool.yaml`, and it names roles and cards, never models.** Which model fills a role is *discovered*: each llama.cpp worker now announces its `llm_profiles.yaml` profile, and the pool confirms it against the server's own `/props`. The mismatch, silent, down and unclaimed cases get no grants.
- **Each lease is a checkpointed LangGraph run.** It covers retry with backoff, dead letter, operator replay, backfill as linked children, and a restart resuming every lease in place.
- **Everything is on the bus.** That covers leases, state, operator control and worker announcements. Lease history persists through sql-writer (`gpu_pool_events`), and lease facts go to grammar in the `capacity` layer. Queue wait is recorded under a label that equilibrium excludes: waiting in line is not transport.
- **Design spec (v2):** `docs/superpowers/specs/2026-09-24-gpu-pool-design.md`. It includes the build order for stages 2–6 and the full delete list for the nine mechanisms it replaces.

## Outcome moved

Nothing could see which model was loaded where. Now the pool can, confirmed three ways: the announcement, the profile's file, and the server's own report. And there is now one scheduler, with every rule tested, where there were nine deciders that didn't know about each other. The policy eval over two simulated hours and about 6,400 leases reports:
- owner starvation beyond grace: 0 s
- leases lost: 0
- big work spilled onto gpu3: 0

## Current architecture

Nine mechanisms picked GPUs, four of them with their own notion of priority:
- the durable-admission broker;
- gateway capacity permits;
- the GPU2 elastic borrow;
- the lane controller;
- the lend-lane Redis gate;
- the gateway in-flight semaphores;
- background slot polling;
- the static route table;
- lease fencing.

On top of those, open PR #2317 added a tenth (now closed). None of them knew what model was actually loaded.

## Architecture touched

- **New service** `services/orion-gpu-pool` (athena, port 8127).
- **New library** `orion/gpu_pool/` (config, scheduler, discovery, lease_graph, client).
- **New schemas** in `orion/schemas/gpu_pool.py`.
- **llamacpp-host** announces its worker (role, profile, port).
- **sql-writer** gets the `gpu_pool_events` table.
- **equilibrium** adds an exclude label.
- **Bus catalog and registry** get the new channels and schemas.

## Files changed

- `config/gpu_pool.yaml`: the whole rule set (cards, roles, classes, routes, defaults).
- `orion/gpu_pool/config.py`: validated loader plus VRAM check.
- `orion/gpu_pool/scheduler.py`: pure scheduler, one test per spec rule.
- `orion/gpu_pool/discovery.py`: announcement × profile × `/props` → confirmed / mismatch / silent / down / unloaded / evicted.
- `orion/gpu_pool/lease_graph.py`: transition table plus the LangGraph builder.
- `orion/gpu_pool/client.py`: the only client, `gpu_lease(...)`.
- `orion/schemas/gpu_pool.py`: all contracts.
- `services/orion-gpu-pool/app/{main,runtime,store,settings}.py`, plus `Dockerfile`, `docker-compose.yml`, `.env_example`, `README.md`, `requirements.txt`.
- `services/orion-gpu-pool/tests/*`: runtime, client round trip, Postgres store and restart.
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`: the policy eval.
- `services/orion-sql-db/manual_migration_gpu_pool_v1.sql`: the projection tables.
- `services/orion-llamacpp-host/app/{main,settings}.py`, `docker-compose.atlas-workers.yml`, `docker-compose.dsv41.yml`, `tests/test_worker_announce.py`: the worker announcement.
- `services/orion-sql-writer/*`: `GpuPoolEventSQL`, route, subscribe guarantee, 30-day retention, shape test.
- `services/orion-equilibrium-service/{app/settings.py,.env_example,docker-compose.yml}`: adds `gpu_pool_wait` to the transport exclude labels.
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `config/metrics/metric_definitions.lock.json`: contracts.
- `scripts/check_gpu_pool_config.py`, `.github/workflows/orion-gpu-pool-tests.yml`: gates.
- `docs/superpowers/specs/2026-09-24-gpu-pool-design.md`: the spec.

## Schema / bus / API changes

- **Added schemas:**
  - `GpuLeaseRequestV1`, `GpuLeaseReplyV1`, `GpuLeaseGrantV1`
  - `GpuPoolEventV1`
  - `GpuPoolStateV1`, `GpuPoolStateRequestV1`
  - `GpuPoolControlV1`, `GpuPoolControlReplyV1`
  - `GpuActuateV1` / `GpuActuateResultV1` (defined, not used until stage 5)
  - `LlmWorkerAnnounceV1`
- **Added channels:**
  - `orion:gpu_pool:lease:request`, with replies on `orion:gpu_pool:reply:*`
  - `orion:gpu_pool:event`
  - `orion:gpu_pool:state`
  - `orion:gpu_pool:state:request`, with replies on `:state:reply:*`
  - `orion:gpu_pool:control:request`, with replies on `:control:reply:*`
  - `orion:llm:worker:announce`
- **Added HTTP:** `GET /health`, `GET /v1/pool` (read-only mirror), `GET /v1/leases/{id}/history`.
- **Removed:** nothing yet (deletions happen in stages 3–5, each in the stage that replaces the piece).
- **Behavior changed:**
  - The metric-lock entry for the shared `orion:grammar:event` channel now lists `orion-gpu-pool` as a producer. The drift gate marks that **HIGH (routing_changed)**, and it was re-locked on purpose (see Risks).
  - Equilibrium's default transport exclude list gains `gpu_pool_wait`. That only matches the pool's own queue-wait hop.
- **Compatibility:** purely additive. No existing caller changes path in stage 1.

## Env/config changes

- **Added keys:**
  - `orion-gpu-pool/.env_example` (new service): `GPU_POOL_*`, `POSTGRES_URI`, `ORION_BUS_URL`, …
  - sql-writer: `GPU_POOL_EVENTS_RETENTION_DAYS=30`
- **Changed default:** equilibrium `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=log_orion_metacognition,gpu_pool_wait`
- **Removed / renamed keys:** none
- **`.env_example` updated:** yes, in all three services.
- **Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer orion-equilibrium-service orion-gpu-pool`:**
  - `+GPU_POOL_EVENTS_RETENTION_DAYS=30` was added to the primary sql-writer `.env`.
  - `orion-gpu-pool` has no primary `.env` (new service). A worktree-local `.env` was bootstrapped (gitignored, mode 600) with `POSTGRES_URI` from durable-runs and a freshly generated operator token.
- **Skipped keys requiring operator action:**
  - `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS`: the live `.env` is `log_orion_metacognition`, which diverges from the new template, and the sync never overwrites a diverged value. **Set it to `log_orion_metacognition,gpu_pool_wait` before deploying the pool.**
  - sql-writer `SQL_WRITER_SUBSCRIBE_CHANNELS` / `SQL_WRITER_ROUTE_MAP_JSON`: these diverged before this PR. No action needed: the route map merges over code defaults, and the subscribe list has a code-level guarantee for `orion:gpu_pool:event`.

## Tests run

```text
python scripts/check_gpu_pool_config.py                          ok (4 cards, 8 roles, 7 classes)
python -m pytest orion/gpu_pool/tests -q                          69 passed
cd services/orion-gpu-pool && GPU_POOL_TEST_POSTGRES_URI=… pytest tests -q   27 passed (real postgres:16:
    migration, fenced projection, single-writer advisory lock + liveness, restart from Postgres checkpoints)
cd services/orion-llamacpp-host && pytest tests -q                 39 passed, 1 failed (pre-existing: test_profile_forwarding
    metacog n_parallel 4 != 1 -- llm_profiles.yaml, untouched here; fails without this branch's changes)
cd services/orion-sql-writer && pytest tests -q                    23 failed / 616 passed; clean origin/main in the same venv:
    23 failed / 612 passed -- identical pre-existing failures, +4 new passing (gpu_pool_events shape)
static gates (from .github/workflows/orion-static-gates.yml): definition drift PASS (re-locked), metric lineage PASS,
    inner_state_registry, stdlib shadow, service hostname refs, compose relative mounts, journal dispatch, schedule
    collisions, sentience instruments, system health producers, control-surface parity, async routes, chat route
    poachers, bus reply channels, env template parity (3 services) -- all PASS
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py     VERDICT: PASS
  2 simulated hours, ~6,400 leases, metacog outage 30-40 min, gpu0 lent 50-70 min, gpu2 swaps (60 s actuator)
  owner_starvation_sec 0 | leases_lost 0 | small_role_violations 0
  metacog spilled to fast in the outage; agent used the gpu2 seat and lent gpu0; retry path exercised (77 requeues)
  leases/sec through runtime + MemorySaver: ~280 (in-memory ceiling; Postgres checkpoint throughput UNVERIFIED until live)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-gpu-pool build                 Built
docker run … python -c "import app.main, app.runtime; load config + 24 llm profiles"   import ok
Live deploy: NOT DONE (production; awaiting Juniper). Live path is UNVERIFIED.
```

## Review findings fixed

The code review subagent reported 13 findings (3 high, 6 medium, 4 low). All are fixed, each with a regression test.

- **Finding (high):** one waiting owner recalled one more borrower every tick of the grace period.
  - Fix: borrowers already in `recalling` count toward the owners waiting.
  - Evidence: `test_one_waiting_owner_recalls_exactly_one_borrower_across_ticks`.
- **Finding (high):** failed, expired or aborted leases retried and were re-granted to callers that had already left. A late `release ok` was ignored.
  - Fix: retry and backlog are opt-in (`retryable`). Non-retryable failures end. Release and cancel are accepted from every waiting state. A retry past its deadline becomes `unavailable`.
  - Evidence: `test_non_retryable_failure_ends_and_is_never_regranted`, `test_release_ok_after_abort_ends_the_lease`, `test_retry_past_deadline_is_unavailable_not_regranted`.
- **Finding (high):** an owner reclaiming gpu2 was backlogged, and the client then cancelled it, so the reclaim never happened.
  - Fix: waiting for its own card back counts as serviceable, and the client keeps backlogs only for `retryable` callers.
  - Evidence: `test_owner_reclaiming_gpu2_waits_instead_of_backlogging`.
- **Finding (medium):** `swap_requested` was emitted at most once per process.
  - Fix: now edge-triggered.
  - Evidence: `test_swap_request_is_reported_again_when_it_recurs`.
- **Finding (medium):** a dead letter reached by expiry or abort was never published as `dead_lettered`.
  - Fix: both the cause and the outcome are published.
  - Evidence: `test_expiry_to_dead_letter_reports_both_facts`.
- **Finding (medium):** operator-only seats were unreachable.
  - Fix: new `hold` / `release` control verbs, and operator holds have no heartbeat expiry.
  - Evidence: `test_operator_hold_via_control_and_release`, `test_operator_hold_has_no_heartbeat_expiry`.
- **Finding (medium):** the projection row could fall behind its checkpoint with no repair.
  - Fix: the row heals from the checkpoint on a rejected transition and at start.
  - Evidence: `test_projection_heals_from_checkpoint`.
- **Finding (medium):** a class or role removed from the YAML crashed every tick, and one bad decision blocked the rest.
  - Fix: orphaned rows end with `config_removed`, and each decision is isolated.
  - Evidence: `test_removed_class_in_live_rows_does_not_kill_the_tick`.
- **Finding (medium):** the leader lock sat on a pooled connection and could vanish silently.
  - Fix: a dedicated connection, re-checked every 10 s. The service exits if the lock is lost.
  - Evidence: Postgres test asserts `leader_alive()` before and after release.
- **Finding (low):** draining ignored whether the owner fits.
  - Fix: now only an owner that fits triggers draining.
- **Finding (low):** a client cancelled mid-acquire left a stray lease behind.
  - Fix: the client withdraws via its `request_id` / `lease_id`.
- **Finding (low):** a heartbeat on a lost lease was ignored.
  - Fix: `lease.lost` and `lease.recalled` are set.
- **Finding (low):** catalog gaps.
  - Fix: gpu-pool is listed as a grammar producer, and the state and control reply wildcards are added.

## Restart required

Stage 1 deploy (production, not performed; Juniper to run or approve):

```bash
# 0. Pre-launch: roll back #2317 pollution in prod (athena llm-gateway built from the #2317 worktree)
sed -i '/^LLM_LANE_CONTENTION_FALLBACK_ENABLED=/d;/^LLM_LANE_CONTENTION_FALLBACK_JSON=/d;/^LLM_LANE_REAL_CAPACITY_JSON=/d' \
  /mnt/scripts/Orion-Sapienform/services/orion-llm-gateway/.env
git -C /mnt/scripts/Orion-Sapienform worktree add --detach ../Orion-Sapienform-prod-gateway-main origin/main
cd /mnt/scripts/Orion-Sapienform-prod-gateway-main && cp ../Orion-Sapienform/.env .env \
  && cp ../Orion-Sapienform/services/orion-llm-gateway/.env services/orion-llm-gateway/.env \
  && scripts/safe_docker_build.sh orion-llm-gateway up -d --build
docker exec orion-llm-gateway sh -c 'ls /app/app/lane_contention.py; env | grep LLM_LANE_'   # expect: no such file, no keys

# 1. projection tables
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_gpu_pool_v1.sql
# 2. equilibrium: exclude the queue-wait hop BEFORE the pool publishes rpc_health
#    set EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=log_orion_metacognition,gpu_pool_wait in its .env, then:
scripts/safe_docker_build.sh orion-equilibrium-service up -d
# 3. sql-writer (gpu_pool_events)
scripts/safe_docker_build.sh orion-sql-writer up -d --build
# 4. the pool (from this worktree; copy its .env into the primary checkout after merge)
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
curl -s localhost:8127/v1/pool | jq '.roles[] | {role, status, profile_name, model_file}'
# 5. circe: recreate llama.cpp workers so they announce (roles read "silent" until then -- correct)
#    (on circe, from a worktree at this branch/main after merge)
scripts/safe_docker_build.sh orion-llamacpp-host -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml up -d
```

## Risks / concerns

- Severity: medium
  - Concern: the metric lock marks the shared `orion:grammar:event` producer list **routing_changed (HIGH)**, because `orion-gpu-pool` was added as a declared producer.
  - Mitigation: the declaration is true. Pool grammar uses layer `capacity`, trace prefix `gpu_pool.lease:`, and roles `gpu_lease_*`. No reducer, field node or equilibrium trigger reads it: substrate-runtime's cursor allowlist excludes it, and equilibrium matches only `rpc_transport_timeout`. It needs Juniper's sign-off as a locked-metric change.
- Severity: medium
  - Concern: Postgres checkpoint throughput at real metacog/fast volume has not been measured. The in-memory ceiling is about 280 leases/s.
  - Mitigation: must be measured live in observe mode before the stage 3 gateway cutover depends on it.
- Severity: medium
  - Concern: 11 other prod containers across athena and circe run from worktrees, not main. #2317 was only one instance of this.
  - Mitigation: out of scope here; listed for a follow-up.
- Severity: low
  - Concern: stage 1 does not actuate swaps (only `swap_requested` events), and no caller uses `retryable` or `replay_payload` yet.
  - Mitigation: by design (observe mode). Stages 3–5 wire them.
- Severity: low
  - Concern: pre-existing failures in llamacpp-host (1) and sql-writer (23) are unrelated to this PR.
  - Mitigation: verified identical on clean main in the same environment.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2318

🤖 Generated with [Claude Code](https://claude.com/claude-code)
