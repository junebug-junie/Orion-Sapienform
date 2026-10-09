# GPU pool stage 5.4: world-model and image generation on pool leases, permits stop, visual_baseline guard deleted

## Summary

The world model and image generation now share circe's gpu2 only through the GPU pool. Nothing asks
durable-runs for a `/capacity` permit any more, and the pool no longer peeks at the visual chain before
loading the 27B.

- **world-model** wraps every CUDA forward pass in a short pool lease (class `world`, priority
  `system`, 2 s deadline). Late or refused -> the same `gpu_contended` answer as before; pool
  unreachable -> `gpu_pool_unreachable`. There is no ungated path.
- **Image generation** (orion-thought): the reverie-visual run's diffusion hold is the grant; the
  generate step attaches a child lease under it for the diffusion call (no second wait) and keeps it
  until the diffusion thread exits, so world stays off gpu2 even if the run gives its hold back. A generate outside a durable run (the `/visual-chain/run-once`
  route, the legacy worker) takes its own pool `diffusion` lease. The gpu-lane-controller slot-status
  pre-check is gone.
- **Pool**: the `visual_baseline` swap guard, `GPU_POOL_VISUAL_ACTIVITY_URL` and the image's copy of
  the baseline policy are deleted. The 27B load is guarded by `thermal` only.
- **Proof**: an end-to-end test (real client, real pool, real `config/gpu_pool.yaml`, in-memory and
  real Postgres) shows world and diffusion never hold gpu2 together, in both directions; a repo gate
  fails if anything calls `/capacity` again; the pool day eval fails on any second of overlap.
- **Carried fix**: `resource_deferred` row in the analytics outcome dimension, plus a test that every
  visual terminal reason has a row.

## Outcome moved

- Image generation asks for gpu2 once instead of twice (hold + permit). World-model's mutex with
  diffusion is now enforced by the same queue as every other GPU user, and is visible: a blocked world
  lease emits `queued reason=serialized:diffusion`.
- After deploy, `durable_gateway_permits` should get **zero** new rows. Baseline (live, read-only,
  2026-09-29): 95 `diffusion` permits in 7 days, last at 21:29:19 UTC; 3 `world-model` permits ever.
- 5.6 can now delete the broker: it has no callers (gate below).

## Current architecture

- world-model: `GpuCapacityPermit` against durable-runs `:8124/capacity`, backend key
  `http://100.112.254.99:8014`, 2 s budget, `gpu_contended` on refusal.
- thought: `generate_visual_bytes` took the same permit (180 s budget) on top of the durable run's pool
  diffusion hold, plus (live: `ORION_VISUAL_ELASTIC_STATUS_ENABLED=true` in the container) a
  `GET circe:8090/v1/gpu-slots/circe-gpu2/status` pre-check.
- pool: `agent-gpu2` swap guards `[thermal, visual_baseline]`; `visual_baseline` read thought's
  `/visual-chain/activity` every 30 s (fired 5 times, last 2026-09-27).
- `serialize_with: [diffusion]` on `world` shipped dormant in 5.1 (no world leases existed; the only
  50 `world` rows in `gpu_pool_leases` are a 2026-09-25 `probe:latency-test`).

## Architecture touched

- Services: orion-world-model, orion-thought, orion-gpu-pool; test fixtures in orion-hub; analytics mart.
- Contracts: `orion/gpu_pool/config.py` `SwapGuard` is now `Literal["thermal"]`. No bus/schema change:
  world-model and thought were already listed as producers/consumers of the lease channels.
- Config: `config/gpu_pool.yaml` (`agent-gpu2` guards, comments). No `launch` block changed, so no
  `launch_digest` moved and circe's controller needs no pull or rebuild for this PR.

## Files changed

- `services/orion-world-model/app/main.py`: lease around the forward pass; `_forward_result` / `_gpu_refused` split.
- `services/orion-world-model/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, `requirements.txt`: `WM_GPU_LEASE_DEADLINE_SEC` replaces `WM_GPU2_CAPACITY_*`.
- `services/orion-world-model/tests/test_service_prediction_gpu_lease.py`: replaces the permit test.
- `services/orion-thought/app/visual_chain.py`: `generate_visual_bytes(..., bus, hold)`; elastic pre-check deleted; unreachable diffusion stays a deferral.
- `services/orion-thought/app/visual_steps.py`: passes the validated hold through; deadline floor uses the lease deadline.
- `services/orion-thought/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `ORION_VISUAL_CHAIN_GPU_LEASE_DEADLINE_SEC` replaces `ORION_VISUAL_CHAIN_GPU2_CAPACITY_*` and `ORION_VISUAL_ELASTIC_*`.
- `services/orion-thought/tests/conftest.py`, `evals/conftest.py`: a granting fake pool replaces the "capacity off" fixture.
- `services/orion-thought/tests/test_visual_chain_gpu_lease.py` (new), `test_visual_chain.py`, `test_visual_steps.py`, `test_visual_steps_db.py`, `test_visual_activity.py`, `test_visual_chain_thermal_gate.py`; `test_visual_chain_gpu2_capacity.py` deleted.
- `orion/schemas/reverie_visual_run.py`: comments only.
- `services/orion-gpu-pool/app/{guards,main,settings}.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`, `README.md`: guard deleted.
- `services/orion-gpu-pool/tests/test_world_diffusion_serialization.py` (new), `test_guards.py`, `test_holds_and_actuation.py`, `test_runtime.py`; `orion/gpu_pool/tests/test_scheduler_{holds,urgent}.py`, `test_stage4_contracts.py`.
- `orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py` (new): the no-caller gate.
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`: `world_diffusion_overlap_sec` target.
- `config/gpu_pool.yaml`, `orion/gpu_pool/{config,scheduler}.py`.
- `services/orion-hub/static/js/gpu_pool.test.js`, `services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py`: guard fixtures.
- `services/orion-analytics/models/marts/dim_reverie_outcomes.sql`, `README.md`; `tests/test_dim_reverie_outcomes_covers_terminal_reasons.py` (new).
- `services/orion-durable-runs/README.md`: `/capacity` has no callers.
- `.github/workflows/orion-gpu-pool-tests.yml` (new `world` job, paths), `.github/workflows/visual-baseline.yml`.

## Schema / bus / API changes

- Added: none.
- Removed: swap guard name `visual_baseline` (config vocabulary, `SwapGuard`).
- Renamed: none.
- Behavior changed:
  - world-model: new `error_code=gpu_pool_unreachable` (bus down / lease RPC unanswered for the full RPC timeout). `gpu_contended` kept for late/refused; its text is `gpu2 contended: <pool reason>`.
  - thought: every generate now runs inside a pool lease (attached under the run's hold, or its own `diffusion` lease with no hold) held until the diffusion thread exits. `resource_deferred:gpu_pool:<reason>` / `resource_deferred:gpu_pool_unreachable...` replace `resource_deferred:gpu2_capacity:...`; `controller_displacement` from the pre-check and `resource_status_unavailable` no longer occur (diffusion-host's own 503 `controller_displacement` still maps).
  - pool: `swap_requested reason=guard:visual_baseline` no longer occurs.
- Compatibility: `/capacity` itself is untouched (5.6 deletes it), so rolling back thought or world-model images works.

## Env/config changes

- Added keys: `WM_GPU_LEASE_DEADLINE_SEC` (world-model), `ORION_VISUAL_CHAIN_GPU_LEASE_DEADLINE_SEC` (thought).
- Removed keys: `WM_GPU2_CAPACITY_{ENABLED,URL,BACKEND_KEY,LANE,BUDGET_SEC,POLL_INTERVAL_SEC}`, `ORION_VISUAL_CHAIN_GPU2_CAPACITY_{ENABLED,URL,BACKEND_KEY,LANE,MAX_INFLIGHT,BUDGET_SEC,POLL_INTERVAL_SEC}`, `ORION_VISUAL_ELASTIC_{STATUS_ENABLED,CONTROLLER_URL}`, `GPU_POOL_VISUAL_ACTIVITY_URL`.
- Renamed keys: none.
- `.env_example` updated: world-model, thought, gpu-pool.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes for orion-thought (`+ORION_VISUAL_CHAIN_GPU_LEASE_DEADLINE_SEC`, then set to the final 90 after review). orion-world-model's `WM_` prefix is outside `SYNC_PREFIXES` and `--all-keys` would also add an unrelated blank `CUDA_VISIBLE_DEVICES`, so `WM_GPU_LEASE_DEADLINE_SEC=2.0` was appended by hand to athena's copy.
- Removed keys left in the local `.env` files on purpose: they are inert for the new code (`extra="ignore"`), and the rollback (previous images) still needs them — notably `ORION_VISUAL_ELASTIC_STATUS_ENABLED=true`, which is live and whose code default is `false`. 5.6 removes them.
- skipped keys requiring operator action: **circe** `services/orion-world-model/.env` needs `WM_GPU_LEASE_DEADLINE_SEC=2.0` (world-model runs on circe; the sync script writes only the host it runs on).

## Tests run

```text
services/orion-world-model (CPU torch venv): pytest tests -> 62 passed
services/orion-thought: pytest tests -> 502 passed, 28 skipped (no PG), 1 failed (test_settings_mind_enrichment::test_mind_enrichment_defaults_off:
  reads the primary checkout's local .env, 'http://mind:6611' vs 'http://orion-mind:6611'; unrelated, env-sensitive)
visual-baseline CI thought step on throwaway postgres:16 (ORION_VISUAL_TEST_DATABASE_URL): 224 passed
tests/test_visual_baseline_pipeline.py + tests/test_dim_reverie_outcomes_covers_terminal_reasons.py: 14 passed
orion/gpu_pool/tests: 293 passed (after merging origin/main incl. 5.1 #2408 and 5.2 #2409)
services/orion-gpu-lane-controller (5.2, merged in): 109 passed
services/orion-gpu-pool (GPU_POOL_TEST_POSTGRES_URI = throwaway postgres:16, free port): 102 passed
  (includes both test_world_diffusion_serialization tests, [memory] and [postgres])
mutation check: removing `serialize_with: [diffusion]` from gpu_pool.yaml -> the e2e test fails
  ("world must not be granted while diffusion is held")
services/orion-hub: node --test static/js/gpu_pool.test.js -> 15 passed; browser smoke skipped (no playwright here)
scripts/check_gpu_pool_config.py: ok (4 cards, 8 roles, 7 classes, 2 launch blocks, digest unchanged 50d8ba8221ae7f20)
check_definition_drift.py: 0 changed; metric gate: no metric added or changed (no locked metric reads gpu_contended,
  resource_deferred, swap guards or durable_gateway_permits)
env template parity: PASS; env key single source: OK; bus reply channels: 0 uncovered; git diff --check: clean
```

## Evals run

```text
services/orion-gpu-pool/evals/run_pool_day_eval.py -> VERDICT: PASS
  world_diffusion_grant_overlap_sec: 0; serialized:diffusion 8839, serialized:world 18; grant:world->world 2402
services/orion-thought/evals (visual chain honesty eval) -> 5 passed
```

## Docker/build/smoke checks

```text
Not run: do-not-deploy task. No image built.
Live read-only baselines (docker exec orion-athena-sql-db psql):
  durable_gateway_permits last 7 d: diffusion 95 (last 2026-09-29 21:29:19), world-model 3 (last 2026-09-24)
  gpu_pool_leases: diffusion holds 22 (last 21:29:19); world requests 50, all probe:latency-test 2026-09-25
  gpu_pool_events guard blocks: guard:visual_baseline 5 (last 09-27), guard:thermal 3
  live thought container: ORION_VISUAL_ELASTIC_STATUS_ENABLED=true (the spec said unset/false)
The spec's 20-minute concurrent world+diffusion CUDA check: UNVERIFIED, not run (it puts real load on
gpu2 in the room; needs Juniper's go). Follow-up evidence, not a gate for this PR.
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent) against `a949a60cc..HEAD`.

- Finding (blocker): diffusion could keep rendering on gpu2 after its pool grant was gone. The durable run releases its hold on a step deadline / recall while the orphaned generate task (and its un-killable thread) keeps going; the old permit lived as long as that task, the hold does not. Same on run-once when the run deadline cancels the call.
  - Fix: the generate now **attaches a child diffusion lease under the hold** (the pool's own "call under a hold" verb: runs in the hold's slot, jumps its queue, so still no second wait) and `_diffusion_call` keeps that lease until the diffusion thread actually exits, even after a cancel (bounded: one more diffusion timeout + 10 s). Verified in the scheduler that a granted child survives its hold's release (only queued/retry_wait children are ended).
  - Evidence: `test_a_render_outlives_its_released_hold_and_still_keeps_world_off[memory|postgres]` (real client + real pool); `test_cancelled_caller_keeps_the_lease_until_the_diffusion_thread_exits`; `test_generate_attaches_under_the_runs_hold`.
- Finding (should-fix): world-model released its lease on `WM_TIMEOUT_S` while the forward-pass thread still ran.
  - Fix: on the card it keeps the lease up to one more `WM_TIMEOUT_S`, and releases as `upstream_error` for timeouts too.
  - Evidence: `test_timed_out_forward_keeps_the_lease_while_it_still_runs`.
- Finding (should-fix): with a 2 s deadline every `PoolRpcTimeout` is the shortened kind (`full=False`), which the client says is not "pool down".
  - Fix: `full=False` -> `gpu_contended`; only `full=True` -> `gpu_pool_unreachable`.
  - Evidence: `test_short_pool_rpc_timeout_is_contention_not_a_dead_pool`.
- Finding (should-fix): `WM_GPU_LEASE_DEADLINE_SEC=${WM_GPU_LEASE_DEADLINE_SEC}` would crash world-model at boot on a circe `.env` without the key.
  - Fix: `${WM_GPU_LEASE_DEADLINE_SEC:-2.0}`.
- Finding (should-fix): narrowing `SwapGuard` makes a YAML-only revert fail to load (`config_unloadable`).
  - Fix: documented rather than kept as an accepted-but-ignored name -- the scheduler fails closed on any listed guard nobody reads, so accepting `visual_baseline` would block every 27B load forever. Rollback for 5.4 is image-only (README, stage 5.4 section).
- Finding (nit): run-once budget cannot fit (lease 180 + diffusion 120 >= run deadline 300).
  - Fix: `ORION_VISUAL_CHAIN_GPU_LEASE_DEADLINE_SEC` default 180 -> 90.
- Finding (nit): eval overlap metric measures grants, not the card.
  - Fix: renamed `world_diffusion_grant_overlap_sec`, docstring says so.
- Finding (nit): spec says the elastic pre-check was off; live it is on. / spec says hold rejection is `resource_deferred`; code says `hold_invalid` retry.
  - Fix: both recorded under "Spec problems" below; the unreachable-diffusion deferral that the live flag gave is kept.
- Finding (nit): report uncommitted. Fix: committed with this PR.

## Restart required

Deploy order (spec): pool first, then thought, then world-model on circe. No circe pull or controller rebuild is needed for the pool/YAML part (no `launch` block or `launch_digest` change). Rollback is image-only: never revert just `config/gpu_pool.yaml` (5.4 code refuses the old `visual_baseline` guard name).

```bash
# athena, from a worktree/checkout at the merged commit
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
# circe: pull to the merged commit, add WM_GPU_LEASE_DEADLINE_SEC=2.0 to services/orion-world-model/.env, then
scripts/safe_docker_build.sh orion-world-model up -d --build
```

Then check `durable_gateway_permits` hourly for 24 h (target 0 new rows) and that reverie-visual images keep landing.
circe's gpu-lane-controller: no pull/rebuild needed (no `launch` change, `launch_digest` unchanged).

## Risks / concerns

- Severity: medium. Concern: world-model's README says **PARKED -- do not deploy**, yet the spec's deploy order redeploys it on circe, and a container is running there with the old permit code. Mitigation: nothing calls it today (24 h of `/health` only). Juniper decides: redeploy it (the lease path) or stop it; either way it must not still be running old code after 5.6 deletes `/capacity`.
- Severity: low. Concern: a generate with no hold (run-once route / legacy worker) now takes a pool `diffusion` lease, which can trigger an owner reclaim of a loaded 27B. Mitigation: the same thing a reverie-visual hold does today; the route is a rollback path and the legacy worker is off.
- Severity: low. Concern: a diffusion or forward-pass thread that hangs past its bounded grace (one more diffusion timeout / one more `WM_TIMEOUT_S`) still has its lease released, so a truly wedged render could overlap world. Mitigation: same bound the old code had (the permit was released on cancel); logged at error level.
- Severity: low. Concern: attach adds one pool RPC to the durable generate; a pool outage now defers the render (`resource_deferred:gpu_pool_unreachable`) where before only the permit broker's outage did. Mitigation: the hold itself already needs the pool.
- Severity: info. Concern: the eval checks grant overlap only; physical overlap rests on callers holding leases for the GPU work's life (pinned by the unit tests above) and is otherwise UNVERIFIED live.

## Spec problems found

1. **Elastic pre-check was live, not off.** The spec says `ORION_VISUAL_ELASTIC_STATUS_ENABLED` is unset (false). The running `orion-athena-thought` has it `=true`, so deleting the controller slot-status pre-check is a live behaviour change. Its "unreachable diffusion -> deferral" half is kept unconditionally.
2. **Hold rejection is not `resource_deferred`.** Decision 1 says `resource_deferred` still fires when `validate_hold_ref` rejects the hold; `generate_step` returns `retry("hold_invalid:...")`, which ends as `retry_window_expired`. Acceptance check 6's "resource_deferred per day" therefore measures pool refusals and diffusion-host busy/unreachable only.
3. **"The hold is the grant, no second gate" is not safe as written.** The run gives its hold back on step deadline/recall while the render thread keeps going; with no lease of its own the card is free for world mid-render. This PR attaches a child lease (no second wait) instead.
4. **world-model is PARKED -- do not deploy** (its README), yet the spec's 5.4 deploy order redeploys it on circe, and an old-image container is running there. Juniper decides: redeploy or stop it.
5. **The 20-minute concurrent-CUDA check** needs real gpu2 load in the room; not run here (UNVERIFIED), left for Juniper's go.


## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2414

🤖 Generated with [Claude Code](https://claude.com/claude-code)
