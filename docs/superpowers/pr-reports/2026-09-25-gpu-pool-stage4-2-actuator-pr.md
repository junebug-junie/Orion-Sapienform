# GPU pool stage 4.2 — circe actuator bridge

**Stacked on #2349** (base branch `feat/gpu-pool-stage4-1-contracts`). Merge #2349 first; this PR's
diff is only the 4.2 slice.

## Summary

The circe controller that physically swaps gpu2 between diffusion and the second 27B can now take
that order from the GPU pool over the bus, not only from durable-runs over HTTP. It runs exactly the
same drain / stop / start / readiness-wait / rollback steps as today. Nothing changes until an
operator flips one switch.

- **New bus consumer** (`app/actuator_bus.py`). It listens on `orion:gpu_pool:actuate:request` and
  answers on `orion:gpu_pool:actuate:result`. The answer sequence is `accepted`, then one `progress`
  per step (`draining`, `stopping`, `starting`, `ready_wait`, `rolling_back`), then one final
  `succeeded`, `failed` or `refused`. The final answer lists which containers are running afterwards.
  - `agent-gpu2 load` runs `gpu2.transition(target="agent-burst")`.
  - `agent-gpu2 unload` runs `transition(target="diffusion")`.
  - The mapping goes through the role's `swap.load`/`swap.unload` bridge verbs in the controller's
    own `config/gpu_pool.yaml`.
- **One switch, `GPU2_AUTHORITY`.**
  - `durable` (the default) is today's behaviour: durable-runs' HTTP route, checked against
    durable-runs `/elastic/status`. Bus requests get an explicit `refused authority_durable`.
  - `pool` accepts only bus requests. The HTTP gpu2 activate route then answers `503 authority_pool`.
- **New fence** (`app/pool_fence.py`). It replaces the call back to durable-runs `/elastic/status`.
  Under `pool`:
  - the controller writes the pool's generation to disk before it touches any container, and refuses
    any generation at or below it;
  - it refuses a `launch_digest` that differs from its own checkout's.
  - `transition()`'s existing checkpoints re-check both. Rollback checks only identity and
    generation, so it can always put diffusion back.
- **Refusals, idempotency, `status`, restart.**
  - Every request addressed to this controller gets a named refusal reason.
  - A replayed `action_id` gets its recorded result back instead of starting a second transition.
  - `status` is a read-only reconcile answer.
  - An action cut off by a controller restart is recorded as `failed` (and `restored=false` if it was
    a load). It is never re-run.
- **Config.**
  - `.env_example`, settings and compose gain `GPU2_AUTHORITY`, `GPU_POOL_ACTUATOR_NAME` and
    `GPU2_POOL_FENCE_STATE_PATH`.
  - A new pinned named volume `orion-gpu-lane-controller-state` holds the fence file.
  - `PyYAML` is now declared, and the README documents all of this.

## Outcome moved

- Stage 4.5 can now make the pool the only decider for gpu2 with an env flip on circe instead of a
  code change. The step that loads the model (Option A) is unchanged code.
- The pool (4.3) gets a tested counterpart before it sends anything, with these guarantees:
  - a stale or replayed request cannot touch a container, even across a controller restart;
  - a checkout mismatch between athena and circe refuses instead of starting the wrong thing;
  - every addressed request gets an answer, so nothing is left to time out.

## Current architecture

Before this PR, durable-runs decided when gpu2 should swap. It called
`POST :8090/v1/gpu-slots/activate` on `orion-gpu-lane-controller` (circe). The controller's
`gpu2.transition()` called back `GPU2_AUTHORITY_URL/elastic/status` at every checkpoint to confirm
the intent. The controller's only bus use was a heartbeat (`HeartbeatOnly`), and it had no writable
volume (the repo is mounted read-only at `/repo`). Live on circe (read-only check, 2026-09-25):
`orion-circe-gpu-lane-controller` had been up 4 days with `GPU2_ENABLED=true`, and the checkout was
at `e5ef19e37` (main, without 4.1's launch blocks).

## Architecture touched

- The `orion-gpu-lane-controller` service only.
- The bus contract is 4.1's, unchanged: `GpuActuateV1` / `GpuActuateResultV1` on
  `orion:gpu_pool:actuate:request` / `:result`, where the catalog already names this service as the
  consumer and producer.
- It reads `orion.gpu_pool.config.launch_digest` / `load_pool_config` from the controller's own
  `/repo` mount.
- Its chassis changed. With the bus enabled, the actuator `Hunter` is the heartbeat source.
  `HeartbeatOnly` now starts only as a fallback if the Hunter fails, so there is still one heartbeat
  stream and one bus connection for heartbeats.

## Files changed

- `services/orion-gpu-lane-controller/app/actuator_bus.py` (new): the bus handler, the refusals,
  admission (serialized under a lock), the background transition, the result sequence, replay and
  `status`.
- `services/orion-gpu-lane-controller/app/pool_fence.py` (new): maps a role and action to a bridge
  target; persists the fence (fsync plus atomic rename); pool-mode `authority()`; recovery of an
  action interrupted by a restart.
- `services/orion-gpu-lane-controller/app/gpu2.py`:
  - `authority()` hands off to the pool fence under `pool`;
  - `status()` skips the durable callback under `pool`;
  - `flip()` refuses under `pool`;
  - `phase()` progress hooks via a contextvar (does nothing on the HTTP path);
  - the first fence failure keeps its reason under `pool`.
- `services/orion-gpu-lane-controller/app/main.py`: actuator Hunter wiring, a single heartbeat
  chassis, and interrupted-action recovery at boot.
- `services/orion-gpu-lane-controller/app/settings.py`: three new keys.
- `services/orion-gpu-lane-controller/.env_example`, `docker-compose.yml`: the keys, plus the state
  volume (pinned name).
- `services/orion-gpu-lane-controller/requirements.txt`: `PyYAML` (imported via
  `orion.gpu_pool.config`).
- `services/orion-gpu-lane-controller/README.md`: a stage 4.2 section covering the authority table,
  the result sequence, refusals, fence and ops.
- `services/orion-gpu-lane-controller/tests/test_actuator_bus.py` (new): 43 tests.
- `orion/schemas/gpu_pool.py`: 4.3's version of the file, taken as-is (status-only
  `in_flight`/`last_action_id` on `GpuActuateResultV1`, plus 4.3's other additive fields), so
  `_status` can set structured state. It is identical to #2352's file, so there is no conflict when
  the stack merges.
- `.github/workflows/gpu2-elastic-tests.yml`: also triggers on `orion/schemas/gpu_pool.py`,
  `orion/gpu_pool/config.py` and `config/gpu_pool.yaml`.

## Schema / bus / API changes

- Added: none to the contracts (these are 4.1's). This is the first producer on
  `orion:gpu_pool:actuate:result` and the first consumer on `:actuate:request`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Under `GPU2_AUTHORITY=pool` only: `POST /v1/gpu-slots/activate` for `circe-gpu2` returns
    `503 {"status":"refused","error":"authority_pool"}`, and `GET /v1/gpu-slots/circe-gpu2/status`
    no longer calls durable-runs.
  - GPU1 is untouched.
- Compatibility notes:
  - `status` replies carry the reconcile state in the structured, status-only fields
    `in_flight: bool` and `last_action_id: str | None`. These were added to `GpuActuateResultV1` by
    4.3 (#2352); this branch carries **only** 4.3's `orion/schemas/gpu_pool.py`, byte-identical, so
    the stack merges without conflict. `reason` is a human summary only. When no action is running,
    the last recorded result is re-published first under its own `action_id`.
  - Deploy order: `in_flight`/`last_action_id` are rejected by a pre-4.3 pool (`extra="forbid"`).
    Only a 4.3 pool sends `status` at all, so in practice the order is fixed by who asks. Still,
    deploy this controller after the 4.3 pool, as that schema's consumer-first note says.
  - On a failed load, `restored=None` means no rollback ran because nothing had been evicted yet.
    `observed` shows the containers.

## Env/config changes

- Added keys: `GPU2_AUTHORITY=durable`, `GPU_POOL_ACTUATOR_NAME=circe`,
  `GPU2_POOL_FENCE_STATE_PATH=/state/gpu2_pool_fence.json`.
- Removed keys: none (`GPU2_AUTHORITY_URL` stays until 4.6).
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran. There was nothing
  to write, because this is a circe-only service and athena's primary checkout has no
  `services/orion-gpu-lane-controller/.env`. **circe's `.env` needs the three lines below. I did not
  edit it.**

  ```text
  GPU2_AUTHORITY=durable
  GPU_POOL_ACTUATOR_NAME=circe
  GPU2_POOL_FENCE_STATE_PATH=/state/gpu2_pool_fence.json
  ```

  Compose has `${…:-default}` fallbacks for all three, so a circe `.env` without them still gets
  exactly these values.
- Skipped keys requiring operator action: the three circe lines above.

## Tests run

```text
.venv/bin/python -m pytest services/orion-gpu-lane-controller/tests -q     -> 83 passed
  (43 pre-existing + 43 new in test_actuator_bus.py)
.venv/bin/python -m pytest services/orion-gpu-lane-controller/tests orion/gpu_pool/tests -q
  -> 218 passed (4.1 contract tests still green on the 4.3 schema file)
Mutation spot-checks (each reverted): removing the authority_durable gate, `<=` -> `<` on the
  generation fence, dropping the deadline check, dropping the HTTP authority_pool refusal, dropping
  the rolling_back phase -> each fails exactly 1 test.
Review regressions: the 8 tests added for review findings all FAIL on the pre-fix commit (52b81594e)
  and pass after.
python scripts/check_gpu_pool_config.py          -> ok (2 launch blocks, digest 0547e17a0751c728)
python scripts/check_env_template_parity.py      -> PASS
python scripts/check_service_env_compose_parity.py orion-gpu-lane-controller -> N/A (env_file: carries all keys)
python scripts/check_compose_no_relative_mounts.py -> PASS
check_async_routes_not_blocking / check_service_hostname_refs / check_system_health_producers /
  check_metric_lineage / check_definition_drift / check_scripts_dir_no_stdlib_shadow -> rc=0
git diff --check -> clean
```

## Evals run

```text
None. The controller has no eval harness (services/orion-gpu-lane-controller/evals/ does not exist).
The behaviour here is a deterministic contract (fence, refusals, sequence) and is fully covered by
the gate tests above. A live end-to-end check belongs to the 4.3/4.5 acceptance checks 4, 6 and 7
(pool emits swap_started -> swapped; kill the pool mid-load -> status adopt; forced readiness
timeout -> swap_failed restored=true), which need the 4.3 producer.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-gpu-lane-controller/Dockerfile .   (athena, throwaway tag, removed)
docker run ... python -c "import app.main, yaml; from app import pool_fence, actuator_bus"
  -> import ok durable /state/gpu2_pool_fence.json 6.0.3
docker compose -f services/orion-gpu-lane-controller/docker-compose.yml config
  -> GPU2_AUTHORITY: durable, GPU_POOL_ACTUATOR_NAME: circe, GPU2_POOL_FENCE_STATE_PATH set,
     volume gpu-lane-controller-state -> /state
Not deployed. The live bus path is UNVERIFIED: nothing produces GpuActuateV1 until 4.3.
```

## Review findings fixed

The code review subagent found no blockers. It raised 6 "should" findings and several nits, and all
are fixed.

- Finding: at the default setting, the service now ran two heartbeat streams (the Hunter also
  heartbeats) and opened two bus connections.
  - Fix: the actuator Hunter is the heartbeat source. `HeartbeatOnly` is only the fallback if the
    Hunter fails to start.
  - Evidence: `test_lifespan_runs_one_heartbeat_chassis`.
- Finding: if publishing `accepted` failed, the action was left marked in flight and never ran, even
  though its generation was already used up.
  - Fix: the task is created first, and a failed ack is logged and does not stop the run.
  - Evidence: `test_accepted_publish_failure_still_runs_and_clears_in_flight`.
- Finding: under `pool`, rollback re-checked the digest, so a `git pull` on circe during a load would
  have left gpu2 empty.
  - Fix: `require_drained=False` (the pre-flight and rollback checks) checks identity and generation
    only.
  - Evidence: `test_rollback_not_blocked_by_checkout_edited_mid_load` (forward progress stops,
    `restored=true`).
- Finding: an interrupted load was reported as `restored=None`, which reads as "nothing evicted".
  - Fix: it is now `restored=false`, so the pool marks the card `fault`.
  - Evidence: `test_restart_mid_action_is_recorded_as_interrupted`.
- Finding: an admission could write an older fence snapshot back over a result that was recorded
  while it waited, which erased that result.
  - Fix: the fence is re-read after the busy check, just before it is written.
  - Evidence: `test_finished_result_not_erased_by_concurrent_admission`.
- Finding: `status` held the admit lock across up to 60s of `docker compose ps`, which could starve a
  load's ack past `actuate_ack_sec`.
  - Fix: the status snapshot is copied under the lock, and observing and publishing happen outside
    it.
  - Evidence: `test_status_observes_outside_admit_lock`.
- Finding: a `deadline_at` without a timezone crashed the handler and published nothing.
  - Fix: it is refused with `invalid_request:deadline_at_naive`.
  - Evidence: `test_naive_deadline_refused_not_crashed`.
- Nits fixed:
  - invalid requests get an answer only when explicitly addressed to this actuator (test added);
  - a recorded row that no longer validates falls back instead of raising;
  - the progress hook is bounded to 5s;
  - under `pool` the first fence failure keeps its own reason;
  - the volume name is pinned;
  - the README says how to inspect or clear a corrupt fence file.
- Nit not fixed: the result for an interrupted action has an empty `observed`. The pool's `status`
  call (which it sends on restart and whenever a result is overdue) returns live `observed`, so this
  was left as it is.

## Restart required

Deploy nothing yet. When Juniper decides to roll this out, which is safe at any time because the
default is a no-op, run this on **circe**, after #2349 and this PR are merged to main:

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
# add the three lines above to services/orion-gpu-lane-controller/.env (GPU2_AUTHORITY=durable)
docker compose \
  --env-file .env \
  --env-file services/orion-gpu-lane-controller/.env \
  -f services/orion-gpu-lane-controller/docker-compose.yml \
  up -d --build gpu-lane-controller
curl -fsS http://localhost:8090/health
docker logs --tail=50 orion-circe-gpu-lane-controller | grep -E "gpu_pool_actuator_started authority=durable|Hunter subscribing"
```

The flip to `GPU2_AUTHORITY=pool` is **not** part of this PR. It is step (3) of the 4.5 cutover
runbook and needs a restart of the same container.

## Risks / concerns

- Severity: medium (for 4.5, not 4.2).
  - Concern: under `pool` the controller no longer re-checks thermal, visual-baseline urgency or the
    closure of durable leases and permits on gpu2. By design those move to pool policy (4.3 guards
    and recall). Flipping `GPU2_AUTHORITY=pool` before the 4.3 guards are live would remove those
    checks. Drain-before-stop and upstream-idle-before-stop still apply here.
  - Mitigation: the 4.5 runbook order (the pool is armed last); the README says to flip only in that
    step.
- Severity: medium.
  - Concern: `launch_digest` is computed by the image's baked-in `orion/gpu_pool/config.py` over the
    `/repo` checkout's YAML. If circe's checkout and its image drift apart (pulled without a rebuild,
    or the reverse), or athena and circe run different parser versions, every request is refused
    `launch_digest_mismatch`.
  - Mitigation: this fails safe (a refusal, never a wrong start). Rebuild on pull. Circe must pull
    the 4.1 YAML, because today's checkout has no launch blocks and every request would be refused
    `role_not_on_this_actuator`.
- Severity: low.
  - Concern: `docker compose down -v` deletes the fence volume and resets the accepted generation to
    0, so an old replayed generation could be admitted.
  - Mitigation: the volume name is pinned and the README warns about it. The pool's generations only
    increase, and it does not replay old action ids.
- Follow-up (coordinator, after 4.3 landed): `status` now sets `in_flight` and `last_action_id`
  and keeps `reason` human-readable only.
  - Evidence: `test_status_republishes_last_result_then_observed`,
    `test_status_reports_in_flight_structurally`, `test_status_on_fresh_controller_says_nothing_ran`,
    `test_non_status_results_never_carry_status_fields`.

- Severity: low.
  - Concern: `restored=None` on a failed load (nothing evicted before the failure) is a third value
    that the spec's pool state machine does not name.
  - Mitigation: documented here and in the README; 4.3 should treat None as "read `observed`".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2350

🤖 Generated with [Claude Code](https://claude.com/claude-code)
