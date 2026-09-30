# GPU pool stage 5.6: delete the dead paths

## Summary

- The old GPU permit broker is gone: durable-runs no longer serves `/capacity/*`, and its store, client,
  schemas, eval, tests and env keys are deleted. Nothing has called it since 5.4 (0 permits granted
  since 2026-09-29 22:45 UTC, checked live).
- The durable-run registry store (the live table Hub's curiosity views read through durable-runs) is
  **moved, not deleted**: `orion/durable_admission/store.py` -> `orion/durable_runs/registry_store.py`,
  class `DurableRunRegistryStore`. The `orion/durable_admission` package is gone.
- circe's GPU controller loses the stage-4 gpu2 bridge (fixed targets with a literal
  `CUDA_VISIBLE_DEVICES "2"`), the GPU1 affect/agent flip, every HTTP route except `/health`, and the
  `GPU2_*` / `GPU_LANE_CONTROLLER_TOKEN` keys. The generic launch executor is the only way a card moves.
- `swap.load` / `swap.unload` are no longer accepted in `config/gpu_pool.yaml` (the validator refuses
  them). `orion/schemas/gpu_slot.py` is deleted.
- Ships, but does **not run**, the drop of the four dead tables: a migration file plus
  `scripts/gpu_pool_stage5_snapshot_and_drop.sh`, which snapshots first and refuses on any late write.
  **[GO] needed from Juniper** (commands below).

## Outcome moved

- Two dead authorities over gpu2 (the permit broker and the bridge) no longer exist, so nothing can
  quietly start using them again. The no-caller gate's allow-list is now empty and it also fails on the
  old package, route strings, schema names and env keys.
- One fewer hard-coded GPU index: the only way the controller picks a card is the card `index` in the YAML.

## Current architecture

- Since 5.4 (live 2026-09-29 22:48 UTC), world-model takes a pool `world` lease and the visual chain runs
  under its run's diffusion hold. The `/capacity` broker in durable-runs was still up
  (`DURABLE_RUNS_CAPACITY_ENABLED=true` in the live container) with no callers.
- Since 5.3, `agent-gpu2` ran through the controller's generic `launch_exec`; the bridge (`gpu2.py`)
  stayed only as the rollback. The GPU1 flip (`lane_control.py`, token-guarded HTTP) had no caller in
  any service.
- Four tables (`durable_gateway_permits` 168,981 rows / 127 MB, `durable_resource_demands` 158,
  `durable_resource_leases` 225, `durable_elastic_slot` 1) were frozen: newest writes 2026-09-29 21:29
  (permits) and 2026-09-26 (the other three), nothing active or pending.

## Architecture touched

- Services: orion-durable-runs, orion-gpu-lane-controller (circe), orion-gpu-pool (tests + a docstring).
- Shared code: `orion/durable_runs/` (new home of the store), `orion/gpu_pool/config.py`,
  `orion/schemas/resource_admission.py`, `orion/schema_skew_discovery.py`.
- Contracts: HTTP routes removed (below). No bus channel, no registered schema changed.
- Config: `config/gpu_pool.yaml` comments only (parsed model and every `launch_digest` unchanged).

## Files changed

(see the "Files changed" list in the PR diff; highlights)

- `orion/durable_runs/registry_store.py`: moved store, `_expire` deleted, `setup()` applies the drop migration so test schemas match production.
- `services/orion-durable-runs/app/{main,settings,admission_runtime}.py`, `.env_example`, `docker-compose.yml`, `README.md`: broker, routes and keys removed; importers updated.
- `services/orion-gpu-lane-controller/app/compose.py` (new): the docker primitives `launch_exec` needs, kept from `lane_control.py`.
- `services/orion-gpu-lane-controller/app/{actuator_bus,pool_fence,launch_exec,main,settings}.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`, `README.md`: bridge, flip, routes, keys removed; two keys renamed.
- `orion/gpu_pool/config.py`: `SwapSpec.load/unload/bridged` and the bridge validator rules removed; digest body kept.
- `services/orion-sql-db/manual_migration_gpu_pool_stage5_drop_legacy_tables.sql`, `scripts/gpu_pool_stage5_snapshot_and_drop.sh` (new).
- `.github/workflows/gpu2-elastic-tests.yml` -> `gpu-lane-controller-tests.yml` (still the only CI for the controller and diffusion-host suites); `orion-durable-runs-tests.yml` loses the capacity eval step.
- Deleted: `orion/durable_admission/{__init__,capacity,capacity_client}.py`, `orion/schemas/gpu_slot.py`, `orion/tests/test_capacity_client.py`, `services/orion-durable-runs/{evals/gateway_capacity.py,tests/test_capacity_postgres.py}`, controller `app/{gpu2,lane_control}.py`, `tests/{test_gpu2,test_lane_control}.py`, `tests/cutover_config.py`.

## Schema / bus / API changes

- Added: none.
- Removed:
  - HTTP (durable-runs): `POST /capacity/{acquire,renew,release}`, `GET /capacity`; `/health` loses `capacity_enabled`.
  - HTTP (controller): `POST /v1/gpu-lane/flip`, `GET /v1/gpu-lane/status`, `GET /v1/gpu-slots/{slot}/status`.
  - Schemas: `CapacityAcquireV1`, `CapacityTokenV1`, `CapacityPermitV1`, `Capacity{Acquire,Renew,Release}ResultV1`, `GpuSlotRequestV1`. None were in either registry (`resolve()` raised before and after).
  - Config vocabulary: `swap.load` / `swap.unload` (now a validation error).
  - Controller refusals: `gpu2_disabled`, `bridge_verb_unsupported`, `bridge_cannot_set_profile`.
- Renamed: `PostgresAdmissionStore` -> `DurableRunRegistryStore` (module moved).
- Behavior changed:
  - The controller has no enable switch any more: a role is actuated iff it has a `launch` block for
    this actuator (spec Env section: `GPU2_ENABLED` -> nothing). A role without one is refused
    `role_not_on_this_actuator`.
  - If circe's checkout YAML can't be parsed, `observed` is now `{}` instead of the fixed gpu2 pair.
- Compatibility: `launch_digest` keeps its always-null `load`/`unload` in the hashed body, so every
  digest is byte-identical to main's (checked for all 8 roles against origin/main's code, and pinned by
  `test_stage5_6_digest_is_unchanged_by_the_bridge_removal`). A 5.5 pool and a 5.6 controller agree;
  no lockstep deploy.

## Env/config changes

- Removed keys: `DURABLE_RUNS_CAPACITY_ENABLED`, `DURABLE_RUNS_LEASE_SECONDS` (durable-runs);
  `GPU2_ENABLED`, `GPU2_DIFFUSION_URL`, `GPU2_AGENT_URL`, `GPU2_MODEL_READY_TIMEOUT_SEC`,
  `GPU_LANE_CONTROLLER_TOKEN` (controller; plus the never-templated `GPU_LANE_HEALTH_POLL_SEC`,
  `AFFECT_*`, `AGENT_*` settings).
- Renamed keys: `GPU2_POOL_FENCE_STATE_PATH` -> `GPU_POOL_FENCE_STATE_PATH`, `GPU2_DRAIN_TIMEOUT_SEC` ->
  `GPU_LANE_DRAIN_TIMEOUT_SEC`. Defaults equal circe's live values (`/state/gpu2_pool_fence.json`, 300),
  so the fence file and its last generation carry over with no `.env` edit.
- `.env_example` updated: orion-durable-runs, orion-gpu-lane-controller.
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-durable-runs orion-gpu-lane-controller`:
  yes (athena: `+GPU_POOL_ACTUATOR_NAME`, `+GPU_POOL_FENCE_STATE_PATH`; `GPU_LANE_DRAIN_TIMEOUT_SEC=300`
  appended by hand, its prefix is outside the script's default set).
- Skipped keys requiring operator action: the dead keys below (the sync script only adds keys).

### Dead keys still in live `.env` files (delete by hand; read-only scan 2026-09-30)

athena (`/mnt/scripts/Orion-Sapienform`):

- `services/orion-durable-runs/.env`: `DURABLE_RUNS_CAPACITY_ENABLED`, `DURABLE_RUNS_LEASE_SECONDS`
- `services/orion-gpu-lane-controller/.env`: `GPU2_AGENT_URL`, `GPU2_AUTHORITY_URL`, `GPU2_DIFFUSION_URL`, `GPU2_DRAIN_TIMEOUT_SEC`, `GPU2_ENABLED`, `GPU2_MODEL_READY_TIMEOUT_SEC`, `GPU_LANE_CONTROLLER_TOKEN`
- `services/orion-thought/.env`: `ORION_VISUAL_CHAIN_GPU2_CAPACITY_{BACKEND_KEY,BUDGET_SEC,ENABLED,LANE,MAX_INFLIGHT,POLL_INTERVAL_SEC,URL}`, `ORION_VISUAL_ELASTIC_CONTROLLER_URL`, `ORION_VISUAL_ELASTIC_STATUS_ENABLED`
- `services/orion-world-model/.env`: `WM_GPU2_CAPACITY_{BACKEND_KEY,BUDGET_SEC,ENABLED,LANE,POLL_INTERVAL_SEC,URL}`
- `services/orion-gpu-pool/.env`: `GPU_POOL_VISUAL_ACTIVITY_URL`
- `services/orion-hub/.env`: `GPU_LANE_MAP_ATHENA_JSON`, `GPU_LANE_MAP_CIRCE_JSON`
- `services/orion-diffusion-host/.env`: `DIFFUSION_POWER_INTENT_GPU_INDEX` (also in `.env.bak.20260830T011819Z`)

circe (`/mnt/scripts/Orion-Sapienform`):

- `services/orion-gpu-lane-controller/.env`: `GPU2_AGENT_URL`, `GPU2_AUTHORITY`, `GPU2_AUTHORITY_URL`, `GPU2_DIFFUSION_URL`, `GPU2_DRAIN_TIMEOUT_SEC`, `GPU2_ENABLED`, `GPU2_MODEL_READY_TIMEOUT_SEC`, `GPU2_POOL_FENCE_STATE_PATH`, `GPU_LANE_CONTROLLER_TOKEN`
- `services/orion-world-model/.env`: `WM_GPU2_CAPACITY_{BACKEND_KEY,BUDGET_SEC,ENABLED,LANE,POLL_INTERVAL_SEC,URL}`
- `services/orion-diffusion-host/.env`: `DIFFUSION_POWER_INTENT_GPU_INDEX` (also in three `.env.bak.*` files)

All are inert for the running code (`extra="ignore"` / unread). Delete them after this PR is deployed
(the old images, i.e. the rollback, still read some). Optional on circe:
`python scripts/sync_local_env_from_example.py orion-gpu-lane-controller` then add
`GPU_LANE_DRAIN_TIMEOUT_SEC=300` (both equal the code defaults, so not required).

## Tests run

```text
orion/gpu_pool/tests                                              299 passed
services/orion-gpu-lane-controller/tests                          79 passed
services/orion-gpu-pool/tests (GPU_POOL_TEST_POSTGRES_URI = throwaway postgres:16, port 55493)   107 passed
services/orion-durable-runs/tests (ORION_ADMISSION_TEST_DSN = same throwaway postgres)            256 passed
  incl. new test_stage5_6_drop_script_postgres.py (10: snapshot-only, drop + rerun, late write,
  active row, row inserted between dump and drop, >100k-row line, 4 injection-shaped cutoffs)
  and test_store_setup_refuses_the_public_schema
services/orion-diffusion-host/tests                               54 passed
orion/world_pulse_read/tests + tests/scripts/test_schema_skew_discovery.py   110 passed
orion-static-gates.yml steps (env sync, registry agreement, lineage, drift, ...)  22/22 OK
  (claude.json-mount gate run directly: PASS)
scripts/check_gpu_pool_config.py      ok (8 roles, 2 launch blocks)
scripts/check_env_template_parity.py  PASS; check_service_env_compose_parity.py: controller N/A (env_file), durable-runs OK
Mutation checks:
  drop launch_digest's null load/unload       -> test_stage5_6_digest_is_unchanged_by_the_bridge_removal FAILS
  drop the in-transaction count re-check      -> test_a_row_written_between_dump_and_drop... FAILS
  drop the --cutoff format check              -> 4 injection tests FAIL
  add a durable_elastic_slot query to field-digester -> test_the_broker_package_and_its_tables_left_the_code FAILS
Digest parity vs origin/main code: all 8 roles byte-identical.
```

## Evals run

```text
services/orion-durable-runs/evals/hold_fairness.py      verdict PASS
services/orion-durable-runs/evals/deploy_order_skew.py  verdict PASS
services/orion-gpu-pool/evals/run_pool_day_eval.py      VERDICT: PASS
(evals/gateway_capacity.py deleted with the broker it measured)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-gpu-lane-controller/Dockerfile (throwaway tag, removed)  -> ok
  in-image: import app.main; routes = ['/health']
docker compose config -q: orion-gpu-lane-controller, orion-durable-runs -> ok
drop script on a throwaway DB via docker exec into a scratch postgres:16 container:
  refuse (cutoff before newest write) exit 2; snapshot-only exit 0 (500/1/2/1 dumped = live);
  refuse (active permit) exit 2; --drop exit 0 (4 tables + sequence gone, runs/events kept);
  rerun exit 0 already_dropped; restoring the dump brought back 500/1/2/1 rows.
Live read-only checks (athena postgres, 2026-09-30 02:24 UTC): durable_gateway_permits 0 rows granted
  after 2026-09-29 22:45 (newest 21:29:19, heartbeat 21:30:04); leases/demands/elastic newest
  2026-09-26; 0 active/pending. Deployed thought (athena) and world-model (circe) images carry no
  /capacity or /v1/gpu-slots caller in app code. NOT deployed: nothing in this PR is live.
```

## Review findings fixed

Code review ran in a subagent (full diff). No blockers. Fixed:

- Finding: the drop takes an ACCESS EXCLUSIVE lock on the live registry `durable_admission_runs`
  (dropping the FK tables removes its triggers), queueing Hub/durable-runs behind it for up to 30 s.
  - Fix: the script locks `durable_admission_runs` up front with the legacy tables and `lock_timeout=3s`
    (refuses instead of queueing); documented in the header.
  - Evidence: drop tests pass; header "LOCKS" section.
- Finding: `--cutoff` was interpolated into SQL (superuser injection).
  - Fix: strict zoned-timestamp regex before any SQL, exit 64.
  - Evidence: 4 parametrized injection/zone-less cases exit 64 with no log written; mutation removing the check fails them.
- Finding: no test for the drop script.
  - Fix: `services/orion-durable-runs/tests/test_stage5_6_drop_script_postgres.py` (fake `docker` shim to real psql/pg_dump), run by the durable-runs CI job (script added to its paths).
  - Evidence: 10 passed; mutation of the in-transaction re-check fails the between-dump test.
- Finding: the 5.3 runbook's YAML-revert rollback now produces a config every component refuses.
  - Fix: "Superseded by 5.6" notes at the top and in its Rollback section.
- Finding: `setup()` now contains a DROP with no guard.
  - Fix: `setup()` refuses when `current_schema()` is `public`, with a test-only docstring.
  - Evidence: `test_store_setup_refuses_the_public_schema`.
- Finding (nits, all fixed): the script now refuses over 100k rows / 100 MB unless `--accept-large-snapshot`
  (s.14); tells the operator to copy the /tmp dump to durable storage; restore note about FKs; an ERR trap
  logs `FAILED` instead of exiting silently; the dead gateway-capacity/gpu2-elastic migrations are marked
  SUPERSEDED and v1 says to apply the drop after it; `observe()` docstring states the pool faults the
  card; README names the remaining emergency stops; `test_api` settings checks use `_env_file=None`;
  the dropped-table gate now scans all of `orion/`, `services/`, `scripts/`, `config/` (.py/.sql/.sh/.yaml/.json).
- Finding (no change): circe `.env` needs no edit for the renamed keys -- its live values equal the new
  code defaults (`/state/gpu2_pool_fence.json`, 300), checked read-only.

## Restart required

Deploy order (nothing here changes behaviour; each step is independent of the others except as noted):

```bash
# 1. athena: durable-runs (drops the /capacity routes; after this nothing can write the four tables)
git -C /mnt/scripts/Orion-Sapienform pull --ff-only
scripts/safe_docker_build.sh orion-durable-runs up -d --build
curl -fsS http://localhost:8124/health    # no capacity_enabled field; admission_enabled true

# 2. circe: the SAME commit, then rebuild the controller (it parses the YAML with code baked into the image)
ssh circe@circe
git -C /mnt/scripts/Orion-Sapienform pull --ff-only
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
curl -fsS http://localhost:8090/health
docker logs orion-circe-gpu-lane-controller 2>&1 | grep -E "gpu_pool_actuator_started|fence_recover"

# 3. athena: the pool picks up the config.py change on its next rebuild; not required (digests unchanged).
#    Probe digest agreement anyway:
ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python \
    scripts/gpu_pool_actuator_probe.py --role agent-gpu2 --check status digest   # expect "OK: launch digests agree"

# 4. [GO] only, after 24 h of zero new permits (spec acceptance 5; 5.4 went live 2026-09-29 22:48 UTC,
#    so not before 2026-09-30 22:48 UTC):
# (169k rows / 127 MB is over the s.14 line; Juniper chose the gzipped snapshot, hence the flag)
scripts/gpu_pool_stage5_snapshot_and_drop.sh --cutoff '2026-09-29 22:45:00+00' --accept-large-snapshot          # snapshot + verify
cp /tmp/gpu-pool-stage5-drop/legacy_tables.sql.gz <durable storage>/                                        # only copy of the rows
scripts/gpu_pool_stage5_snapshot_and_drop.sh --cutoff '2026-09-29 22:45:00+00' --accept-large-snapshot --drop   # then the drop
tail -f /tmp/gpu-pool-stage5-drop/progress.log
cat /tmp/gpu-pool-stage5-drop/report.md /tmp/gpu-pool-stage5-drop/before_after.csv

# 5. delete the dead env keys listed above, by hand, on both hosts.
```

Rollback: the 5.3 "put the bridge verbs back in the YAML" rollback no longer exists (5.6 deletes the
bridge; such a YAML is now refused). Rolling back 5.6 = redeploy the previous durable-runs and
controller images. After the drop, restore the tables with
`gzip -dc /tmp/gpu-pool-stage5-drop/legacy_tables.sql.gz | docker exec -i orion-athena-sql-db psql -U postgres -d conjourney`
(verified on a throwaway database).

## Risks / concerns

- Severity: medium. Concern: the drop is irreversible except from the dump in /tmp.
  Mitigation: the script refuses without a verified dump, on any late write or live row, and under
  lock contention; operator copies the dump to durable storage before `--drop`; restore verified.
- Severity: medium. Concern: 24 h of zero new permits (spec acceptance 5) had NOT elapsed when this was
  written (5.4 live 2026-09-29 22:48 UTC; checked at 02:24 UTC, ~3.5 h: 0 rows). Mitigation: run step 4
  only after 2026-09-30 22:48 UTC; the script refuses anyway if a permit was written after the cutoff.
- Severity: low. Concern: the controller has no enable switch any more (spec: `GPU2_ENABLED` -> nothing).
  Mitigation: emergency stops are `ORION_BUS_ENABLED=false`, stopping the container, or the pool's
  `GPU_POOL_ACTUATE_ROLES=` (README).
- Severity: low. Concern: if circe's checkout YAML can't be parsed, `observed` is now `{}`, so the pool
  faults the card (before: the fixed gpu2 pair). Stricter, never looser; operator `clear_fault` after
  rebuilding the image.
- Severity: low. Concern: the "0 GPU1 flip calls in 7 days" figure is the spec's. The controller keeps no
  access log and restarted at 22:48, so it could not be re-counted; no service or container env on
  athena references the flip route or its token (only the dead `ORION_VISUAL_ELASTIC_CONTROLLER_URL` in
  thought's env, unread by its 5.4 code).
- Severity: info. Concern: the 5.3 YAML-revert rollback is gone; roll back 5.6 by images.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2426

🤖 Generated with [Claude Code](https://claude.com/claude-code)
