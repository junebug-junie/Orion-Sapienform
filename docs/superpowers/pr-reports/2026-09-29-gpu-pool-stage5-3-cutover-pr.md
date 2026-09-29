## Summary

GPU pool stage 5.3: circe's second 27B seat (`agent-gpu2`) stops using the two hard-coded gpu2 moves
and is loaded/unloaded by the generic "run this role's `launch:` block" executor from 5.2.

- `config/gpu_pool.yaml`: `agent-gpu2` loses its `swap.load/unload` bridge verbs and lists one allowed
  model, `launch.profiles: [qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex]` — the model the seat has
  announced on every load (live `discovery_confirmed`, and circe's `ATLAS_AGENT_PROFILE_NAME`), so the
  cutover does not change the model.
- The pool sends that profile with every `load` (`PoolConfig.load_profile` = `launch.profiles[0]`;
  choosing by vision/ctx/VRAM is not built) and records it on the card's `actuation` and on
  `swap_started`/`swapped`/`swap_failed` `detail.profile`. The Hub panel shows it ("model …").
- Validator: a seat that still has bridge verbs may not list `profiles` (the bridge refuses any
  profile, so every load would fail). This guards a half-done rollback.
- Tests: an end-to-end test drives the real pool runtime into the real circe controller with fake
  docker (load with profile, busy unload with the new `upstream_not_idle:agent-gpu2` reason, unload,
  failed load rolled back, deadline math); the bridge tests now run on the rollback shape, which
  proves the rollback still works.
- Runbook `docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md` and a read-only probe script
  (`scripts/gpu_pool_actuator_probe.py`) that asks the controller "status" and "do our digests agree"
  over the bus without spending a generation or touching a container.

## Outcome moved

Adding a model option or a card no longer needs controller code for gpu2: agent-gpu2 runs on the same
YAML-driven path any future seat will (spec worked examples A/B). Behaviour on agent-gpu2 after deploy:

- ready wait 600 s -> 900 s (`launch.timeout_sec`); pool deadline 1500 s, stuck ceiling 3000 s,
  controller's realistic worst case (failed load: 2 `docker ps` + 300 s drain + 900 s + 600 s restore
  + ~60 s docker calls ≈ 1920 s) is past the first deadline (pool polls `status`) and inside the ceiling;
- failure reasons carry the role (`upstream_not_idle:agent-gpu2`, `model_readiness_timeout:agent-gpu2`);
- the controller's `/v1/gpu-slots/circe-gpu2/status` no longer shows swap progress.

## Current architecture

- 5.1 (#2408) and 5.2 (#2409) merged to `main` 2026-09-29 22:05 UTC. 5.2 already built
  `launch_exec.py` and `pool_fence.resolve() -> LaunchPlan` for seats without bridge verbs; agent-gpu2
  still had the verbs, so it went through `gpu2.transition()`.
- The pool sent `profile=None` on every action (`runtime.py _begin_actuation`).
- Live: athena pool rebuilt 22:06 UTC on 5.1+5.2 (`config_digest=e58f5238846b0bb7`); circe checkout still
  at `f720356c1` with a 09-26 controller image, so agent-gpu2 loads are currently refused on the launch
  digest until circe catches up (runbook step 1).

## Architecture touched

- Pool core config (`orion/gpu_pool/config.py`), pool runtime actuation send path, committed YAML.
- No bus channel, schema field or env key changes. `GpuActuateV1.profile` existed since stage 4 and is
  now produced; the controller has accepted it since 5.2.
- Hub panel (render only).

## Files changed

- `config/gpu_pool.yaml`: agent-gpu2 off the bridge; `launch.profiles` allow-list.
- `orion/gpu_pool/config.py`: `PoolConfig.load_profile`; validator refuses bridged seat + profiles.
- `orion/schemas/gpu_pool.py`: comment only (`actuation.profile`).
- `services/orion-gpu-pool/app/runtime.py`: send `profile` on loads; record it in actuation + swap_* detail; deadline-math docstring.
- `services/orion-gpu-pool/README.md`, `services/orion-gpu-lane-controller/README.md`: 5.3 behaviour, the slot-status route, rollback.
- `services/orion-hub/static/js/gpu_pool.js`: show the load's model.
- `scripts/gpu_pool_actuator_probe.py` (new): read-only status + digest-agreement probe used by the runbook.
- `docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md` (new).
- Tests: `services/orion-gpu-pool/tests/test_stage5_3_cutover_e2e.py` (new), `test_holds_and_actuation.py`;
  `services/orion-gpu-lane-controller/tests/cutover_config.py` (new rollback-shape helper),
  `test_actuator_bus.py`, `test_launch_exec.py`; `orion/gpu_pool/tests/test_config.py`,
  `test_stage4_contracts.py`, `test_stage5_config.py`; `services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py`.
- `.github/workflows/orion-gpu-pool-tests.yml`: installs `loguru` (controller logger, for the e2e
  test); triggers on controller app + probe script changes.

## Schema / bus / API changes

- Added: none (`GpuActuateV1.profile` now non-null on agent-gpu2 loads; `GpuPoolEventV1.detail.profile`, free-form dict).
- Removed: none.
- Renamed: none.
- Behaviour changed: see "Outcome moved".
- Compatibility notes: pool and controller must run the same commit (launch digest). A 5.2+ controller
  image parses both the 5.3 and the rollback YAML.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: circe still lacks the 5.1 keys `ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES` /
  `ATLAS_AGENT_BURST_PROFILE_NAME` (read 2026-09-29); runbook step 1 syncs them with `--all-keys`.
  Compose defaults cover both meanwhile.

## Tests run

```text
python scripts/check_gpu_pool_config.py                              ok (digest 08b098deef36cb6a)
pytest orion/gpu_pool/tests -q                                       292 passed
(cd services/orion-gpu-pool && pytest tests -q)                      98 passed, 7 skipped (Postgres store tests: no DSN locally; CI runs them)
pytest services/orion-gpu-lane-controller/tests -q                   114 passed
(cd services/orion-hub && pytest tests/test_gpu_pool_routes.py -q)   10 passed, 1 skipped
node --test services/orion-hub/static/js/gpu_pool.test.js            15 pass
pytest services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py 2 passed (playwright 1.63, scratch venv)
python scripts/check_definition_drift.py                             no definition changes
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py           VERDICT: PASS
```

No new eval: 5.3 changes the actuation path, not scheduling policy. The e2e test is the behavioural
check for it; live acceptance checks 1–4 are in the runbook.

## Docker/build/smoke checks

```text
Not run. Deploy is a runbook step (Juniper's go), not part of this PR.
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent) on the committed diff; suites re-run after fixes.

- Finding (should-fix): the "fits inside the 3000 s stuck ceiling" claim ignored `GPU_LANE_COMMAND_TIMEOUT_SEC`
  (900 s per compose call, four on a failed load, ~4100 s if they all hang), and the test used hand-picked constants.
  - Fix: `_action_timeout` docstring now states both bounds; a hung docker call faulting the card is the ceiling's
    purpose (pre-existing; the bridge had it with 300 s less room). The test reads the shipped
    `Settings.model_fields` defaults and pins both: realistic (~1920 s) inside, wedged (~4260 s) outside.
  - Evidence: `test_actuation_deadline_and_stuck_ceiling_cover_the_controllers_worst_case` passes.
- Finding (should-fix): runbook and probe script were referenced but not yet committed at review time.
  - Fix: committed (`0532993f2`).
  - Evidence: `git ls-files docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md scripts/gpu_pool_actuator_probe.py`.
- Finding (nit): a bridge-era `gpu2._state = failed` (no `restored`) would keep orion-thought deferring every
  image until a controller restart, since the generic path never updates it.
  - Fix: `actuator_bus._run` resets it to `neither` before every generic action; README says so.
  - Evidence: new `test_generic_action_resets_the_stale_bridge_state_thought_reads`.
- Finding (nit): the test's `Settings()` read the local `.env`.
  - Fix: uses `Settings.model_fields[...].default`.
  - Evidence: same deadline test.
- Finding (nit): Hub browser smoke assertion never ran (no Playwright locally, no workflow runs that file).
  - Fix: ran it in a scratch venv with Playwright 1.63.
  - Evidence: `pytest services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py` -> 2 passed. (Still no CI
    job runs it outside `schedule-browser-smoke.yml`'s own path filter.)
- Finding (nit): `GpuActuateV1.profile` comment said "stage 4 always None".
  - Fix: updated.
  - Evidence: `orion/schemas/gpu_pool.py`.
- Checked by the reviewer, no issue: model/recreate unchanged (46/46 live discovery events announce the same
  profile; resolved `LLM_PROFILE_NAME` identical); digest agreement pinned by the e2e test; no consumer parses the
  failure reasons; rollback shape valid; no module collision; fake/real clock not flaky (3/3 repeats).

## Restart required

Follow `docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md`. In order:

```bash
# step 1 (circe to the pool's 5.1+5.2 commit) -- needed today regardless of this PR
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && git fetch origin && git checkout --detach 38d65a36e \
  && python3 scripts/sync_local_env_from_example.py orion-llamacpp-host --all-keys'
# then from a circe worktree at that commit: scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
# step 3 (after merge, $SHA = merge commit): circe controller rebuild at $SHA, then athena:
scripts/safe_docker_build.sh orion-gpu-pool up -d --build gpu-pool
```

## Risks / concerns

- Severity: medium
  - Concern: pool and controller on different commits refuse every agent-gpu2 action (`launch_digest_mismatch`). True today already (athena on 5.2, circe on 4.5).
  - Mitigation: runbook step 1 first; the probe script reports MISMATCH/OK before and after each step; refusals are safe (card untouched, 600 s cooldown).
- Severity: low
  - Concern: orion-thought's pre-image check (`ORION_VISUAL_ELASTIC_STATUS_ENABLED=true` live — the spec says it is off) reads the controller slot route, whose `state` no longer moves.
  - Mitigation: it defers on `active != "diffusion"`, and `active` is read from `docker compose ps`; `state` is reset to `neither` by the controller restart and by every generic action, which triggers none of its defer conditions. The draining window it no longer sees is covered by the image run's pool hold (a load cannot evict diffusion while its hold is granted: scheduler `residents_idle`). Deleted in 5.4.
- Severity: low
  - Concern: a busy-27B unload still faults the card briefly (as with the bridge) before discovery clears it.
  - Mitigation: unchanged behaviour, only the reason text differs; covered by the e2e test.
- Severity: low
  - Concern: a failed load whose four docker calls each hang near their 900 s command timeout exceeds the 3000 s stuck ceiling; the pool faults the card and ignores the late result (pre-existing; 300 s tighter than on the bridge).
  - Mitigation: by design (a wedged actuator must not hold the card); operator `clear_fault` reconciles via `status`.
- Severity: low
  - Concern: acceptance check 3 (failed load, live) is not forced; UNVERIFIED live until a natural failure.
  - Mitigation: e2e test covers the rollback path through the real controller code.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2415

🤖 Generated with [Claude Code](https://claude.com/claude-code)
