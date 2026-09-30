# GPU pool stage 5.7: enforce is the end state, config decides what the pool loads, one emergency stop

## Summary

- The pool now decides which models it may load and unload from `config/gpu_pool.yaml` alone: a swap seat is
  actuated if and only if it has a `launch:` block. `GPU_POOL_ACTUATE_ROLES` is deleted everywhere (settings,
  compose, `.env_example`, README), with no fallback.
- `GPU_POOL_MODE` defaults to `enforce`. In enforce the pool asks circe's controller, once and read-only, what each
  seat's cards hold at every start (and on resume) and adopts the answer, instead of guessing from liveness or
  reloading. `observe` stays only as the documented rollback; an unknown mode fails the boot.
- One emergency stop: control verbs `pause_actuation` / `resume_actuation` (Hub button, or
  `scripts/gpu_pool_pause.py`). Persisted on `gpu_pool_cards`, so it survives a restart. While paused nothing is
  loaded or unloaded and nothing is drained for a swap.
- The latent "experiment hold empties every card" bug (stage 5 spec, correction 5.1 #5) is fixed: its hold is
  refused `not_actuatable:experiment` in every mode, and the scheduler never drains residents for a seat that has
  no launch block. The Hub greys the button out with that reason.
- Runbook: `docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md`. **Not deployed.**

## Outcome moved

- Before: stopping all model moves meant editing an env list and restarting the pool; after 5.6 and this PR that
  list is gone, so without this PR there would be no fast stop at all. Now: one click, persisted, visible in the
  panel header, `/health`, pool state and two new events.
- Before: an operator hold on `experiment` would drain chat/agent/metacog/fast and load nothing (a full LLM outage
  until someone clicked release). It was only unreachable because observe refused all holds. Now refused, named,
  and structurally impossible in the scheduler. Eval: `experiment_resident_recalls=[]`, `resident_grants=420`
  (the mutation with the fix removed fails `residents_starved`).
- Before: in enforce there was no way to adopt a seat loaded by hand (the spec assumed a boot `status` that did not
  exist). Now pinned by tests: one `status`, no reload.

## Current architecture

- athena pool live with `GPU_POOL_MODE=observe`, `GPU_POOL_ACTUATE_ROLES=agent-gpu2`. "observe" already actuated
  agent-gpu2; it switched two behaviours: liveness adoption of never-actuated seats (`_observe_swap_seats`) and
  refusing operator holds (`hold_requires_swap_actuation`).
- The controller has had no enable switch since 5.6; the README's emergency stops were `ORION_BUS_ENABLED=false`
  on the controller, stopping it, or emptying `GPU_POOL_ACTUATE_ROLES`.
- The scheduler drained every resident for any queued operator lease on an operator-only seat, launch block or not.
- Live 2026-09-30: gpu2 `swapped_in={agent-gpu2}`, generation 79; first real generic-path load succeeded 06:32:26;
  the first generic **unload** had not happened yet when this was written (runbook precondition P1).

## Architecture touched

- Service: `orion-gpu-pool` (runtime, settings, store, main, compose, `.env_example`, README, tests, eval).
- Shared: `orion/gpu_pool/config.py` (`actuated_seats`, `not_actuatable_reason`), `orion/gpu_pool/scheduler.py`
  (`frozen`), `orion/schemas/gpu_pool.py`.
- Hub: GPU pool panel (JS, template, control route verb list).
- Postgres: additive migration v3 on `gpu_pool_cards`.
- Test fixtures in `orion-durable-runs` (the in-process pool).
- Docs: `orion-gpu-lane-controller/README.md` (emergency stop), `orion/bus/channels.yaml` descriptions, the stage 5
  spec ("Corrections from building 5.7").
- Not touched: circe's controller code and `config/gpu_pool.yaml` (every `launch_digest` unchanged; config gate
  prints the same digest).

## Files changed

- `services/orion-gpu-pool/app/runtime.py`: actuated set from config; enforce boot/resume reconcile
  (`_reconcile_idle_seats`, `_on_reconcile`, stale-replay guard, timeout); `_set_paused`; `frozen` passed to the
  scheduler; hold refusals; `not_actuatable` acquire refusal; `actuation_paused` in state.
- `services/orion-gpu-pool/app/{settings,main,store}.py`: `mode` is `Literal["enforce","observe"]` default enforce;
  `actuate_roles` removed; `/health.actuation`; v3 card columns + `check_schema`.
- `services/orion-gpu-pool/{.env_example,docker-compose.yml,README.md}`: enforce default, key deleted, mode table,
  emergency stop, v3 migration in the deploy block.
- `services/orion-sql-db/manual_migration_gpu_pool_v3_actuation_pause.sql` (new).
- `orion/gpu_pool/config.py`: `actuated_seats()`, `not_actuatable_reason()`.
- `orion/gpu_pool/scheduler.py`: `frozen` keyword; swap seats without a launch always frozen (never drained).
- `orion/schemas/gpu_pool.py`: control verbs, event literals, `GpuPoolStateV1.actuation_paused`, state mode default.
- `orion/bus/channels.yaml`: control/event channel descriptions (control description was stale: it still named a
  removed token).
- `services/orion-hub/static/js/gpu_pool.js`, `templates/gpu_pool.html`, `scripts/gpu_pool_routes.py`: emergency
  stop button + paused banner + header badge; hold control via `holdControlFor` (release / hold / greyed-out refusal).
- `scripts/gpu_pool_pause.py` (new): the same verb from a shell.
- Tests: `services/orion-gpu-pool/tests/test_stage5_7_enforce.py` (new), `test_runtime.py`,
  `test_holds_and_actuation.py`, `test_store_postgres.py`; `orion/gpu_pool/tests/test_scheduler.py`,
  `test_stage5_config.py`; Hub `gpu_pool.test.js`, `test_gpu_pool_panel_browser_smoke.py`, `test_gpu_pool_routes.py`;
  durable-runs `tests/pool_fixture.py`, `tests/test_durable_acceptance.py`, `evals/hold_fairness.py`.
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`: `enforce_scenario` + failures; `_views` now carries
  `operator` (it silently dropped it, so an operator lease never behaved as one in the eval).
- `.github/workflows/orion-gpu-pool-tests.yml`: path triggers for the v3 migration and the pause script.
- `docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md` (new), the stage 5 spec (corrections section).

## Schema / bus / API changes

- Added: `GpuPoolControlV1.verb` `pause_actuation`, `resume_actuation`; `GpuPoolEventV1.event`
  `actuation_paused`, `actuation_resumed`; `GpuPoolStateV1.actuation_paused` (`{paused, since, by}` or null);
  `/health.actuation` (`{seats, paused, since?, by?}`); `gpu_pool_cards.actuation_paused_at`, `.actuation_paused_by`.
- Removed: nothing on the wire.
- Renamed: control refusal `hold_requires_swap_actuation` -> `hold_refused_observe_mode` (it was no longer true).
- Behavior changed: `swap_requested` reason is `actuation_paused` (with `detail.paused/wanted`) while paused; operator
  leases on a launch-less seat are `unavailable reason=not_actuatable:<role>`; enforce boot/resume sends one
  `GpuActuateV1 action=status` per idle actuated seat (`action_id=<seat>:reconcile:<hex>`, `reason=reconcile:<why>`)
  and may emit `swapped reason=adopted:<why>` or `swap_failed reason=reconcile_{ambiguous,foreign_action}:<why>`.
- Compatibility: `GpuPoolEventV1` is extra=forbid with a literal list and sql-writer validates it, so **sql-writer
  deploys before the pool** (an old one only drops the two new event rows). The Hub builds `GpuPoolControlV1`
  locally, so **Hub deploys after the pool**. Hub reads state as a dict, so the new state field is safe either way.
  No channel added or removed; registry entries unchanged (same models).

## Env/config changes

- Added keys: none.
- Removed keys: `GPU_POOL_ACTUATE_ROLES` (orion-gpu-pool).
- Renamed keys: none.
- Changed default: `GPU_POOL_MODE` `observe` -> `enforce` (settings, compose, `.env_example`).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-gpu-pool --all-keys`: yes; no keys
  added. Reported diverged and left alone (on purpose): `GPU_POOL_MODE local=observe example=enforce` -- flipping it is
  runbook step 3 [GO], not a sync.
- skipped keys requiring operator action: **delete line 37 `GPU_POOL_ACTUATE_ROLES=agent-gpu2`** from
  `/mnt/scripts/Orion-Sapienform/services/orion-gpu-pool/.env` (dead: nothing reads it; the compose file no longer
  passes it), and flip line 21 `GPU_POOL_MODE=observe` -> `enforce` at deploy time.

## Tests run

```text
PYTHONPATH=<wt> pytest orion/gpu_pool/tests -q                                  326 passed
cd services/orion-gpu-pool && GPU_POOL_TEST_POSTGRES_URI=<throwaway postgres:16 on free port 59311> pytest tests -q
                                                                                 132 passed (incl. Postgres store + v3 migration)
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<same throwaway> pytest tests -q   262 passed
pytest services/orion-gpu-lane-controller/tests -q                               79 passed
cd services/orion-hub && pytest tests/test_gpu_pool_routes.py tests/test_biometrics_preview_api.py tests/test_urgent_evidence.py -k "not router_registered"
                                                                                 57 passed
node --test $(find services/orion-hub/static/js -name '*.test.js')               217 pass, 0 fail
playwright: pytest services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py 3 passed (new: emergency stop + refused hold)
static gates: env sync/parity tests, schema registry, schema skew discovery, grammar catalog  123 passed;
  check_metric_lineage --gate PASS; check_definition_drift --gate PASS; check_env_template_parity PASS;
  check_compose_no_relative_mounts PASS; check_service_hostname_refs OK; check_async_routes_not_blocking OK;
  check_control_surface_store_parity OK; check_gpu_pool_config ok (digest 0fe539fbd2368aa8, unchanged); git diff --check clean
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py                      VERDICT: PASS
  enforce_scenario: experiment_resident_recalls=[], experiment_granted=false, resident_grants=420,
                    seat_recalls_while_paused=[], seat_reclaimed_after_resume_sec=0, diffusion_granted_at_sec=1202
  mutation check (scheduler `frozen` removed): FAIL ['pause_drained_the_seat', 'resume_did_not_restore_reclaim']
                                         and    FAIL ['residents_starved', 'resume_did_not_restore_reclaim']
python services/orion-durable-runs/evals/hold_fairness.py                       PASS
python services/orion-durable-runs/evals/deploy_order_skew.py                   PASS
```

## Docker/build/smoke checks

```text
Not run: this PR is not deployed (by instruction). The runbook's steps 1-7 are the deploy and live checks.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`: 1 blocker, 5 should-fix, 7 nits.

- Finding (blocker B1): a committed test loaded `scripts/gpu_pool_pause.py`, which was not committed yet at the commit
  the reviewer read; same for the runbook and this report.
  - Fix: all committed (script and runbook in 550fcc8fe, this report with the fixes).
  - Evidence: `git ls-files scripts/gpu_pool_pause.py docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md` lists both.
- Finding (S1): a reconcile that agreed still overwrote the card's action record with an unfinished `status`, so after
  every ordinary restart the Hub would show "status agent-gpu2 ... in flight" forever, and memory disagreed with the DB.
  - Fix: the record is replaced only when the card changes (adopt or fault).
  - Evidence: `test_boot_reconcile_that_agrees_changes_nothing` now pins memory, store and published state to the
    previous `unload` record.
- Finding (S2): a pause row for a card since removed from the YAML would re-pause the pool on every restart, even after
  Resume.
  - Fix: boot reads only configured cards; pause/resume writes every row.
  - Evidence: `test_a_pause_row_for_a_card_no_longer_configured_does_not_pause_and_resume_clears_every_row`.
- Finding (S3): memory flipped before four separate row writes; a failed write mid-resume could leave rows paused.
  - Fix: `store.set_actuation_paused` is one `UPDATE` over all rows, run before memory flips; a failure answers
    `not_persisted:<error>` and changes nothing.
  - Evidence: `test_a_pause_that_cannot_be_persisted_changes_nothing`; Postgres round trip in
    `test_holds_children_and_a_mid_load_card_survive_a_restart_on_postgres`.
- Finding (S4): while a reconcile was open, the scheduler could still drain the seat on the stored belief the
  reconcile was checking (up to 90 s of pointless recalls after a resume).
  - Fix: seats with an open reconcile are passed to the scheduler as `frozen`.
  - Evidence: `test_nothing_is_drained_on_a_seat_while_its_reconcile_is_open` (fails with the fix reverted -- checked).
- Finding (S5): the engine tests and the durable-runs acceptance test only ran in observe.
  - Fix: both restart-mid-swap tests are parametrized over observe/enforce; the durable-runs gpu2 acceptance test runs
    enforce with its actuator fixture answering the boot `status`.
  - Evidence: pool 132 passed; durable-runs 262 passed.
- Finding (N1): `in_flight=None` on a reconcile answer was read as "nothing running".
  - Fix: keep the stored state.
  - Evidence: `test_a_reconcile_that_cannot_say_whether_something_runs_keeps_the_card`.
- Finding (N2): the docs said observe "restores the liveness shortcut"; for gpu2 it restores nothing, because
  agent-gpu2 is already pool-owned (generation 79).
  - Fix: `.env_example`, settings comment, README table, runbook rollback 1, spec corrections all say so.
- Finding (N4): hold refusals were read outside the runtime lock.
  - Fix: moved into `acquire` (`_operator_refusal`) under the lock.
- Finding (N5): the runbook's inline shell snippet had a `reply::` double colon and duplicated logic.
  - Fix: replaced by `scripts/gpu_pool_pause.py`, whose round trip through the real control path is tested.
- Finding (N6): no Postgres pause round trip. Already present (`test_store_postgres.py`, pause -> restart -> resume ->
  restart); the reviewer had no Postgres URI, so it was skipped for them. Ran here on a throwaway postgres:16.
- Finding (N7): a paused, loaded operator seat would still recall its holders at max_hold.
  - Fix: gated by `frozen`.
  - Evidence: `test_a_frozen_operator_seat_keeps_its_holders_past_max_hold`.
- Not fixed (N3, low): while paused, a lease whose only role is an unloaded frozen seat counts as serviceable and
  waits instead of backlogging/failing. No class is served only by a swap seat today (agent-gpu2 is always behind
  agent; experiment is refused at admission), so there is no live effect. Follow-up if a seat-only class appears.

## Restart required

```bash
# All [GO] steps from docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md, in order, from a worktree of merged main:
scripts/safe_docker_build.sh orion-sql-writer up -d --build
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_gpu_pool_v3_actuation_pause.sql
# primary .env: GPU_POOL_MODE=enforce, delete GPU_POOL_ACTUATE_ROLES
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# circe: nothing
```

## Risks / concerns

- Severity: medium. Concern: a 5.7 pool refuses to boot without migration v3, and a pool that will not boot is an
  LLM outage. Mitigation: runbook step 2 before step 4, with a read-back query; the migration is additive and
  re-runnable.
- Severity: medium. Concern: the boot reconcile faults gpu2 if circe reports a half-done card (e.g. diffusion
  mid-restart at the moment of the pool deploy). Mitigation: discovery auto-clears the fault once the card is
  consistent; "Clear fault" in the Hub; rollback 1 (observe) turns the reconcile off.
- Severity: low. Concern: while paused, `agent-gpu2`'s 2.5 h `max_hold_sec` does not recall holds and diffusion waits
  for gpu2 until resume. Mitigation: by design (a drain that cannot end in an unload only empties the card);
  documented in README and runbook.
- Severity: low. Concern: the pause does not stop an action already running on circe. Mitigation: documented second
  step (stop the controller container); the pool keeps following the running action to its result.
- Severity: low. Concern: `scripts/gpu_pool_pause.py` has not been run against the live bus (UNVERIFIED live); its
  round trip through the pool's real control path is tested. The Hub button is the primary path.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2433

🤖 Generated with [Claude Code](https://claude.com/claude-code)
