## Summary

- Before each image, Orion's diffusion service guesses how many watts it will draw (`expected_watts` on the power intent). The guess is the median of what recent images really drew.
- That memory lived only inside the running container. The GPU pool stops the diffusion container whenever gpu2 goes back to the agent model (after ~15 idle minutes), and starts it again for the next image. Almost every image ran in a fresh container with an empty memory, so the guess said "unknown".
- The memory is now saved to a small file on a host directory and loaded on start. Every sample carries the time it was seen. A sample older than 7 days stops counting.
- The file is stamped with the model, dtype, steps and image size. A file from a different setup is ignored, so a model swap still starts from "unknown".
- This is one of the source fixes for the self-monitor organ (PR #2827). It makes the GPU-power bet gradeable: guess first, meter second.

## Outcome moved

Live numbers for `power_intent_settled`, 2026-11-03:

- There is only one intent kind: `reverie_diffusion` on circe, gpu 2. No other producer declares power intents, so no kind lacks an estimate by nature.
- All time: 968 of 1,297 rows have a guess. Almost all of those come from before the GPU pool's warm standby (2026-10-14). On 10-12 and 10-13 every row had one.
- Since 10-23: 5 of 159 rows (0 of 9 on 11-02, 1 of 17 on 10-31).
- Pool history shows the cycle: `owner_reclaim` unloads the agent model, then a settlement lands within ~2 minutes, then the agent model reloads ~15 minutes later. Example: 11-03 01:22 reclaim, 01:23:51 settlement, 01:39 agent reload.
- Review confirmed the cause in code: `services/orion-gpu-lane-controller/app/launch_exec.py` `stop()` runs `docker compose ... stop diffusion-host`. The process restarts, so the memory was lost.

The guess is never filled in after the outcome. It is still decided at intent time, from settlements that came before.

## Current architecture

`services/orion-diffusion-host/app/power_prior.py` held a per-(workload, node, gpu) deque of settled peaks, in memory only. `main.py` subscribed to `orion:power:intent:settled` to feed it, and read its median into `PowerIntentV1.expected_watts`. The 2026-08-30 design chose in-memory on purpose and expected a re-warm after a restart within ~30 minutes. With the pool relaunching per image, that re-warm never finished.

## Architecture touched

- orion-diffusion-host only: the prior, its wiring, settings, env template, compose (one new bind mount), README, design-doc amendment.
- No schema or bus change.

## Files changed

- `services/orion-diffusion-host/app/power_prior.py`: timestamped samples, age limit, `to_state`/`load_state`, fingerprint, `load_prior_file`/`save_prior_file` (atomic, fsync).
- `services/orion-diffusion-host/app/main.py`: load on first use, save after each counted settlement, pass `now`, fingerprint.
- `services/orion-diffusion-host/app/settings.py`, `.env_example`: `DIFFUSION_POWER_PRIOR_STATE_PATH`, `DIFFUSION_POWER_PRIOR_MAX_AGE_SEC`.
- `services/orion-diffusion-host/docker-compose.yml`: `/mnt/storage-warm/orion-diffusion-host:/var/lib/orion-diffusion-host`.
- `services/orion-diffusion-host/README.md`: new "Power intent and its watt guess" section.
- `docs/superpowers/specs/2026-08-30-power-intent-prior-design.md`: dated amendment; non-goal marked superseded.
- `services/orion-diffusion-host/tests/test_power_prior.py`, `tests/test_power_intent_publish.py`: persistence, age, fingerprint, bad-file, and relaunch regression tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `expected_watts` is now set on intents from a relaunched container once 3 settlements within 7 days exist for the same setup.
- Compatibility notes: none. `PowerIntentV1.expected_watts` was already `Optional[float]`.

## Env/config changes

- Added keys: `DIFFUSION_POWER_PRIOR_STATE_PATH` (default `/var/lib/orion-diffusion-host/power_prior.json`, empty disables), `DIFFUSION_POWER_PRIOR_MAX_AGE_SEC` (default `604800`, 0 = no limit).
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: yes. The keys were added to `services/orion-diffusion-host/.env` in the primary checkout. The worktree has no local `.env`, so the sync script had nothing to write there. Circe's `.env` is not reachable from this host. If it lacks the keys, the `settings.py` defaults apply (same values).
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-diffusion-host && PYTHONPATH=../.. pytest tests -q   -> 75 passed
Relaunch regression test on origin/main's main.py + power_prior.py -> FAILED (expected_watts None); on this branch -> passed
python scripts/check_env_template_parity.py -> PASS (94 services)
```

## Evals run

```text
No eval harness exists for orion-diffusion-host. The live acceptance check below is the eval: the share of new
power_intent_settled rows with expected_watts. Grading the bet belongs to the self-monitor organ (PR #2827).
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-diffusion-host/.env -f services/orion-diffusion-host/docker-compose.yml config
-> rc=0; renders the new bind mount and both keys.
Live (UNVERIFIED until deployed on circe):
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), count(expected_watts) from power_intent_settled where created_at > now() - interval '1 day'"
diffusion-host log: power_prior_loaded path=... samples=N (N>0 after the first 3 settlements)
```

## Review findings fixed

- Finding (minor): the design doc still said persistence was a non-goal.
  - Fix: added a dated amendment and marked the non-goal superseded.
  - Evidence: `docs/superpowers/specs/2026-08-30-power-intent-prior-design.md`.
- Finding (minor): a model swap would now inherit the old model's watts. Before, a restart forgot them by accident.
  - Fix: the state file carries a fingerprint (model, dtype, steps, width, height). A mismatch is ignored on load.
  - Evidence: `test_a_saved_window_from_another_model_is_ignored`.
- Finding (minor): `observe(at=0.0)` default would silently stamp a forgotten time as 1970.
  - Fix: `at` is required.
  - Evidence: `test_observe_requires_a_time`.
- Finding (nit): no fsync, fixed temp name.
  - Fix: unique `NamedTemporaryFile` in the same directory, flush + fsync, then `os.replace`.
  - Evidence: `test_save_leaves_no_temp_file_behind`.
- Finding (nit): older publish tests could read a real `/var/lib/...` file on a deployed host.
  - Fix: an autouse fixture points the path at `tmp_path` and resets the prior.
  - Evidence: the test file fixture.

## Restart required

The GPU pool launches diffusion-host on circe itself. The first launch after circe's checkout pulls this change recreates the container with the new mount. To force it now, on circe from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-diffusion-host up -d --build
```

## Risks / concerns

- Severity: low
- Concern: if a settlement arrived after the pool already stopped the container, this fix cannot recover it. The deadline is 120 s; the agent model reloads ~15 minutes after the last image, so this is unlikely. UNVERIFIED.
- Mitigation: watch the live acceptance query for a day after deploy.
- Severity: low
- Concern: `/mnt/storage-warm/orion-diffusion-host` is created by Docker on circe on first mount (the container runs as root).
- Mitigation: a failed save only logs `power_prior_save_failed`. Generation is never affected.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3048

🤖 Generated with [Claude Code](https://claude.com/claude-code)
