## Summary

- Retro's game-goal planner switches now default **on**: `RETRO_COGNITION_ENABLED` and `HUB_RETRO_COGNITION_ENABLED`, in settings, compose and `.env_example`. #2801 shipped them off, which broke the ships-on rule.
- A per-session handoff (**Try Orion goals · 60s**) is still required, so nothing changes until a player opts in.
- Updated the stale #2796 continuation spec for #2801 (goal inbox), #2825 and #2832 (save/load).
- Re-ran the real-planner eval. It is still BLOCKED by the cabinet heat shed. The spec now records the exact rerun command.

## Outcome moved

Before, the goal experiment was deployed but could not be used without a config edit and restart. Now it is available per session, and the switch is only for rollback. The spec no longer reports built features as NOT IMPLEMENTED.

## Current architecture

#2796 added the scripted baseline, #2801 added attempts and goals behind two default-off switches, and #2825 added save/load. The continuation spec was written before #2801 and #2825 merged.

## Architecture touched

Retro and Hub settings defaults only. No contract, bus channel or schema change.

## Files changed

- `services/orion-retro/settings.py`, `docker-compose.yml`, `.env_example`: Retro switch default changed to true.
- `services/orion-hub/app/settings.py`, `.env_example`: Hub switch default changed to true.
- `services/orion-retro/tests/test_goals.py`: regression test. The switch defaults on, and setting the env value to false turns it off.
- `services/orion-retro/README.md` and two Retro specs: docs and acceptance reconciliation.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the `/goal` and `/plan` routes are enabled by default. They still require a live, owned, explicitly handed-off session.
- Compatibility notes: set either switch to `false` to disable.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes. Both keys now `true`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran. The sync only adds missing keys and does not change existing values, so the primary checkout's `services/orion-retro/.env` and `services/orion-hub/.env` were set to `true` by hand.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-retro/tests -q            82 passed
pytest services/orion-hub/tests/retro tests/retro -q   32 passed
git diff --check                                  clean
```

## Evals run

```text
services/orion-retro/evals/cognitive_loop.py (real planner, pinned Nova v1.0.6a + FCEUmm ac0d00a7)
FAIL/BLOCKED: planner_empty_result:gpu_pool_unavailable
orion:regulation:latest 2026-10-21T05:16Z: arousal strained, cabinet_hot, 33.63 C
```

The thermal shed is correct behavior and was not bypassed. Mistake → Orion revision → changed input → checked outcome is still UNVERIFIED.

## Docker/build/smoke checks

```text
Not run: defaults-only change. The deploy below picks up the .env values.
```

## Review findings fixed

- Subagent review found no material findings. Notes:
  - The `getattr(..., False)` fallback in `retro_routes.py` is unreachable.
  - The dataclass default `cognition_enabled=False` is used only by tests. Production uses `from_env`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the planner route is now reachable by default.
- Mitigation: the per-session handoff, epoch/attempt checks and immediate human takeover are unchanged. The switch is the rollback lever.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2837
