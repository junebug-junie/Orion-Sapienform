## Summary

- Turns on Orion's first learned world action (#2461) in the operator env templates, at Juniper's call: the cabinet can compete for attention, the GPU pool accepts Orion's shed, and dispatch lets Orion act.
- `SUBSTRATE_CABINET_HEAT_ATTENTION_ENABLED=true` (substrate-runtime), `GPU_POOL_ORION_SHED_ENABLED=true` (gpu-pool), `ORION_WORLD_ACTIONS_ENABLED=true` + `ORION_WORLD_ACTIONS_ALLOWED=shed_background_gpu` (execution-dispatch-runtime).
- Code defaults and compose fallbacks stay off; local `.env` files in the primary checkout flipped to match.

## Outcome moved

The attend-to-act loop can run live: cabinet warming wins attention → `shed_background_gpu` proposal → policy → allocator → pool `orion_self_shed` → settlement → sensor-scored outcome vs a 0.5 holdback.

## Current architecture

#2461 merged and deployed 2026-10-01 03:06-03:09 with all action flags off; both migrations applied (`gpu_pool_orion_shed`, `substrate_world_action_episodes`). Reflex (`cooling_incident`, #2424) live and idle.

## Architecture touched

Env templates only: orion-substrate-runtime, orion-gpu-pool, orion-execution-dispatch-runtime.

## Files changed

- `services/orion-substrate-runtime/.env_example`: cabinet heat attention on.
- `services/orion-gpu-pool/.env_example`: Orion shed accepted.
- `services/orion-execution-dispatch-runtime/.env_example`: world actions on, allow-list `shed_background_gpu`.
- this report.

## Schema / bus / API changes

- None.

## Env/config changes

- Changed values: the four keys above, off → on. No keys added/removed/renamed.
- `.env_example` updated: yes. Local `.env` synced: yes (primary checkout, same four keys).
- Skipped keys: none.

## Tests run

```text
python scripts/check_env_template_parity.py -> PASS (94 service(s) compared)
git diff --check -> clean
```

## Evals run

```text
None: config-only. #2461 shipped run_attend_act_loop_eval.py (fixture 20/20); live mode after restart.
```

## Docker/build/smoke checks

```text
Recreate (no rebuild needed; images already carry #2461), see Restart required.
```

## Review findings fixed

- None: four config values.

## Restart required

From the primary checkout on main (images already current, so no `--build`):

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-gpu-pool up -d
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d
```

Kill: `ORION_WORLD_ACTIONS_ENABLED=false` + recreate dispatch, or `GPU_POOL_ORION_SHED_ENABLED=false` + recreate pool (active Orion shed settles `cancelled`). Neither touches the reflex.

## Risks / concerns

- Severity: medium. Concern: the action can now really hold back background GPU work (≤15 min, ≤1 h/day). Mitigation: pool caps, reflex precedence, 0.5 holdback, kill switches above.
- Severity: medium. Concern: proposals bind attention's winner by exact timestamp match, unverified live — if every proposal carries `winner_unbindable:no_broadcast_log_row`, the action never fires. Check after restart.
- Severity: medium. Concern: stage 1 proves the loop closes; the allocator retires the action after ~12-24 treated rows, before the ~47/arm needed to learn the effect (stage 3 needs Juniper's yes).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2464
