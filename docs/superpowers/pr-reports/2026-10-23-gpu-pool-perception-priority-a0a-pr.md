# feat(gpu-pool): teach every loader the `perception` priority (A0a, no user yet)

## Summary

- Adds a fifth GPU-pool priority, `perception`, between `interactive` (Juniper's turn) and `system`. Final order: `urgent > interactive > perception > system > background`.
- The heat reflex cannot shed it. `orion/gpu_pool/shed.py` can only ever block `background`/`system` (`SHEDDABLE`), so this needs no shed change.
- A config file that does not name `perception` still loads: the validator inserts it just below `interactive`. This is what makes the PR safe to deploy everywhere before any YAML uses the new value.
- Nothing uses `perception` yet. A0b (stacked PR) switches vision-council onto it.

## Outcome moved

No live behaviour change. This is the consumer-first half of Patch A0 of the body-economy design (`docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`, PR #2855). Juniper decided on 10-23 that vision is Tier 0 and must keep working through heat. In the 7 days to 10-23, the 34 °C reflex refused vision-council 25,430 times.

## Current architecture

- Four priorities, held as a hardcoded tuple and two `Literal` types: `orion/gpu_pool/config.py`, `orion/schemas/gpu_pool.py`.
- `PoolConfig` is strict: `priorities:` must be an exact permutation of the tuple, and `RouteSpec.priority` is a `Literal`.
- circe's lane controller mounts the live checkout read-only (`/repo`) and re-reads `config/gpu_pool.yaml` with its own baked code.
- So if one PR introduced a new priority name and also used it in the YAML, a `git pull` on circe would make the old controller refuse its config (`config_unloadable`). That was the 10-09 incident.

## Architecture touched

- `orion/gpu_pool/config.py`: `PRIORITIES`, a shared `Priority` Literal, and a `priorities` field validator that inserts `perception` when absent.
- `orion/schemas/gpu_pool.py`: the bus `Priority` Literal. Lease requests, events and state snapshots accept `perception`.
- `orion/gpu_pool/shed.py`: docstring only.

## Files changed

- `orion/gpu_pool/config.py`: the new priority, the shared Literal, and the compat validator.
- `orion/schemas/gpu_pool.py`: the bus Literal.
- `orion/gpu_pool/shed.py`: doc line naming perception as never shed.
- `orion/gpu_pool/tests/test_perception_priority.py`: tests for the ordering, old-config compatibility, route acceptance, schema acceptance, the shed board never blocking perception, the scheduler granting perception while system is shed, and interactive winning the last slot. The ordering test runs the real scheduler.
- `docs/superpowers/pr-reports/2026-10-23-gpu-pool-perception-priority-a0a-pr.md`: this report.

## Schema / bus / API changes

- Added: `perception` to `Priority` in `orion/schemas/gpu_pool.py` (`GpuLeaseRequestV1`, `GpuPoolEventV1`, state).
- Removed: none.
- Renamed: none.
- Behavior changed: none until a caller sends `perception` (A0b).
- Compatibility notes: additive Literal on forbid models. This PR is the **consumer** half; it must be deployed on every loader before A0b, which is the producer.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env change).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/gpu_pool/tests + orion/llm                    518 passed
services/orion-gpu-pool/tests                       194 passed, 15 skipped
services/orion-llm-gateway/tests                    459 passed
scripts/check_gpu_pool_config.py                    ok (digest 13e162d05c7a1382, identical to main)
launch_digest(agent-gpu2), launch_digest(diffusion) byte-identical to main
```

## Evals run

```text
None. No behaviour change to evaluate. The pool's day eval (services/orion-gpu-pool/evals/run_pool_day_eval.py) is unaffected.
```

## Docker/build/smoke checks

```text
Not run in this session. Pure Python validation change. The real gate is the deploy order below.
```

## Review findings fixed

- Finding: the "interactive before perception" test sorted by rank itself and never ran the scheduler.
  - Fix: it now runs `schedule()` with exactly one free metacog slot (spill roles down) and asserts only the interactive lease is granted.
  - Evidence: the first rewrite failed because perception spilled to `fast`, which is correct pool behaviour. The final test takes the spill roles away.
- Rollout check (reviewer, NOTE): A0a alone is safe. Old loaders never see `perception`, and new loaders accept both YAML shapes. Confirmed by the identical digests above.

## Restart required

Deploy A0a on **every** host that loads the pool config, before A0b merges. Each host runs from its clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```
```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```
```bash
ssh circe@circe "cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build"
```

The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh`) normally rebuilds the services whose code changed. Check that the lane controller on circe was rebuilt before merging A0b.

## Risks / concerns

- Severity: medium.
  - Concern: A0b merges before circe's lane controller runs A0a code.
  - Mitigation: A0b's PR says so in its first line. The controller fails safe: it refuses and alerts via the controller-health card, and swaps pause until it is rebuilt.
- Severity: low.
  - Concern: some other service bakes an old `orion` package and validates a whole pool state snapshot after A0b ships.
  - Mitigation: those services copy the YAML into their image, so they only see `perception` after being rebuilt with A0a+ code.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2856

🤖 Generated with [Claude Code](https://claude.com/claude-code)
