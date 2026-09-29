# GPU pool stage 5.2: the generic "load this role on this card" actuator

## Summary

circe's GPU controller can now load or unload any model seat described in `config/gpu_pool.yaml`,
not just the two hard-coded gpu2 moves. Nothing live changes in this PR. The 27B seat on gpu2 still
carries its old bridge verbs, so it still goes through the old path until 5.3 removes them.

- **New `app/launch_exec.py` runs a role's `launch:` block.** The steps are the ones `gpu2.py`
  already runs, applied to any role:
  - **load:** for every running evicted role, drain it (if it has a drain) and stop it. Start the
    seat with `up -d --no-build --no-deps`, resume it, and wait until it is ready.
  - **rollback on failure:** stop the seat, restart the stopped evicted roles in reverse order, and
    un-drain any that were drained but never stopped.
- **unload:** a running LLM seat must have every llama.cpp slot idle before it stops. Then stop the
  seat and start the evicted roles.
- **The request can only name a role and a profile.** The profile must be in the role's
  `launch.profiles`, or the request is refused with `profile_not_allowed`. The controller's own YAML
  supplies every compose file, service, compose profile, and env var name. The only values it sets
  are the card `index` (in `launch.cuda_env`) and the chosen profile (in `launch.profile_var`).
  There is no free-form Docker.
- **`pool_fence.resolve()` picks the path.** It returns a `LaunchPlan` for any swap seat without
  bridge verbs. Seats that still have `swap.load/unload` keep the stage-4 bridge (`gpu2.transition`)
  unchanged.
- **`observed` (the container state the pool reconciles from) is now computed from the YAML.** It
  covers every launch role on this actuator; today that is still `agent-gpu2` and `diffusion`. It
  replaces `TARGET_ROLES`. If the YAML can't be parsed, it falls back to the old fixed pair.
- **Unchanged:** generation fencing, `launch_digest` checkpoints before every mutation, idempotency
  by `action_id`, persisted state, and the `GpuActuateResultV1` meaning of `restored`, phases,
  `in_flight` and `last_action_id`.

## Outcome moved

Adding a card (for example a gpu4 that hosts either an 8B or a vision model) or a second model option
for an existing seat no longer needs controller code. The spec's worked examples A and B run
end-to-end through the real bus handler against a fake Docker runner. They issue the exact compose
calls and env shown in the tests. No live behaviour moves until 5.3.

## Current architecture

- `actuator_bus._dispatch` → `pool_fence.resolve()` only accepted bridge verbs. Everything else was
  refused (`not_a_bridge_role`), and so was any profile (`profile_unsupported`).
- `gpu2.transition` ran two fixed targets. `observed` came from a fixed `TARGET_ROLES` map.

## Architecture touched

- Service: `orion-gpu-lane-controller` (circe) only.
- No schema, bus channel, env key or YAML change.

## Files changed

- `services/orion-gpu-lane-controller/app/launch_exec.py` (new): the generic executor, from
  `gpu2.py`'s steps.
- `services/orion-gpu-lane-controller/app/pool_fence.py`: `RolePlan`/`LaunchPlan`, `build_plan`,
  `role_plans`, and profile + swap-seat checks in `resolve()`. `TARGET_ROLES` is removed.
- `services/orion-gpu-lane-controller/app/actuator_bus.py`: dispatches to the bridge or
  `launch_exec`, a shared busy check, and the generic `observe()` with the bridge-pair fallback.
- `services/orion-gpu-lane-controller/tests/test_launch_exec.py` (new): 32 tests covering worked
  examples A and B, refusals, fencing and rollback edges.
- `services/orion-gpu-lane-controller/tests/test_actuator_bus.py`: two refusal names, renamed per
  the spec (see below). No other bridge test changed.
- `services/orion-gpu-lane-controller/README.md`: new generic-actuator section and updated refusal
  list.

## Schema / bus / API changes

- **Added refusals:**
  - `profile_not_allowed`;
  - `not_a_swap_seat`: a resident role can't be loaded directly;
  - `no_launch_block:<evicted role>`;
  - `bridge_cannot_set_profile`.
- **Renamed:** `not_a_bridge_role` → `not_a_swap_seat`, and `profile_unsupported` →
  `profile_not_allowed` (spec, "Proposed schema / API changes").
- **Behaviour changed:**
  - `observed` is built from the YAML's launch roles; the key set is identical today;
  - on the generic path, failure reasons carry the role (`drain_timeout:<role>`,
    `upstream_not_idle:<role>`, `model_readiness_timeout:<role>`, …).
- **Compatibility:**
  - `GpuActuateV1`/`GpuActuateResultV1` are unchanged;
  - the pool (`runtime.py`) reads `observed` by role name and `restored`, and neither changed shape.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no. No local `.env` sync is needed, and circe needs no `.env` lines for
  5.2. (The 5.1 `ATLAS_AGENT_BURST_*` keys still need
  `sync_local_env_from_example.py orion-llamacpp-host --all-keys` on circe, per 5.1.)
- Skipped keys: none.
- Note: the generic drain wait uses `GPU2_DRAIN_TIMEOUT_SEC` (a gpu2-named key; its meaning is
  unchanged). 5.6 should rename it.

## Tests run

```text
cd services/orion-gpu-lane-controller && python -m pytest tests -q        -> 109 passed (77 before + 32 new)
python -m pytest services/orion-diffusion-host/tests -q                    -> 41 passed (same CI `safety` job)
python scripts/check_gpu_pool_config.py                                     -> ok (2 launch blocks)
```

Mutation checks, run by hand:
- dropping the reverse-order restart fails a test;
- dropping the profile-var strip fails a test;
- dropping the unload fence checkpoint before `stop` fails a test.

## Evals run

```text
None. The controller has no eval harness (unchanged from stage 4). The worked-example tests drive
the full bus → resolve → execute path with a fake Docker runner and fake HTTP; the live proof is
5.3's acceptance checks 1-4.
```

## Docker/build/smoke checks

```text
Not run. The controller runs on circe and this PR changes no live path (agent-gpu2 is still
bridged). Not deployed, per instructions. UNVERIFIED live.
```

## Review findings fixed

The code-review subagent found no must-fix issues.

- **Finding:** unload had no fence checkpoint between the llama.cpp idle check and `stop`.
  - Fix: `authority()` now runs immediately before the seat stop.
  - Evidence: `test_unload_rechecks_the_fence_between_idle_check_and_stop`, which fails on mutation.
- **Finding:** `start()` gave resume and ready-wait each a full `timeout_sec`, which could outrun
  the pool's stuck-actuator timer.
  - Fix: one shared deadline per role.
- **Finding:** `observe()` returned `{}` on an unparseable checkout, so a `status` reconcile would
  fault the card as ambiguous.
  - Fix: it falls back to the fixed bridge pair; this is deleted in 5.6.
  - Evidence: `test_config_unloadable_observe_falls_back_to_the_bridge_pair`.
- **Finding:** untested paths:
  - bridge-parity `observed`;
  - multi-evictee reverse order;
  - drained-but-not-stopped un-drain;
  - superseded generation during rollback;
  - unload with an evicted role that fails to start;
  - un-drain of a draining seat whose stop fails;
  - busy while the generic lock is held.
  - Fix: a test for each.
- **Finding (nit):** with no profile, an `ATLAS_*_PROFILE_NAME` inherited from the controller's
  env would silently override compose's default.
  - Fix: `profile_var` is removed from the process env when there is no profile.
  - Evidence: `test_no_profile_strips_an_inherited_profile_var`.
- **Finding (nit):** the `FakeDocker` env filter could break on a host that exports
  `CUDA_VISIBLE_DEVICES`.
  - Fix: it now records only the vars the controller changed.
- **Finding (nit):** a `LaunchPlan` docstring gave the wrong order for unload.
  - Fix: docstring corrected.
- **Not changed (noted for 5.3):**
  - the generic path doesn't update `gpu2._state`, so `GET /v1/gpu-slots/circe-gpu2/status` won't
    show generic progress (the pool reads bus results, not that route);
  - `ready()` accepts `{"ready": true}` or `{"status": "ok"}` for any role, which is safe for
    diffusion because its `/ready` returns 503 when not ready.

## Restart required

```bash
# circe, only when deploying (not required for 5.2 to be harmless). Same commit as athena.
git -C /mnt/scripts/Orion-Sapienform pull
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
```

A pull alone is not enough: the controller parses the YAML with code baked into its image (5.1
correction 1).

## Risks / concerns

- **Severity: low.**
  - Concern: two failure-reason names change once 5.3 moves agent-gpu2 onto this path.
    `burst_upstream_not_idle` becomes `upstream_not_idle:agent-gpu2`, and the reasons gain
    `:<role>` suffixes.
  - Mitigation: the pool doesn't branch on these strings; only human `gpu_pool_events` queries do.
- **Severity: low.**
  - Concern: on the generic path the 27B's readiness wait becomes the YAML `timeout_sec: 900`,
    where the bridge used `GPU2_MODEL_READY_TIMEOUT_SEC=600`.
  - Mitigation: the pool already budgets from the YAML value.
- **Severity: low.**
  - Concern: `gpu2.py` and `launch_exec.py` duplicate the same steps until 5.6.
  - Mitigation: the duplication is deliberate, so the bridge stays byte-for-byte unchanged while
    both exist.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2409

🤖 Generated with [Claude Code](https://claude.com/claude-code)
