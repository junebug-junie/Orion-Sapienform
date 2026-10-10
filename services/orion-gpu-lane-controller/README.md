# orion-gpu-lane-controller

The GPU pool's actuator on **circe**. When the pool (`orion-gpu-pool`, athena) decides a card should
hold a different model, it sends a `GpuActuateV1` on the bus; this service runs that role's `launch:`
block from `config/gpu_pool.yaml` against circe's own docker daemon (drain, stop, start, wait for
ready, roll back on failure) and answers on the bus. It has no HTTP control surface: `GET /health`
(503 when it cannot load its own `config/gpu_pool.yaml`, with the error and the fix)
is the only route.

## Why this exists

The pool and everything that asks for GPUs run on **athena**; the containers that move live on
circe. `docker compose` can only reach its own host's daemon, so this is the seam that crosses the
host boundary -- a bus consumer that only ever runs compose files and services named in the checked-in
YAML, never a caller-supplied container, path or env value.

## Bus actuation (stage 4.2 onward)

Specs: [`docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md`](../../docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md),
[`docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md`](../../docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md).

**Bus contract** (`orion/schemas/gpu_pool.py`). Requests for another `actuator` name are ignored.
For ours the controller publishes on `orion:gpu_pool:actuate:result`: `accepted`, then one
`progress` per phase (`draining`, `stopping`, `starting`, `ready_wait`, `rolling_back`), then one
terminal `succeeded` | `failed` | `refused`, with `observed` = container state
(`running|exited|absent|unknown`) of every role whose `launch` names this actuator (stage 5.2; today
`agent-gpu2` and `diffusion`). A failed load carries `restored` (were diffusion's containers put
back); `restored` absent on a failed load means no rollback ran because nothing had been evicted
yet -- read `observed`.

- **Checkout and image move together.** The controller re-reads `config/gpu_pool.yaml` from the
  bind-mounted host checkout on every request, but parses it with the `orion/` code baked into its
  image. A `git pull` that changes the YAML schema without rebuilding the image makes every action
  refuse `config_unloadable:*` (safe, but nothing loads). Pull, then
  `scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build`, at the same commit athena's
  pool runs (the `launch_digest` fence).
- **Refusals** (`reason`): `invalid_request:<field>`, `unknown_role`, `role_not_on_this_actuator`
  (includes a role with no `launch` block for this actuator -- the only "off switch"),
  `cards_mismatch`, `launch_digest_mismatch`, `profile_not_allowed`, `not_a_swap_seat`,
  `no_launch_block:<evicted role>`, `deadline_passed`, `stale_generation`, `busy`,
  `fence_state_unreadable:*`, `fence_state_unwritable:*`, `config_unloadable:*`. A refusal never
  advances the generation.
- **Generation fence.** The accepted generation is written (fsync + atomic rename) to
  `GPU_POOL_FENCE_STATE_PATH` on the `gpu-lane-controller-state` volume *before* any container is
  touched; anything `<=` it is refused. `launch_exec`'s authority checkpoints re-check that the running
  action is still the newest generation and that the checkout's digest has not changed.
- **Idempotency.** A replayed `action_id` gets its recorded terminal result back; a replay of the
  in-flight one gets `progress`. Neither starts a second action.
- **`status`** is a read: no digest check, no fence. It re-publishes the card set's last recorded
  result (so a restarted pool adopts it by its own `action_id`; skipped while an action is running),
  then answers with `observed`, `in_flight` and `last_action_id`. `reason` is only a human summary.
  If this checkout's YAML cannot be parsed, `observed` is empty (no fixed fallback pair since 5.6).
- **Controller restart mid-action**: on boot the in-flight action is recorded as
  `failed reason=interrupted_by_controller_restart`, never re-run.
- **Not re-checked here:** thermal and lease recall -- pool policy. Drain-before-stop and
  upstream-idle-before-stop stay here.

## Generic launch actuator (stage 5.2)

`app/launch_exec.py` runs any swap seat's `launch:` block, so a new card or a new model option on a
card is YAML + compose only. All docker calls go through `app/compose.py` (`docker` is the only
binary; subcommands `ps`/`stop`/`up -d --no-build --no-deps`, each naming exactly one service --
`docker-compose.atlas-workers.yml` holds several llama.cpp workers, and a bare `stop` would take them
all down).

- **What the bus message can choose:** a role and (for `load`) a profile. The profile must be in the
  role's `launch.profiles`, else `profile_not_allowed`. Compose file, service, compose profile, env
  file and env var *names* all come from this checkout's YAML (fenced by `launch_digest`). The only
  values the controller sets, as compose interpolation variables in the `docker compose` process
  env: `launch.cuda_env` = the role's card `index`es joined by commas, and `launch.profile_var` =
  the chosen profile (when none, the variable is removed from the process env so compose's own
  default applies, never a value inherited from the controller's environment).
- **load:** each evicted role that is running is drained (if it has `launch.drain`) and stopped;
  the seat is `up -d --no-build --no-deps`'d with its env, un-drained if it has `drain`, and waited on
  for `launch.ready` up to `launch.timeout_sec` (one budget covering resume + ready). On a failure after an evicted role was touched: stop
  the seat, restart the stopped evicted roles in reverse order (each waits for ready), un-drain any
  drained but not stopped -> `failed restored=true|false`.
- **unload:** a running `kind: llm` seat must have every llama.cpp `/slots` idle
  (`upstream_not_idle:<role>`); drain (if any) -> stop the seat, then start every evicted role.
- **Ready** means the `ready` path answers `{"ready": true}` (diffusion-host `/ready`) or
  `{"status": "ok"}` (llama.cpp `/health`). A new service's ready endpoint must return one of those.
- **Failure reasons** carry the role: `container_state_not_safe:<role>`, `drain_timeout:<role>`
  (`GPU_LANE_DRAIN_TIMEOUT_SEC`), `stop_failed|stop_unconfirmed|startup_failed:<role>`,
  `model_readiness_timeout:<role>`, `seat_and_evicted_both_running`, plus `:restoration_failed`.
- Residents (`swap: null`) are never loaded directly (`not_a_swap_seat`): they come back only as a
  seat's evictions.
- Log line per action: `launch_exec load role=agent-gpu2 service=atlas-agent-burst
  env=ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2 ATLAS_AGENT_BURST_PROFILE_NAME=<profile>`.

Keys: `GPU_POOL_ACTUATOR_NAME`, `GPU_POOL_FENCE_STATE_PATH`, `GPU_LANE_DRAIN_TIMEOUT_SEC`,
`GPU_LANE_COMMAND_TIMEOUT_SEC`, `GPU_LANE_REPO_ROOT` (see `.env_example`).

## Deleted in stage 5.6 (kill means kill)

- The stage-4 gpu2 bridge: `app/gpu2.py` (fixed `diffusion`/`agent-burst` targets with a literal
  `CUDA_VISIBLE_DEVICES=2`), `BRIDGE_TARGETS`, the `swap.load`/`swap.unload` YAML verbs (now refused by
  the config validator), the `bridge_verb_unsupported`/`bridge_cannot_set_profile` refusals and the
  `gpu2_disabled` switch.
- The GPU1 affect/agent flip: `app/lane_control.py`, `POST /v1/gpu-lane/flip`,
  `GET /v1/gpu-lane/status`, `GET /v1/gpu-slots/{slot}/status` (no caller in any service; orion-thought's
  slot pre-check was deleted in 5.4) and `GPU_LANE_CONTROLLER_TOKEN`.
- Keys: `GPU2_ENABLED`, `GPU2_DIFFUSION_URL`, `GPU2_AGENT_URL`, `GPU2_MODEL_READY_TIMEOUT_SEC`,
  `GPU_LANE_CONTROLLER_TOKEN`. Renamed: `GPU2_POOL_FENCE_STATE_PATH` -> `GPU_POOL_FENCE_STATE_PATH`
  (same file), `GPU2_DRAIN_TIMEOUT_SEC` -> `GPU_LANE_DRAIN_TIMEOUT_SEC` (same meaning, default 300).
- No enable/disable key remains. **Emergency stop (stage 5.7): pause actuation on the pool** -- Hub GPU
  pool panel "Emergency stop", or control verb `pause_actuation` (persisted; see
  `services/orion-gpu-pool/README.md`). That stops every new load/unload but lets one already running
  finish; to stop that too, stop this container (`docker stop orion-circe-gpu-lane-controller`).
  `GPU_POOL_ACTUATE_ROLES` no longer exists.
- `launch_digest` is unchanged by the deletion (the always-null bridge fields stay in its hashed
  body), so a pool and a controller on either side of 5.6 still agree.

## Docker-outside-of-docker

This container talks to circe's **host** docker daemon over a bind-mounted
`/var/run/docker.sock` — it never runs a nested daemon of its own. Every
container it starts or stops is a normal top-level container on circe, a
sibling of this one, not a child of it. The repo checkout is bind-mounted
read-only at `/repo` so `docker compose` can see the same
`services/*/docker-compose.yml` / `.env` files an operator running commands
by hand on circe would.

## Deploy (circe only)

```bash
git -C /mnt/scripts/Orion-Sapienform pull --ff-only     # the same commit athena's pool runs
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
curl http://localhost:8090/health   # 200 {"ok": true, "config_loadable": true}; 503 + config_error + fix when gpu_pool.yaml will not load
```

**Fence file operations.** It lives on the pinned volume `orion-gpu-lane-controller-state`
(`docker compose down -v` deletes it and resets the accepted generation to 0 -- don't). If it is
corrupt, every request is refused `fence_state_unreadable:*` until an operator inspects it:
`docker exec orion-circe-gpu-lane-controller cat /state/gpu2_pool_fence.json`. Only delete it with
the pool's actuation paused (`pause_actuation`), since the pool's next generation must then be above
whatever the controller last accepted.
