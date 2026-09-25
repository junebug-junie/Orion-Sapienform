# orion-gpu-lane-controller

Circe-local control surface that exclusively flips GPU1 (Tesla V100-SXM2-32GB)
between two lanes:

- **`affect`** — `orion-affectgpt-worker` (rare, on-demand affect assessment)
- **`agent`** — `orion-llamacpp-host`'s `atlas-agent` worker (the LLM-gateway
  `agent` route), forced onto GPU1 via an explicit
  `ATLAS_AGENT_CUDA_VISIBLE_DEVICES` override at invocation time regardless
  of whatever's already in `orion-llamacpp-host/.env` (the README there says
  `.env.atlas`; confirmed live on circe 2026-09-02 that's not actually what's
  deployed -- runtime truth over documented convention)

## Why this exists

`orion-cortex-exec` (which already has a generic
`skills.docker.compose_service_bringup.v1` docker-compose skill) and
`orion-hub` both run on **athena**, not circe. That skill can only ever run
`docker compose` against its own host's repo checkout — it structurally
cannot reach circe's containers. This service is the seam that crosses that
host boundary: a small, purpose-built HTTP API, not a generic remote-docker
passthrough. It only ever touches the two compose targets named above.

See `docs/superpowers/pr-reports/` for the fuller design writeup (GPU1 flex
lane, PR TBD).

## API

- `GET /health` — liveness, no auth.
- `GET /v1/gpu-lane/status` — `{"active": "affect"|"agent"|"neither"|"both", "affect": {...}, "agent": {...}}`, no auth (read-only).
- `POST /v1/gpu-lane/flip` `{"target": "affect"|"agent"}` — requires
  `Authorization: Bearer <GPU_LANE_CONTROLLER_TOKEN>`. Idempotent (a no-op if
  `target` is already the sole running lane); otherwise stops the other lane,
  brings the requested one up on GPU1, and polls until it settles (or the
  poll window expires). **Fails closed**: with `GPU_LANE_CONTROLLER_TOKEN`
  unset, every flip request gets `503`, not an open route.

## `docker-compose.atlas-workers.yml` has four services in one file

Every invocation this service makes against that file names `atlas-agent`
explicitly — `stop`, `build`, `up -d`, and `ps` all take the service name as
an argument. A bare `docker compose -f docker-compose.atlas-workers.yml stop`
with no service name would stop `atlas-chat`/`atlas-metacog`/`atlas-fast` too
(the always-on chat/metacog/quick lanes elsewhere on circe) — this service
never does that.

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
cp services/orion-gpu-lane-controller/.env_example services/orion-gpu-lane-controller/.env
# set GPU_LANE_CONTROLLER_TOKEN -- same value goes on cortex-exec's
# GPU_LANE_CONTROLLER_TOKEN (athena side)

docker compose \
  --env-file .env \
  --env-file services/orion-gpu-lane-controller/.env \
  -f services/orion-gpu-lane-controller/docker-compose.yml \
  up -d --build

curl http://localhost:8090/health
curl http://localhost:8090/v1/gpu-lane/status
```

## Non-goals

- **No generic remote-docker API.** Two fixed targets, not a caller-supplied
  service name — see `app/settings.py`'s `_targets()`.
- **No auto-idle flip.** Manual only, matching what Hub's button calls.
  An idle-based auto-trigger is a real, separate, larger patch (needs an
  idle detector and a race check against `orion-hub`'s ambient vision loop)
  — not built here.
- **No concurrent GPU1 sharing.** Exclusive swap only; `affect` and `agent`
  are never both resident on GPU1 by design.


## Optional GPU2 elastic admission

GPU2 diffusion/agent-burst borrowing is additive and defaults off. See the
[ownership ADR](../../docs/architecture/gpu2-elastic-admission.md),
[pre-edit repository/live evidence](../../docs/architecture/gpu2-elastic-evidence.md),
and [consumer-first rollout and rollback](../../docs/runbooks/gpu2-elastic-admission.md)
for this service's exact flags, HTTP contracts and operator commands.
No production env sync, migration, GPU transition or deployment was performed.

GPU2 ordinary transitions never build images. They require a matching durable
intent from `/elastic/target` or the admission broker, atomic diffusion draining,
and authoritative lease/permit closure plus upstream idle evidence to restore.
The "manual only" and build-on-flip behavior above describe the legacy GPU1
adapter only. GPU2 restoration belongs to durable admission reconciliation.

GPU2 control endpoints use the existing internal service/tailnet boundary without
bearer tokens. Durable intent, generation fencing, lease/permit protection, and
atomic diffusion draining remain enforced. The existing GPU1 API is unchanged.

## GPU pool actuation bridge (stage 4.2)

Spec: [`docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md`](../../docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md).

The controller also listens on the bus for the GPU pool asking it to load or unload a model on
gpu2, and does it with the same drain / stop / start / readiness-wait / rollback steps
(`app/gpu2.py` `transition()`) the durable-runs HTTP route uses. Only who asks and the fence differ.

**Who may move gpu2 is one switch, `GPU2_AUTHORITY`:**

| value | gpu2 moved by | fence | bus actuation requests |
| --- | --- | --- | --- |
| `durable` (default, today) | durable-runs `POST /v1/gpu-slots/activate` | callback to `GPU2_AUTHORITY_URL/elastic/status` | answered `refused reason=authority_durable` |
| `pool` | `GpuActuateV1` on `orion:gpu_pool:actuate:request` | pool generation (persisted) + `launch_digest` vs this checkout's `config/gpu_pool.yaml` | executed; HTTP gpu2 activate answers `503 authority_pool` |

Flip to `pool` only as step (3) of the stage 4.5 cutover runbook. GPU1's flip is unaffected either way.

**Bus contract** (`orion/schemas/gpu_pool.py`). Requests for another `actuator` name are ignored.
For ours the controller publishes on `orion:gpu_pool:actuate:result`: `accepted`, then one
`progress` per phase (`draining`, `stopping`, `starting`, `ready_wait`, `rolling_back`), then one
terminal `succeeded` | `failed` | `refused`, with `observed` = container state of `agent-gpu2` and
`diffusion` after the action. A failed load carries `restored` (were diffusion's containers put
back); `restored` absent on a failed load means no rollback ran because nothing had been evicted
yet -- read `observed`.

- **Bridge mapping** (stage 4 only; stage 5 builds the compose call from the role's `launch`):
  `agent-gpu2 load` -> `transition(target="agent-burst")`, `agent-gpu2 unload` ->
  `transition(target="diffusion")`, via the role's `swap.load`/`swap.unload` verbs.
- **Refusals** (`reason`): `authority_durable`, `gpu2_disabled`, `invalid_request:<field>`,
  `unknown_role`, `role_not_on_this_actuator`, `cards_mismatch`, `launch_digest_mismatch`,
  `not_a_bridge_role`, `bridge_verb_unsupported`, `profile_unsupported`, `deadline_passed`,
  `stale_generation`, `busy`, `fence_state_unreadable:*`, `fence_state_unwritable:*`,
  `config_unloadable:*`. A refusal never advances the generation.
- **Generation fence.** The accepted generation is written (fsync + atomic rename) to
  `GPU2_POOL_FENCE_STATE_PATH` on the `gpu-lane-controller-state` volume *before* any container is
  touched; anything `<=` it is refused. `transition()`'s own authority checkpoints re-check that the
  running action is still the newest generation and that the checkout's digest has not changed.
- **Idempotency.** A replayed `action_id` gets its recorded terminal result back; a replay of the
  in-flight one gets `progress`. Neither starts a second transition.
- **`status`** is a read: no digest check, no fence. It re-publishes the card set's last recorded
  result (so a restarted pool adopts it by its own `action_id`), then answers with `observed`.
- **Controller restart mid-action**: on boot (pool authority) the in-flight action is recorded as
  `failed reason=interrupted_by_controller_restart`, never re-run.
- **Not re-checked here any more under `pool`:** thermal, visual-baseline urgency and lease/permit
  closure. Those are pool policy (stage 4.3 guards + recall). Drain-before-stop and
  upstream-idle-before-stop stay here.

Keys: `GPU2_AUTHORITY`, `GPU_POOL_ACTUATOR_NAME`, `GPU2_POOL_FENCE_STATE_PATH` (see `.env_example`).
