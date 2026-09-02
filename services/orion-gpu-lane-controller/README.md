# orion-gpu-lane-controller

Circe-local control surface that exclusively flips GPU1 (Tesla V100-SXM2-32GB)
between two lanes:

- **`affect`** — `orion-affectgpt-worker` (rare, on-demand affect assessment)
- **`agent`** — `orion-llamacpp-host`'s `atlas-agent` worker (the LLM-gateway
  `agent` route), forced onto GPU1 via an explicit
  `ATLAS_AGENT_CUDA_VISIBLE_DEVICES` override at invocation time regardless
  of whatever's already in `.env.atlas`

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
