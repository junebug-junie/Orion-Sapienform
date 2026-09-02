# PR report: `orion-gpu-lane-controller` (Phase 1 of the GPU1 flex-lane pattern)

## Summary

- New service `orion-gpu-lane-controller`, deployed on **circe only**: a
  narrow HTTP control surface that exclusively flips GPU1
  (V100-SXM2-32GB) between `orion-affectgpt-worker` and `orion-llamacpp-host`'s
  `atlas-agent` worker (the LLM-gateway `agent` route).
- `GET /health`, `GET /v1/gpu-lane/status` (no auth, read-only), `POST
  /v1/gpu-lane/flip` (bearer-token auth, **fails closed** with a 503 if
  `GPU_LANE_CONTROLLER_TOKEN` is unset).
- `flip()` is idempotent (no-op if the target is already the sole running
  lane), serialized against concurrent calls, always names `atlas-agent`
  explicitly in every command against the 4-service
  `docker-compose.atlas-workers.yml` file, and forces
  `ATLAS_AGENT_CUDA_VISIBLE_DEVICES=1` at invocation time regardless of
  whatever's already in `.env.atlas`.
- Docker-outside-of-docker: talks to circe's host docker daemon over a
  bind-mounted socket; never runs a nested daemon.

This is Phase 1 of 3 (see the approved plan) — the mechanism itself, not yet
wired into cortex-exec/Hub. Standalone and independently deployable/curlable.

## Outcome moved

GPU1 currently sits pinned to `orion-affectgpt-worker` permanently, holding
~18.4GB VRAM whether or not it's in use, with no way to reclaim it for the
`agent` LLM lane without an operator manually editing compose files on
circe. This service makes that swap a single authenticated HTTP call.

## Current architecture (before this patch)

- `services/orion-affectgpt-worker/docker-compose.yml` pins GPU1 via
  `device_ids: ["1"]`, `restart: unless-stopped` — no programmatic way to
  free it.
- `orion-llamacpp-host`'s `atlas-agent` worker (port 8014, the `agent`
  LLM-gateway route) is a separate compose service in
  `docker-compose.atlas-workers.yml`, GPU-selected via
  `ATLAS_AGENT_CUDA_VISIBLE_DEVICES`, gated behind compose profile
  `agent-split`.
- `orion-cortex-exec`/`orion-hub` run on **athena**, not circe. cortex-exec's
  existing `skills.docker.compose_service_bringup.v1` verb can only ever run
  `docker compose` against its own host's repo checkout — it structurally
  cannot reach circe's containers. No SSH/remote-exec skill exists anywhere
  in the repo (checked before choosing this design).

## Architecture touched

New service only — nothing existing was modified in this phase.

## Files changed

- `services/orion-gpu-lane-controller/{README.md,.env_example,docker-compose.yml,Dockerfile,requirements.txt}` — new service scaffold
- `services/orion-gpu-lane-controller/app/{settings.py,lane_control.py,main.py}` — settings, the flip/status mechanism, the FastAPI surface
- `services/orion-gpu-lane-controller/tests/{test_lane_control.py,test_api.py}` — 21 tests, all mocked (no real docker calls)

## Schema / bus / API changes

- Added: new HTTP API (`GET /health`, `GET /v1/gpu-lane/status`, `POST
  /v1/gpu-lane/flip`) on a brand-new service. No bus channel/schema changes
  — this service takes heartbeat-only bus participation (no intake).
- Removed: none.
- Renamed: none.
- Behavior changed: none (nothing existing wired to this yet — Phase 2).
- Compatibility notes: n/a, new service.

## Env/config changes

- Added keys (`services/orion-gpu-lane-controller/.env_example`, all new):
  `LOG_LEVEL`, `NODE_NAME`, `ORION_BUS_ENABLED`, `ORION_BUS_ENFORCE_CATALOG`,
  `ORION_BUS_URL`, `HEARTBEAT_INTERVAL_SEC`, `GPU_LANE_CONTROLLER_TOKEN`
  (**required** for the flip endpoint to do anything), `GPU_LANE_CONTROLLER_HOST_PORT`,
  `GPU_LANE_HOST_REPO_ROOT`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (new file).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  **could not** — this is a brand-new service with no existing `.env`
  anywhere for the script to bootstrap from (known limitation: the sync
  script reads `.env_example` from the *primary checkout*, not this
  worktree, and has no path to create a `.env` for a service it's never
  seen before).
- Skipped keys requiring operator action: **all of them, on circe
  specifically.** This service is circe-only. Juniper needs to, on circe:
  ```bash
  ssh circe@circe
  cd /mnt/scripts/Orion-Sapienform
  cp services/orion-gpu-lane-controller/.env_example services/orion-gpu-lane-controller/.env
  # generate and set GPU_LANE_CONTROLLER_TOKEN, e.g.:
  python3 -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
  The same token value will need to go on cortex-exec's (athena) side in
  Phase 2 — not yet built, so nothing to set there yet.

## Tests run

```text
PYTHONPATH=/mnt/scripts/Orion-Sapienform-circe-gpu1-lane-flex \
  /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-gpu-lane-controller/tests -q
21 passed, 18 warnings (pre-existing pydantic protected-namespace warnings, unrelated to this change)
```

```text
PYTHONPATH=. .venv/bin/python scripts/check_service_env_compose_parity.py orion-gpu-lane-controller
-> N/A (declares env_file:, no compose-level overrides to drift)

PYTHONPATH=. .venv/bin/python scripts/check_env_key_single_source.py
-> OK: 1 owned env key(s), no drifted copies.

PYTHONPATH=. .venv/bin/python scripts/check_settings_defaults.py orion-gpu-lane-controller
-> N/A (script is allowlist-scoped to orion-actions only, not a general gate)

git diff --check
-> clean
```

Note: CLAUDE.md §17 names `check_env_template_parity.py`/
`check_schema_registry.py`/`check_bus_channels.py` as the `make agent-check`
chain — confirmed these three don't exist in this repo (the `Makefile`
itself documents this gap). Ran the real equivalents above instead.

## Evals run

None — no eval harness exists for this service (it has no model-quality
surface; it's pure control-plane orchestration). Not adding one for this
phase; flagging as a gap only if Juniper wants one later.

## Docker/build/smoke checks

Not run — this service must build/run on circe (docker-outside-of-docker
against circe's own host socket), which I can't reach interactively (`sudo`
on circe needs a password per prior sessions' notes). Deploy commands for
Juniper:

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform
git fetch && git checkout feat/circe-gpu1-lane-flex   # once pushed, or merged
cp services/orion-gpu-lane-controller/.env_example services/orion-gpu-lane-controller/.env
# set GPU_LANE_CONTROLLER_TOKEN in that .env

docker compose \
  --env-file .env \
  --env-file services/orion-gpu-lane-controller/.env \
  -f services/orion-gpu-lane-controller/docker-compose.yml \
  up -d --build

curl http://localhost:8090/health
curl http://localhost:8090/v1/gpu-lane/status
```

Then a live flip test (needs the real token):

```bash
curl -X POST http://localhost:8090/v1/gpu-lane/flip \
  -H "Authorization: Bearer $GPU_LANE_CONTROLLER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"target": "agent"}'
# then confirm on circe: nvidia-smi (GPU1 tenant should change),
# and orion-llm-gateway's GET /routes should report "agent" up.
```

## Review findings fixed

- Finding: `GET /v1/gpu-lane/status` called the synchronous
  `lane_control.get_status()` directly from an async handler — two blocking
  `docker compose ps` subprocess calls (up to 60s combined) on uvicorn's
  single event loop (no `--workers`) would freeze every other concurrent
  request, including `/health`.
  - Fix: wrapped in `asyncio.to_thread` (`app/main.py`), matching the
    pattern `flip()` already used correctly.
  - Evidence: `services/orion-gpu-lane-controller/app/main.py` — `get_status`
    route.
- Finding: `flip()` had no mutual-exclusion guard — a read-then-act
  (TOCTOU) snapshot with no lock, so two overlapping `POST
  /v1/gpu-lane/flip` calls (double-click, or a client retry after its own
  timeout) could both read a stale snapshot, both conclude "not a no-op",
  and both run `docker compose stop`/`build`/`up -d` concurrently against
  the same two targets — producing exactly the "both"/"neither" state the
  module's own docstring calls "a bug, not a valid state."
  - Fix: added `_FLIP_LOCK` (module-level `asyncio.Lock`); a flip already in
    progress makes a concurrent call return `"busy"` (HTTP 409) immediately
    rather than queueing behind a sequence that can legitimately take
    several minutes, or racing it.
  - Evidence: `services/orion-gpu-lane-controller/app/lane_control.py` —
    `_FLIP_LOCK`, `flip`, `_flip_locked`; new tests
    `test_flip_returns_busy_without_racing_when_already_in_progress`
    (`tests/test_lane_control.py`) and `test_flip_returns_409_when_busy`
    (`tests/test_api.py`).

## Restart required

No restart required on athena (nothing there yet touches this). On circe:
the `docker compose up -d --build` command above is the first deploy, not a
restart of anything existing.

## Risks / concerns

- Severity: medium
  - Concern: `flip()`'s health-poll classifies "settled" purely on
    container state (`running` + healthy-or-no-healthcheck), not on the
    llama-server actually finishing model load and accepting inference
    requests. A 27B GGUF cold load can outlast the container reaching
    "running."
  - Mitigation: `atlas-agent`'s own `docker-compose.atlas-workers.yml`
    healthcheck (`curl -f http://localhost:8080/health`, `start_period:
    60s`) is what actually gates "healthy" vs "running_no_healthcheck" — if
    llama-server's own `/health` doesn't come up until the model is loaded
    (need to confirm this against `orion-llamacpp-host`'s actual server
    code in a later phase), this is already covered; flagging as unverified
    rather than assumed.
- Severity: low
  - Concern: no eval/smoke harness for this service.
  - Mitigation: not needed yet — pure control-plane orchestration, no
    model-quality surface. Revisit if Phase 3's profile work suggests one.

## PR link

Not yet pushed/opened — will push `feat/circe-gpu1-lane-flex` and open the
PR once Juniper confirms this Phase 1 checkpoint before I continue to Phase 2.
