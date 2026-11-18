# Standard heartbeat for five services

## Summary

- The task said these five services send no heartbeat. That was wrong for four of them. A 40-second listen on `orion:system:health` (2026-10-20) showed gpu-cluster-power, whisper-tts, vision-edge and vision-frame-router already heartbeating. Each one used its own hand-written loop and sent a made-up host name ("psu-node", "whisper-node", "vision-edge-node", or no name at all). orion-rag also has a loop, but no container runs it anywhere.
- gpu-cluster-power, orion-rag, whisper-tts and vision-frame-router now use the shared heartbeat (`HeartbeatOnly` from `orion/core/bus/bus_service_chassis.py`). Each one has its own bus connection and reports its real host name (`NODE_NAME`). Each old loop is deleted, so every service has only one publisher.
- vision-edge keeps its own loop on purpose. The shared heartbeat always says "ok". vision-edge says "degraded" when its camera is blind, which is the fix for the 2026-10-11 frozen-frame incident. vision-edge now reports the real host name too.
- vision-frame-router keeps its router counters in the heartbeat `details`. They now come from a provider function that copies the data safely, because the shared heartbeat reads it from a worker thread.
- Equilibrium now expects gpu-cluster-power, vision-edge, vision-frame-router and whisper-tts. If one of them goes silent, it counts as "down". orion-rag is not on the list.

## Outcome moved

Equilibrium keys each service by `service@node`. Before this patch the live keys were `gpu-cluster-power@psu-node`, `whisper-tts@whisper-node`, `vision-edge@vision-edge-node` and `vision-frame-router@unknown`. After deploy they will be `<service>@athena`. Before, all four could go silent and no alarm would fire, because none of them was on the expected list. After, their silence raises `distress_score`.

## Current architecture

- gpu-cluster-power (`app/api.py`): a FastAPI startup hook ran a 30s loop on the PSU command listener's bus.
- orion-rag (`app/main.py`): a daemon thread ran its own asyncio loop every 30s.
- whisper-tts (`app/main.py`): a 30s task on the TTS/STT listeners' bus.
- vision-frame-router (`app/main.py`, `app/metrics.py`): `FrameRouterService._health_loop` ran every `HEALTH_INTERVAL_SECONDS` with a rich `details` dict and no node.
- vision-edge (`app/main.py`): a 30s loop that sends ok or degraded based on the camera.
- Equilibrium's expected list had 20 names, and none of these five was on it.

## Architecture touched

- Five services: lifespan or startup/shutdown wiring, settings, and compose.
- Equilibrium's `.env_example`: the expected list.
- `scripts/check_system_health_producers.py`: the floor `MIN_EXPECTED_SITES` goes from 8 to 6. Four `SystemHealthV1(...)` construction sites moved into the chassis. The gate still checks the chassis's single construction site.
- No bus channel or schema changes. `orion:system:health` and `SystemHealthV1` are unchanged.
- I did not edit `bus_service_chassis.py` or `resilience.py`.

## Files changed

- `services/orion-gpu-cluster-power/app/api.py`: `build_heartbeat_chassis()`; start the chassis at startup and stop it at shutdown; skip it when the bus is disabled; delete the old loop.
- `services/orion-gpu-cluster-power/app/settings.py`, `.env_example`, `docker-compose.yml`: add `HEARTBEAT_INTERVAL_SEC`; compose now passes `NODE_NAME`.
- `services/orion-rag/app/main.py`: async startup and shutdown hooks for the chassis; delete the heartbeat thread.
- `services/orion-rag/app/settings.py`, `.env_example`, `docker-compose.yml`: add `NODE_NAME` and `HEARTBEAT_INTERVAL_SEC`.
- `services/orion-whisper-tts/app/main.py`: chassis in `startup()`/`shutdown()`; `/health` and the heartbeat share `BOOT_ID`; delete the old loop.
- `services/orion-whisper-tts/app/settings.py`, `.env_example`, `docker-compose.yml`: add `NODE_NAME` and `HEARTBEAT_INTERVAL_SEC`.
- `services/orion-whisper-tts/README.md`, `app/cuda_watchdog.py`: update stale `heartbeat_loop` mentions.
- `services/orion-whisper-tts/tests/test_main_startup_cuda_gating.py`: stub the chassis so these tests never open a real Redis connection.
- `services/orion-vision-frame-router/app/main.py`: chassis in the module lifespan, with `details` from `service.health_details`; delete `_health_loop`.
- `services/orion-vision-frame-router/app/metrics.py`: `make_health_envelope` becomes `make_health_details`, which builds its counts from copies of the data.
- `services/orion-vision-frame-router/app/settings.py`, `docker-compose.yml`: add `NODE_NAME` (compose only, read from the root `.env`).
- `services/orion-vision-edge/app/main.py`, `routes.py`, `settings.py`, `docker-compose.yml`: send the real `NODE_NAME` in the heartbeat and `/health`; a docstring says why vision-edge does not use the shared heartbeat.
- `services/orion-equilibrium-service/.env_example`: four names added to the expected list.
- `scripts/check_system_health_producers.py`: the floor goes from 8 to 6, with the reason in a comment.
- New tests: `test_heartbeat_chassis.py` in gpu-cluster-power, rag, whisper-tts and vision-frame-router; `test_heartbeat_node.py` in vision-edge; `test_expected_services_template.py` in equilibrium.

## Expected-services decisions

- **gpu-cluster-power: added.** It runs on athena with `restart: unless-stopped` and is heartbeating live.
- **whisper-tts: added.** It moved from circe to athena's T10 on 2026-09-21 (PR #2276). athena's bring-up now manages it (`mesh-utilities/common/exclude_services.txt`). It is running now. It is not on demand.
- **vision-edge: added.** It runs on athena with `restart: unless-stopped` and is heartbeating live. It is only excluded from the env-refresh script, not from bring-up.
- **vision-frame-router: added.** It runs on athena with `restart: unless-stopped` and is heartbeating live.
- **orion-rag: not added.** It is in athena's `exclude_services.txt`, it has no container, and it sends no heartbeat. Adding it would make it read as "down" forever.
- **Mesh-guardian:** none of the four is in `config/mesh_remediation_roster.yaml`. The guardian only restarts services listed in that roster. Being on the expected list changes `distress_score` only. A test pins this.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the heartbeat `node` field now holds the real host name for all five services. The heartbeat interval goes from 30s to 10s for gpu-cluster-power, rag and whisper-tts. Each payload carries its own `heartbeat_interval_sec`, so equilibrium's grace period adjusts by itself.
- Compatibility notes: nothing reads the old fake node names (checked with grep across services, orion, config and scripts).

## Env/config changes

- Added keys: `HEARTBEAT_INTERVAL_SEC=10` for orion-gpu-cluster-power, orion-rag and orion-whisper-tts. `NODE_NAME` is passed through compose from the root `.env`; it is not a new service key.
- Changed value: `EQUILIBRIUM_EXPECTED_SERVICES` has four more names.
- Removed or renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced:
  - Running `python scripts/sync_local_env_from_example.py` from the worktree reads the live `.env` files in the primary checkout. The default key set does not include these keys, so I ran it again with `--all-keys` for the three services, which added `HEARTBEAT_INTERVAL_SEC`.
  - `EQUILIBRIUM_EXPECTED_SERVICES` showed as "diverged", but the live value was exactly the old template value. I ran `--all-keys --force` for orion-equilibrium-service only. The dry run showed that this key was the only change.
  - Running the sync from `/mnt/scripts/Orion-Sapienform` changed nothing.
- Skipped keys requiring operator action: none.

## Tests run

All runs use `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/<svc>/tests -q -p no:cacheprovider` with `PYTHONPATH=services/<svc>:.`. None of these services has its own CI workflow.

```text
service                   baseline (origin/main)   this branch
orion-gpu-cluster-power   1 passed                 7 passed
orion-rag                 1 passed                 6 passed
orion-whisper-tts         87 passed                91 passed
orion-vision-frame-router 109 passed               115 passed
orion-vision-edge         22 passed (SOURCE set)   25 passed (SOURCE set)
orion-equilibrium-service 238 passed               241 passed
tests/test_system_health_producers.py + tests/test_heartbeat_only_chassis.py: 9 passed
scripts/check_system_health_producers.py: OK (6 sites)
scripts/check_env_template_parity.py: PASS
```

vision-edge without `SOURCE` set fails at collection in `test_camera_reconnect.py`. That failure is the same on origin/main. The new vision-edge test passes on its own without `SOURCE`.

`check_service_env_compose_parity.py` fails for gpu-cluster-power (`API_TOKEN`) and rag (`VECTOR_DB_CREATE_IF_MISSING`). Both keys are already missing from compose on origin/main, and neither key is from this patch.

## Evals run

```text
None. These services have no eval harness. This is liveness plumbing; the live check is the post-deploy subscribe below.
```

## Docker/build/smoke checks

```text
Pre-patch live evidence: a 40s subscribe on orion:system:health; equilibrium:state HKEYS shows
gpu-cluster-power@psu-node, whisper-tts@whisper-node, vision-edge@vision-edge-node, vision-frame-router@unknown.
Post-patch heartbeat: UNVERIFIED. Not deployed, per instructions.
Verify after deploy:
  timeout 40 redis-cli -u redis://100.92.216.81:6379/0 subscribe orion:system:health | grep -E 'gpu-cluster-power|whisper-tts|vision-edge|vision-frame-router'
Expected result: "node":"athena" for all four.
```

## Review findings fixed

- Finding: gpu-cluster-power built a heartbeat even with `ORION_BUS_ENABLED=false`, which ran a loop of no-op publishes. rag skips it in that case.
  - Fix: skip the heartbeat when the bus is disabled.
  - Evidence: `test_bus_disabled_starts_no_heartbeat` (gpu-cluster-power: 7 passed).
- The reviewer found no material bugs. The other two notes are listed under Risks.

## Restart required

Run after merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-gpu-cluster-power up -d --build
scripts/safe_docker_build.sh orion-whisper-tts up -d --build
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
scripts/safe_docker_build.sh orion-vision-edge up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

orion-rag runs nowhere, so it needs no restart.

## Risks / concerns

- Severity: low. Concern: after deploy, the old keys (`gpu-cluster-power@psu-node` and the others) stay in equilibrium state as "down" until `EQUILIBRIUM_STATE_RETENTION_SEC` (1h) prunes them. This gives a one-time bump in `distress_score`. Mitigation: none needed; the bump clears itself.
- Severity: low. Concern: if the heartbeat cannot connect at boot, nothing retries it until the process restarts. Mitigation: equilibrium now shows that state honestly as "down". A retry belongs in the chassis, which was out of scope here.
- Severity: low. Concern: vision-edge still uses its own loop. Mitigation: move it to the shared heartbeat when the chassis can report a per-tick status.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2811
