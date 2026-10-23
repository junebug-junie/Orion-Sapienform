## Summary

- The mesh guardian (athena) can now see crash loops on **circe**. Before, it read only athena's own Docker, so circe's AffectGPT worker restarted 601 times over 9 days and nothing said so.
- circe's `orion-gpu-lane-controller` already holds circe's Docker socket and already sends a heartbeat. It now reads every container's restart count every 30 s and puts it on that heartbeat.
- The equilibrium service already copies heartbeat details into the snapshot the guardian already reads. The guardian folds circe's counts into its existing crash-loop detector. No new channel, no SSH, no remote Docker socket.
- Every container key now names its host (`circe/<container>@<id12>`), so an athena container and a circe container with the same name never share a history or a card.
- Silence is not calm. If circe's count stops arriving, arrives stale, or reports a failed read, the guardian raises a "blind" card instead of reading zero restarts.
- First CI workflow for the guardian, plus an eval that replays this incident.

## Outcome moved

Replayed through the real hops (30 s read on circe, ~15 s equilibrium lag, 60 s guardian cycle):

- At the conservative cadence (601 restarts over 235 h, one per ~23.5 min) the first card lands at **72 min**, inside the 2 h crash-loop window. At the reported ~22 s cadence it lands at **cycle 2**.
- The whole 9.8-day incident would have raised **40 cards** (one per 6 h re-alert window). Actual: **0**.
- If circe's reporter stops heartbeating, a `container_watch_blind:circe` card lands within **5 min**.

## Current architecture

- `services/orion-mesh-guardian/app/stability.py` `CrashLoopTracker`: alerts when a container's `RestartCount` rises by 3 within 2 h. Built after the 2026-10-02 bus-mirror incident (581 restarts).
- `service.py` `_check_crash_loops` read only `/var/run/docker.sock` on athena.
- Existing seams already in place (verified live 2026-10-23 22:41Z by reading one `orion:equilibrium:snapshot` message): the snapshot holds 85 services, including `gpu-lane-controller@circe` (status ok, heartbeat 5 s old) and `biometrics@circe` with its own `details` keys. So equilibrium does carry per-heartbeat `details` from circe today. No hecate service was in the snapshot (hecate is on-demand).

## Existing-mechanism check (why this seam)

| Candidate on circe | Has circe's Docker socket? | Heartbeats? | Verdict |
|---|---|---|---|
| `orion-gpu-lane-controller` | yes (DooD, for actuation) | yes, SystemHealthV1 via its bus chassis | **chosen**: zero new privilege, zero new channel |
| `orion-biometrics` | no (only `/mnt/docker` read-only for disk usage) | yes | would need a new socket mount |
| `orion-power-guard` | no | yes | would need a new socket mount |
| hardware-watch | runs on athena | - | not on circe |
| SSH / exposed remote socket | - | - | rejected: new credential surface; tailnet SSH from athena is also refused by policy |

The guardian already consumes `orion:equilibrium:snapshot`, and `EquilibriumServiceState.details` already carries the heartbeat's `details` (`services/orion-equilibrium-service/app/service.py` `_evaluate_state` / `_build_service_state`). So the producer is one heartbeat hook and the consumer is one function over a snapshot the guardian already holds.

hecate: no service there holds a Docker socket today, and hecate is on-demand. It is **not** in the expected-hosts list, so it never reads as blind. If a hecate service ever adds the same heartbeat detail, the guardian watches it with no config change.

## Architecture touched

- Contract: `ContainerRestartsV1` + `CONTAINER_RESTARTS_DETAIL_KEY` in `orion/schemas/telemetry/system_health.py` (a typed shape for `SystemHealthV1.details["container_restarts"]`). Registered in `orion/schemas/registry.py`. No new channel.
- Chassis: `Hunter` now accepts `heartbeat_details` (as `BaseChassis` and `Clock` already do). Additive keyword, default None.
- Producer: `orion-gpu-lane-controller` (circe).
- Consumer: `orion-mesh-guardian` (athena).

## Files changed

- `orion/schemas/telemetry/system_health.py`: `ContainerRestartsV1`, detail key.
- `orion/schemas/registry.py`: register it.
- `orion/core/bus/bus_service_chassis.py`: `Hunter(heartbeat_details=...)`.
- `services/orion-gpu-lane-controller/app/container_census.py`: new. Read-only `docker ps -aq` + `docker inspect`; own 30 s loop; heartbeat hook copies the last read; a failed read is published as `error` with empty counts.
- `services/orion-gpu-lane-controller/app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: wire the loop and the hook; `GPU_LANE_CONTAINER_CENSUS_ENABLED=true`, `GPU_LANE_CONTAINER_CENSUS_INTERVAL_SEC=30`.
- `services/orion-gpu-lane-controller/tests/test_container_census.py`: new. `tests/test_api.py`: disable the census in the HTTP fixture.
- `services/orion-mesh-guardian/app/stability.py`: `qualify`, `remote_restart_counts` (blind detection), `CrashLoopTracker.observe(read_hosts=...)` so an unread host keeps its history; card text names the host.
- `services/orion-mesh-guardian/app/service.py`: `_check_crash_loops` merges local + remote; a local socket error no longer blinds circe.
- `services/orion-mesh-guardian/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: three new keys (below).
- `services/orion-mesh-guardian/tests/test_remote_crash_loops.py`: new.
- `services/orion-mesh-guardian/evals/test_remote_crash_loop_incident_replay.py`: new.
- `.github/workflows/mesh-guardian-tests.yml`: new (the guardian had no CI). `.github/workflows/gpu-lane-controller-tests.yml`: also triggers on `system_health.py`.

## Schema / bus / API changes

- Added: `ContainerRestartsV1` (`read_at`, `counts: {"<name>@<id12>": int}`, `error`), carried in `SystemHealthV1.details["container_restarts"]`.
- Removed: none
- Renamed: none
- Behavior changed: guardian crash-loop keys and cards are host-qualified, **including athena's own**: `crash_loop:<name>` is now `crash_loop:athena/<name>`, and the Hub card subject (`service_id`) is now `athena/<name>` instead of `<name>`. Anything that matched on the old subject sees a new identity. circe cards are `crash_loop:circe/<name>`. New card kind `container_watch_blind:<host>` (warning).
- Compatibility notes: `details` is a free-form dict on `SystemHealthV1`, so old consumers ignore the new key. Alert-gate keys change once on deploy (in-memory only), so a crash loop already in progress may get one extra card.

## Env/config changes

- Added keys: `GPU_LANE_CONTAINER_CENSUS_ENABLED=true`, `GPU_LANE_CONTAINER_CENSUS_INTERVAL_SEC=30` (gpu-lane-controller); `MESH_GUARDIAN_REMOTE_CONTAINER_WATCH_ENABLED=true`, `MESH_GUARDIAN_REMOTE_CONTAINER_HOSTS=circe`, `MESH_GUARDIAN_REMOTE_CONTAINER_STALE_SEC=300` (mesh-guardian)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes, both services
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-mesh-guardian orion-gpu-lane-controller`: yes (athena)
- skipped keys requiring operator action: circe's own `services/orion-gpu-lane-controller/.env` is on circe; the settings defaults are ON, so the census runs even if circe's `.env` lacks the keys.

## Metric quality gate (RestartCount from a remote host)

1. Provenance: Docker's own `RestartCount` from `docker inspect` on circe (`container_census.read_restart_counts`), the same field the guardian already reads on athena through the Engine API.
2. Independence: one remote copy of an already-gated signal, not a new signal. Host-qualified keys keep it from merging with athena's.
3. Theory anchor: a restart-policy container that keeps exiting raises `RestartCount`. That was the observed failure on 10-02 (581) and 10-14 (601).
4. Live data: the transport is verified live (equilibrium snapshot carries circe heartbeat `details`). The new field itself is **UNVERIFIED** until the controller is rebuilt on circe. Rest state: a healthy container's count is flat, so the tracker reads 0 rise; this is not a mean of noise and has no floor.
5. Existing mechanism: reuses the existing tracker, heartbeat hook, equilibrium snapshot.
6. Reversibility: two flags; the detail key is ignored by everything else.

## Tests run

```text
pytest services/orion-mesh-guardian/tests services/orion-mesh-guardian/evals -q   -> 136 passed (also in a clean venv from requirements only, as CI installs it)
pytest services/orion-gpu-lane-controller/tests -q                                -> 110 passed
.github/workflows/orion-static-gates.yml, every run step locally                  -> 37 steps, 0 failures
```

## Evals run

```text
services/orion-mesh-guardian/evals/test_remote_crash_loop_incident_replay.py
  10-14 circe replay, one restart per 23.5 min: first card at cycle 72 (72 min)
  10-14 circe replay, one restart per 22 s: first card at cycle 2
  10-14 circe replay, whole incident: 40 cards over 235 h (was 0)
  circe silent from cycle 10: blind card at cycle 15
services/orion-mesh-guardian/evals/test_stability_incident_replay.py (10-02, unchanged): crash_loop at cycle 56
```

The two cadences disagree (the report says "every ~22 s"; 601 restarts in 235 h averages 23.5 min). I could not read circe's Docker from athena to settle it, so both are replayed.

## Docker/build/smoke checks

```text
Not built or deployed (task said do not deploy). Live end-to-end path: UNVERIFIED until the controller is rebuilt on circe and the guardian on athena.
```

## Review findings fixed

Review ran in a subagent (manual review; no code-review skill was available). No must-fix findings.

- Finding: two census reporters on one host were merged last-writer-wins; an older, lower count could read as a container recreate and reset the loop history (latent alert suppressor).
  - Fix: keep only the freshest error-free reading per host.
  - Evidence: `test_two_reporters_on_one_host_freshest_wins_no_merge`.
- Finding: `ContainerRestartsV1` forbids extra fields, so a newer producer adding a field would blind an older guardian, and the card would wrongly say "no census reached".
  - Fix: unknown fields are dropped before validating; a truly malformed census is reported as "malformed census".
  - Evidence: `test_newer_producer_field_is_ignored_not_blinding`, `test_malformed_census_card_says_malformed`.
- Finding: the eval claimed to replay "the real path" but hand-built the snapshot.
  - Fix: each tick now runs through the controller's real census module, `SystemHealthV1` JSON, `EquilibriumServiceState`, and `EquilibriumSnapshotV1.model_validate` of JSON. Same results (72 min / cycle 2 / 40 cards / blind at cycle 15).
- Finding: freshness compares circe's clock with athena's; a future `read_at` counted as fresh.
  - Fix: `read_at` more than 60 s in the future is reported blind as clock skew. A remote clock more than 300 s behind still reads blind (loud; the card shows the age). Not fixed: a full skew-proof "has read_at advanced in guardian time" tracker.
  - Evidence: `test_future_read_at_is_clock_skew_not_fresh`, `test_small_future_read_at_is_tolerated`.
- Finding: controller shutdown could re-raise a census-task exception and skip the bus cleanup.
  - Fix: suppress it on shutdown; the loop also survives an unexpected exception.
  - Evidence: `test_census_loop_keeps_reading_after_an_error`.
- Finding: athena's own card keys/subjects changed without being stated.
  - Fix: stated under "Behavior changed" above.
- Finding (nit): `--no-trunc` ids made argv larger than needed.
  - Fix: dropped; 12-char ids are what the key uses.
- Finding (nit): boot grace depended on `run_stability_checks` being the caller.
  - Fix: `_check_crash_loops` starts the grace itself.
- Not fixed (nits): histories of a host that goes silent forever are kept (tiny); a partial `docker inspect` failure re-baselines the missing containers; the restart cadence is not checked against circe's real restart timestamps (no circe shell from athena) -- both cadences are replayed instead.

## Restart required

```bash
# circe (controller first: it produces the census)
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
# athena
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build
```

Proof after deploy: one `orion:equilibrium:snapshot` message shows `gpu-lane-controller@circe` with `details.container_restarts.counts` non-empty, and the guardian's hourly `stability cycle` log line shows `'remote': {'circe': '<n>s'}`.

## Risks / concerns

- Severity: low. Concern: equilibrium's snapshot is the only carrier, so if equilibrium dies the guardian goes blind on circe. Mitigation: that is surfaced as a blind card within 5 min, not silence.
- Severity: low. Concern: the controller now runs `docker inspect` every 30 s on circe. Mitigation: two short read-only CLI calls; off switch `GPU_LANE_CONTAINER_CENSUS_ENABLED=false`.
- Severity: low. Concern: clock skew between circe and athena shifts reading age. Mitigation: 300 s stale threshold; tailnet hosts run NTP.
- hecate has no reporter and is not watched (on-demand host).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2895

Companion PR (GPU UUID pin for the worker, no card chosen): https://github.com/junebug-junie/Orion-Sapienform/pull/2894

🤖 Generated with [Claude Code](https://claude.com/claude-code)
