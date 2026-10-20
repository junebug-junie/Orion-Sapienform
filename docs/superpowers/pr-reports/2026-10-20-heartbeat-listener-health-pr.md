## Summary

- The shared heartbeat that every chassis service sends to equilibrium now reports whether that process's bus listeners are connected.
- While any listener is not subscribed, the heartbeat says `status="degraded"`. It also carries a small summary, `details.listeners = {ok, down, count}`.
- A process with no listeners sends exactly what it sent before.
- A test now pins mesh-guardian's behaviour: an on-time heartbeat that says "degraded" never triggers a container restart.

## Outcome moved

Before this change, the heartbeat always said "ok". On 2026-10-20, equilibrium showed substrate-runtime as healthy for hours while its finalize listener was dead and chat lost reflection. Now a dead listener shows in the equilibrium snapshot as `degraded`, with the listener's name, for all 19 converted services that send heartbeats. #2786 already added this for substrate-runtime's and hub's own `/health`. This change puts it on the shared bus heartbeat, so all those services report it the same way.

This change gives **visibility only**. Nothing turns "degraded" into distress, a notification or a restart. Equilibrium's distress/zen and uptime come from heartbeat age alone. substrate-runtime's own monitor already pages on a down listener (#2786).

## Current architecture

`BaseChassis._heartbeat_loop` (`orion/core/bus/bus_service_chassis.py`) published SystemHealthV1 with a hard-coded `status="ok"`. Equilibrium stores each heartbeat's status and details in its snapshot. It overrides the status to "down" only when the heartbeat is late.

## Architecture touched

- `orion/core/bus/bus_service_chassis.py`: new `_listener_heartbeat_status()`, which reads `orion.core.bus.resilience.subscription_health()`.
- mesh-guardian: a comment and a test that pin the no-restart rule.

## Files changed

- `orion/core/bus/bus_service_chassis.py`: the heartbeat status and the listener summary.
- `tests/test_chassis_heartbeat_listener_status.py`: four cases. A listener is down, all are up, there are no listeners, and the registry raises (the heartbeat must survive).
- `tests/test_chassis_heartbeat_details.py`: isolates the process-wide listener registry, so other tests' listeners can't leak in.
- `services/orion-mesh-guardian/app/equilibrium_watch.py`, `tests/test_equilibrium_watch.py`: pin that a fresh degraded heartbeat is not `equilibrium_bad`.

## Schema / bus / API changes

- Added: `SystemHealthV1.details.listeners` (free-form `details` dict; no schema change).
- Removed: none
- Renamed: none
- Behavior changed: SystemHealthV1 `status` can now be `degraded`. The schema already allowed it, but no producer sent it until now.
- Compatibility notes:
  - Reviewed every consumer of the status. Equilibrium passes it through.
  - Uptime and distress use heartbeat age only.
  - Mesh-guardian's degraded branch needs `down_for_ms > grace`. `down_for_ms` measures heartbeat lateness, so it is 0 while heartbeats are on time. Now pinned by a test.
  - `downtime_transition_tracker` will write ok↔degraded rows to `equilibrium_service_transitions` (sql-writer). Downtime math covers "down" only, so it is unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
tests/test_chassis_heartbeat_* + tests/test_heartbeat_only_chassis.py + orion/core/bus/tests -> 125 passed
tests/test_system_health_producers.py, tests/test_vision_retina_health.py (+ chassis)    -> 20 passed
services/orion-equilibrium-service/tests  -> 238 passed
services/orion-heartbeat/tests            -> 151 passed
services/orion-mesh-guardian/tests        -> 102 passed
mutation: status hard-coded back to "ok" -> the degraded test fails
```

## Evals run

```text
No eval harness covers heartbeat status. Live check after deploy, below.
```

## Docker/build/smoke checks

```text
After deploy: read one equilibrium snapshot on orion:equilibrium:snapshot (or its debug route) and confirm
services report details.listeners with ok=true, count>0, for e.g. orion-substrate-runtime and orion-hub.
```
UNVERIFIED until this is deployed.

## Review findings fixed

- Finding: mesh-guardian's degraded-restart branch is safe today only because `down_for_ms` measures heartbeat lateness. If that ever changes, a stuck listener becomes a restart loop.
  - Fix: a comment at the branch plus `test_fresh_degraded_heartbeat_never_triggers_remediation`.
  - Evidence: mesh-guardian 102 passed.
- Finding: do not claim more than visibility.
  - Fix: this report says it is visibility only.

Recorded, not fixed:
- A listener that fails before it registers is invisible. Example: `bus.connect()` raising inside a daemon-thread starter, as in sql_router, voip and rag. Same failure class, but upstream of the registry.
- A listener that stops by design keeps its process degraded until restart. No live case today. The hub's optional caches can trip it if their startup throws after registering.
- The registry is per process. A process hosting two chassis, or a chassis plus a library listener, reports one dead listener for all of them.
- Short ok→degraded→ok transition rows during Redis blips and restarts. They go to SQL only, and nothing alerts on them.

## Restart required

Every chassis service picks this up on its next rebuild. To see it right away on the main two:
```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```
Run from the primary main checkout after merge. The post-pull rebuild script picks up every service that imports the chassis.

## Risks / concerns

- Severity: low
  - Concern: services may show `degraded` for a few seconds around a Redis blip or a restart.
  - Mitigation: accurate and short. It only adds SQL transition rows.
- Severity: low
  - Concern: a service with a listener that stopped by design stays degraded.
  - Mitigation: no current case; documented.

## PR link

<link>
