## Summary

- The Ring doorbell retina (PR #2747) never built or started after merge. It sat behind a `doorbell` compose profile. The post-pull rebuild runs a plain `up` and re-ups only profile services that are already running, so it skipped the doorbell. Log evidence (`~/.orion/post_pull_rebuild.log`, 2026-10-17 03:02 UTC): only `orion-vision-retina` and the walkway were brought up.
- Removed the profile. The next post-pull rebuild of `orion-vision-retina` now builds and starts `orion-vision-retina-doorbell` on athena.
- This is safe without a Ring login. The doorbell waits, reports `degraded` health with `ring not authenticated`, retries every 60 s, never contacts Ring or a camera, and never exits. A new test pins that.
- A missing `ring_doorbell` import now names itself in health. Before, it silently killed the supervisor.

## Outcome moved

The doorbell deploys itself like every other unprofiled service. Juniper's only manual step is the one-time Ring login.

## Current architecture

`orion-vision-retina-doorbell` had `profiles: ["doorbell"]`, copied from the walkway. The walkway needs its profile because an empty `WALKWAY_RTSP_URL` is refused at boot. The doorbell has no such failure.

## Architecture touched

`services/orion-vision-retina` only (compose, source, docs, tests).

## Files changed

- `services/orion-vision-retina/docker-compose.yml`: profile removed; comment explains why.
- `services/orion-vision-retina/app/ring_source.py`: ImportError inside the connect attempt sets `last_error`.
- `services/orion-vision-retina/README.md`, `.env_example` (comment only): login flow without a profile.
- `services/orion-vision-retina/tests/test_ring_source.py`: asserts no profile.
- `services/orion-vision-retina/tests/test_capture_task_supervision.py`: an un-logged-in doorbell waits degraded and never exits.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the doorbell container starts on a plain `up`.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: comment only
- local `.env` synced: ran `sync_local_env_from_example.py --all-keys orion-vision-retina`; no changes needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. python -m pytest services/orion-vision-retina/tests -q   -> 127 passed
docker compose ... -f services/orion-vision-retina/docker-compose.yml config --services
  -> orion-vision-retina, orion-vision-retina-doorbell   (doorbell now selected by plain up)
```

## Evals run

```text
No eval harness for retina deploy wiring; the supervision test covers the safety claim.
```

## Docker/build/smoke checks

```text
compose config (above). Reviewer ran the real source in the production image with no token and
--network none: _connect_once returns before any Ring call; supervisor alive; nothing written.
```

## Review findings fixed

- Finding: the central safety claim ("an unauthenticated doorbell never exits") had no test.
  - Fix: `test_unauthenticated_ring_doorbell_never_exits_for_restart`.
  - Evidence: 127 passed.
- Finding: a `ring_doorbell` ImportError outside the try killed `_supervise` silently.
  - Fix: catch it and set `last_error`.
  - Evidence: same suite.
- Reviewer verified that the unauthenticated doorbell never pages anyone:
  - equilibrium computes distress from heartbeat freshness, not from status;
  - mesh-guardian acts only on roster services;
  - orion-notify does not consume system health.

## Restart required

None by hand. The post-pull rebuild after merge starts the doorbell. Then, once:

```bash
docker exec -it orion-vision-retina-doorbell python -m app.ring_login
```

## Risks / concerns

- Severity: low.
  - Concern: until the login, health shows `degraded` every 10 s, and one ERROR log line appears per minute (about 1,440 a day).
  - Mitigation: nothing pages on it. It stops the moment the login lands.
- Severity: low.
  - Concern: the live Ring path is still UNVERIFIED until the login and a real event.
  - Mitigation: none in this PR; the next step is the login plus a real visit.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
