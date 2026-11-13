## Summary

- `evals/retro_play.mjs`: a real-browser play harness. It drives the live Hub Retro tab and records input cadence, acknowledgment latency, socket closes, status text, long tasks and the server journal. `--override` serves working-copy JS to the live site, so client fixes can be tested without a deploy.
- `hub-retro-gateway`: Retro's relay in its own process (same router and checks), loopback on 8092, so game traffic no longer queues behind Hub's event loop.
- The Retro tab unloads Hub's other iframes while open (the `/spark/ui` WebGL visualizer took about 22% of the shared browser thread).
- New game replaces an existing game instead of failing with `session_active`.
- The worker now says why a game is paused (`session_paused:<why>`). The browser heals lapsed leases, dropped controllers and overload pauses, and honors an explicit Pause.

## Outcome moved

Live A/B with `retro_play.mjs` on 2026-10-20, Super Mario Bros, a Hub page open in another tab:

| path | inputs applied | ack p50 | drops |
|---|---|---|---|
| through the Hub process | 55 | 720 ms | 4, then stuck on `session_not_running` |
| through the dedicated relay | 944 | 17 ms | 0 |

Hub-tab run before and after the iframe suspension:
- Retro frame long tasks: 13.9 s → 0.1 s per run.
- Timer lag max: 5.5 s → 52 ms.

## Current architecture

Browser → tailscale serve `/` → Hub process (one event loop for every Hub page, poll and socket) → Retro UDS. The Retro tab shared its browser thread with Hub's other iframes.

## Architecture touched

- orion-hub: the new compose service `hub-retro-gateway` and `scripts/retro_gateway.py`; `retro-tab.js`; `retro-panel.js`.
- orion-retro: `worker.py` error reason.
- CI: `retro-preflight.yml` paths, plus pydantic-settings for the gateway test.

## Files changed

- `services/orion-hub/scripts/retro_gateway.py`: Retro-only app plus a health route.
- `services/orion-hub/docker-compose.yml`: `hub-retro-gateway` (host network, 127.0.0.1, socket and auth mounts only, no bus-wait entrypoint).
- `services/orion-hub/.env_example`: `HUB_RETRO_GATEWAY_PORT=8092`.
- `services/orion-hub/static/js/retro-tab.js`: suspend and restore other iframes.
- `services/orion-hub/static/js/retro-panel.js`: New game stops an existing game; `SERVER_PAUSE` updated.
- `services/orion-retro/app/worker.py`: `paused_by` and `session_paused:<why>`; an explicit Pause on an already-paused game sticks.
- `services/orion-hub/evals/retro_play.mjs`: the harness (`--url` required).
- Tests: `services/orion-retro/tests/test_worker.py` (2) and `services/orion-hub/tests/retro/test_boundary.py` (2).
- `docs/runbooks/retro-browser.md`, `.github/workflows/retro-preflight.yml`.

## Schema / bus / API changes

- Behavior changed: input and control rejections on a paused game say `session_paused:<pause|disconnect|no_controller|deadline_overrun>`. `session_not_running` remains for non-paused states. No other consumer matches the old string (checked container_smoke.py, operator_smoke.py).
- No bus or schema changes.

## Env/config changes

- Added keys: `HUB_RETRO_GATEWAY_PORT` (orion-hub).
- `.env_example` updated: yes.
- Local `.env` synced: yes, key added to the primary checkout's `services/orion-hub/.env`.
- Skipped keys requiring operator action: none. The tailscale route is an operator step (below).

## Tests run

```text
services/orion-retro/tests: 36 passed (new paused-reason tests fail on old worker)
services/orion-hub/tests/retro: 3 passed
node --check retro-panel.js retro-tab.js retro_play.mjs: OK
```

## Evals run

```text
retro_browser.py (worktree retro image): delay 0 PASS, delay 350 PASS x2
retro_play.mjs live A/B: see Outcome moved
```

## Docker/build/smoke checks

```text
docker compose config: hub-retro-gateway resolves (127.0.0.1:8092, ipc+auth mounts)
gateway image boot without entrypoint: health ok in 0.65 s; reaches the live Retro session; wrong Origin -> 403
```

## Review findings fixed

- Finding: the gateway test needs pydantic-settings, which the CI job did not install; the gateway files were not in the workflow paths.
  - Fix: install pinned pydantic-settings==2.5.2 (Hub's pin); add the gateway, compose and harness to the paths filters.
  - Evidence: workflow diff.
- Finding: comment and PR text claimed overload pauses were respected; the code heals them.
  - Fix: comment and runbook now say the overload pause heals, bounded at 3 heals per 10 s.
- Finding: Pause on an already auto-paused game kept the old `paused_by`, so a browser could resume it.
  - Fix: an explicit Pause always sets `paused_by='pause'`.
  - Evidence: `test_explicit_pause_on_an_auto_paused_game_sticks` fails without the fix.
- Finding: the gateway waited for the Redis bus at boot through Hub's entrypoint.
  - Fix: `entrypoint: []`.
  - Evidence: compose test; image boots without it.
- Finding: the harness defaulted to the live URL.
  - Fix: `--url` is required.
- Documented, not changed: iframe state resets on leaving Retro; New game ends a game from another Hub window (shared credential); per-process socket caps.

## Restart required

Post-pull rebuild brings up orion-hub (including the new hub-retro-gateway) and orion-retro. Then, once, on athena:

```bash
sudo tailscale serve --bg --set-path /api/retro http://127.0.0.1:8092/api/retro
```

## Risks / concerns

- Severity: low. Concern: until the tailscale route exists, play still goes through Hub (works, but slow under load). Mitigation: the route command above; check with `tailscale serve status`.
- Severity: medium (separate). Concern: the Hub event-loop stalls themselves remain. A graph-agent finding: `concept_atlas_summary` calls `snapshot()` on the loop (up to about 107 s for aitown), and startup takes about 4 min (graph hydration). Mitigation: follow-up PRs; this PR only moves Retro off that loop.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
