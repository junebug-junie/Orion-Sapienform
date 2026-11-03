## Summary

- The resonance health monitor stops sending a fresh "recovered" note every time `orion-thought` restarts.
- At startup, a theme now counts as still open only if its newest pending item is a "worsening" page, not a recovery note.
- The same rule now covers the reverie-timeout and painting-failure monitors. All three had their own copy of the lookup with the same bug. They now share `app/notify_pending.py` and one `RECOVERED_PREFIX`.

## Outcome moved

Live 2026-11-03: 273 of 311 unacked resonance items were repeat recovery notes. Each one landed within minutes of an `orion-thought` restart (11 restarts since 10-30, each matched by a recovery note per open theme). All 311 were acked by hand after review (`/tmp/resonance-ack-cleanup/before.csv`, 311 ok, 0 errors).

## Current architecture

`ResonanceHealthMonitor._bootstrap_from_notify` read every pending item for orion-thought. Pages and recovery notes share one `reason` key and both require ack. So a pending recovery note made the theme look still open, and the next healthy tick sent another recovery note.

## Architecture touched

- New `services/orion-thought/app/notify_pending.py:open_alert_reasons`: keeps the newest pending item per reason (sql-writer returns newest first) and drops reasons whose newest item is a recovery note. Fail-open (`[]`) as before.
- `resonance_monitor.py`, `reverie_health_monitor.py`, `visual_chain_health_monitor.py`: `_has_open_alert` / bootstrap call the shared helper; recovery text built from `RECOVERED_PREFIX`.

## Files changed

- `services/orion-thought/app/notify_pending.py`: shared open-page rule (new).
- `services/orion-thought/app/resonance_monitor.py`, `reverie_health_monitor.py`, `visual_chain_health_monitor.py`: use the shared helper; duplicated lookup removed.
- `services/orion-thought/tests/test_notify_pending.py`: rule tests + wiring tests for the reverie and visual monitors (new).
- `services/orion-thought/tests/test_resonance_monitor.py`: restart-loop regression test (fails on old code, passes now).
- Other monitor tests and `evals/test_attempt_streak_replay.py`, `evals/test_painting_gap_replay.py`: patch target moved to `app.notify_pending.requests.get`.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
services/orion-thought: pytest -q tests evals -> 820 passed, 32 skipped
new restart-loop test against old code -> 1 failed (as expected)
```

## Evals run

```text
services/orion-thought/evals -> pass
```

## Docker/build/smoke checks

```text
Not rebuilt from branch; deploys from main after merge.
```

## Review findings fixed

- Round 1 finding: the reverie-timeout and painting monitors had the same "any pending item = open" bug (silent on a real new page after a restart if a recovery note was unacked).
  - Fix: shared `open_alert_reasons` used by all three monitors.
  - Evidence: `test_reverie_monitor_ignores_answered_page`, `test_visual_monitor_ignores_answered_page`.
- Round 2: no material findings. Reviewer confirmed every recovery message starts with `RECOVERED_PREFIX`, fail-open is unchanged, rendered log names are unchanged, and no stale patch targets remain.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the lookup reads only the newest 200 pending items (orion-notify caps `limit` at 200). A page older than that is not seen.
- Mitigation: the lookup fails open, so the worst case is a duplicate page, not a missed one. The 311-item backlog that crowded the window was acked on 2026-11-03, and this fix stops recovery notes piling up. Removing the cap needs a server-side `source_service` filter in sql-writer, which is a separate API change.
- Concern: the log lines keep their text, but the logger name is now `orion-thought.notify_pending`.
- Mitigation: nothing in the repo filters on the old logger names.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3047
