## Summary

- The resonance health monitor stops sending a fresh "recovered" note every time `orion-thought` restarts.
- At startup, a theme now counts as still open only if its newest pending item is a "worsening" page, not a recovery note.

## Outcome moved

Live 2026-11-03: 273 of 311 unacked resonance items were repeat recovery notes. Each one landed within minutes of an `orion-thought` restart (11 restarts since 10-30, each matched by a recovery note per open theme). All 311 were acked by hand after review (`/tmp/resonance-ack-cleanup/before.csv`, 311 ok, 0 errors).

## Current architecture

`ResonanceHealthMonitor._bootstrap_from_notify` read every pending item for orion-thought. Pages and recovery notes share one `reason` key and both require ack. So a pending recovery note made the theme look still open, and the next healthy tick sent another recovery note.

## Architecture touched

- `services/orion-thought/app/resonance_monitor.py`: `_fetch_pending_reasons` keeps the newest item per reason (list is newest first from sql-writer) and drops reasons whose newest item is a recovery note. `_has_open_alert` uses the same rule.

## Files changed

- `services/orion-thought/app/resonance_monitor.py`: newest-item-decides rule; `_RECOVERED_PREFIX` constant.
- `services/orion-thought/tests/test_resonance_monitor.py`: regression test (fails on old code, passes now).

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
tests/test_resonance_monitor.py tests/test_reverie_chain.py evals -> 80 passed
new test against old code -> 1 failed (as expected)
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

- No material findings. Reviewer confirmed `reason`/`message` fields end to end and that the 200-item cap can drop a reason but cannot invert newest-first.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low (follow-up)
- Concern: `reverie_health_monitor.py:115` and `visual_chain_health_monitor.py:332` still treat any pending item as open. They do not bootstrap, so no spam, but after a restart they may stay silent on a real new page if an unacked recovery note exists.
- Mitigation: move the newest-is-not-recovery helper into a shared module for all three.

## PR link

TBD
