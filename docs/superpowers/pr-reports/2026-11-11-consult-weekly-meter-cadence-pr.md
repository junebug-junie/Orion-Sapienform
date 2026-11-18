## Summary

- The weekly Claude-usage reading is now published every 15 minutes instead of every hour.
- Why: readings are pub/sub only, with no backlog. After the #3205 deploy, Hub restarted 12 s after the meter's first weekly reading, so it missed it. The consult loop then sat at `budget:weekly_meter_missing` for up to an hour. Every deploy that restarts both services repeats this.
- Hub's freshness limit for the weekly reading moves with it: 3 missed publishes (45 min).

## Outcome moved

- After a deploy, consults wait at most 15 minutes for the meter, not an hour.
- Cost: about 4 more scans of the weekly window per hour, roughly 2.4 s each.

## Files changed

- `services/orion-cocreation-signals/.env_example`, `app/settings.py`: `weekly_limit:168:900`
- `orion/consult/core.py`: `WEEKLY_READING_MAX_AGE_SEC = 3 * 900`
- `orion/consult/tests/test_core.py`: the stale case uses the new limit
- `services/orion-hub/README.md`: the stated wait

## Env/config changes

- Changed: `COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS` to `session_limit:5:300,weekly_limit:168:900`.
- Local `.env`: updated by hand. The sync does not overwrite a diverged value.

## Tests run

```text
orion/consult/tests                         33 passed
services/orion-cocreation-signals/tests     111 passed
services/orion-hub/tests/test_consult_loop  14 passed
```

## Evals run

None. This is a cadence change. The live evidence is the deploy log: `weekly_limit` was published at 18:51:24, Hub's consult loop started at 18:53:41, and every tick until 19:51 read `budget:weekly_meter_missing`.

## Review findings fixed

Reviewed inline. It is a five-line config and constant change, and no subagent review was run. Checked: the compose file passes the variable through with no fallback (`${COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS}`), and the settings validator accepts the spec.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
```

The post-pull rebuild does this when it sees the change.

## Risks / concerns

- Severity: low.
  - Concern: about 4x the weekly scan I/O.
  - Mitigation: about 9.6 s of scanning per hour in total.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
