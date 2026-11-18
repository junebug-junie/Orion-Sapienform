# Self-monitor: a failed turn is not a broken guess

## Summary

- Live right after the 11-03 deploy, the organ marked two of stance's guess skills as **broken**:
  "how many steps my turn takes" and "how long my reply runs". The outreach loop queued a message
  telling Juniper.
- That was false. At 04:00 that day stance stopped stating those two ranges, because the relative
  "bigger than usual" bets from #3042 replaced them. After that, the only rows left for those skills
  were the 62 turns that errored. A failed turn voids all four of its guesses at once, so each guess
  skill showed a run of 26 "didn't run".
- **Fix, part 1:** a failed turn no longer counts against the guess skills.
- **Fix, part 2:** turn failures get one skill of their own, `turn.run`. It counts each turn once and
  compares the turns that failed with the turns that ran. Live: 13% of turns failed, which is not
  broken.

## Outcome moved

Live report after the fix:
- No broken skill.
- `turn.self.steps` and `turn.self.reply_length` read miscalibrated again. That is real: their
  ranges held 5% and 8% of the time, against a stated 80%.
- `turn.run` reads unknown, with 13% of turns failing.

## Current architecture

`orion/self_monitor/adapters.turn_self_bet` mapped `void / failed_to_execute` onto every guess
faculty of that turn.

## Architecture touched

- `orion/self_monitor/adapters.py`: `turn_self_bet` drops failed turns. A new `turn_run` adapter
  makes one bet per turn.
- `orion/self_monitor/collect.py`: a new source, `prediction_turn_run`. It is one SQL query that
  groups `prediction_turn_self_bet` by correlation_id, read-only.

## Files changed

- `orion/self_monitor/adapters.py`
- `orion/self_monitor/collect.py`
- `orion/self_monitor/tests/test_self_monitor_record.py`
- `orion/self_monitor/tests/test_self_monitor_host.py`

## Schema / bus / API changes

- Added: faculty `turn.run`; source name `prediction_turn_run`.
- Removed: none
- Renamed: none
- Behavior changed: failed turns no longer count against guess faculties.
- Compatibility notes: none (no schema change).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/self_monitor/tests                           -> 108 passed (incl. new regression test from the live case)
services/orion-durable-runs: pytest tests                 -> 686 passed, 121 skipped
services/orion-hub: test_self_finding_outreach.py         -> 23 passed
services/orion-thought: cue tests + truth eval            -> 15 passed
```

## Evals run

```text
PYTHONPATH=. POSTGRES_URI=... python -P -m orion.self_monitor.report  (live, read-only)
  no broken faculty; turn.self.steps/reply_length miscalibrated (5%/8% vs 80%); turn.run unknown, 13% failed
```

## Docker/build/smoke checks

```text
Rebuilt by the post-pull rebuild after merge.
```

## Review findings fixed

- No subagent review: this is a small adapter fix, done under live pressure. It is pinned by a
  regression test built from the live case.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: the two false "broken" findings are already pending in Hub's outreach state, in Redis
  key `orion:hub:self_finding_outreach:state`. Nothing removes them on its own, and the held status
  in durable-runs stays broken for 3 more hourly evaluations.
- Mitigation: Juniper clears the two pending entries with the one-line command in the session
  reply. The cue re-checks its numbers, so it shows no broken line.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3078

🤖 Generated with [Claude Code](https://claude.com/claude-code)
