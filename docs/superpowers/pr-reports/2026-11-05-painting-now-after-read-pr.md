# Fix: the painting loop never painted (negative activity age read as stale)

## Summary

- **The problem:** the self-start loop from #3134 issued no painting after the 2026-11-05 05:48Z
  deploy. Each tick's `/health` result was `visual_activity_unavailable_or_stale`.
- **The cause:** the tick takes `now`, then reads the painting activity. The activity's
  `observed_at` defaults to the read time, a few milliseconds later. So `now - observed_at` is
  negative, and `orion/reverie/baseline.schedule` refuses a negative age as stale.
  - Live, inside the container: the age was -0.57 s.
- **The fix:** decide at the later of the two instants, `now = max(now, activity.observed_at)`.
- **Why tests and the replay missed it:** their fake activity was stamped at exactly `now`.
  The fixture now has a `read_lag`, and a regression test reads the activity 600 ms after the tick
  clock, the way the live loop does.

## Outcome moved

Orion paints on its own again. Before the fix: 0 paintings since 05:48Z. One was due about 06:13Z
(the last painting closed at 04:43:57Z).

## Tests run

```text
services/orion-thought tests/test_painting_self_start.py: 29 passed
  new test_activity_read_after_the_tick_clock_still_paints: FAILS with the fix removed (mutation check), passes with it
services/orion-thought full suite: 840 passed, 32 skipped
```

## Evals run

```text
None changed. The replay eval injects activity at exactly `now`, so it could not see this bug.
The fixture's read_lag is the guard against it.
```

## Review findings fixed

The code review ran in a subagent at medium effort. It found no blocker, and confirmed two things
by putting the old tick back: only the new test fails, and later uses of `now` are all safe (the
shift is milliseconds).
- **SHOULD:** the shift had no limit. A future `observed_at` from another host's clock would
  silently push deadlines and dispatch ids forward.
  - Fix: the clock moves only for a local read lag of 5 s or less (`MAX_READ_LAG`). Anything
    further ahead stays refused.
  - Evidence: `test_an_activity_far_in_the_future_is_still_refused_as_stale`.
- **Checked:** no other caller of `schedule()` or `validate_eligibility` has this ordering. Each
  takes its clock after the observation.
- **Not done (NIT):** a read lag in the replay eval. The test fixture covers it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

Then check: `curl -s http://127.0.0.1:7155/health`. Under `painting_self_start.last.result` it
should read `submitted` (or `not_due`); a `painting_self_start_submitted` log line should appear,
and a `thought.painting` act should open.

## Risks / concerns

- Severity: low. Concern: `now` moves forward by the read time, a few milliseconds. The gate and
  the due checks use that later time. That is harmless: it is the true decision time.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
