## Summary

- A timer look whose note exactly repeats the previous one is still stored, so it keeps the note fresh. Its act now closes 0 instead of 1. It says nothing new, so it should not make `hub.retro_watch` look better at looking.
- The timer path now counts the looks it holds back. A 2-second check where a timer look was due but no button changed in 60 s now counts `input_stale_skips`. Before, the timer path counted nothing, so a short sitting could not say why it got fewer looks than minutes.
- Both come from Orion's review of PR #3995 (questions 2 and 4).

## Outcome moved

Live 2026-11-25: sitting `9e28...` got 2 timer looks in about 6 minutes. All of its bids were funded. The cause of the 4-minute gap was never recorded, because `input_stale_skips` stayed 0 on the timer path. That counter now shows those checks. No exact repeat has happened live yet, so nothing has inflated the family's outcome so far.

## Current architecture

#3906 added the no-floor timer (`orion/retro/watch.py` `gate`, `services/orion-hub/scripts/retro_watch.py`). It mapped a timer look's `copied` to `stored`, and that look closed 1 like any stored look. `count()` returned early for every timer gate.

## Architecture touched

- `orion/retro/watch.py`: new pure `timer_due()`; `count(..., timer_due=)`.
- `services/orion-hub/scripts/retro_watch.py`: the repeat closes 0 (close note `look repeat: ...`); passes `timer_due` into `count()`.

## Files changed

- `orion/retro/watch.py`, `services/orion-hub/scripts/retro_watch.py`: as above.
- `tests/retro/test_watch_rules.py`, `services/orion-hub/tests/test_retro_watch_service.py`: tests for both changes.
- `docs/runbooks/retro-browser.md`: says what the close value and the counter now mean.

## Schema / bus / API changes

- Added / removed / renamed: none. The DB column `retro_sitting.input_stale_skips` already exists.
- Behavior changed:
  - A timer repeat's act now closes `produced=false`.
  - On the timer path, `input_stale_skips` counts checks where a due look was held back. There it is not a subset of `checks_over_floor`, as it is on the floor path.

## Env/config changes

None.

## Tests run

```text
tests/retro services/orion-hub/tests/retro orion/dig/tests/test_watch.py -> 373 passed
services/orion-hub/tests/test_retro_watch_service.py (from services/orion-hub) -> 37 passed
scripts/check_body_economy_contract.py -> OK; scripts/check_integration_seam.py -> OK
```

## Evals run

None new. The look replay eval covers the floor statistic, which did not change.

## Docker/build/smoke checks

No dependency change and no image build. The live path is UNVERIFIED until deploy and a no-floor sitting.

## Review findings fixed

The code review ran in a subagent. Verdict: OK to ship, no material findings.
- Nit: the runbook said the counter counts "missed looks", but it counts 2-second checks.
  - Fix: reworded, with the scale (10 idle minutes add about 300).
- Nit: `_timer_sec(cal)` was called twice.
  - Fix: it is now stored in a local.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull rebuild does this after merge.

## Risks / concerns

- Severity: low.
  - Concern: the line lock that holds back looks is still not counted.
  - Mitigation: ticket #4006 (play's "watching" flickers during a line).
- Issues opened: #4005 (the line gate's exact-text dedup), #4006 (watching flickers during a line).
- Issues closed: none.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
