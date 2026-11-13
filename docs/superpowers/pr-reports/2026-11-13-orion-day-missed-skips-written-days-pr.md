## Summary

- Hub drops a day from a letter order's `missed_letters` when that day has an `orion_day_letter` row by the time of the order.
- Why: on 11-13, the 11-11 letter was re-ordered by hand (run `orion-day-2026-11-11-2`) after Prime had recorded the day as `not_written`. Prime's next order (11-12) still lists 11-11 as missed. Without this filter, the 11-12 letter would tell Juniper it could not write a day that now has a letter.
- Prime marks a day "told" only after a written letter carried it, so its own list converges after one more letter. Hub is the authority on whether a row exists.

## Outcome moved

No false "I could not write that day" when a missed day is filled later.

## Current architecture

`missed_letters_for` passed Prime's list through as given. It added the day before only when that day had no row.

## Architecture touched

`services/orion-hub/scripts/orion_day_order.py` only (`missed_letters_for`, `_order_locked`).

## Files changed

- `services/orion-hub/scripts/orion_day_order.py`: checks for a row on the day before and on each listed day (at most 8 `EXISTS` reads).
- `services/orion-hub/tests/test_orion_day_order.py`: regression test.

## Schema / bus / API changes

None. The `missed_letters_for` keyword changed from `previous_day_has_letter` to `days_with_letter`. It has one caller.

## Env/config changes

None.

## Tests run

```text
pytest services/orion-hub/tests/test_orion_day_order.py services/orion-hub/tests/test_orion_day_mailer.py   42 passed
```

## Evals run

None. A behavior gate test covers it.

## Docker/build/smoke checks

The post-pull rebuild covers orion-hub.

## Review findings fixed

Review: no material findings. It confirmed:
- the async set comprehension is valid, and errors become `unavailable`;
- Prime converges, because `mark_told(missed_sent)` covers the day Hub dropped;
- nothing downstream compares the brief's list with Prime's list.

Nits fixed:
- Finding: the test used the day before as the filled day, so a day-before-only filter would also have passed.
  - Fix: the filled day is now 3 days back, and the test also asserts that the day before is still auto-named.
- Finding: the test's import order. Fix: moved.

Noted, not changed: Prime's own record keeps the filled day as `not_written`, and its status lines call it missed. Hub keeps that out of the letter; Prime's record is not corrected.

## Restart required

```text
No manual restart: git pull rebuilds orion-hub.
```

## Risks / concerns

- Severity: low. Concern: a store error on one of the extra reads makes the order `unavailable`. Mitigation: the reads share the existing try block, and Prime re-sends an unavailable order.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
