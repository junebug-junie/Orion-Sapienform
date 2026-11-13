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

See the review commit, if any.

## Restart required

```text
No manual restart: git pull rebuilds orion-hub.
```

## Risks / concerns

- Severity: low. Concern: a store error on one of the extra reads makes the order `unavailable`. Mitigation: the reads share the existing try block, and Prime re-sends an unavailable order.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
