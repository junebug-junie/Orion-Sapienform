## Summary

- The lane door report no longer counts a context-overflow re-lease as a gap. When a request is too big for its slot, the gateway gives the lease back and takes a bigger one. The door sees two leases, and the gateway writes one row.
- No gateway or schema change. The data already exists: the pool records every overflowed lease in `gpu_pool_events` as `released` with reason `context_overflow`.
- The request's last lease carries the gateway row. When that last lease also overflowed, the row has `outcome = 'context_overflow'`. So per lane per day: overflow extras = overflowed leases the door saw, minus the lane's `context_overflow` rows (never below 0).
- The gap now compares distinct leases with gateway rows plus overflow extras. A new `overflow` column shows the extras, and `--json` carries `overflow_leases`, `overflow_rows`, `overflow_extras`.

Fixes #3942

## Outcome moved

A lane with overflows no longer fails the 1 % gap for a benign reason. chat and agent-gpu2 still read READY TO ENFORCE on live data.

Live overflow count: since the door logs began (2026-11-16), there are **zero** overflow re-leases on any door lane. The only overflows were two on agent-gpu2 on 11-20. Both were final (each request returned the overflow and wrote its own `context_overflow` row), so they net to 0 extras. Before the door logs, from 10-30 to 11-12, the pool had 66 non-final overflow re-leases: 53 on fast and 13 on metacog. None since 11-12.

## Current architecture

`scripts/lane_door_report.py` compared distinct door lease ids per lane-day with `llm_call_usage` rows for `pool_role = lane`. The gateway overflow loops (`services/orion-llm-gateway/app/main.py` `_dispatch_on_pool`, `passthrough_proxy.py` pool loop) re-lease up to 3 times and write one row with the last grant's role.

## Architecture touched

Read-only operator script only. Existing-mechanism check: option (b) from the issue. `gpu_pool_events` (durable since #3920) plus `llm_call_usage.outcome` already identify overflow re-leases, so no new column was needed.

## Files changed

- `scripts/lane_door_report.py`: overflow lookup in the existing lease join, overflow rows in the existing gateway count, `overflow_extras()`, gap formula, `overflow` column, JSON fields, docstring.
- `tests/test_lane_door_report.py`: overflow re-lease reads 0 % gap; final overflow is not an extra; a non-overflowed extra lease still fails; two-lane overflow-then-served-elsewhere; rows > overflowed leases floors at 0; `join_db` SQL wiring with a fake driver.
- `docs/runbooks/2026-11-14-lane-door.md`: explains the `overflow` column.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: report gap formula and output columns only.
- Compatibility notes: `--json` gains three fields per day.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest tests/test_lane_door_report.py -q -p no:cacheprovider
18 passed in 0.16s
```

## Evals run

Live report (read-only), circe door logs plus the athena DSN:
```text
== lane agent: NOT READY -- 2026-11-24: 11 lease ids not in gpu_pool_events (made up?)
== lane agent-deep: NOT READY -- 2026-11-24: no log (door down, or not deployed yet) -- unknown is not zero
== lane agent-gpu2: READY TO ENFORCE -- 7 clean days ending 2026-11-24
== lane chat: READY TO ENFORCE -- 7 clean days ending 2026-11-24
== lane fast: NOT READY -- 2026-11-24: 5 lease ids not in gpu_pool_events (made up?)
--json: only lane-day with overflow activity = agent-gpu2 2026-11-20: overflow_leases 2, overflow_rows 2, extras 0
```
The unit test checks the SQL wiring with a fake driver only. The `bool_or` / `FILTER` semantics were checked against live Postgres, by the live run above and by the reviewer.

## Docker/build/smoke checks

Not applicable (operator script).

## Review findings fixed

- Finding: "fails safe" was too strong. A lost usage row of an overflowed request (or a final-overflow row closing after 00:00Z) hides itself as an extra.
  - Fix: docstring and runbook now state the bound: an over-count only covers pool-verified gateway leases, never a bypass.
  - Evidence: `scripts/lane_door_report.py` docstring, runbook paragraph.
- Finding: tests were single-lane; the common live shape is overflow on lane A, served on lane B. The `max(0, ...)` floor was never exercised.
  - Fix: added `test_an_overflow_on_one_lane_served_on_another_reads_zero_on_both` and `test_more_overflow_rows_than_overflowed_leases_never_goes_negative`.
  - Evidence: 18 passed.
- Finding (nit): the fake-driver test does not prove the SQL semantics.
  - Fix: stated above; checked live.
- Finding (nit): a test docstring described a shape that does not occur live.
  - Fix: reworded to what the test asserts.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: a lost gateway row of an overflowed request can hide itself as an overflow extra. Mitigation: bounded to pool-recorded gateway leases; documented.
- Issues opened: none. Closed: #3942 (on merge).

## PR link

See the PR this report is attached to.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
