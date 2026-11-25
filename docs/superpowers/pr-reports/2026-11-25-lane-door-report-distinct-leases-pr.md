## Summary

- `scripts/lane_door_report.py` now checks each lane's daily number of **distinct lease ids** against the gateway's own call count, instead of raw leased calls.
- A Strata JSON corrective retry sends the call again under the same lease (`llm_backend.py:1323-1325`). The door logged it twice and the gateway once, so the chat lane failed the 1 % gap check on most days.
- Gateway rows are now counted by `recorded_at` (the gateway's own close time, which is indexed) instead of `created_at` (sql-writer's insert time). A call ending near 00:00Z no longer slips into the next day.
- The table gains a `leases` column next to `leased`, so the re-sends stay visible.

Fixes #3919

## Outcome moved

The chat lane reads READY TO ENFORCE: on every day from 11-16 to 11-25, its distinct leases match the gateway's count exactly (for example 989/989 and 297/297). Before the fix it read NOT READY because of the raw re-send count (for example 307 vs 297, 3.4 %).

## Current architecture

The report compared `Day.leased` (one per door log line) with `llm_call_usage` rows, which are one per gateway request, counted by `created_at`.

## Architecture touched

Read-only operator script only.

## Files changed

- `scripts/lane_door_report.py`: the distinct-lease gap (`gap_pct`), the `recorded_at` day bucket, the `leases` column, and a docstring that states the row unit and the remaining limits.
- `tests/test_lane_door_report.py`: a re-send under one lease is not a gap; an extra distinct lease still is.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest tests/test_lane_door_report.py -q -> 12 passed
New re-send test against origin/main's script -> fails (as it should)
```

## Evals run

Live report (read-only), with the circe door logs and the athena DSN:
```text
== lane agent-gpu2: READY TO ENFORCE -- 7 clean days ending 2026-11-24
== lane chat: READY TO ENFORCE -- 7 clean days ending 2026-11-24
== lane fast: NOT READY -- 2026-11-24: 5 lease ids not in gpu_pool_events (#3920 rebuild loss)
== lane agent: NOT READY -- 2026-11-24: 11 lease ids not in gpu_pool_events
== lane agent-deep: NOT READY -- no log
```

## Docker/build/smoke checks

Not applicable (operator script).

## Review findings fixed

- Finding: the docstring said `llm_call_usage` has one row per lease. It has one row per gateway request (`call_usage_emit.py:78`).
  - Fix: the docstring now states the row unit and both benign mismatches.
- Finding: a context-overflow re-lease (up to 3 per request, `main.py:436-471`) gives one gateway row but several distinct leases, so it can still read as a gap.
  - Fix: documented in the docstring and covered by a test. Netting it out needs gateway data, so it is filed as #3942.
- Finding: counting distinct ids hides a caller that replays a real lease id.
  - Fix: documented. The door guards against accidental bypass, not an adversary.
- Finding: `created_at` is sql-writer's insert time and can move a boundary call into D+1.
  - Fix: switched to `recorded_at` (indexed; 0 nulls over the last 7 days; at most 6 s behind `created_at`).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: overflow re-leases can still read as a small gap. Mitigation: #3942.
- Issues opened: #3942. Closed: #3919 (on merge).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
