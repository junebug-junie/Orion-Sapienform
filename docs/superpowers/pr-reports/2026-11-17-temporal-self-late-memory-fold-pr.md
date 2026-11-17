# Temporal Self: a late memory about a closed day joins the day it was written (#3584)

## Summary

- Episode memories about an earlier, already-closed day were dropped from every day record. Now
  they join the day they were written: "I remembered this today, about yesterday".
- `_flush_buffer` no longer drops unbound day-context items dated before the current day's start.
- The live frame shows the same items before they leave the buffer.
- Regression test uses the 6 live rows of 2026-11-16.

## Outcome moved

Before: 0 of the 6 live late memories (written 2026-11-16 15:10:43Z, about 18:57-20:26Z on 11-15)
were in any day or arc. After: all 6 land once, in day 2026-11-16 (proved by the reducer test with
the live shapes). Nothing is dropped on this path any more, so there is nothing left to count.

## Current architecture

PR #3469 (#3400 part 2) ordered `memory_episode` by its write time (`payload.available_at`) and kept
binding by `occurred_at`. A memory with no covering arc in the current day goes to the day's unbound
context when it leaves the 6 h context buffer. A guard (`item.at >= day_start`) kept memories about
an earlier day out of today's list by design ("that day is closed and is not rewritten"), so they
were in no day at all.

**Root cause, with evidence.** Denver day 11-15 closed at 2026-11-16 07:00Z. Each of the 6 rows is
folded at 15:10:43Z; its `occurred_at` is already outside the buffer horizon, so `_flush_buffer`
flushes it at once, and the guard (`19:xx 11-15 < 07:00 11-16`) drops it. `f5ea4fcd` was never
folded either: it shows in `temporal_self_day` 2026-11-16 only as
`frame.source_cursors.memory_episode = "2026-11-16T15:10:43.395995+00:00|f5ea4fcd-..."`, the
reducer's read cursor (it sorts last of the 6), which is why the issue's `like` query matched it.

## Architecture touched

- `orion/temporal_self/arcs.py` `_flush_buffer`: guard removed. The buffer only holds items folded
  into the current day (`_roll_day` flushes it fully), so every unbound item joins the current day.
- `orion/temporal_self/frame.py` `build_frame`: buffered unbound items no longer need `at >= day start`.

## Files changed

- `orion/temporal_self/arcs.py`: the rule above.
- `orion/temporal_self/frame.py`: live frame matches the rule.
- `orion/temporal_self/tests/test_tick_gap_and_late_memory.py`: new live-shape regression test; the
  #3469 test that pinned the drop now asserts the memory about yesterday lands today.
- `orion/temporal_self/README.md`: the rule, stated.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `day_context_event_ids` / `unbound_context_event_ids` can now hold a
  `memory_episode` whose `occurred_at` is before the day. The event row keeps `occurred_at` and its
  own `day_id` (the day it is about).
- Compatibility notes: no field changes; old stored state reads unchanged.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no. local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
.venv/bin/python -m pytest orion/temporal_self/tests orion/temporal_self/evals -q   -> 333 passed (334 after the review test)
  (new test fails before the fix: 0 of 6 ids in the frame; passes after)
.venv/bin/python -m pytest orion/orion_day/tests orion/dig/tests/test_organs.py orion/personality/tests/test_situation.py -q -> 182 passed
services/orion-hub: pytest tests/test_orion_day_temporal_day.py tests/test_orion_day_mailer.py -q -> 42 passed
scripts/check_integration_seam.py -> OK (16 organs; 29 source kinds covered)
```

## Evals run

```text
orion/temporal_self/evals (included in the 333 above) -> passed
```

## Docker/build/smoke checks

```text
Pure-code change in the shared orion/ package; no compose or dependency change. Live fold UNVERIFIED
until the temporal self reducer host is rebuilt and a new memory_episode is written.
```

## Review findings fixed

- Finding: review (code-review subagent) found no defects. It confirmed that the buffer only holds current-day items, that other kinds cannot land in the wrong day, and that no consumer assumes the ids fall inside the day window. One test gap: nothing checked that the result is the same when the fold is chunked across midnight with a late memory.
  - Fix: added `test_late_memories_about_a_closed_day_do_not_depend_on_chunking_across_midnight`. It uses the 6 live rows plus a memory written at 00:30 local about 23:00 the night before. It asserts the state is identical whether the rows are folded at once or in chunks with a boundary on midnight. It also asserts the 00:30 memory lands in the day it was written, not in the day before.
  - Evidence: `pytest orion/temporal_self -q` -> 334 passed.

## Restart required

Rebuild `orion-durable-runs` (runs the chronicle reducer) from clean main after merge (the
post-pull rebuild does this). The 6 rows of 11-16 are already behind the cursor and stay unfolded;
this fixes new rows only.

## Risks / concerns

- Severity: low
- Concern: rows already past the cursor (the 6 of 11-16) are not refolded.
- Mitigation: none needed for correctness going forward; verify on the next written memory with the
  issue's query (`#3584` stays open as the live check).
- Issues opened: none. Issues referenced: #3584 (Refs; close with live proof).

## PR link

PR_LINK_PLACEHOLDER
