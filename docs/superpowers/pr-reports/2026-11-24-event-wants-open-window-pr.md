## Summary

- Prime's event-wants menu now keeps a funded event want whose window is still open, however old its decision is. Before, a camera ask fell off Prime's menu 24 hours after it was decided, while it stayed open for Juniper for 7 days.
- `GET /physiology/event-wants` searches `want_ledger` back 192 hours (`OPEN_WINDOW_SEARCH_HOURS`: the 7-day camera-ask window plus one day). `hours` still bounds how old a decision may be for a want with no known window end.
- A want decided before `hours` is kept only when its window end is known and still in the future. The `gone` rule is unchanged: an answered or dismissed ask, or an ask whose card never opened, is not on the menu.
- The `hours` cap went from 72 to 192. Prime still sends `hours=24`; `prime_loop.py` is unchanged.

Fixes #3558

## Outcome moved

Prime can see open camera questions to Juniper for their whole 7-day life, not only their first day. Live replay at 2026-11-22 12:00Z: 16 asks were open and older than 24 h. 10 of them had a funded want. The old SQL returned 0 of them to Prime. The new SQL returns all 10.

## Current architecture

`read_menu` (`services/orion-durable-runs/app/event_wants_menu.py`) ran `LATEST_EVENT_DECISIONS_SQL`: the latest decision per event want, limited to `decided_at >= now - hours`. `menu_from_rows` then kept funded rows that were not expired and not `gone`. Orion Prime (`services/orion-mind/app/prime_loop.py`) calls it with `hours=24&limit=8&care=1`. A camera ask gets one ledger decision when it opens (`services/orion-sql-writer/app/ask_bind.py`, `latest = ask.expires_at` = created + 7 days), so on day 2 it was outside the search.

## Architecture touched

- `services/orion-durable-runs` only. Read-only SQL change plus one pure-function argument.
- No schema, bus, or env change.

## Files changed

- `services/orion-durable-runs/app/event_wants_menu.py`: the SQL picks each want's latest decision within 192 h, then keeps it when decided within `hours` or when funded with its own `window.latest` still ahead (the cast is guarded by a `CASE`, so one malformed value cannot fail the read). `menu_from_rows(..., since=)` drops an older row with no known window end. `MAX_HOURS` 72 -> 192.
- `services/orion-durable-runs/app/main.py`: endpoint docstring.
- `services/orion-durable-runs/README.md`: menu section explains the open-window rule.
- `services/orion-durable-runs/tests/test_event_wants_menu.py`: regression tests (3-day-old open ask kept; answered and expired asks not kept; old want with no window end not kept; SQL shape).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `GET /physiology/event-wants` also lists funded wants decided before `hours` whose window is still open. `considered` now counts those rows too. `hours` accepted up to 192 (was 72).
- Compatibility notes: same payload shape (`FundedEventWantsV1`). `limit` and net ordering unchanged, so Prime's list is still at most 8 wants.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs: pytest tests/test_event_wants_menu.py tests/test_retro_sitting_event_menu.py -q -> 20 passed
services/orion-mind: pytest tests/test_prime_event_intake.py tests/test_prime_hunger_loop.py -q -> 18 passed
```

## Evals run

```text
python services/orion-durable-runs/evals/event_rank_grading_mutations.py -> all mutations KILLED
python services/orion-sql-writer/evals/ask_bind_mutations.py -> needs ORION_SQL_WRITER_TEST_DATABASE_URI (not set here); not run
```

## Docker/build/smoke checks

Live read-only check (`BEGIN READ ONLY`, SELECT only on `want_ledger`, `orion_ask`), old SQL vs new SQL at `hours=24`:

```text
now (2026-11-24 ~19:30Z): rows old=94 new=154 (411 before the review fix); open funded asks old=8 new=8; open asks older than 24 h old=0 new=0
  (orion_ask has 0 open asks older than 24 h right now, so there is nothing to recover today)
replay at past times (decisions up to T only; ask open at T and older than 24 h):
  2026-11-18..21 12:00Z  truth=0      old=0 new=0
  2026-11-22 12:00Z      truth=16 (10 with a funded want)  old=0 new=10
  2026-11-23 12:00Z      truth=0      old=0 new=0
last 30 days: 61 answered + 25 dismissed asks stayed open longer than 24 h (~1,237 ask-hours Prime could not see)
malformed window.latest values in want_ledger: 0
EXPLAIN ANALYZE (review agent, first version of the SQL): idx_want_ledger_decided index scan, 697 rows read, 63 ms
```

Docker build not run: no dependency, Dockerfile, or env change.

## Review findings fixed

- Finding: `latest::timestamptz` in an `OR` could fail the whole read if one `latest` string were malformed (Postgres does not promise short-circuit order). Prime would lose its whole concern source.
  - Fix: the cast sits inside `CASE WHEN latest ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN ... END`.
  - Evidence: 20 tests pass; the live SQL runs; 0 malformed values today.
- Finding: the `latest IS NULL` branch returned about 250 dead world-pulse read rows on every read (their 3 h family window had closed), and they fed the label query.
  - Fix: an old row is read only when its own `window.latest` is still ahead. Family windows are at most 24 h. `menu_from_rows` still drops an old row with no window end.
  - Evidence: live rows at `hours=24` went from 411 to 154. The 11-22 replay still recovers all 10 asks.
- Finding (minor, not changed): the fake-pool tests do not run the SQL, so its meaning is pinned by the SQL-text test plus the live replay above. The repo has no real-Postgres fixture for orion-durable-runs.
- Finding (nit, noted): `considered` now counts rows from up to 8 days back. Nothing reads it (Prime reads `funded_open`).
- Finding (nit, noted): an unheld old ask can now build Prime's horizon drive as its 7-day window closes (`orion/prime/hunger.py` `event_horizon_needs`). This is the intended effect: Prime keeps the ask in view until it closes.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run from the clean primary `main` checkout after merge (the post-pull rebuild does this). orion-mind needs no restart.

## Risks / concerns

- Severity: low
- Concern: the menu query now reads up to 8 days of `want_ledger` event decisions (about 400 rows today vs about 100). It uses `idx_want_ledger_decided`.
- Mitigation: rows are filtered in SQL before they leave the database (154 rows today); `limit` still caps the result.
- Severity: medium
- Concern: the fix is UNVERIFIED live. Today no open ask is older than 24 h, so Prime's current menu does not change. The proof waits for an ask that stays open more than a day after deploy.
- Mitigation: verify-live ticket #3856.
- Issues opened: #3856 (verify-live). Issues closed: #3558.

## PR link

PR_LINK
