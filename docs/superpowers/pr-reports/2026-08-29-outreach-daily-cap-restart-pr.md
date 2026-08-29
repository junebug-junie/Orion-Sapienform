# A restart no longer hands Orion a fresh daily cap

Branch: `fix/outreach-daily-cap`

## Summary

- `EndogenousOutreach._sent_today` is an in-process counter initialised to `0` and zeroed only on a day rollover. **Nothing rehydrated it**, so every container restart reset the gate that bounds how often Orion interrupts Juniper.
- This repo deploys several times a day, so the cap was bounded by deploy frequency rather than by `daily_cap`.
- Recovers the count from `endogenous_outreach_decisions`, which already held it.
- Retried each tick until it lands; a failed read stays **unknown**, never zero.
- Also removes a comment orphaned by #1938 that described a deleted field, and the *rejected* design, as if current.

## Outcome moved

The cap was exceeded on **4 of the last 7 days**, peaking at nearly 3x:

```text
 mdt_day    | sent | cap
 2026-08-22 |    3 |   4
 2026-08-23 |    4 |   4
 2026-08-24 |    4 |   4
 2026-08-25 |   11 |   4
 2026-08-26 |    8 |   4
 2026-08-27 |    7 |   4
 2026-08-28 |    5 |   4
```

The 2026-08-28 row is traceable end to end: 4 sends by 12:29 MDT, `daily_cap` blocking at 20:12, then a 5th at **20:54** — four minutes after the #1938 deploy restart, with no day change in between.

This is a gate that protects Juniper, not the system. Every excess row is an interruption she was not supposed to get.

## Current architecture

`_sent_today` lives only in memory. `_roll_daily_counter` zeroes it when the local date changes; `__init__` sets `0`. `endogenous_outreach_decisions` (added 2026-08-22) has recorded every `sent` row since — it was write-only, with no reader.

## Architecture touched

- `endogenous_outreach_decisions.py` gains its first read (`count_sent_on`) and a shared `decision_log_enabled()` the writer now uses too.
- `EndogenousOutreach` gains `_recover_sent_today()`, awaited by both delivery paths.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: recovery, awaited in `maybe_outreach` and `offer_message`; `status()` surfaces recovery state; orphaned #1938 comment removed.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: `count_sent_on`, `decision_log_enabled`.
- `services/orion-hub/tests/test_endogenous_outreach.py`: 10 tests.

## Schema / bus / API changes

- Added: `status()` gains `sent_today_recovered` and `sent_today_recovery_failures` (additive, backs the existing debug endpoint).
- No migration, no table change, no bus channel, no schema registry entry.

## Env/config changes

- No keys added, removed, or renamed. `HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED` is now read by the reader as well as the writer; `.env_example` unchanged, so no sync required.

## Tests run

```text
services/orion-hub/tests/test_endogenous_outreach.py -> 173 passed
full hub suite -> see below
```

Mutation matrix — every fix has a test that fails without it:

| Mutation | Result |
|---|---|
| never recover (the live bug) | 6 failed |
| treat a failed read as zero | 1 failed |
| `offer_message` stops recovering | 1 failed |
| absolute assignment (drop the delta) | 1 failed |
| flag check removed from the reader | 1 failed |

The flag test asserts the engine is never **consulted**, not that the result is `None`: with no Postgres in the test environment `count_sent_on` returns `None` either way, so the result-only assertion I wrote first passed against the bug.

## Evals run

```text
none
```

`services/orion-hub/evals/` exists, so §11's "no harness" escape does not apply. Stated as a gap rather than skipped silently: this patch's behaviour is a deterministic gate, fully covered by unit tests, and the eval lane there measures caption quality — a daily-cap eval would need a multi-day live window to say anything the tests do not.

## Docker/build/smoke checks

```text
not run -- no Dockerfile, requirements, port, or boot-config change
```

The recovery SQL was verified directly against the live database instead:

```sql
SELECT count(*) FROM endogenous_outreach_decisions
WHERE reason='sent' AND (decided_at AT TIME ZONE 'America/Denver')::date = CAST('2026-08-28' AS date);
-- 5
```

## Review findings fixed

- Finding: **MUST-FIX 1** — `offer_message` (curiosity loop) increments the same shared counter but never triggered recovery, so a restarted hub delivered against a counter at `0`. `blocked_reason()` is sync and cannot await, so it cannot be the gate.
  - Fix: `await self._recover_sent_today()` at the top of `offer_message`.
  - Evidence: `test_offer_message_on_a_fresh_process_respects_the_cap` fails without it.
- Finding: **MUST-FIX 2** — recovery read the count, awaited a thread hop, then wrote an absolute value, discarding any send that landed in that window.
  - Fix: snapshot before the await, add the delta back. No await between the delta read and the write.
  - Evidence: `test_a_send_during_recovery_is_not_discarded`.
- Finding: **SHOULD-FIX 3** — with the decision log switched off the read still succeeds and returns `0`, marking the count recovered at zero.
  - Fix: shared `decision_log_enabled()`; the reader returns `None`.
  - Evidence: `test_the_reader_and_writer_agree_on_the_decision_log_flag`.
- Finding: **SHOULD-FIX 4** — a permanently-failing recovery (missing migration) leaves the cap unenforced while Postgres is healthy, with only a log line as evidence.
  - Fix: `status()` exposes `sent_today_recovered` and `sent_today_recovery_failures`.
- Finding: **SHOULD-FIX 5** — a comment claimed `str(self._tz)` differs from `self.timezone_name` on the fallback path. It does not; `__init__` reassigns the latter to `"UTC"`. The test built on it passed against a hardcoded `"UTC"` and guarded nothing.
  - Fix: comment corrected to say it is defensive style; test now pins a real zone reaching Postgres.
- Finding: **NIT 7/8** — a no-op rollover marking with a 10-line comment and a test, and an assertion guaranteed by construction.
  - Fix: both removed; `_roll_daily_counter` is back to three lines.
- Not taken: the review's suggestion to hold `_send_lock` across recovery. The delta approach fixes the same lost update without blocking sends for the query, and `_send_lock` can be held for minutes during `_generate`.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: **low**. Concern: the recovered count is a **lower bound**. `record_decision` is fire-and-forget and swallows INSERT failures, so a delivered message whose row never landed is a send with no row. Errs toward under-counting, i.e. toward allowing an extra send — same direction as the bug, so it narrows the gap without closing it. Stated in the docstring rather than claimed as equality.
- Severity: **low**. Concern: `local_date` is computed before the await, so a query spanning midnight would set the previous day. Self-corrects on the next `_gate_inputs`; the write now re-reads the clock.
- Severity: **low**. Concern: while recovery is failing, each unauthenticated `POST /api/debug/endogenous-outreach/trigger` spawns a DB attempt. The same amplification `endogenous_outreach_decisions.py` already discloses and accepts for its writer.

## PR link

<to be filled>
