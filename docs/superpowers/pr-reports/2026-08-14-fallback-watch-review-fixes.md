# Review fixes for the bus_fallback_log backlog watcher

Branch: `fix/fallback-watch-review-fixes`
Service: `orion-sql-writer`
Follows: PR #1653 (merged 2026-08-14 05:23:13Z)

## Why this is a separate PR

PR #1653 was merged before its review fixes were pushed. The merge captured the
first two commits only. **`main` therefore carries a watcher that cannot send an
alert** — verified:

```text
$ git show origin/main:services/orion-sql-writer/requirements.txt | grep -c "^requests"
0
```

The live container is fine: it was built from the worktree that already had
these fixes, and the alert fired for real at 05:31:02Z. But `main` does not
match what is deployed, and a rebuild from `main` would silently revert to a
watcher that logs an error every 300s and mails nothing.

This PR carries the two commits that missed the merge.

## Summary

- **Must-fix: the alert could never have been sent on `main`.** `requests` is
  not a declared dependency of `orion-sql-writer`, and
  `orion/notify/client.py:10` imports it at module level.
- **The escalation rule re-alerted on oscillation.** Replayed over real data:
  12 alerts in 20 days, six of them "crossed 5".
- `_send_alert`'s "never raises" docstring was false.
- Shutdown could skip cancelling the watch task (`CancelledError` is a
  `BaseException`).
- The severity rationale cited a code path `POST /notify` does not execute, and
  two tests asserted against that same wrong path.

## Outcome moved

Before: a rebuild of `orion-sql-writer` from `main` produces a watcher that
raises `ModuleNotFoundError` on its first threshold crossing, logs
`fallback_watch_tick_failed` every 300 seconds forever, writes no state row, and
sends nothing. The monitor built to end silent failures, failing silently.

After: the alert path works, and is proven to work against the real
`should_send_email` gate rather than against a policy file that is not on the
delivery path.

## Files changed

- `services/orion-sql-writer/requirements.txt`: `requests==2.32.3`.
- `services/orion-sql-writer/app/fallback_watch.py`: hysteresis in
  `next_alert_threshold`; `build_alert_request()` extracted; everything inside
  `_send_alert`'s `try`; commit-before-send ordering; unconfigured-notify path;
  `last_alert_sent_at` only on a real send; non-positive `step` logs.
- `services/orion-sql-writer/app/main.py`: `gather(..., return_exceptions=True)`
  on shutdown.
- `services/orion-sql-writer/tests/test_fallback_watch.py`: 32 → 40 tests.
- `docs/superpowers/pr-reports/2026-08-14-fallback-backlog-alerts.md`: corrected
  and extended with review findings + live deploy evidence.

## Review findings fixed

### 1. (must-fix) The alert could never have been sent

`orion/notify/client.py:10` imports `requests` at module level.
`services/orion-sql-writer/requirements.txt` did not list it and nothing pulled
it in transitively. Every other `NotifyClient` consumer (orion-actions,
orion-security-watcher, orion-notify-digest, orion-world-pulse) declares it;
this service was the exception.

The import also sat *outside* `_send_alert`'s `try`, so the first crossing would
have raised, propagated through `evaluate_once`'s `rollback; raise`, and
discarded the diagnostic state write along with the alert — 288 ERROR lines a
day, zero alerts, and an empty `bus_fallback_alert_state` implying the watcher
had never run.

- Evidence, before: `docker exec orion-athena-sql-writer python -c "from
  orion.notify.client import NotifyClient"` → `ModuleNotFoundError: No module
  named 'requests'`.
- Evidence, after rebuild: `NotifyClient import: ok`.
- Fix: dependency added; everything moved inside the `try`; two tests — one
  asserts the dependency is declared, one performs the import eagerly so it
  fails at test time rather than at first-alert time.
- Note: my own in-image smoke check missed this because `fallback_watch.py`
  imports `NotifyClient` lazily inside the send path. Importing the module
  proved nothing about the module it needed.

### 2. The escalation rule re-alerted on oscillation

The high-water mark ratcheted down to whatever level the count currently sat at.
The count is taken over a **trailing** window, so it drifts in both directions
all day as rows age out — no incident required.

- Evidence: replayed the shipped functions over the real 87 `created_at_ts`
  values in live `bus_fallback_log` at 300s/86400s/step-5 — **12 alerts across
  20 days, six of them "crossed 5", three "crossed 10"**. A count alternating
  14/15 between polls alerted every other poll, ≈144 emails/day. Precisely the
  alert fatigue the module docstring claimed to prevent.
- Fix: only a full drain below `step` re-arms. A dip from 15 to 11 and back is
  one incident.
- Evidence after: in-image, `oscillation 15,14,15 -> [None, None, None]`.
- Every original test passed this bug because each fed a monotonic sequence.

### 3. `_send_alert`'s "Never raises" was false

Both imports and the `NotificationRequest` construction were outside the `try`.
Not a crash-loop risk (the loop's catch-all holds) but it discarded the state
write with the alert. Fix: everything inside.

### 4. Shutdown could skip cancelling `watch_task`

`CancelledError` derives from `BaseException`, so `contextlib.suppress(Exception)`
does not catch it, and the bus chassis's `start()` has no handler — awaiting the
first task re-raised out of the loop before the second was ever cancelled,
leaving it pending at loop close, possibly mid-DB-query. Converting the
single-task shutdown into a loop is what made the pre-existing suppress bug
bite.

Fix: `await asyncio.gather(*pending, return_exceptions=True)`.

### 5. The severity rationale cited a path `/notify` does not execute

`Policy.evaluate` runs only on `/attention/request` (`main.py:274`). On
`/notify`:

- **email** is gated solely by `should_send_email()` (`email_delivery.py:15`):
  severity in `{error, critical}`, or `channels_requested` contains `email`.
- **in_app** is published unconditionally (`main.py:233`) — "we do this blindly
  as a router". Severity is not consulted.
- The `channels` lists, `throttle` blocks, and quiet hours never run here.

The conclusion (`error`) was right; the mechanism was wrong, and **both tests
asserted against `rules.yaml`** — they would have kept passing if
`should_send_email` changed to exclude `error`.

Fix: docstring corrected to the real path; `build_alert_request()` extracted so
the outbound request is fed to the *real* gate in a test;
`channels_requested=["email","in_app"]` added so the mail survives an edit to
either gate.

### Nits also taken

- An unconfigured `NOTIFY_SERVICE_URL` no longer consumes the crossing — there
  is no network cost to retrying, and burning the only alert for a level
  because a key was blank would lose it permanently.
- `last_alert_sent_at` is stamped only on a real send. A column named "sent_at"
  set on failure lies to whoever queries it.
- The state commit now happens before the blocking 10s HTTP call, rather than
  holding a pool connection and a row lock on `bus_fallback_alert_state` across
  it.
- A non-positive `step` logs at ERROR instead of going silently inert.
- The production `now` default is exercised by a test; every other DB test
  passes an explicit `now`.

### Reviewed clean, no findings

Privacy (traced end to end — `count_backlog` never has
`payload`/`correlation_id`/`error` in scope), the env-sync change, import-time
side effects, event-loop blocking, session cleanup, and pool pressure. Review
also independently confirmed the `dedupe_key`-is-never-enforced finding and
noted it is understated: since `/notify` skips policy entirely, the `throttle`
blocks are equally decorative there.

## Tests run

```text
$ pytest tests/test_fallback_watch.py -q
40 passed, 8 warnings in 1.14s

$ pytest tests -q
10 failed, 280 passed, 3 skipped, 34 warnings in 13.76s

$ pytest tests -q --ignore=tests/test_fallback_watch.py
10 failed, 240 passed, 3 skipped, 27 warnings in 12.74s
```

The 10 failures are byte-identical with and without this file — pre-existing
cross-test isolation failures in `test_grammar_truth.py`,
`test_journal_entry_payload_boundary.py`, `test_notify_attention_ack.py`, and
`test_notify_attention_escalate.py`.

### Mutation testing

```text
revert to level-by-level ratchet-down (the original bug) -> 3 failed
soften severity to warning and drop channels_requested   -> 1 failed
unconfigured notify consumes the crossing anyway         -> 1 failed
stamp last_alert_sent_at even when the send failed       -> 1 failed
```

The last one initially passed — no test covered it. Assertion added, then
re-run to confirm it fails.

## Evals run

```text
No eval harness exists for orion-sql-writer (services/orion-sql-writer/evals/
does not exist). Not created here: the escalation rule is exact arithmetic and
is mutation-tested.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-sql-writer build
Image orion-sql-writer-sql-writer Built

$ docker run --rm --env-file services/orion-sql-writer/.env <image> python -c "
    from orion.notify.client import NotifyClient
    from app.fallback_watch import build_alert_request, next_alert_threshold ..."
NotifyClient import: ok
severity= error channels= ['email', 'in_app']
context= {'threshold': 15, 'count': 17, 'window_seconds': 86400}
oscillation 15,14,15 -> [None, None, None]
```

### Live, post-deploy — the alert actually fired

```text
$ docker inspect -f 'Running={{.State.Running}} Restarts={{.RestartCount}}'
Running=true Restarts=0

$ docker logs orion-athena-sql-writer | grep fallback_watch
fallback_watch_started interval_sec=300 window_sec=86400 step=5
fallback_watch_alert_sent threshold=15 count=17 window_sec=86400
fallback_watch_tick count=17 high_water=15 alerted=15
```

State row:

| last_alerted_threshold | last_count | last_alert_status | last_alert_sent_at |
|---|---|---|---|
| 15 | 17 | sent | 2026-08-14 05:31:02+00 |

Persisted at the receiving end (`notify_requests`):

```text
orion-sql-writer | sqlwriter.fallback_backlog | error |
[Orion] Bus fallback backlog crossed 15 (17 in 24h) | 2026-08-14 05:31:02
```

Hub leg: `PUBSUB NUMSUB orion:notify:in_app` → **1** subscriber, with
`NOTIFY_IN_APP_ENABLED=true`. That is the exact check that returned **0** for
`orion:substrate:juniper_affective_state` and was the original silent-drop bug.

Email leg: SMTP login verified from inside `orion-athena-notify` against the
live credentials (`smtp.gmail.com:587`, login OK), and the exact outbound
request passes the real `should_send_email` gate. `transport.send()` itself is
unverifiable from the host — see Risks.

## Restart required

Already deployed from this worktree — the running container is built from this
code, which is why the live alert fired. No restart needed for the current host.

Any other host, or a rebuild from `main` before this merges:

```bash
cd <worktree>
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- **Severity: medium. `main` is currently wrong.** Until this merges, a rebuild
  from `main` produces a watcher that cannot send. The deployed container is
  correct; the repo is not.
- **Severity: medium, pre-existing, not introduced here — `orion-notify` cannot
  report whether an email sent.** `services/orion-notify/app/main.py:46` gets a
  logger but the module never calls `logging.basicConfig`, so under uvicorn
  **zero `[NOTIFY]` lines have ever appeared** in that container's logs —
  confirmed by `grep -c "\[NOTIFY\]"` → 0 across its whole history, including
  for other services' notifications. That silences
  `email_send_eligible`/`_attempted`/`_succeeded`/`_failed`, so a
  `transport.send()` exception is caught and logged into nothing. The
  `notify_attempts` table that would hold a durable record is empty. Net: a 200
  from `/notify` proves acceptance, not delivery. Worth its own small patch.
- **Severity: low, pre-existing.** `/health` on orion-sql-writer reports
  `degraded=true` for `grammar_retention_failed` — the startup `grammar_events`
  retention pass dies on the 10s `statement_timeout`
  (`psycopg2.errors.QueryCanceled`, `rows_pruned=0 elapsed_sec=10.04`). The log
  itself recommends applying `idx_grammar_events_source_created` via
  `services/orion-sql-db/manual_migration_grammar_atlas.sql`. Unrelated to this
  patch.
- **Severity: low (was written as medium), pre-existing, fixed in a follow-up —
  `legacy.message`.**

  > **CORRECTION (2026-08-14, same day).** This bullet originally said these
  > rows were "persisted nowhere else." **That was wrong**, and the same false
  > claim appears in `2026-08-14-fallback-backlog-alerts.md`. Measured against
  > live Postgres afterwards: all 80 rows have a matching `chat_history_log`
  > row with a byte-identical response length — `total=80, no_history_row=0,
  > response_matches_exactly=80, differs=0`. No data was lost. Severity is
  > downgraded accordingly: the real cost was one WARNING per chat turn plus
  > alert noise, not lost cognition. Root-caused and deleted in
  > `2026-08-14-hub-legacy-chat-publish-kill.md`.

  `orion/core/bus/codec.py:72` names an envelope `legacy.message` when a
  producer publishes a raw dict with no `kind`. Those rows carry `prompt`,
  `response`, `reasoning_trace`, `spark_meta` — chat turns, landing in
  the fallback log since at least 2026-07-24. 12 of
  the 17 events in the alert that just fired were these.
  `tests/test_route_map_completeness.py:47` lists it under
  `LEGACY_KIND_ALIASES` as "resolved at runtime"; it is not — it reaches the
  terminal `_write_fallback(..., "Unknown kind")`. The one test that would have
  caught it exempts it.
