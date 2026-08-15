# Alert on bus_fallback_log backlog at 5, 10, 15, ...

Branch: `feat/fallback-backlog-alerts`
Service: `orion-sql-writer`

## Summary

- Adds a background watcher to `orion-sql-writer` that polls `bus_fallback_log`
  every 300s over a 24h trailing window and sends an alert to **Hub (in_app) and
  email** each time the count crosses a new multiple of 5.
- One alert per level. The high-water mark is persisted in a new single-row
  table `bus_fallback_alert_state`, so redeploys do not re-send, and only a full
  drain below the step re-arms — a partial dip does not, because the window is
  trailing and dips constantly on its own.
- Severity is `error` — traced to the gate that actually decides
  (`should_send_email()` on the `/notify` path), not to the policy file, which
  looks authoritative and is not on this path.
- Alerts carry kinds and counts only, never payloads.
- Found and documented (not fixed here) that `legacy.message` — 80 of the 87
  fallback rows — is an off-contract raw-dict publish (**corrected same day:
  originally written as "real cognition being silently dropped"; it is a
  redundant copy of an already-persisted turn, see the correction below**), and
  that the one
  test which should have caught it explicitly exempts it.
- Found and documented that the notify service's `dedupe_key` /
  `dedupe_window_seconds` are accepted and stored but never enforced.

## Outcome moved

`bus_fallback_log` had no watcher at all. Two separate routing failures on
2026-08-13/14 each ran for hours while every surface reported healthy —
container up, producer ticking on schedule, `PUBSUB NUMSUB` showing a live
subscriber, one WARNING per dropped event and no error. Both were found only
because someone happened to run a row count by hand.

After this patch a backlog of 5 produces an email and a Hub card within 5
minutes.

## Current architecture

`handle_envelope` in `services/orion-sql-writer/app/worker.py` has two dispatch
paths (`settings.route_map` → a SQL model, and the adapter branch →
`evidence_units`). Anything neither handles falls to `_write_fallback`
(`worker.py:1484`), which writes the raw payload to `bus_fallback_log` and logs
a single WARNING.

That table is a good safety net — a routing mistake loses the *destination*, not
the *data*. Nothing read it.

`app/route_coverage.py` (PR #1648, merged 2026-08-14) closed half the gap: it
compares the subscribe list against the route map at startup, so *config drift*
between a durable `.env` and the code is now loud at boot.

It cannot see the other half. A kind that arrives on a subscribed channel but was
never in the route map at all produces no drift to detect — the config is
self-consistent and simply incomplete. Route coverage reports
`route_coverage_ok subscribed=72 all_routed=true` while `legacy.message` events
land in the fallback log every few hours.

## Architecture touched

- New periodic task in `orion-sql-writer`'s lifespan, alongside the existing
  Hunter task. Independent of `orion_bus_enabled` — the rows it reads are
  already in Postgres.
- First outbound notify call from this service. It already *persisted* notify
  records (`app/models/notify_models.py`); it had never *sent* one, so
  `NOTIFY_SERVICE_URL` / `NOTIFY_API_TOKEN` are new keys here.
- No bus channel, no schema registry entry, no new event kind on the wire. The
  alert is an HTTP POST to `orion-notify`.

## Files changed

- `services/orion-sql-writer/app/fallback_watch.py` (new): the watcher. The
  escalation rule lives in two pure functions with no clock, database, or
  network in reach.
- `services/orion-sql-writer/app/models/fallback_alert_state.py` (new):
  single-row high-water-mark table.
- `services/orion-sql-writer/tests/test_fallback_watch.py` (new): 40 tests.
- `services/orion-sql-writer/requirements.txt`: `requests`, which
  `NotifyClient` needs and this service did not have.
- `services/orion-sql-writer/app/main.py`: start/cancel the watch task.
- `services/orion-sql-writer/app/settings.py`: 4 watcher keys + 2 notify keys.
- `services/orion-sql-writer/app/models/__init__.py`: export the new model.
- `services/orion-sql-writer/.env_example`, `docker-compose.yml`: the same keys.
- `scripts/sync_local_env_from_example.py`: allowlist the new keys so the sync
  actually propagates them.

## The escalation rule

Thresholds are multiples of `step` (5, 10, 15, ...). An alert fires when the
windowed count first *reaches* a multiple higher than the highest already
alerted. Consequences, all pinned by tests:

- One alert per level, not one per poll. A backlog parked at 7 for a week is one
  email, not 2,016.
- A jump past several levels (0 → 17) sends **one** alert naming 15, not three.
- A drain below `step` resets the mark to 0, and a later climb alerts again —
  that is a new incident, not a repeat.
- A partial drain (15 → 11) re-arms **nothing**. Only a full drain below `step`
  does. Ratcheting down level-by-level is the intuitive implementation and it is
  wrong — see the oscillation finding under review, where it produced six
  "crossed 5" alerts in three weeks against the real data.
- `count >= threshold`, not `count > threshold`. Off by one in the noisy
  direction, which is the correct direction for a monitor whose entire history
  is failing silently.

## Why severity is `error`

Traced through the handler this actually calls — `POST /notify`
(`services/orion-notify/app/main.py:194`). The policy file looks authoritative
and is **not on this path**; an earlier draft of this report justified the
choice from `rules.yaml` and was right about the answer for the wrong reason
(caught in review).

- **email** is gated solely by `should_send_email()`
  (`app/email_delivery.py:15`): true when severity is `error` or `critical`, or
  when `channels_requested` contains `email`.
- **in_app** is published unconditionally at `main.py:233` whenever the bus and
  `NOTIFY_IN_APP_ENABLED` are up — "we do this blindly as a router", per the
  comment there. Severity is not consulted.
- `Policy.evaluate` — and therefore the `channels` lists, the `throttle` blocks,
  and quiet hours — runs only on `/attention/request` (`main.py:274`).

So `error` is what makes the email go out. Both the severity *and*
`channels_requested=["email","in_app"]` are set, so the alert survives an edit
to either gate. Deliberately **not** escalating to `critical` at higher
thresholds: it buys no additional channel on this path, so it would be
decoration.

Pinned by `test_error_severity_is_what_actually_triggers_the_email` and
`test_the_real_outbound_request_passes_the_real_email_gate`, which build the
real outbound request and feed it to the real `should_send_email` — not to a
copy, and not to `rules.yaml`.

## Privacy

Alerts carry **kinds and counts only**. Fallback rows keep whatever the
undelivered event held, and for the largest current contributor that is Orion's
own prompts, responses, and reasoning traces. Those do not belong in an email.

`test_no_payload_content_reaches_the_alert` seeds rows with sentinel payloads and
`correlation_id`s and asserts on the fully-rendered title and body, so the
natural "make this alert more useful" edit — pasting in an example row — fails
loudly.

## Metric quality gate (CLAUDE.md §0A)

1. **Provenance.** `_write_fallback` (`worker.py:1484`), reached from the
   `"Unknown kind"` branch at `worker.py:2370` and from six exception handlers.
   Real code, one producing function.
2. **Independence.** Not redundant with `route_coverage` (config drift at boot;
   blind to a kind that was never routed) or `orion-mesh-guardian` (HTTP/Redis
   liveness; both incidents had a live subscriber and a healthy container).
3. **Theory anchor.** Direct measurement, not a proxy: a row in this table is by
   construction an event that had no route.
4. **Live-data sanity.** 87 rows over 3 weeks, 0–23/day, with many days at
   exactly zero — a real rest state, not a floor artifact. Non-degenerate in
   both directions.
5. **Existing mechanism.** Searched. `orion-mesh-guardian` is the watcher
   precedent and supplied the `NotifyClient` pattern; it has no Postgres access
   and cannot see this.
6. **Reversibility.** One module, one table, one env flag. Cheap to remove.

## Two findings surfaced, not fixed

### `legacy.message` — 80 of 87 rows are a duplicate publish, not lost data

> **CORRECTION (2026-08-14, same day).** As originally written this section
> claimed these rows were "real cognition being dropped" and "never persisted
> anywhere else." **Both claims were wrong**, and the same false claim appears
> in `2026-08-14-fallback-watch-review-fixes.md`. Measured afterwards against
> live Postgres: all 80 rows have a matching `chat_history_log` row with a
> byte-identical response length — `total=80, no_history_row=0,
> response_matches_exactly=80, differs=0`. Nothing was lost. They were a third,
> redundant publish of a turn already persisted by two properly-enveloped
> publishes on the same channel. Root-caused and deleted in
> `2026-08-14-hub-legacy-chat-publish-kill.md`. The rest of this section stands.

`orion/core/bus/codec.py:72` names an envelope `legacy.message` when a producer
publishes a **raw dict with no `kind` field**. Those rows carry `prompt`,
`response`, `reasoning_trace`, `spark_meta`, `session_id`, `mode`, `source` —
chat turns. They have been landing in the fallback log since at least
2026-07-24.

`services/orion-sql-writer/tests/test_route_map_completeness.py:47` lists
`legacy.message` under `LEGACY_KIND_ALIASES`, described as "legacy / multi-kind
channels where envelope kind is resolved at runtime." At runtime it is not
resolved — it reaches the terminal `_write_fallback(..., "Unknown kind")`. The
one test that would have caught this explicitly exempts it.

Not fixed here: finding the unwrapped producer is a separate patch. This
watcher's first alert will be about it.

### notify's `dedupe_key` is decorative

`NotificationRequest.dedupe_key` and `dedupe_window_seconds` are accepted,
stored on the record, and set by `attention_escalation.py:71-72`, but **nothing
in `services/orion-notify` ever reads them back to suppress a duplicate** —
verified by search across `app/*.py`. The policy rules carry
`dedupe_window_seconds: 60` on the `critical` and `error` paths and it does
nothing.

This watcher therefore does not rely on it. The persisted high-water mark is the
only real dedupe in the path. `dedupe_key` is still set, so it starts working
for free if enforcement ever lands.

## Env/config changes

- Added keys (`services/orion-sql-writer`): `SQL_WRITER_FALLBACK_WATCH_ENABLED`
  (true), `..._INTERVAL_SEC` (300), `..._WINDOW_SEC` (86400),
  `..._THRESHOLD_STEP` (5), `NOTIFY_SERVICE_URL` (`http://notify:7140`),
  `NOTIFY_API_TOKEN` (empty).
- Removed / renamed: none.
- `.env_example` updated: yes.
- Local `.env` synced via `python scripts/sync_local_env_from_example.py`: yes —
  5 of 6 keys written and verified present.
- Skipped keys requiring operator action: `NOTIFY_API_TOKEN`, deliberately left
  out of the sync allowlist because it is a credential and `.env_example`
  carries it empty; the settings default of `""` makes it optional, and the live
  notify server has `API_TOKEN` empty so no token is required today.

The sync script needed a change to work at all here: `should_sync_key` gates on
an allowlist, so new keys are skipped silently. Added the
`SQL_WRITER_FALLBACK_WATCH_` prefix and `NOTIFY_SERVICE_URL` as an exact key.
The first run reported success while writing nothing — worth knowing.

`NOTIFY_SERVICE_URL` host is the compose **service** name `notify`, not the
directory name `orion-notify` and not the container name `orion-athena-notify`.
Both of those have already broken this exact URL in other services (see the
comment block in `services/orion-notify-digest/.env_example`).

## Tests run

```text
$ pytest tests/test_fallback_watch.py -q
40 passed, 8 warnings in 1.14s

$ pytest tests -q
10 failed, 280 passed, 3 skipped, 34 warnings in 13.76s

$ pytest tests -q --ignore=tests/test_fallback_watch.py
10 failed, 240 passed, 3 skipped, 27 warnings in 12.74s
```

The 10 failures are byte-identical with and without this patch — pre-existing
cross-test isolation failures in `test_grammar_truth.py`,
`test_journal_entry_payload_boundary.py`, `test_notify_attention_ack.py`, and
`test_notify_attention_escalate.py`. This patch adds 40 passing tests and no
failures.

### Mutation testing

Every test that matters was checked against a deliberately broken
implementation. All five mutations were caught:

```text
re-alert on every poll (no high-water dedupe)     -> 6 failed
strict > instead of >= at the threshold            -> 5 failed
drop the trailing-window filter (count forever)    -> 1 failed
sort kinds ascending instead of busiest-first      -> 1 failed
state never remembers across evaluations (restart) -> 2 failed
```

## Evals run

```text
No eval harness exists for orion-sql-writer (services/orion-sql-writer/evals/
does not exist). Not created here: this patch has no quality dimension that
unit tests do not already cover -- the escalation rule is exact arithmetic and
is mutation-tested.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-sql-writer build
Image orion-sql-writer-sql-writer Built

$ docker run --rm --env-file services/orion-sql-writer/.env \
    orion-sql-writer-sql-writer python -c "import app.fallback_watch; ..."
import ok
enabled= True
interval= 300
window= 86400
step= 5
notify_url= http://notify:7140
table= bus_fallback_alert_state
rule 0->17: (15, 15)
```

The in-image import check is not ceremony. Two days ago the route-coverage
module crash-looped this service because module-level `parents[3]` path
arithmetic was correct in the repo layout and an `IndexError` at `/app/app/`.
The image, not the worktree, is where that shows up.

Notify reachability verified live from inside the running container:

```text
$ docker exec orion-athena-sql-writer python3 -c "socket.gethostbyname('notify')"
172.18.0.5
$ docker exec orion-athena-sql-writer python3 -c "urlopen('http://notify:7140/health')"
{"ok":true,"service":"notify","mode":"router_with_escalation","smtp_configured":true}
```

The real query against live Postgres, run from inside the built image before
deploy:

```text
live 24h count = 17
by kind        = [('legacy.message', 12), ('juniper.affective_state.v1', 5)]
would alert at = (15, 15)
```

Post-deploy:

```text
$ scripts/safe_docker_build.sh orion-sql-writer up -d --build
Container orion-athena-sql-writer Started

$ docker inspect -f 'Running={{.State.Running}} Restarts={{.RestartCount}}'
Running=true Restarts=0

$ docker logs orion-athena-sql-writer | grep fallback_watch
[SQL_WRITER] INFO - sql-writer.fallback_watch -
  fallback_watch_started interval_sec=300 window_sec=86400 step=5
```

Unrelated pre-existing condition observed while verifying, NOT caused by this
patch and not fixed here: `/health` reports `degraded=true` with
`degraded_reasons: ['grammar_retention_failed']` — the startup `grammar_events`
retention pass dies on the 10s `statement_timeout`
(`psycopg2.errors.QueryCanceled`, `rows_pruned=0 batches=0 elapsed_sec=10.04`).
The same log recommends applying `idx_grammar_events_source_created` via
`services/orion-sql-db/manual_migration_grammar_atlas.sql`. Worth its own patch.

## Review findings fixed

Five findings, one of which meant the feature could never have worked.

- **Finding (must-fix): the alert could never have been sent.**
  `orion/notify/client.py:10` imports `requests` at module level;
  `services/orion-sql-writer/requirements.txt` did not list it and nothing
  pulled it in transitively. Every other `NotifyClient` consumer declares it;
  this service was the exception. The import sat *outside* `_send_alert`'s
  `try`, so the first crossing would have raised `ModuleNotFoundError`,
  propagated through `evaluate_once`'s `rollback; raise`, and discarded the
  diagnostic state write along with the alert — 288 ERROR lines a day, zero
  alerts, and an empty state row implying the watcher had never run. The
  monitor built to end silent failures, failing silently.
  - Fix: `requests==2.32.3` added; `_send_alert` now has everything inside the
    `try`; two tests added (one asserts the dependency is declared, one performs
    the import eagerly).
  - Evidence: before — `docker exec orion-athena-sql-writer python -c "from
    orion.notify.client import NotifyClient"` → `ModuleNotFoundError: No module
    named 'requests'`. After rebuild — `NotifyClient import: ok`.
  - My own in-image check missed this because `app/fallback_watch.py` imports
    `NotifyClient` lazily inside the send path, so importing the module proved
    nothing about the module it needed.

- **Finding (should-fix): the escalation rule re-alerted on oscillation.**
  The mark ratcheted down to whatever level the count currently sat at. The
  count is taken over a **trailing** window and therefore drifts in both
  directions all day as rows age out, so a backlog crossing a level boundary
  re-armed and re-alerted that level every crossing.
  - Evidence: review replayed the shipped functions over the real 87
    `created_at_ts` values in live `bus_fallback_log` at 300s/86400s/step-5 —
    **12 alerts across 20 days, six of them "crossed 5"**. A count alternating
    14/15 between polls alerted every other poll, ≈144 emails/day. Exactly the
    alert fatigue this module's docstring claims to prevent.
  - Fix: only a full drain below `step` re-arms. Two new tests, including the
    literal 15/14/15 oscillation.
  - Confirmed in-image after the fix: `oscillation 15,14,15 -> [None, None, None]`.

- **Finding (should-fix): `_send_alert`'s "Never raises" docstring was false.**
  Both imports and the `NotificationRequest` construction were outside the
  `try`. Not a crash-loop risk (the loop's catch-all holds) but it threw away
  the state write with the alert.
  - Fix: everything inside the `try`.

- **Finding (should-fix): shutdown could skip cancelling `watch_task`.**
  `CancelledError` derives from `BaseException`, so `contextlib.suppress(
  Exception)` does not catch it, and the bus chassis's `start()` has no handler
  — awaiting the first task re-raised out of the loop before the second was
  ever cancelled, leaving it pending at loop close, possibly mid-DB-query.
  Converting the single-task shutdown into a loop is what made the pre-existing
  suppress bug bite.
  - Fix: `await asyncio.gather(*pending, return_exceptions=True)`.

- **Finding (should-fix): the severity rationale documented a path `/notify`
  does not execute.** `Policy.evaluate` runs only on `/attention/request`
  (`main.py:274`). On `/notify`, email is gated solely by `should_send_email()`
  (`email_delivery.py:15`) and in_app is published unconditionally by the
  router. The conclusion (`error`) was right; the stated mechanism was wrong,
  and **both tests asserted against `rules.yaml`** — they would have kept
  passing if `should_send_email` changed to exclude `error`.
  - Fix: docstring corrected to the real path; request construction extracted to
    `build_alert_request()` so the outbound request is fed to the *real*
    `should_send_email` gate in a test; `channels_requested=["email","in_app"]`
    added as belt-and-braces so the mail survives an edit to either gate.

Nits also taken: an unconfigured `NOTIFY_SERVICE_URL` no longer consumes the
crossing (no network cost to retrying, and burning the level permanently would
be worse); `last_alert_sent_at` is stamped only on a real send; the state commit
now happens before the blocking 10s HTTP call rather than holding a pool
connection and a row lock across it; a non-positive `step` logs at ERROR instead
of going silently inert; and the production `now` default is exercised by a test.

Review confirmed clean, with no findings, on: privacy (traced end to end —
`count_backlog` never has `payload`/`correlation_id`/`error` in scope), the env
sync change (`NOTIFY_SERVICE_URL` appears in three `.env_example` files all with
the identical value, only one is in `DEFAULT_SERVICES`, and diverged-detection
protects the rest), import-time side effects, event-loop blocking, session
cleanup, and pool pressure. Review also independently confirmed the
`dedupe_key`-is-never-enforced finding and noted it is understated: since
`/notify` skips policy entirely, the `throttle` blocks are equally decorative
there.

### Re-run mutation testing after the fixes

```text
revert to level-by-level ratchet-down (the original bug) -> 3 failed
soften severity to warning and drop channels_requested   -> 1 failed
unconfigured notify consumes the crossing anyway         -> 1 failed
stamp last_alert_sent_at even when the send failed       -> 1 failed
```

The last mutation initially passed — no test covered it. Added the assertion,
then re-ran to confirm it fails.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-fallback-backlog-alerts
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

`Base.metadata.create_all()` creates `bus_fallback_alert_state` on boot. This is
a **new table**, not a new column, so no hand-applied `ALTER TABLE` is needed —
`create_all` does handle this case.

## Risks / concerns

- **Severity: low.** The watcher runs inside the service it monitors. If
  `orion-sql-writer` is down entirely, no alert. Accepted: a dead writer stops
  producing fallback rows anyway, and the failure mode actually observed twice
  is a *healthy* writer misrouting, which this does catch. `orion-mesh-guardian`
  already covers liveness.
- **Severity: low.** The first alert after deploy fires immediately on the
  existing `legacy.message` backlog (13 rows in the last 24h at time of
  writing). That is correct behaviour, not a false positive — but it is expected
  and should not be read as a new incident.
- **Severity: low.** Rows with a NULL `created_at_ts` are invisible to the count.
  `_write_fallback` always sets it and there are zero such rows live, but a row
  inserted by another path would be missed rather than counted as recent.
  Documented in `count_backlog`'s docstring and pinned by a test.
- **Severity: low (corrected down from medium), pre-existing, not introduced
  here, now fixed.** `legacy.message` events reach the fallback log and have for
  at least 3 weeks. Originally written as data loss; measured afterwards to be a
  redundant publish of an already-persisted turn — see the correction above and
  `2026-08-14-hub-legacy-chat-publish-kill.md`.
