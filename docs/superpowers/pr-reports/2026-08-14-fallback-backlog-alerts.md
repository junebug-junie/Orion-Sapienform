# Alert on bus_fallback_log backlog at 5, 10, 15, ...

Branch: `feat/fallback-backlog-alerts`
Service: `orion-sql-writer`

## Summary

- Adds a background watcher to `orion-sql-writer` that polls `bus_fallback_log`
  every 300s over a 24h trailing window and sends an alert to **Hub (in_app) and
  email** each time the count crosses a new multiple of 5.
- One alert per level. The high-water mark is persisted in a new single-row
  table `bus_fallback_alert_state`, so redeploys do not re-send, and it ratchets
  back down as the backlog drains so recovery re-arms the lower levels.
- Severity is `error` — read off the live notify policy, not chosen by feel. It
  is the lowest severity that routes to both channels and the lowest that
  survives quiet hours.
- Alerts carry kinds and counts only, never payloads.
- Found and documented (not fixed here) that `legacy.message` — 80 of the 87
  fallback rows — is real cognition being silently dropped, and that the one
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
- `services/orion-sql-writer/tests/test_fallback_watch.py` (new): 32 tests.
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
- A partial drain (15 → 11) only gives back the levels actually released.
- `count >= threshold`, not `count > threshold`. Off by one in the noisy
  direction, which is the correct direction for a monitor whose entire history
  is failing silently.

## Why severity is `error`

From `services/orion-notify/app/policy/rules.yaml`:

| severity | channels |
|---|---|
| `critical` | `email`, `in_app` |
| `error` | `email`, `in_app` |
| `warning` | `in_app` only |
| `info` | `[]` — delivered nowhere |

and `Policy.evaluate` (`policy.py:81`) drops channels to `[]` during quiet hours
(22:00–07:00 America/Denver) for anything that is not `critical` or `error`.

Hub *and* email, at any hour, therefore means `error` at minimum. Deliberately
**not** escalating to `critical` at higher thresholds: it buys no additional
channel and no additional urgency in the policy, so it would be decoration.

Pinned by `test_severity_reaches_both_email_and_in_app_in_the_live_policy` and
`test_severity_survives_quiet_hours`, which assert against the real rules file
and the real `Policy` class rather than a copy — a plausible "don't be alarmist"
edit to `warning` would silently switch the email off and mute it overnight.

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

### `legacy.message` — 80 of 87 rows are real cognition being dropped

`orion/core/bus/codec.py:72` names an envelope `legacy.message` when a producer
publishes a **raw dict with no `kind` field**. Those rows carry `prompt`,
`response`, `reasoning_trace`, `spark_meta`, `session_id`, `mode`, `source` —
chat/cortex turns. They have been landing in the fallback log since at least
2026-07-24 and are never persisted anywhere else.

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
32 passed, 3 warnings in 1.06s

$ pytest tests -q
10 failed, 272 passed, 3 skipped, 29 warnings in 13.29s

$ pytest tests -q --ignore=tests/test_fallback_watch.py
10 failed, 240 passed, 3 skipped, 27 warnings in 12.74s
```

The 10 failures are byte-identical with and without this patch — pre-existing
cross-test isolation failures in `test_grammar_truth.py`,
`test_journal_entry_payload_boundary.py`, `test_notify_attention_ack.py`, and
`test_notify_attention_escalate.py`. This patch adds 32 passing tests and no
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

## Review findings fixed

_(pending — code review subagent running)_

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
- **Severity: medium, pre-existing, not introduced here.** `legacy.message`
  events are being dropped and have been for at least 3 weeks. See above.
