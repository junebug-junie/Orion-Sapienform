## Summary

- The resonance health monitor's edge-trigger logic (only page on a real
  healthy->unhealthy transition) is correct and already tested, but the
  signal it watches — `violation_count` over `detect_resonance`'s 200-row
  lookback window — is noisy enough on its own to flip healthy<->unhealthy
  repeatedly for the same theme. Live 2026-09-10: one theme (
  `open-loop-7376a3da4050`) paged 12 times in 24 hours, 29 total unacked
  historically.
- `NotificationRequest.dedupe_key`/`dedupe_window_seconds` look like the
  built-in fix and are already accepted by `orion-notify` and stored on the
  row — but nothing in `orion-notify` ever reads them back to suppress a
  repeat. Confirmed by search; `orion-sql-writer`'s own
  `BusFallbackAlertState` docstring documents the exact same finding for a
  different caller ("nothing in `services/orion-notify` ever reads them
  back to suppress a duplicate -- verified by search, 2026-08-14").
- Adds a real, durable per-check-key cooldown instead: a new table
  (`substrate_reverie_resonance_alert_cooldown`), two `store.py` functions
  mirroring the existing `reverie_refractory_is_suppressed`/`_suppress`
  pair, wired into `ResonanceHealthMonitor._publish`.
- The cooldown only gates the "worsening" (error-severity, ack-required)
  publish path. A genuine recovery note always goes out.
- Default cooldown: 3600s (`ORION_REVERIE_RESONANCE_ALERT_COOLDOWN_SEC`).

## Outcome moved

Repeat pages for the same still-flapping theme, within the cooldown
window, are suppressed at the source instead of each becoming a fresh
Pending Attention item requiring its own acknowledgment.

## Current architecture

`ResonanceHealthMonitor` (`services/orion-thought/app/resonance_monitor.py`)
tracks per-theme healthy/unhealthy state in memory (`_last_healthy`,
reconstructed at startup from `orion-notify`'s pending list) and calls
`_publish()` -> `NotifyClient.attention_request()` only on a transition.
That part already has a passing regression test
(`test_monitor_alerts_only_on_worsening_transition_not_every_tick`). The
gap was one layer up: nothing stopped the underlying `violation_count`
metric itself from oscillating across the edge-trigger boundary many times
a day, and the `dedupe_key`/`dedupe_window_seconds` fields that look like
they'd catch this are decorative everywhere in this codebase (confirmed by
grep — every caller that sets them either doesn't need real suppression,
or, like `BusFallbackAlertState` and `orion-actions`' various
`deduper.try_acquire`, already builds its own real dedupe state because
this one doesn't work).

## Architecture touched

- `services/orion-thought` only. No bus/schema contract changes.

## Files changed

- `services/orion-sql-db/manual_migration_reverie_resonance_alert_cooldown.sql`:
  new table `substrate_reverie_resonance_alert_cooldown(check_key pk,
  last_alerted_at, updated_at)` + index on `last_alerted_at`. Applied live
  against `conjourney` before deploy.
- `services/orion-thought/app/store.py`: `resonance_alert_cooldown_active`
  (fail-open read) / `resonance_alert_cooldown_mark` (upsert), mirroring
  `reverie_refractory_is_suppressed`/`_suppress`.
- `services/orion-thought/app/resonance_monitor.py`: `_publish` checks the
  cooldown before sending a non-recovered page; marks it after a
  successful send. Module docstring updated with the live finding and
  rationale.
- `services/orion-thought/app/settings.py`: new
  `reverie_resonance_alert_cooldown_sec` field (default 3600.0).
- `services/orion-thought/.env_example`: new
  `ORION_REVERIE_RESONANCE_ALERT_COOLDOWN_SEC=3600`, with a comment
  pointing at the migration.
- `services/orion-thought/tests/test_store.py`: 8 new tests for the two
  store functions (active/inactive/elapsed/disabled/fail-open on both
  sides).
- `services/orion-thought/tests/test_resonance_monitor.py`: 3 new tests —
  cooldown-active skips the network call but still advances edge-tracking
  state, cooldown-inactive sends and marks, recovery is never gated even
  when the cooldown mock reports active.

## Schema / bus / API changes

- Added: `substrate_reverie_resonance_alert_cooldown` table (new, additive,
  no existing reader/writer affected).
- Removed: none.
- Renamed: none.
- Behavior changed: `ResonanceHealthMonitor` now skips sending a repeat
  "worsening" `attention_request` within the cooldown window of the last
  one that actually sent for the same theme. Recovery notes unaffected.
- Compatibility notes: fully additive; a missing/unmigrated table
  degrades to `resonance_alert_cooldown_active` always returning `False`
  (fail-open), so a deploy without the migration just behaves like before
  this patch, not broken.

## Env/config changes

- Added keys: `ORION_REVERIE_RESONANCE_ALERT_COOLDOWN_SEC=3600`
  (`services/orion-thought/.env_example`, `services/orion-thought/.env`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`:
  yes (writes to the primary checkout's `.env`; manually mirrored into
  this worktree's `.env` afterward, per this repo's known sync-script
  limitation).
- skipped keys requiring operator action: none.

## Tests run

```text
venv/bin/python -m pytest services/orion-thought/tests/test_store.py services/orion-thought/tests/test_resonance_monitor.py -q
  73 passed

venv/bin/python -m pytest services/orion-thought/tests -q
  389 passed, 3 failed (pre-existing, unrelated -- env-default drift in
  test_settings_mind_enrichment.py / test_settings_salience_flags.py /
  test_reverie_spontaneous_thought.py, confirmed by inspection: local
  ORION_MIND_BASE_URL etc. diverge from .env_example on this host,
  untouched by this patch)

python3 scripts/check_env_template_parity.py
  PASS (86 services compared; orion-thought clean)
```

## Evals run

No eval harness exists for `orion-thought`'s notification/health-monitor
path; not adding one for a single cooldown gate (thin patch, not a new
capability class).

## Docker/build/smoke checks

```text
docker exec orion-athena-sql-db psql -U postgres -d conjourney -f - < services/orion-sql-db/manual_migration_reverie_resonance_alert_cooldown.sql
  CREATE TABLE
  CREATE INDEX

bash scripts/safe_docker_build.sh orion-thought build
  Image orion-thought-thought Built

bash scripts/safe_docker_build.sh orion-thought up -d
  Container orion-athena-thought Recreated / Started

curl -fsS http://localhost:7155/health
  {"ok":true,"service":"orion-thought", ...}

docker exec orion-athena-thought python -c "import app.store as s; print('resonance_alert_cooldown_active' in dir(s))"
  True   # confirms the running container actually has the new code
```

## Review findings fixed

Code review (subagent) found nothing material. Two accepted, non-blocking
observations, not changed:

- Finding: `_publish` discards `resonance_alert_cooldown_mark`'s return
  value, so a DB write failure right after a successful send is silent
  and could let one duplicate page through on the very next flap.
  - Fix: none applied. Matches the existing fire-and-forget pattern for
    `reverie_refractory_suppress` elsewhere in this same file; the
    failure is still logged at `logger.warning` inside the store
    function itself. Narrow window (only the write, not the read, has to
    fail), consistent with established precedent, not worth the added
    complexity for this patch.
  - Evidence: `services/orion-thought/app/resonance_monitor.py` (mark
    call site), `services/orion-thought/app/store.py::reverie_refractory_suppress`
    (the pre-existing pattern being matched).
- Finding: the new `idx_substrate_reverie_resonance_alert_cooldown_last_alerted`
  index isn't used by any current query (all lookups are by `check_key`,
  the primary key).
  - Fix: none applied. Harmless, cheap, left in place for a future
    prune/TTL job on stale cooldown rows that doesn't exist yet.
  - Evidence: `services/orion-sql-db/manual_migration_reverie_resonance_alert_cooldown.sql`.

## Restart required

Already restarted live as part of this change (see Docker/build/smoke
checks above). No further action needed.

## Risks / concerns

- Severity: low
- Concern: a DB write failure on `resonance_alert_cooldown_mark` right
  after a successful notification send is silent to the caller (see
  Review findings above) and could let one duplicate page through.
- Mitigation: logged at `logger.warning`; narrow window; matches existing
  precedent elsewhere in the same file. Revisit if it's ever observed live.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2192
