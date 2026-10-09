# PR report -- watchdog escalation failures are loud (exit 4 + fallback card)

## Summary

- When the substrate ladder watch cannot tell a human (its memory of sent cards is unreadable/unwritable, or orion-notify refuses the card), it now exits **4** and prints `ESCALATION FAILED: ...` instead of exiting 0/1 with one quiet log line.
- If its memory is broken it still sends the card for everything currently red, every run, with no dedupe. A card every 10 minutes is annoying; no card for 34 hours is what actually happened.
- The disk watchdog and the Postgres connection-headroom watch had the same shape; both get the same exit-4 contract and the same fallback.
- `check_substrate_ladder_liveness.py --test-escalation` sends exactly one labelled test card through the real orion-notify path, so the card path can be proven on demand.

## Outcome moved

Failure mode closed: 2026-09-26 -> 09-27, `/mnt/telemetry/orion-athena` was root-owned, so creating the ladder watch's state dir raised `PermissionError` on **572/572** cron runs. 204 of those were RED (cortex-orch running an image older than `orion/schemas/reading_turn.py`, ~34h) and raised **zero** Hub cards; each exited 1 with one `escalation failed (...)` line in `logs/orion-substrate-ladder-liveness.log`. Replayed that exact shape in tests: the run now exits 4, attempts the card anyway, and the next tick cards again.

## Current architecture

Three host-cron watchers (`make disk-threshold-watchdog`, `make postgres-headroom-watch`, `make substrate-ladder-watch`) each check something, then raise a Hub Pending Attention card via orion-notify `POST /attention/request`, deduped by a JSON state file under `${TELEMETRY_ROOT}/${PROJECT}/<name>/`. All three wrapped escalation in a catch-all that logged and moved on:

- ladder: any exception -> one stderr line, exit code unchanged (0 green / 1 red).
- headroom: any exception -> one stderr line; orion-notify `ok=False` -> exit 1 (same as a delivered alarm); missing notify client -> "alarm not escalated", exit 1.
- disk: unwritable state -> exit 2 *before measuring anything* (a full disk was never carded); `ok=False` -> exit 1.

## Architecture touched

Host scripts only. No service, bus, schema, or env change.

## Files changed

- `scripts/check_substrate_ladder_liveness.py`: `notify()` returns an `Escalation` outcome; stateless fallback (`_notify_stateless`) cards all red keys when state is unusable; save-failure after a send does not double-card; `EXIT_ESCALATION_FAILED = 4`; `--test-escalation`.
- `scripts/check_postgres_connection_headroom.py`: `notify_alarm`/`clear_alarm` return failure; undeduped card on unusable state; exit 4 on any escalation failure.
- `scripts/disk_threshold_watchdog.py`: `run()` returns `(state, any_bad, failures)`; unusable state still measures every path and cards bad ones undeduped; exit 4 replaces exit 2.
- `tests/scripts/test_substrate_ladder_liveness.py`, `tests/test_check_postgres_connection_headroom.py`, `tests/test_disk_threshold_watchdog.py`: incident replays (0o555 parent dir -> real `PermissionError`), notify refused / raising, no double card, plain-red when already carded.
- `scripts/README.md`, `Makefile`: exit-4 contract documented.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: exit code 4 on escalation failure (all three watchers); disk watchdog no longer returns 2; ladder gains `--test-escalation`.
- Compatibility notes: nothing on this host reads these exit codes except the cron log; `crontab -l` entries unchanged. Two deliberate changes for anything keying off exit codes: a GREEN tick with unusable state now exits 4 (was 0), and `postgres-headroom` without `--gate` exits 4 (was 0) when escalation fails.
- `.github/workflows/orion-static-gates.yml`: new step runs the disk + headroom tests.

## Exit-code contract (all three watchers)

| code | meaning |
|---|---|
| 0 | fine (disk: also "skipped, another run holds the lock") |
| 1 | red / breached / alarm, and the card is delivered or already delivered |
| 2 | could not complete the check (ladder, headroom). Disk: no longer returned |
| 3 | disk only: the watchdog itself crashed |
| 4 | the check ran but a human may not have been told: state unusable, or orion-notify refused/raised. Wins over 1/2; the stdout verdict (RED/GREEN, status lines) is still printed |

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest tests/scripts/test_substrate_ladder_liveness.py tests/test_check_postgres_connection_headroom.py tests/test_disk_threshold_watchdog.py -q
146 passed, 2 skipped
fresh venv, CI deps only (pydantic pydantic-settings PyYAML pytest requests), + test_schema_skew_discovery.py:
164 passed, 2 skipped   (skips = live-Postgres tests needing ORION_TEST_POSTGRES_URI)
python scripts/check_definition_drift.py --gate           rc=0
python scripts/check_metric_lineage.py --gate             rc=0
python scripts/check_scripts_dir_no_stdlib_shadow.py      rc=0
python scripts/check_system_health_producers.py           rc=0
python scripts/check_sentience_instruments.py --static-only rc=0
```

CLI smoke against a real 0o555 dir:

```text
ladder --skip-db --skip-docker --notify --state-file <ro>/x/state.json
  ESCALATION FAILED: dedupe state unusable (PermissionError: [Errno 13] ...). Nothing is red now, ...
  GREEN  rc=4
ladder same, writable state:  GREEN rc=0
disk --threshold-pct 0 --state-file <ro>/d/state.json --notify-base-url http://127.0.0.1:9
  ESCALATION FAILED -- dedupe state unusable (PermissionError ...)
  ESCALATION FAILED -- /tmp: orion-notify did not accept the breached card   rc=4
```

## Evals run

```text
No eval harness for host watchdog scripts; the live card proof below is the end-to-end check.
```

## Docker/build/smoke checks

Live card proof (one card, sent once):

```text
$ .venv/bin/python scripts/check_substrate_ladder_liveness.py --test-escalation
test card accepted by orion-notify at http://localhost:7140 (notification_id=5b7b0dc4-e917-42ff-84d6-448270fca687)
$ curl -s localhost:7140/attention?limit=20   # filtered
attention_id d04ab6c0-2264-4fdb-8aa6-40ed6a9898a2, source_service check_substrate_ladder_liveness,
reason substrate_ladder_liveness_test, severity warning, status pending, require_ack true
message: "TEST: substrate ladder watch escalation check -- safe to dismiss ..."
(sent before the review nit that also put TEST into `reason`; the live card's reason is `substrate_ladder_liveness_test`)
```

Dismiss from Hub Pending Attention, or:

```bash
curl -s -X POST localhost:7140/attention/d04ab6c0-2264-4fdb-8aa6-40ed6a9898a2/ack \
  -H 'Content-Type: application/json' \
  -d '{"attention_id":"d04ab6c0-2264-4fdb-8aa6-40ed6a9898a2","ack_type":"dismissed"}'
```

## Review findings fixed

Review ran in a subagent against `git diff d0782f542~1 d0782f542`. No must-fix. Should-fix items all fixed:

- Finding: a wrong-shaped state file (`{"notified_keys": 5}`, non-numeric `episode_rank`) raised inside the dedupe path before any send, so a red tick exited 4 with **zero** cards, every tick.
  - Fix: `_load_state` in ladder/headroom (and disk's per-path entries) treats wrong shapes as empty and rewrites; a crash anywhere in the dedupe path now routes to the undeduped fallback card.
  - Evidence: `test_malformed_state_still_cards_the_red_and_is_rewritten` (4 shapes), `test_a_crash_inside_the_dedupe_path_still_cards_the_red`, `test_a_malformed_state_file_still_cards_the_alarm`, `test_a_crash_in_the_dedupe_path_still_cards_the_alarm`, `test_wrong_shaped_path_entry_does_not_kill_the_card`.
- Finding: ladder crash message blamed orion-notify when notify was never called.
  - Fix: crash now reports as `dedupe state unusable (unexpected ...)`.
  - Evidence: asserted in `test_a_crash_inside_the_dedupe_path_still_cards_the_red`.
- Finding: any `flock` OSError counted as "another run holds the lock" -> quiet skip forever on ENOLCK/NFS.
  - Fix: only EWOULDBLOCK/EAGAIN is contention in all three; anything else takes the fallback path.
  - Evidence: `test_non_contention_flock_error_cards_instead_of_skipping`, `test_contention_is_a_quiet_skip`, `test_non_contention_flock_error_takes_the_stateless_path`.
- Finding: disk and headroom tests ran in no CI workflow.
  - Fix: added a step to `orion-static-gates.yml` (installs `requests`, which `orion.notify.client` needs).
  - Evidence: all four test files pass in a fresh venv with only the CI deps: 164 passed, 2 skipped (live-Postgres tests).
- Nits fixed: disk exit-1/exit-3 docstrings; `TEST` now also in the card's `reason`; shared class attribute removed from a test fake.

## Restart required

```text
No restart required. Cron runs the scripts from the primary checkout; `git pull` on main there after merge picks this up on the next tick.
```

## Risks / concerns

- Severity: low
- Concern: with a broken state dir and a sustained red, Hub gets one card per tick (6/h ladder, 6/h headroom, 4/h per bad path disk).
- Mitigation: deliberate; every such card says why it repeats and the fix (make the state dir writable) is one `chown`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2392

🤖 Generated with [Claude Code](https://claude.com/claude-code)
