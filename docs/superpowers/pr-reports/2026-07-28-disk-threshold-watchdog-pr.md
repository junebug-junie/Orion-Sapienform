# PR report — host-level disk threshold watchdog -> Hub Pending Attention

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1425
Branch: `chore/disk-threshold-watchdog`
Status: **DONE**

## Summary

- `/mnt/docker`, `/mnt/scripts/`, and `/mnt/telemetry` (three distinct
  physical mounts on this host, confirmed via `df -h`) had no threshold
  monitoring that surfaced a breach anywhere an operator would actually
  see it.
- Added `scripts/disk_threshold_watchdog.py`: a standalone, host-level
  (non-containerized), cron-run script that checks `shutil.disk_usage()`
  per path and, the first time a path crosses `--threshold-pct` (default
  90), fires `orion-notify`'s `POST /attention/request` -- the same
  mechanism `orion-mesh-guardian` already uses -- which lands as a Hub
  Pending Attention card.
- Mirrors the pattern already proven and code-reviewed in
  `scripts/bus_core_health_watchdog.py`: pure `evaluate_path()` for the
  threshold/debounce logic, `flock`-guarded atomic state writes to survive
  overlapping cron invocations, distinct exit codes (0/1/2/3) so a
  monitoring wrapper can tell "breach detected" apart from "tooling
  failure" apart from "watchdog bug."
- Live-verified end to end against the real running `orion-notify`: real
  `attention_request` calls landed with `status: "pending"`, confirmed via
  `GET /attention`, then ack'd/dismissed to clean up smoke-test noise.
- A high-effort code-review subagent (`orion-repo-agent`) found one
  **critical** bug before merge: a failed/unconfirmed notify call was
  being persisted as "already notified," so a breach that first occurred
  while `orion-notify` was down would never retry -- no Pending Attention
  card would ever land for the entire duration of the outage, exactly the
  failure window this feature exists to survive. Fixed and live-reproduced
  (see below).

## Outcome moved

Previously: nothing watched host disk usage on these three mounts. Now: a
cron-run script that checks all three independently every 15 minutes,
publishes a real Hub Pending Attention card on first breach, debounces
repeat notification while already confirmed, and -- critically -- retries
on every subsequent tick if the prior notify attempt never actually
succeeded, rather than silently going quiet.

## Current architecture

No prior disk-threshold monitoring existed. Two lookalikes were
investigated and ruled out during design:

- `skills.storage.disk_health_snapshot.v1`
  (`orion/cognition/verbs/skills.storage.disk_health_snapshot.v1.yaml`) --
  a reactive cognition skill, goal-driven, invoked by Orion's own autonomy
  loop on demand, not a threshold watchdog.
- `resource_pressure`/`cpu_pressure`/`gpu_pressure` (Scarcity Economy) --
  Orion's internal cognitive-economy signal feeding drives, not an
  operator infra alert. Wrong layer entirely.

`orion.notify.client.NotifyClient.attention_request()` ->
`orion-notify`'s `POST /attention/request` -> Hub Pending Attention is the
real, live, proven mechanism, already used by `orion-mesh-guardian`
(`services/orion-mesh-guardian/app/attention.py`) and the Fuseki recover
job.

## Architecture touched

Host-level only. No new service, no schema/bus contract change, no Docker
container. `NotifyClient`/`ChatAttentionRequest`
(`orion/schemas/notify.py`, `orion/notify/client.py`) reused as-is.
`PendingAttentionCardV1` (`orion/schemas/attention_salience.py`) was
considered and ruled out -- it's hard-locked to `source:
Literal["cognitive_loop"]` with `extra="forbid"`, a different subsystem
entirely.

## Files changed

- `scripts/disk_threshold_watchdog.py` (new): the watchdog.
  `measure_path()` (never raises), `evaluate_path()` (pure
  threshold/debounce/retry state machine), `_publish_attention()` (returns
  whether orion-notify actually confirmed), `run()` (flock-guarded
  read-evaluate-write cycle), `main()` (CLI + exit codes).
- `tests/test_disk_threshold_watchdog.py` (new, 35 tests): pure-logic
  tests for every state transition (breach/recovery/re-breach,
  error/ok/error, breach->error), concurrency-lock tests, `load_state`
  corruption-resilience tests, and -- added during the review-fix pass --
  explicit retry-on-failed-notify regression tests mirroring the real
  `NotifyClient` contract (`NotificationAccepted(ok=False, ...)`, not an
  exception).
- `Makefile`: new `disk-threshold-watchdog` target (sets `PYTHONPATH=.`,
  since this script imports `orion.notify.client`, unlike
  `bus_core_health_watchdog.py` which deliberately imports nothing from
  `orion.*`).
- `scripts/README.md`: new "Disk Threshold Watchdog" discoverability
  section -- description, usage, expected output, one-time `chown`
  prerequisite (same `root:root` 755 gotcha confirmed live for the
  bus-core watchdog's telemetry directory), and the crontab install line.

## Schema / bus / API changes

None. Reuses `ChatAttentionRequest`/`NotifyClient.attention_request()` as
is -- no new bus channel, no schema registry change, no
`PendingAttentionCardV1` widening.

## Env/config changes

None added. Reads `$PROJECT`/`$TELEMETRY_ROOT`/`$NOTIFY_BASE_URL`/
`$NOTIFY_API_TOKEN` (existing keys) purely as CLI-flag defaults,
overridable via `--project`/`--telemetry-root`/`--notify-base-url`/
`--notify-api-token`/`--paths`/`--threshold-pct`/`--state-file`.

## Tests run

```text
source venv/bin/activate && python -m py_compile scripts/disk_threshold_watchdog.py
=> OK

PYTHONPATH=. python -m pytest tests/test_disk_threshold_watchdog.py -q
=> 35 passed

PYTHONPATH=. python -m pytest tests/test_disk_threshold_watchdog.py tests/test_bus_core_health_watchdog.py -q
=> 84 passed

git diff --check
=> clean
```

## Evals run

None applicable -- deterministic gate-style script (matches AGENTS.md
section 11's "Gate tests" category, same as `bus_core_health_watchdog.py`),
not a quality/behavior measurement needing an eval harness.

## Docker/build/smoke checks

No Docker involved -- host-level script, nothing to build.

Live smoke against the real host and real running `orion-notify` (port
7140):

```text
$ PYTHONPATH=. python3 scripts/disk_threshold_watchdog.py --threshold-pct 90 --json
disk_threshold_watchdog: /mnt/docker status=ok used=79.6%   (/dev/sda)
disk_threshold_watchdog: /mnt/scripts status=ok used=6.0%   (/dev/sde1)
disk_threshold_watchdog: /mnt/telemetry status=ok used=18.6% (/dev/sdf1)
exit: 0
```

Forced-breach end-to-end (threshold-pct=0): 3 real `attention_request`
calls landed in `orion-notify`'s `/attention` list with `status:
"pending"`, confirmed via `GET /attention`, then ack'd/dismissed via
`POST /attention/{id}/ack` to clean up.

Retry-on-failure regression, reproduced live against an unreachable
notify port (`--notify-base-url http://127.0.0.1:1`) both BEFORE and
AFTER the fix:

```text
BEFORE FIX:
  tick 1 (notify unreachable): last_status=breached persisted, notify attempted
  tick 2 (notify still unreachable): NO notify attempt at all -- silently swallowed

AFTER FIX:
  tick 1 (notify unreachable): last_status=breached, notified=false persisted
  tick 2 (notify still unreachable): notify RETRIED for all 3 paths, notified=false
  tick 3 (notify now reachable): notify RETRIED, notified=true persisted
  tick 4 (still breached): no retry -- already confirmed notified
```

## Review findings fixed

Code review (`orion-repo-agent`, high effort) against
`/mnt/scripts/Orion-Sapienform-disk-threshold-watchdog`:

- **Finding (CRITICAL)**: `_publish_attention()` discarded
  `NotifyClient.attention_request()`'s return value entirely, and `run()`
  persisted the debounce transition to "breached"/"notified" before the
  notify call even happened. The real `NotifyClient` never raises on
  network failure -- it catches everything internally and returns
  `NotificationAccepted(ok=False, ...)`. So a breach occurring while
  `orion-notify` is down would get `last_status="breached"` committed on
  tick 1, and tick 2 would see `status == prev_status` and never retry --
  permanently silent, zero Pending Attention card, for the entire outage.
  Live-reproduced by the reviewer against a mocked unreachable client.
  - **Fix**: `_publish_attention()` now returns `bool(result.ok)` (False
    on both exception and `ok=False`). `evaluate_path()` now tracks a
    separate `notified` field per path, only ever set `True` by `run()`
    after a confirmed-successful call; `should_notify` fires on a status
    transition OR whenever the current bad status has `notified != True`,
    so a failed/unconfirmed attempt retries every subsequent tick until it
    succeeds.
  - **Evidence**:
    `test_evaluate_path_repeated_breach_retries_if_previous_notify_never_confirmed`,
    `test_evaluate_path_repeated_error_retries_if_previous_notify_never_confirmed`,
    `test_run_retries_notify_next_tick_when_notify_returns_ok_false`,
    `test_run_stops_retrying_once_a_notify_attempt_succeeds` -- plus the
    live before/after reproduction above against a real unreachable port.
- **Finding (MEDIUM)**: the one existing test claiming to cover "notify
  unreachable" (`test_run_notify_failure_does_not_crash_watchdog`) used
  `side_effect = RuntimeError(...)`, which the real client never actually
  raises -- it gave false confidence for exactly the broken scenario.
  - **Fix**: `_fake_notify()` now defaults to mirroring the real contract
    (`MagicMock(ok=True/False)`, not an exception); the exception-path
    test is kept (still valid coverage for `_publish_attention`'s
    `except` branch) alongside new `ok=False`-based tests that match the
    realistic failure mode.
  - **Evidence**: see tests listed above.
- **Finding (LOW, fixed)**: the Makefile target's comment pointed to a
  `scripts/README.md` cron-install section that didn't exist yet.
  - **Fix**: added the actual crontab line plus the one-time `chown`
    prerequisite (same gotcha class already documented for the bus-core
    watchdog) to `scripts/README.md`.
  - **Evidence**: confirmed live -- `/mnt/telemetry/orion-athena/` is
    `root:root` 755 on this host, same as documented for
    `bus_core_health_watchdog.py`.
- **Finding (MEDIUM, accepted not fixed)**: `_StateLock` wraps the full
  per-path loop including the `orion-notify` HTTP call (up to 10s timeout
  per path), unlike `bus_core_health_watchdog.py`'s lock which only ever
  guards fast local I/O (its one blocking call, `docker inspect`, happens
  outside the lock, and its alert path is a local file write, not a
  network call). During a slow/hanging (not just refusing) `orion-notify`,
  an overlapping cron tick that can't acquire the lock skips cleanly
  rather than measuring disk at all for that tick.
  - **Not fixed**: a correct two-phase lock (measure+decide under lock,
    notify outside, re-lock to commit outcome) is real complexity for a
    lower-severity, cron-cadence-dependent (default every 15 min, not
    every 1 min like bus-core-watchdog) risk. The reviewer explicitly
    assessed this as shippable with a documented follow-up rather than
    blocking.
  - **Mitigation**: documented here and in the script's own docstring; a
    future patch could split measurement from notification if the
    every-15-min cadence proves too coarse in practice.
- **Finding (MINOR, informational, accepted)**: `run()` does a single
  `_atomic_write_json` after the full per-path loop rather than per-item
  like the sibling script. If `measure_path`/`evaluate_path`/
  `_publish_attention` ever raised unexpectedly partway through a
  multi-path tick (not currently reachable given `measure_path`'s
  documented "never raises" contract), in-memory progress for paths
  already processed that tick would be lost. Reviewer noted this fails in
  the safe direction (duplicate notify next tick, not silently dropped) --
  not urgent.

## Restart required

```text
No restart required.
```

This is a new, standalone host-level script -- nothing to restart. It
only becomes live once the crontab line in `scripts/README.md`'s "Disk
Threshold Watchdog" section is installed by hand (host crontab is shared
infrastructure, deliberately not modified automatically).

**Exact install commands (report only, not run by this session):**

```bash
sudo mkdir -p /mnt/telemetry/orion-athena/disk-watchdog
sudo chown "$(whoami)":"$(whoami)" /mnt/telemetry/orion-athena/disk-watchdog

crontab -e
# then paste:
*/15 * * * * cd /mnt/scripts/Orion-Sapienform && PATH=/mnt/scripts/Orion-Sapienform/venv/bin:$PATH make disk-threshold-watchdog >> /mnt/scripts/Orion-Sapienform/logs/orion-disk-threshold-watchdog.log 2>&1
```

**Post-push correction (2026-07-28):** the crontab line originally
documented here omitted the `PATH=.../venv/bin:$PATH` prefix. When
Juniper actually installed it, `make disk-threshold-watchdog` ran under
cron's own minimal `PATH` (not an activated shell), which resolved
`python3` to the system interpreter and crashed on
`ModuleNotFoundError: No module named 'pydantic'` (this script imports
`orion.notify.client`, unlike `bus_core_health_watchdog.py`, whose
crontab line has no such requirement since it imports nothing outside the
stdlib). Live-confirmed with `env -i PATH="/usr/bin:/bin" ... python3 -c
"import pydantic"` failing, and the venv's `python3 -c "import pydantic"`
succeeding. Fixed in this file and in `scripts/README.md`; the line above
is the corrected version.

## Risks / concerns

- Severity: low
- Concern: `/mnt/docker` is already at ~80% used on this host today. At
  the default `--threshold-pct 90`, this won't fire immediately, but it's
  close enough that a real card is plausible soon after this ships --
  expected behavior, not a bug, flagging so it isn't mistaken for a
  smoke-test artifact when it lands.
- Severity: medium (documented, not blocking)
- Concern: `_StateLock` holds across the notify HTTP call (see Review
  Findings above) -- a slow/hanging (not refusing) `orion-notify` could
  cause a skipped measurement tick under overlapping cron runs. Low
  real-world likelihood at the default 15-minute cadence.
- Severity: low
- Concern: every-15-minute cron cadence assumes cron itself is running --
  same inherited gap as every other cron-based gate in this repo
  (`bus_core_health_watchdog.py`, `check_concept_relation_digest_liveness.py`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1425
