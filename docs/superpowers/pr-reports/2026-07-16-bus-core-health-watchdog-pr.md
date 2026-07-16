# PR report — host-level bus-core crash-loop watchdog

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/chore/bus-core-health-watchdog)
Branch: `chore/bus-core-health-watchdog`
Status: **DONE**

## Summary

- `bus-core` (Redis, `services/orion-bus/docker-compose.yml`) periodically
  crash-loops from AOF corruption (a parallel task is adding auto-repair for
  that -- not touched here). The real gap: detecting "bus is stuck in a crash
  loop" today depends on either a human noticing cascading failures across
  many *other* services, or on signals that themselves route through the bus
  or Postgres -- both of which can be down *at the same time* in dev
  (confirmed by Juniper, not hypothetical).
- Added `scripts/bus_core_health_watchdog.py`: a standalone, host-level
  (non-containerized) script that reads `docker inspect`'s
  `.State.Health.Status`/`.RestartCount` for the real
  `orion-orion-athena-bus-core` container only -- zero Redis connection, zero
  Postgres connection, ever. Two independent crash-loop signatures (either
  fires an alert): a consecutive-unhealthy streak (default 3), and a
  restart-count-within-window (default 3 restarts / 10 minutes) that catches
  a loop fast enough it never settles into `unhealthy` between polls.
- Grepped the repo first for an existing desktop-notification mechanism
  (`notify-send`, `osascript`) or house alerting pattern -- none exists.
  The one HTTP alert path that does exist (`orion-notify`'s
  `/attention/request`, used by `orion-mesh-guardian` and the Fuseki
  recover job) is itself infrastructure that could plausibly be down in the
  same incident this watchdog exists to catch, so it was deliberately not
  reused. On alert: a plain, loud, never-auto-deleted marker file at
  `${TELEMETRY_ROOT}/${PROJECT}/bus/ORION_BUS_CRASH_LOOP_ALERT.txt`.
- State persists to `${TELEMETRY_ROOT}/${PROJECT}/bus/watchdog/health_watchdog_state.json`
  -- a sibling of, not inside, the AOF data mount at `.../bus/data` that a
  parallel AOF-repair task may also be touching.
- Matched this repo's dominant existing convention for this exact job shape
  (`scripts/check_concept_relation_digest_liveness.py` +
  `services/orion-rdf-store/README.md`'s "Scheduled maintenance" crontab
  section) rather than the one systemd-timer precedent found (a single
  long-running nightly backup job, not this repeated-fast-poll shape, and
  only present in a gitignored worktree, not a fresh checkout).
- A high-effort code-review subagent pass found 5 real Important-severity
  gaps (state-file/marker-file concurrency race, an uncaught-exception
  exit-code collision, an unguarded `KeyError` on malformed persisted data)
  -- all fixed, with new regression tests locking each fix in. See "Review
  findings fixed" below.

## Outcome moved

Detecting a `bus-core` crash loop no longer requires the bus or Postgres to
be reachable. Previously: nothing. Now: a host cron job that reads container
health directly from Docker, persists its own state to local disk, and
writes an impossible-to-miss marker file the moment a crash-loop signature
is met -- verified end to end against both the real, currently-running
`orion-orion-athena-bus-core` container (healthy path) and a fake `docker`
binary simulating 3 consecutive unhealthy checks (alert path, including
recovery clearing the state but not deleting the marker).

## Current architecture

`services/orion-bus/docker-compose.yml` defines `bus-core` (Redis 7, AOF
persistence, `container_name: orion-${PROJECT}-bus-core`) with a native
Docker healthcheck (`redis-cli ping`, 5s interval, 5 retries -> ~25s to
`unhealthy`), plus `bus-exporter` (Prometheus) and `bus-observer` (a
containerized service that itself depends on the bus being up to observe
it -- exactly the blind spot this watchdog exists outside of). No prior
host-level, bus/Postgres-independent liveness check existed for this
container. The closest analogous pattern in this repo is
`scripts/check_concept_relation_digest_liveness.py`, a standalone
Postgres-backed liveness gate for a different cron job, run the same way
(host cron, not a live service loop).

## Architecture touched

Host-level only. No container, service, schema, or bus contract was
touched. `services/orion-bus/docker-compose.yml`'s own definition and
`bus-core`'s entrypoint were explicitly left alone per the task brief (a
parallel task owns AOF auto-repair there).

## Files changed

- `scripts/bus_core_health_watchdog.py` (new): the watchdog. Pure
  `evaluate()` function (no I/O, takes `now` as a parameter) for the
  threshold/alert logic; `inspect_container()` wraps `docker inspect` via
  `subprocess`; `run()`/`main()` wire persistence, locking, and the CLI.
- `tests/test_bus_core_health_watchdog.py` (new, 40 tests): pure-logic
  tests against fake health-check sequences (streaks, neutral statuses,
  restart-count windows including a container-recreation/RestartCount-reset
  scenario, malformed persisted samples), mocked-subprocess tests for the
  `docker inspect` parsing boundary (including the real lowercase
  `"no such object"` wording observed from this host's docker CLI), and
  end-to-end tests against real temp files for state persistence, atomic
  writes, locking, and `main()`'s exit codes.
- `Makefile`: new `bus-core-health-watchdog` target, mirroring the existing
  `concept-relation-digest` / `check-concept-relation-digest-liveness`
  pattern (`PROJECT` passthrough).
- `services/orion-bus/README.md`: new "Host-level crash-loop watchdog"
  section under Monitoring -- what it detects and why, default paths, the
  first-install permission gotcha on this host, and the crontab install
  block.
- `scripts/README.md`: short discoverability entry alongside this repo's
  other host-level gate/check scripts.

## Schema / bus / API changes

None. This script has no bus, schema, or API surface -- it is intentionally
outside all of those (that independence is the entire point).

## Env/config changes

None added. The script reads `$PROJECT`/`$TELEMETRY_ROOT` (both already
present in `.env_example`/`.env`, no new keys) purely as CLI-flag defaults,
overridable via `--project`/`--telemetry-root`/`--container`/`--state-file`/
`--alert-marker`.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest tests/test_bus_core_health_watchdog.py -q
=> 40 passed

/mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest tests/test_bus_core_health_watchdog.py tests/test_check_concept_relation_digest_liveness.py -q
=> 48 passed

/mnt/scripts/Orion-Sapienform/venv/bin/python -m py_compile scripts/bus_core_health_watchdog.py
=> OK

git diff --check
=> clean
```

Full `tests/` collection was also run for context: 33 pre-existing
collection errors on unrelated service test files (missing per-service
`conftest`/path setup), confirmed identical on `main` before this branch --
not caused by this patch.

## Evals run

None applicable -- this is a deterministic gate script (matches
`AGENTS.md` §11's "Gate tests" category, same as
`check_concept_relation_digest_liveness.py`), not a quality/behavior
measurement that needs an eval harness.

## Docker/build/smoke checks

Live, read-only `docker inspect` smoke against the real running container:

```text
$ python3 scripts/bus_core_health_watchdog.py --state-file <scratch>/state.json --alert-marker <scratch>/ALERT.txt --json
{"container": "orion-orion-athena-bus-core", ..., "crash_loop_detected": false,
 "last_health_status": "healthy", "last_restart_count": 0, "container_status": "running", ...}
exit: 0
```

Confirmed real container name is `orion-orion-athena-bus-core`
(`orion-${PROJECT}-bus-core` with `PROJECT=orion-athena` from `.env`,
matching `services/orion-bus/docker-compose.yml`'s `container_name`), and
the resulting state-file JSON has the exact shape asserted by the test
suite's shape-regression test.

End-to-end crash-loop simulation via a fake `docker` binary (real
subprocess, real file writes, not mocked):

```text
run 1: unhealthy, streak=1 -> exit 0
run 2: unhealthy, streak=2 -> exit 0
run 3: unhealthy, streak=3 -> CRASH LOOP DETECTED, marker written -> exit 1
recovery run: healthy -> exit 0, marker NOT deleted, RESOLVED footer appended
```

No Docker build/compose changes -- nothing to build; this script runs
outside any container.

`scripts/safe_graphify_update.sh` was run per repo convention: it correctly
refused (node count 32529 -> 2522, ~92% drop -- the same known,
root-cause-unidentified `graphify update .` bug from the 2026-07-14
incident, unrelated to this patch) and auto-restored `graph.json`. Stray
generated artifacts from the refused run (`GRAPH_REPORT.md` diff,
`.graphify_labels.json`, `.graphify_root`, `graph.html`) were reverted/removed
so they don't pollute this PR. Per the documented guidance, did not force a
full re-extraction (out of scope, expensive, and the wrapper's own note says
not to just re-run it blindly).

## Review findings fixed

High-effort code-review subagent (`orion-repo-agent`, git-range
`020501bf` vs its parent), full findings below:

- **Finding (Important #1)**: No lock guarded the state-file
  read-evaluate-write cycle in `run()`. Two overlapping cron invocations
  (e.g. one run still waiting on a hung `docker inspect`, up to the 15s
  subprocess timeout, when the next minute's cron fires) could silently
  clobber each other's streak increment -- the exact race shape
  `services/orion-rdf-store/README.md` documents a real incident from (a
  plain existence-check lock, not `flock`, raced its own cron job).
  - **Fix**: added `_StateLock`, a non-blocking `flock` on
    `<state_file>.lock`. A run that can't acquire it raises
    `WatchdogLockedError` and `main()` treats that as a clean skip (exit 0,
    no state mutation) rather than a failure.
  - **Evidence**: `test_state_lock_blocks_concurrent_acquisition`,
    `test_state_lock_is_released_after_context_exit`,
    `test_run_skips_cleanly_when_lock_already_held`,
    `test_main_exits_zero_when_lock_already_held` -- all pass.
- **Finding (Important #2)**: the alert marker (`write_alert_marker`,
  `append_resolved_footer`) used plain `write_text()`/`open(..., "a")`,
  unlike the state file's atomic `mkstemp`+`os.replace`, despite being the
  file a human reads live during an incident.
  - **Fix**: extracted a shared `_atomic_write_text()` helper; both marker
    functions now use it (the resolved-footer append reads existing
    content, concatenates, and atomically replaces rather than appending in
    place).
  - **Evidence**: `test_alert_marker_write_is_atomic_no_leftover_temp_files`,
    `test_resolved_footer_append_is_atomic_and_preserves_original_content`.
- **Finding (Important #3)**: uncaught `OSError`/`PermissionError` from
  `mkdir`/write (e.g. the telemetry directory not yet created/chowned on a
  fresh host -- confirmed live and already documented in
  `services/orion-bus/README.md` as a real first-install gotcha on this
  exact host) would exit 1 by Python's default uncaught-exception behavior,
  indistinguishable from "crash loop detected" (also exit 1) to anything
  consuming only the exit code.
  - **Fix**: `main()` now explicitly catches `OSError` and maps it to exit
    2 (same family as `DockerUnavailableError`), with a message pointing at
    the README's permissions section.
  - **Evidence**: `test_main_exits_two_not_one_on_permission_error_writing_state`.
- **Finding (Important #4)**: `evaluate()`'s
  `min(s["restart_count"] for s in samples)` would raise `KeyError` on a
  persisted sample missing/with a non-int `restart_count` (e.g. a
  hand-edited or partially-migrated state file) -- the one place in the
  module that didn't match the "never raises except for genuine tooling
  failure" contract already established for `inspect_container`/`load_state`.
  - **Fix**: `prune_restart_samples()` now validates `restart_count` is an
    `int`, not just that `at` parses, dropping malformed samples instead of
    letting them reach `min()`.
  - **Evidence**: `test_evaluate_ignores_malformed_restart_samples_without_crashing`.
- **Finding (Important #5)**: no test exercised the RestartCount-reset
  (container recreation, not just restart) scenario, even though the task
  brief specifically flagged it -- the reviewer confirmed by hand-tracing
  that the existing `min()`-over-window logic happens to handle it
  correctly, but nothing locked that in against a future "simplification"
  that would silently break it.
  - **Fix**: added an explicit code comment on why `min()` over the
    window's values (not the earliest sample) is deliberate, plus a
    regression test simulating a real recreation (`RestartCount` climbing
    47->48->49, then resetting to 0, then climbing 1->2->3, correctly
    re-baselining and firing at the right point).
  - **Evidence**: `test_restart_count_reset_from_container_recreation_rebaselines`.
- **Finding (Minor)**: `services/orion-bus/README.md` cited
  `.worktrees/local-mnt-scripts-backup/deploy/systemd/` as the repo's
  systemd-timer precedent -- that path only exists in a gitignored worktree
  on this host, not on a fresh checkout, making the citation fragile.
  - **Fix**: now cites the tracked plan doc
    (`docs/superpowers/plans/2026-05-09-local-mnt-scripts-backup-implementation.md`)
    instead.
  - **Evidence**: confirmed via `find` that the `.worktrees/` path is
    real-but-gitignored on this host, while the plan doc is tracked.
- **Finding (Minor, not fixed)**: `entry = parsed[0]` in
  `inspect_container()` assumes `docker inspect <name>` on a single name
  always returns exactly one array element matching a container. If a
  differently-namespaced Docker object (volume, network) happened to share
  the exact name, this would silently read as `health_status="none"`
  rather than surfacing anything unusual. Reviewer flagged this as
  extremely unlikely given the container name is
  `orion-${PROJECT}-bus-core` -- accepted as-is, not worth the added
  complexity for a near-zero-probability collision.

## Restart required

```text
No restart required.
```

This is a new, standalone host-level script -- nothing to restart. It only
becomes live once the crontab line below is installed by hand (see
"Confusion protocol"-adjacent note below: host crontab is shared
infrastructure and was deliberately not modified automatically, per
`AGENTS.md`).

**Exact install command (report only, not run by this session):**

```bash
crontab -e
# then paste:
* * * * * cd /mnt/scripts/Orion-Sapienform && make bus-core-health-watchdog >> /mnt/scripts/Orion-Sapienform/logs/orion-bus-core-health-watchdog.log 2>&1
```

**One-time prerequisite** (the default state/marker directory is
`root:root` 755 on this host):

```bash
sudo mkdir -p /mnt/telemetry/orion-athena/bus/watchdog
sudo chown "$(whoami)":"$(whoami)" /mnt/telemetry/orion-athena/bus/watchdog
```

## Risks / concerns

- Severity: low
- Concern: the every-minute cron cadence assumes cron itself and the host's
  clock are reliable -- if cron is not running (a different failure mode
  entirely, outside this patch's scope), this watchdog is silent, same as
  every other cron-based gate in this repo (`check_concept_relation_digest_liveness.py`
  included). Not a new gap, an inherited one.
- Severity: low
- Concern: the alert marker is never auto-deleted by design (so an incident
  is never silently lost), which means a human must remember to `rm` it
  after reviewing. If that's forgotten, a *second*, unrelated crash loop
  weeks later would still write into (technically: atomically replace) the
  same marker file, but the `detected_first_at` timestamp inside it would
  be stale/misleading relative to the new incident until a human notices
  the marker already existed. Acceptable tradeoff per the task's explicit
  "impossible to miss, human decides when to clear" design goal, but worth
  flagging as a manual-process dependency.
- Severity: low
- Concern: `entry = parsed[0]` in `inspect_container()` assumes exactly one
  matching Docker object for the given name (see Review findings, last
  Minor item, not fixed) -- near-zero real-world probability given the
  container-naming convention, documented as an accepted gap rather than
  silently ignored.

## PR link

`gh` is authenticated in this environment; PR to be opened from this branch.
