# The headroom gate's escalation path shipped dead on arrival

## Summary

- PR #2025 gave `check_postgres_connection_headroom.py` a `--notify` path. It
  could never have fired: the script has no `sys.path` setup, so
  `from orion.notify.client import NotifyClient` raised `ModuleNotFoundError`
  whenever it was run as `python scripts/...` — which is exactly how
  `make postgres-headroom-watch`, and therefore cron, invokes it.
- Every alarm would have printed `notify unavailable; alarm not escalated` and
  raised no card. The gate would have run every 10 minutes and escalated nothing.
- Found by running the real cron target against the live database, not by any
  test. The 37-test suite passed straight through it.
- Fixed in the script rather than with `PYTHONPATH=.` in the Makefile, plus a
  subprocess regression test that actually reproduces the invocation.

## Outcome moved

The escalation path added in PR #2025 goes from inert to working. Verified live:
invoked from `/tmp` with `--notify` and a forced alarm against a dead notify
port, the client imported, made a real HTTP attempt, failed as expected, and
recorded `"notified": false` — the retry-until-confirmed rule working end to end
against the real state directory.

## Current architecture

`notify_alarm()` imports `orion.notify.client` lazily. Run as a script,
`sys.path[0]` is `scripts/`, not the repo root, so `orion` is not importable.
`scripts/disk_threshold_watchdog.py` solves this with `PYTHONPATH=.` set by its
Makefile target; that only holds when cwd happens to be the repo root.

## Files changed

- `scripts/check_postgres_connection_headroom.py`: drops `scripts/` from
  `sys.path[0]` (mirroring `disk_threshold_watchdog.py`, so a file in `scripts/`
  cannot shadow a stdlib module — `check_scripts_dir_no_stdlib_shadow.py` is the
  CI gate for that) and inserts the repo root. Done in the script because this
  script's design contract, stated in `connection_params()`, is that an operator
  can run it from anywhere with no setup.
- `tests/test_check_postgres_connection_headroom.py`: subprocess regression test.

## Why the existing tests were blind to this

This is the part worth reading. All 37 tests passed with the bug present, and
they could not have failed:

1. The test module does `sys.path.insert(0, REPO_ROOT)` at import time, so
   `orion` is importable inside pytest no matter what the script does.
2. Every notify test then does
   `monkeypatch.setitem(sys.modules, "orion.notify.client", stub)`, so the
   import being stood in for resolves from `sys.modules` and never touches the
   path at all.

The test environment supplied exactly the thing production lacked. The only test
that can catch this is one that shells out, from a foreign cwd, with `PYTHONPATH`
cleared — which is what was added.

## Tests run

```text
python -m pytest tests/test_check_postgres_connection_headroom.py -q
38 passed, 2 skipped in 0.55s
```

All 10 CI static gates from `.github/workflows/orion-static-gates.yml` pass.

Mutation (removing the repo-root insert, i.e. restoring the shipped bug):

```text
CAUGHT: the sys.path repo-root insert removed (the shipped bug)
    by: test_the_notify_client_is_importable_when_run_as_a_script
```

Note on that result: the first mutation run reported GREEN, and the harness was
wrong, not the test. Its `failing()` helper matched only `^test_\w+`, but pytest
`-q` reports failures as `FAILED tests/x.py::name`, which does not match at line
start. Same class as a previously recorded incident where a mutation harness that
could not run read as all-green — a harness needs its own baseline check as much
as the code under test does.

## Live verification

```text
$ cd /tmp && .venv/bin/python .../check_postgres_connection_headroom.py \
    --gate --notify --min-free-pct 99 --notify-base-url http://localhost:9 ...

FAIL: only 81% of connection slots free (threshold 99%). 56/300 used ...
[NOTIFY] Failed to send attention request to http://localhost:9/... Connection refused
  attention card FAILED to send (will retry next tick)

$ cat /mnt/telemetry/orion-athena/postgres-headroom/_probe.json
{"episode_rank": 1, "last_alarm_at": "...", "notified": false, "reason": "headroom_low"}
```

Before the fix the same command printed
`notify unavailable (No module named 'orion'); alarm not escalated` and wrote
nothing.

## Env/config changes

None. Related host setup, done out of band: the state directory
`/mnt/telemetry/orion-athena/postgres-headroom` did not exist and could not be
created by the service user (`/mnt/telemetry/orion-athena` is `root:root 755`).
Created and chowned to match its `disk-watchdog` sibling.

## Restart required

```text
No restart required. Cron picks up the new code on its next tick.
```

## Risks / concerns

- Severity: low
  Concern: two `sys.path` conventions now exist for host scripts — this one is
  self-contained, `disk_threshold_watchdog.py` depends on `PYTHONPATH=.`.
  Mitigation: the self-contained form is strictly more robust; converging the
  older script is a separate change, noted not silently done.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2029
