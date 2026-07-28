# PR report — disk threshold watchdog follow-up: recover stranded crontab fix + expand to 8 mounts

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/chore/disk-threshold-watchdog-followup)
Branch: `chore/disk-threshold-watchdog-followup`
Status: **DONE**

## Summary

- PR #1425 (the original disk threshold watchdog) merged into `main`
  *before* two follow-up commits pushed to that branch landed --
  `docs(bootstrap): flag host crontab jobs must be reinstalled on a new
  machine` did make it into `main` (a keyword-grep on the commit message
  falsely suggested otherwise; verified via `git merge-base
  --is-ancestor`), but `fix(docs): disk-threshold-watchdog crontab line
  needs venv PATH prefix` did not. GitHub does not retroactively pull in
  commits pushed to a branch after that branch's PR has already merged.
  Recovered by cherry-picking the missing commit onto a fresh branch off
  current `main` (the old branch itself is stale relative to `main` by
  ~90 unrelated files from other merged work, so opening a PR directly
  from it would have shown a misleading diff).
- Expanded `scripts/disk_threshold_watchdog.py`'s `DEFAULT_PATHS` from 3
  mounts to 8: added `/` (root), `/mnt/postgres`, `/mnt/graphdb`,
  `/mnt/storage-warm`, and `/mnt/storage-lukewarm` per Juniper's request.
  All confirmed via `df -h` to be genuinely distinct physical filesystems
  on this host (`/dev/mapper/ubuntu--vg-ubuntu--lv`, `/dev/sdg1`,
  `/dev/sdg2`, `/dev/sdc`, `/dev/sdh` respectively), not bind mounts of an
  already-monitored path.
- Purely additive to the data, not the logic -- `evaluate_path()`,
  `run()`, `_publish_attention()`, and the flock-guarded state machine
  from PR #1425 (already code-reviewed, including the critical
  retry-on-failed-notify fix) are unchanged.
- Live crontab on this host was also directly edited (with explicit
  operator go-ahead) to add the `PATH=`/`cd` prefixes the earlier fix
  documented -- confirmed via `crontab -l` before and after, only the
  disk-threshold-watchdog line touched, the other 3 jobs (fuseki
  recover/compact, concept-relation-digest) left untouched.

## Outcome moved

The disk threshold watchdog now covers every distinct physical mount on
this host, not just the original 3. The previously-documented crontab fix
(required for the watchdog to actually run under cron rather than crash
on `ModuleNotFoundError: No module named 'pydantic'`) is now actually
reachable from `main`, not stranded on a closed PR's branch.

## Current architecture

`scripts/disk_threshold_watchdog.py` (PR #1425, merged) already
implements the full threshold/debounce/retry/notify state machine.
Nothing in that logic touches path count or path shape -- `DEFAULT_PATHS`
is a flat tuple consumed only via `--paths` CLI parsing (comma-split) and
iterated as opaque dict keys in `run()`'s per-path loop. Verified (by
code review, see below) that no code anywhere assumes a `/mnt/...` shape
or does prefix-matching between paths, so adding `/` and more `/mnt/*`
mounts required zero logic changes.

## Architecture touched

None beyond the one script's default config and its tests/docs. No
schema, bus, or service changes.

## Files changed

- `scripts/disk_threshold_watchdog.py`: `DEFAULT_PATHS` grown from 3 to 8
  entries; module docstring's mount list and `df -h` device mapping
  updated to match.
- `tests/test_disk_threshold_watchdog.py`: renamed
  `test_default_paths_include_docker_scripts_telemetry` ->
  `test_default_paths_include_all_eight_host_mounts`, asserts the full
  8-tuple in order.
- `scripts/README.md`: "Disk Threshold Watchdog" section's prose and
  expected-output example updated to all 8 paths with a fresh live
  snapshot; also carries the recovered `PATH=` crontab fix (was already
  correct content, just needed to actually land in `main`).
- `docs/superpowers/pr-reports/2026-07-28-disk-threshold-watchdog-pr.md`:
  carries the recovered `PATH=` fix's post-push correction note (same
  reason as above).

## Schema / bus / API changes

None.

## Env/config changes

None. `DEFAULT_PATHS` is a Python constant, not an env-driven value
(`--paths`/`$DISK_WATCHDOG_PATHS` already existed as an override
mechanism from PR #1425 and needed no changes).

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

None applicable -- same deterministic gate-style script as PR #1425, no
eval harness needed.

## Docker/build/smoke checks

No Docker involved. Live smoke against all 8 real mounts on this host:

```text
$ PYTHONPATH=. python3 scripts/disk_threshold_watchdog.py --threshold-pct 90 --json
/               status=ok used=20.1%
/mnt/docker           status=ok used=81.1%
/mnt/scripts          status=ok used=6.8%
/mnt/telemetry        status=ok used=18.6%
/mnt/postgres         status=ok used=14.8%
/mnt/graphdb          status=ok used=0.0%
/mnt/storage-warm     status=ok used=34.1%
/mnt/storage-lukewarm status=ok used=7.4%
exit: 0
```

Cross-checked against `df -h` for the same 8 mounts at review time -- all
within normal drift, no inversions or copy-paste errors between lines
(`/mnt/graphdb`'s 0.0% vs. df's rounded 1% is expected:
`shutil.disk_usage()`'s `used/total` differs slightly from df's
`used/(used+avail)` reserved-blocks formula, most visible on a
near-empty filesystem).

Also directly edited and verified the live host crontab (explicit
operator request, "put it in the cron for me" for the earlier PATH= fix):

```text
$ crontab -l | tail -3
# alerts for disk fullness on /mnt/[docker,scripts,telemetry] -- needs the venv
# PATH prefix because disk_threshold_watchdog.py imports orion.notify.client
# (pydantic/requests), unlike bus_core_health_watchdog.py which is stdlib-only
*/15 * * * * cd /mnt/scripts/Orion-Sapienform && PATH=/mnt/scripts/Orion-Sapienform/venv/bin:$PATH make disk-threshold-watchdog >> /mnt/scripts/Orion-Sapienform/logs/orion-disk-threshold-watchdog.log 2>&1
```

Confirmed via a fully cron-simulated environment (`env -i`, minimal
`PATH`, no inherited cwd) both broken (bare `make`, no `cd`/`PATH=`) and
fixed, before installing the fixed line live.

## Review findings fixed

Code review (`orion-repo-agent`, medium effort, scoped to the additive
mount-expansion diff since the underlying state-machine logic already got
a full high-effort review in PR #1425):

- **Finding (SHOULD FIX)**: the mount-expansion diff existed only as
  uncommitted working-tree changes at review time, not a commit -- a
  `git worktree remove` or accidental `git checkout --` could have
  silently lost it with nothing to recover.
  - **Fix**: committed (`261bb51af`) immediately after the review
    returned.
  - **Evidence**: `git log --oneline -1` / `git status --short` now
    clean.
- **Finding (informational, confirmed clean)**: verified no code path
  anywhere in the script special-cases or prefix-matches on path shape --
  `/` (root) and `/mnt/storage-warm`/`/mnt/storage-lukewarm` (prefix
  substrings of each other) are all handled as opaque exact-match dict
  keys throughout `evaluate_path()`, `run()`, `_publish_attention()`, and
  the log-line f-string. No double-slash or malformed output for `/`.
- **Finding (informational, confirmed clean)**: diff scope is genuinely
  narrow -- confirmed no changes to any state-machine/locking/notify
  logic, only the data (`DEFAULT_PATHS`) and matching docs/tests.

## Restart required

```text
No restart required.
```

Standalone host-level script, same as PR #1425. The live crontab line was
already updated as part of this same session (see Docker/build/smoke
checks above) -- the expanded 8-mount coverage takes effect on the next
`*/15 * * * *` tick once this branch's code is what the crontab's
`cd /mnt/scripts/Orion-Sapienform` resolves to (i.e. once merged and the
primary checkout is on the merged commit).

## Risks / concerns

- Severity: low
- Concern: `/mnt/docker` remains the mount closest to the default 90%
  threshold on this host (81-86% observed across this session) --
  expected to be the first real Pending Attention card once this ships,
  not a bug.
- Severity: low
- Concern: this is the second time in one day a commit got stranded by a
  PR merging before a later push landed. No process gap identified beyond
  "check `git merge-base --is-ancestor <sha> origin/main` before assuming
  a pushed commit made it into a merged PR" -- not proposing new tooling
  for a two-occurrence pattern, just noting it.

## PR link

<link>
