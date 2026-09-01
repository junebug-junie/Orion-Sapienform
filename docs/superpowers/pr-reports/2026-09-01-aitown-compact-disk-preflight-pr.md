# AI Town compaction: disk preflight, signal-safe recovery hint, and a resume path

## Summary

- `compact_convex_data.sh` gains a **step 0 disk preflight** that refuses to start
  when the job-dir filesystem cannot hold step 3's full database copy, or when
  `$HOME` cannot hold step 5b's npm cache — **summing the two when they are the
  same filesystem**, which is the live topology on circe.
- Any failure past step 4 — including `SIGINT`/`SIGTERM`/`SIGHUP` — now prints an
  explicit "do NOT re-run this script" hint naming the resume path, and can no
  longer exit `0`.
- New `resume_compact_convex_data.sh` replays steps 5b–7 against an existing job
  dir, guarded against importing an empty, stale, or wrong-target export.
- New `AITOWN_COMPACT_SKIP_RAW_BACKUP` so the gate cannot block the only
  operation that shrinks an oversized database.
- 20 new tests that **execute** both scripts under stubbed `docker`/`npx`/`curl`/
  `df`/`stat` rather than grepping them.

## Outcome moved

A compaction that runs out of disk now fails **before** step 1 instead of after
step 4. Before this patch the failure mode was: live database renamed aside,
backend restarted on an empty one, `npm error nospc`, and a container reporting
`healthy` while serving nothing — with no recovery hint and no resume path, in a
state where the obvious next action (re-run the script) destroys the data.

## Current architecture

`services/orion-ai-town/scripts/compact_convex_data.sh` reclaims Convex revision
bloat: export live data (1) → capture env vars (1b) → stop backend (2) → copy
`db.sqlite3` to the job dir (3) → rename it inside the volume (4) → start backend
on a fresh empty DB (5) → redeploy functions (5b) → restore env (5c) → reimport
(6) → heartbeat the world (7). It is `set -euo pipefail` with no resume path, and
step 1 always exports whatever is currently live.

## Architecture touched

`services/orion-ai-town/` only. No bus channel, schema, or API surface — §6 does
not apply.

## Files changed

- `services/orion-ai-town/scripts/compact_convex_data.sh`: step 0 preflight;
  same-filesystem summing; `RESUMABLE`/`COMPLETED` flags with a signal-safe EXIT
  trap; `AITOWN_COMPACT_JOB_DIR_BASE`, `AITOWN_COMPACT_SKIP_RAW_BACKUP`,
  `AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE`; step 5's recovery line defers to the
  trap rather than contradicting it.
- `services/orion-ai-town/scripts/resume_compact_convex_data.sh`: new.
- `services/orion-ai-town/tests/test_compact_convex_data_disk_preflight.py`: new.
- `services/orion-ai-town/tests/test_resume_compact_convex_data_script.py`: new.
- `services/orion-ai-town/README.md`: env table, recovery procedure, the
  "healthy is not evidence" check, incident record.
- `services/orion-ai-town/.env_example`: corrected — these knobs are **not** read
  from `.env` and never were; listing them there was config that does nothing.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the compactor can now refuse to start on insufficient disk,
  and exits non-zero on signals past step 4 where `SIGINT` previously exited `0`.
- Compatibility: default behaviour is unchanged for a run that fits.

## Env/config changes

- Added keys: none in `.env`. Three new **environment** overrides
  (`AITOWN_COMPACT_JOB_DIR_BASE`, `AITOWN_COMPACT_SKIP_RAW_BACKUP`,
  `AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE`) plus two for the resume script
  (`AITOWN_RESUME_CONFIRM`, `AITOWN_RESUME_MIN_FREE_BYTES`).
- `.env_example` updated: yes — to state these are environment-only and remove
  the pre-existing implication that `.env` is read.
- local `.env` synced: run; no key changes required (see above). Three
  pre-existing `Diverged` entries (`AITOWN_UPSTREAM_REF`, `URL_BASE`,
  `INSTANCE_SECRET`) are host-local and untouched by this patch.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-aitown-compact-disk-preflight
python -m pytest services/orion-ai-town/tests/ -q
119 passed, 1 skipped in 19.01s      (was 91 passed, 1 skipped on main)
```

Mutation checks against the real files (each reverted after):

```text
COMBINED_NEED = BACKUP_NEED + HOME_NEED  ->  BACKUP_NEED
  FAILED test_shared_filesystem_requirements_are_summed          (1 failed, 14 passed)

trap on_exit EXIT INT TERM HUP           ->  trap on_exit EXIT
  FAILED test_signal_past_step_4_still_warns_and_does_not_exit_zero[SIGINT]
                                                                 (1 failed, 14 passed)

resume: healthy-deployment refusal       ->  if false
  FAILED test_refuses_to_run_against_a_healthy_deployment        (1 failed, 17 passed)

compact: if (( JOB_AVAIL < BACKUP_NEED )) -> if false && ...
  FAILED test_preflight_aborts_when_job_dir_cannot_hold_the_backup
resume: (( DOCS >= MIN_EXPORT_DOCS ))     -> true
  FAILED test_refuses_an_export_of_an_already_emptied_database
  FAILED test_refuses_a_corrupt_export
```

## Evals run

`services/orion-ai-town/` has no `evals/` directory. Not added here — the
behaviour under change is a deterministic operational gate, which is the tests'
lane, not an eval's. Flagged as a pre-existing gap.

## Docker/build/smoke checks

No image or compose change. The scripts were exercised end-to-end under stubs by
the test suite; the resume script was additionally run for real on circe against
job dir `/tmp/aitown-compact-20260831-130002` (see below).

```text
CI static gates, all 10 from .github/workflows/orion-static-gates.yml:
  check_metric_lineage.py --gate            PASS
  check_definition_drift.py --gate          PASS
  check_inner_state_registry.py             PASS
  check_scripts_dir_no_stdlib_shadow.py     PASS
  check_service_hostname_refs.py            PASS
  check_compose_no_relative_mounts.py       PASS
  check_journal_dispatch_registry.py        PASS
  check_daily_schedule_collisions.py        PASS
  check_system_health_producers.py          PASS
  check_control_surface_store_parity.py     PASS
```

## Live incident this closes

2026-08-31 13:00 UTC, circe. The compaction wrote a 10,994,872,320-byte
`db.sqlite3` backup into `/tmp`, filling `/` to 100% (90G, 0 bytes free), then
died at step 5b:

```text
[13:03:50] step 5/7: starting backend fresh and waiting for health
[13:03:53] backend healthy
[13:03:53] step 5b/7: redeploying Convex functions
npm error nospc ENOSPC: no space left on device
```

`orion-ai-town-backend-1` then reported `Up 33 hours (healthy)` while serving a
131,072-byte empty database with no deployed functions.

Recovered 2026-09-01 with the resume script in this PR: **117,679 documents**
restored (the export held 117,697 including the 18-row `_tables` system table,
which is not imported as data), world heartbeated to `running`, engine confirmed
ticking by `db.sqlite3` growth (+262 KB / 10s) rather than by the healthcheck.
The compaction's own goal was met in the process: **10.2 GB → 314 MB**.

## Review findings fixed

Ran the code-review skill in a subagent. Three must-fix findings, all reproduced
before fixing:

- Finding: the resume tests ran the **repo** script, so `ROOT` resolved to the
  real `services/orion-ai-town` and `UPSTREAM` to its real `upstream/`. On any
  host where that exists — the main checkout and every deploy host; it is absent
  in the worktree only because `upstream/` is gitignored — a test clearing the
  doc-count and env.backup guards would reach `convex env set --from-file` with a
  fake env.backup and `convex import --replace-all` with a synthetic export,
  against whatever Convex deployment was on :3210.
  - Fix: every test now runs a copy staged in `tmp_path` with a fake `upstream/`
    and a stub PATH, plus a meta-test that fails if a future test references the
    repo script anywhere but `_stage()`'s `copy2`.
  - Evidence: `services/orion-ai-town/upstream/node_modules/convex/bin/main.js`
    confirmed present (755) in the main checkout; the reviewer's run cleared
    every guard and stopped only at the refused TCP connection to :3210.

- Finding: `$?` inside an `EXIT` trap is the last completed command's status, not
  the signal, so the "do not re-run" warning was suppressed on
  `SIGINT`/`SIGTERM`/`SIGHUP` — Ctrl-C on a hung step 5b, a dropped SSH session,
  a cron timeout. `SIGINT` additionally exited `0`.
  - Fix: `trap on_exit EXIT INT TERM HUP`, gate on a `COMPLETED` flag instead of
    the exit status, force a non-zero status on signal death, and re-`exit "$rc"`.
  - Evidence: measured directly — `TRAP rc=0` on all three signals, and
    `script_exit=0` for `SIGINT`. Now covered by a parametrised test over all
    three; reverting to `trap on_exit EXIT` makes the `SIGINT` case fail (the
    script does not even terminate — the health loop continues).

- Finding: **the patch did not close the incident it was written for.** The
  job-dir and `$HOME` requirements were checked independently against what is one
  filesystem, never summed. Each passes while the sum does not fit, step 3
  consumes the shared pool, and step 5b hits `ENOSPC` with the preflight green.
  - Fix: resolve `stat -c%d` for both, and when they match, check
    `BACKUP_NEED + HOME_NEED` against the single figure.
  - Evidence: verified 2026-09-01 that `/tmp`, `$HOME` and `/` are all device
    64512 on **circe**, the deploy host — the incident's exact topology. Covered
    by `test_shared_filesystem_requirements_are_summed`, which sets free space to
    `BACKUP_NEED + 1GiB` (enough for either demand alone, not both).

Should-fix findings also addressed: the preflight had no door for an oversized
database (`AITOWN_COMPACT_SKIP_RAW_BACKUP`, which now genuinely skips step 3);
the `$HOME` check failed **open** while the job-dir check failed closed, on the
check guarding the step that actually broke (now fails closed, with
`AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE`); the resume script never checked that
the town was actually broken before `--replace-all`, could import a stale export,
hardcoded `/` for its free-space check with no override, and blamed a missing
`python3` on the export; step 5 printed a recovery line that contradicted the
trap's; and `.env_example` documented knobs the script cannot read.

Two review nits deliberately not taken: `test_preflight_runs_before_the_export`
is kept despite being subsumed by a behavioural assertion, because it names the
ordering invariant explicitly; and the trap's terminal-ordering interleave with
`tee` is cosmetic — the message reaches `progress.log` and captured output
intact, which is what recovery depends on.

## Restart required

```text
No restart required.
```

Both files are host scripts invoked on demand or by cron; nothing running needs
to be recreated. Note the host crontab entry is **not** in circe's `circe` user
crontab — README's "see `crontab -l`" is stale, pre-existing drift worth chasing
separately, since it determines which user's `$HOME` the new check measures.

## Risks / concerns

- Severity: low. Concern: the preflight can refuse a run that would previously
  have started. Mitigation: two documented overrides, both covered by positive
  controls; a refused run leaves the database untouched, which is the point.
- Severity: low. Concern: `AITOWN_COMPACT_SKIP_RAW_BACKUP=1` drops one of three
  recovery artifacts. Mitigation: `export.zip` and step 4's in-volume rename both
  remain; the flag is opt-in and logged loudly in `progress.log`.
- Severity: medium, pre-existing, not fixed here. Concern: the Convex
  healthcheck reports `healthy` for a backend with no functions and no data —
  that is what hid a 33-hour outage. This PR documents the real check
  (`world:defaultWorldStatus` plus `db.sqlite3` growth) but does not change the
  healthcheck itself. Worth its own patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2020
