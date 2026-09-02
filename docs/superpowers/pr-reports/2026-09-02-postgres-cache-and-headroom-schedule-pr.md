# Postgres cache sizing, and a gate that had never run

## Summary

- `shared_buffers` and `effective_cache_size` were still at the postgres:15
  defaults (128MB / 4GB) on a 41GB database. Now 4GB / 12GB, sized against the
  container's own `mem_limit: 16g` rather than the host's 503GB.
- The connection-headroom gate from PR #2010 **had never run**. The Makefile
  carried its intended cron line as a *comment*; nothing installed it.
- Scheduling it alone would not have fixed that — the script had no way to reach
  a human, only an exit code. Added `--notify`, which raises one Hub Pending
  Attention card per alarm episode.
- Separate `postgres-headroom-watch` target so an operator running the check by
  hand never raises a card.
- 10 new tests (37 total), 7/7 mutations caught against the real file.

## Outcome moved

A slow crawl toward `max_connections` now has a watcher. It previously had none:
PR #2010 raised the ceiling to 300 and shipped a gate to warn on approach, and
between then and 2026-09-02 that gate was never executed a single time. The
condition it watches had already produced 217 `FATAL: sorry, too many clients`
refusals over five days.

Separately, the database's cache is no longer 128MB. Everything outside those
128MB was being served from kernel page cache *inside the 16g cgroup*, where it
competed with backend memory and was first to be evicted.

## Current architecture

`services/orion-sql-db/docker-compose.yml` passes tuning as `-c` flags on the
`postgres` command. `max_connections`, autovacuum and maintenance settings were
declared there; the two cache settings were not, so they silently took the
image defaults. `scripts/check_postgres_connection_headroom.py` read headroom
and returned exit 0/1/2, with no alerting path of any kind.

## Architecture touched

- Postgres postmaster configuration (restart required).
- A cron-run host gate gains the `orion.notify` seam already used by
  `scripts/disk_threshold_watchdog.py` and `scripts/bus_core_health_watchdog.py`.

## Files changed

- `services/orion-sql-db/docker-compose.yml`: adds `shared_buffers=4GB` and
  `effective_cache_size=12GB`, with the sizing rationale and the reason
  `work_mem` was deliberately left alone.
- `scripts/check_postgres_connection_headroom.py`: adds `--notify`,
  `--notify-base-url`, `--notify-api-token`, `--state-file`; flock-guarded
  debounce state; `notify_alarm()` / `clear_alarm()`.
- `Makefile`: adds `postgres-headroom-watch`; removes the stale never-installed
  cron comment from `postgres-headroom`.
- `tests/test_check_postgres_connection_headroom.py`: 6 tests for the escalation
  path.

## Sizing rationale (the part worth checking)

Sized against `mem_limit: 16g`, **not** host RAM. The host has 503GB; this
container cannot use it, and sizing to the host is the standard way to earn a
cgroup OOM. 4GB is the conventional 25%-of-available applied to the real
ceiling. Accounting: 4GB shared + ~0.8GB backends (300 x ~2.6MB measured anon
RSS) + 1GB `shm_size` leaves ~10GB of the 16g for page cache and work memory.

`effective_cache_size` allocates nothing — it is a planner hint. The 4GB default
made the planner systematically overprice index scans.

`work_mem` was **not** raised. It is per-sort, not per-backend; at
`max_connections=300` it is the one knob on this list where the cgroup cap
genuinely bites.

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: none. `--notify-base-url` / `--notify-api-token` read the existing
  `NOTIFY_BASE_URL` / `NOTIFY_API_TOKEN`, same as `disk_threshold_watchdog.py`.
- `.env_example` updated: not applicable, no new keys.
- Postgres flags are compose-literal, not env-driven, matching the existing
  settings in that command block.

## Tests run

```text
python -m pytest tests/test_check_postgres_connection_headroom.py -q
37 passed, 2 skipped in 0.18s      (was 27 passed, 2 skipped)
```

All 10 CI static gates from `.github/workflows/orion-static-gates.yml` pass.

Mutation test against the real file, 7/7 caught. The three marked REGRESSION
reproduce defects code review found in the first cut of this patch:

```text
CAUGHT: notified=True on ATTEMPT not confirmation
CAUGHT: debounce removed -- card every tick
CAUGHT: REGRESSION: debounce keyed on reason again (the flapping bug)
CAUGHT: recovery never clears the episode
CAUGHT: --notify becomes always-on
CAUGHT: REGRESSION: reserve-hazard clause deleted from the card
CAUGHT: REGRESSION: alarm re-fused to --gate (ungated tick wipes the episode)
```

The first mutation is the one that matters: `notified=ok` -> `notified=True`
records a card as delivered when the send actually failed, which is worse than
having no debounce at all — an alarm that first fires while orion-notify is down
would be debounced into permanent silence.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-db config
  command:
    - max_connections=300
    - shared_buffers=4GB
    - effective_cache_size=12GB
```

Boot-tested on an isolated container with the exact flag set (not the production
one), since a bad `shared_buffers` fails at postmaster startup:

```text
docker run --memory=16g --shm-size=1gb postgres:15 postgres -c shared_buffers=4GB ...
READY after 3s
  effective_cache_size = 1572864 8kB   (12GB)
  max_connections      = 300
  shared_buffers       = 524288 8kB    (4GB)
  startup errors: none
  probe mem: 126.6MiB / 16GiB
```

126MB at rest confirms `shared_buffers` is mapped lazily — this displaces page
cache as it fills rather than spiking at restart.

## Review findings fixed

Review ran against commit `a5a3ec42f` and returned DONE_WITH_CONCERNS with three
`[must]` items. All are fixed. It also drove two live defects rather than
inferring them, which is why they were found at all.

- Finding: **`postgres-headroom-watch` hardcoded `.venv/bin/python`, which does
  not exist in a linked worktree.** This Makefile already contains
  `$(METRIC_PYTHON)` built for exactly this, with a header explaining that a
  bare `.venv/bin/python` "would fail exactly where this repo does most of its
  work." I propagated the broken form from PR #2010 into the *cron-facing*
  target, where a silent exit 127 in a log file is the failure mode that matters.
  - Fix: both targets now use `$(METRIC_PYTHON)`.
  - Evidence: before, from the worktree: `make: .venv/bin/python: No such file or
    directory ... Error 127`. After: runs, `62/300 used (238 free, 79%)`.

- Finding: **the documented cron line had no `cd` and no `-C`.** Every one of the
  six real entries in this crontab uses one or the other; cron starts in `$HOME`,
  where there is no Makefile. Reproduced by review as `make: *** No rule to make
  target 'postgres-headroom-watch'`.
  - Fix: the recipe now uses `make -C /mnt/scripts/Orion-Sapienform`, and says why.
  - Evidence: `make -C` verified working from an arbitrary cwd.

- Finding: **the Makefile comment said "Installed now:" and nothing was
  installed.** That reproduces the exact failure this patch exists to fix -- a
  schedule living in a comment nobody executed.
  - Fix: the comment now says NOT INSTALLED BY THIS COMMIT, in those words, and
    that the line is "a recipe, not a record."
  - Evidence: `crontab -l` has 41 lines and zero matches for `headroom`.

- Finding: **flapping fired a card every tick.** `saturated` and `headroom_low`
  are two readings of one incident -- at the wall, whether this script's own
  connection wins a slot is close to a coin flip -- and the debounce was keyed on
  `reason`. Review drove 8 alternating ticks and got 8 cards, indefinitely,
  during precisely the incident this exists for.
  - Fix: the episode is keyed on SEVERITY RANK. The escalation warning ->
    critical still fires once; the flap back down is silent.
  - Evidence: `test_a_flapping_incident_does_not_card_every_tick` (8 ticks -> 1
    card) and `test_escalation_to_critical_still_fires_once`. A regression
    mutation restoring the `reason` key is caught by both.

- Finding: **`--notify` without `--gate` wiped a live alarm.** The alarm
  condition was fused to `args.gate`, so an ungated tick on a still-alarming
  reading fell through to `clear_alarm()`, letting the next gated tick raise a
  second card for the same episode.
  - Fix: `alarm` is evaluated independently; `--gate` decides the exit code and
    nothing else.
  - Evidence: `test_notify_without_gate_does_not_wipe_a_live_alarm`; the
    re-fusing mutation is caught.

- Finding: **`assert "superuser" in call["message"]` was vacuous.**
  `headroom.summary()` always emits `superuser clients={n}`, so it passed with
  the entire reserve-hazard clause deleted.
  - Fix: asserts the clause (`"will not hold a door open"`).
  - Evidence: deleting the clause now fails the test; before, `33 passed`.

- Finding: **nothing tested the debounce key at all**, so its correctness was
  unpinned even though the implementation was right.
  - Fix: covered by the two flapping/escalation tests above.

- Finding: **a notify-path exception exited 1, which is `EXIT_ALARM`.** An
  unwritable telemetry root surfaced as an unhandled traceback whose exit status
  is indistinguishable from a full database.
  - Fix: `notify_alarm`/`clear_alarm` catch and report; the alarm is still
    reported truthfully.
  - Evidence: `test_escalation_failure_is_not_reported_as_an_alarm`.

- Finding: **`~2.6MB` per backend was ~1.8x optimistic** (live cgroup:
  `anon=333MB` / 69 processes = ~4.8MB). The budget still closes; the number did
  not.
  - Fix: comment now carries the re-measured figure and the recomputed budget
    (~6.4GB of 16g, not ~5.8GB).

- Finding: **"roughly memory-neutral" was an accounting claim doing risk work.**
  `memswap_limit == mem_limit` means zero swap, and under cgroup v2 shmem is
  reclaimable only by swapping, so the 3.9GB becomes *pinned*, not re-labelled.
  - Fix: comment says so explicitly, with the pinned/reclaimable split.

- Finding: **the pre-existing `max_connections` note contradicted the new
  block**, still asserting 128MB and 40GB in the present tense.
  - Fix: amended in place, with a dated note saying what changed and why the old
    sentence is gone rather than left standing.

- Finding: **`mem_limit` is `${POSTGRES_MEM_LIMIT:-16g}`, not a constant.**
  Lowering it without lowering `shared_buffers` puts 4GB pinned inside a smaller
  cap, and nothing validates the relationship.
  - Fix: documented, with a threshold (below ~8g, drop `shared_buffers` too).

Review also confirmed, against the live server rather than from memory, the one
thing that could have made this fail to boot: `shared_memory_type=mmap`, so
`shared_buffers` is an anonymous mapping and is **not** bounded by
`shm_size: 1gb` (that bounds only `dynamic_shared_memory_type=posix`, used by
parallel workers). Corroborated by the cgroup's `shmem=161MB` matching today's
128MB `shared_buffers` plus slop.

Not fixed, and accepted: the "217 refusals over five days" figure traces to a
2026-08-31 server-log reading that is not captured anywhere in the tree. It is
load-bearing for motivation, not for behaviour. Left as-is rather than deleted,
but it is unfalsifiable as written.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-postgres-cache-and-headroom-schedule
scripts/safe_docker_build.sh orion-sql-db up -d
```

`shared_buffers` and `effective_cache_size` are postmaster-level; a `restart`
will not pick them up, the container must be recreated. Expect a brief
connection blip for every service.

## Still not scheduled

**This commit does not install the cron entry.** `crontab -l` still has zero
matches for `headroom`. The `crontab` write was blocked by the session's
permission classifier, and the target does not exist on `main` until this merges
anyway. The recipe is in the Makefile and the ready-to-install line is in the
PR discussion. Until it is installed and `crontab -l` shows it, the gate still
has no watcher and this half of the commit is `UNVERIFIED`.

## Risks / concerns

- Severity: medium
  Concern: recreating the Postgres container drops every open connection.
  Mitigation: services reconnect via their pools; do it at a quiet moment.
- Severity: low
  Concern: this is a third copy of the notify-debounce pattern (after
  `disk_threshold_watchdog.py` and `bus_core_health_watchdog.py`), plus three
  service-side `health_monitor.py` variants. Extraction into a shared helper is
  the right follow-up; it was deliberately not done here because refactoring a
  live watchdog is not in scope for scheduling a check.
  Mitigation: noted as follow-up, not silently duplicated.
- Severity: low
  Concern: `EXIT_CANNOT_CHECK` does not raise a card, so a fully-down Postgres
  is not escalated by this script (a *full* one is — that is `EXIT_ALARM`).
  Mitigation: deliberate, to keep the state surface single-subject. Worth
  revisiting; `orion-notify` does not share this database, so it would work.

## Not fixed here, and why

**Services run as the `postgres` superuser.** Named as a hazard in PR #2010 and
still true. Measured this time rather than assumed:

- 24 service `.env` files carry a `POSTGRES_URI`.
- All 214 tables in `public` are owned by `postgres`.
- 17 files perform runtime DDL (`CREATE TABLE/INDEX IF NOT EXISTS`).

So a non-superuser role cannot simply be granted DML — it would need ownership
of all 214 tables (`REASSIGN OWNED`), or membership in an owning group, before
the boot-time DDL in those 17 files would succeed. That is a staged ownership
migration plus a fleet-wide restart, not a config change. `pgcrypto` is already
installed, so `CREATE EXTENSION IF NOT EXISTS` is a privilege-free no-op and does
**not** block it.

Recommend a dedicated proposal-mode change with a rollback plan.

## PR link

<pending>
