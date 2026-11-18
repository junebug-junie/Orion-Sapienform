## Summary

- A pull whose rebuild never ran no longer loses its services. Each completed rebuild run in the primary checkout now writes its head and the services that failed to `~/.orion/post_pull_deployed_head`.
- The next run plans from that recorded head when it is older than the previous pull (`ORIG_HEAD`), which means a run never finished. This only widens the diff. It never narrows it.
- The next run also retries the recorded failed services, and only those. Services that already deployed are not rebuilt again. A `--services` run clears the services it deployed.
- Rebuild runs now take a lock, so a pull that lands mid-run waits for the earlier run instead of racing it.
- With no record file (first run), behavior is the same as before.

## Outcome moved

The failure mode in #3704 is closed: on 2026-11-18 the 02:28Z pull (df45f4f71 -> 96dac92aa) ran no rebuild. The next pull rebuilt only orion-hub, and orion-prediction and orion-mind stayed on old containers until a manual rebuild at ~05:10Z. With this patch the 03:51Z run would have planned from df45f4f71 and rebuilt the union.

### Root cause of the skipped 02:28Z rebuild

The `git pull` was interrupted 3 s after its fast-forward. The post-merge hook was killed before the rebuild script wrote its first log line.

Evidence:
- Session transcript `7347b6db…/subagents/agent-a28d4e15daeada2e1.jsonl`: an agent ran `git pull --ff-only 2>&1 | tail -15` in the primary checkout at 02:28:25.663Z. At 02:28:35.366Z the tool call returned "User rejected tool use" / "[Request interrupted by user for tool use]". At 02:28:50Z the coordinator relayed "Juniper says: ... Don't run git pull on the primary checkout".
- Reflog: `HEAD@{2026-11-18 02:28:32}: pull --ff-only: Fast-forward`. The ref moved 3 s before the interrupt.
- A healthy hook run takes 4-13 s from the fast-forward to its `RUN` line, e.g. 01:43:07 -> 01:43:17, 03:51:30 -> 03:51:34, 05:07:16 -> 05:07:29. The run first does `check_rebuild_log.py --mark`, `git fetch origin main`, and the planner. The interrupt landed inside that window.
- Nothing is logged before `RUN`, and the `ABORT` trap is armed only after `RUN`. That is why `~/.orion/post_pull_rebuild.log` has no line at all for 96dac92aa.
- Ruled out: `--ff-only` fast-forwards do fire post-merge. The 01:36:31Z `pull --ff-only` logged a RUN at 01:36:44Z. The marker `.orion/post-merge-rebuild-enabled` exists. `orion-git-shim` passes `pull` straight through to real git. `core.hooksPath` is unset.

Second gap found in the same log: at 05:10:07Z a pull landed while the 05:07Z run was mid-flight. Its own run diffed 02e006803..8893fee9c and logged `NOTHING`. Meanwhile the first run's remaining 13 `up`s were refused because main had moved. Those services were lost the same way. The failed-services list plus the new lock cover that case.

## Current architecture

`scripts/git_hooks/post-merge` (opt-in marker) runs `scripts/rebuild_services_from_git_diff.sh` synchronously. The script asks `scripts/rebuild_affected_services.py --plan` for the services changed between `ORIG_HEAD` (or MERGE_HEAD, `HEAD@{1}`, `HEAD~1`) and `HEAD`. There was no memory of what was actually deployed, so a pull without a completed run lost its changes. Runs could also overlap.

## Architecture touched

- `scripts/rebuild_services_from_git_diff.sh` only. The planner, the deploy wrapper, the migration applier order, pinned-head rules and sandbox sync are unchanged.

## Files changed

- `scripts/rebuild_services_from_git_diff.sh`: deploy state record (head + `pending=` failed services; read before planning, written after every completed run in the primary checkout), `CATCHUP` and `PENDING_RETRY` log lines, `base=deployed:<sha>` in RUN/NOTHING lines, run lock (`LOCK_WAIT`/`LOCK_ACQUIRED`/`LOCK_TIMEOUT`/`LOCK_UNAVAILABLE`), `check_rebuild_log --mark` skipped after a lock timeout, and the detached sandbox retry no longer inherits the lock fd.
- `tests/scripts/test_rebuild_deployed_head_catchup.py`: new. It uses real git repos with a bare origin, a planner fake that answers from a real `git diff`, and a fake deploy wrapper. No docker is touched.
- `CLAUDE.md` section 8: one sentence on the record and the lock.
- `docs/superpowers/pr-reports/2026-11-18-post-pull-skipped-rebuild-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the post-pull rebuild may plan from an older base (the last deployed head) and logs `CATCHUP`. A second concurrent run waits on a lock.
- Compatibility notes: `scripts/check_rebuild_log.py` parses `RUN base=\S*`, and `base=deployed:<sha>` matches it.

## Env/config changes

- Added keys: none in any `.env_example`. Optional shell overrides: `ORION_REBUILD_STATE` (record path), `ORION_REBUILD_LOCK` (lock path), `ORION_REBUILD_LOCK_WAIT_S` (default 7200).
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/scripts/test_rebuild_deployed_head_catchup.py -q
13 passed
(first version of the file against the pre-patch script: 5 failed, including test_skipped_pull_services_are_rebuilt_by_the_next_pull)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/scripts/test_rebuild_deployed_head_catchup.py tests/scripts/test_rebuild_services_from_git_diff.py tests/scripts/test_check_rebuild_log.py -q
79 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/scripts/ tests/test_deploy_migration_gate.py tests/test_post_pull_apply_migrations.py -q   (first version)
927 passed, 2 failed, 1 skipped -- the 2 failures are tests/scripts/test_eval_vision_organ_replay.py (pydantic schema error), which pass alone and do not touch this code (order-dependent, pre-existing)
```

## Evals run

```text
No eval harness covers deploy scripts; the regression tests drive the real script end to end against throwaway git repos.
```

## Docker/build/smoke checks

```text
None run. No real rebuild or deploy (forbidden for this task). The first live proof is the next post-pull run: ~/.orion/post_pull_deployed_head appears after its DONE or NOTHING line. UNVERIFIED live until then.
```

## Review findings fixed

- Finding (material): one stack that kept failing froze the record, so every later pull rebuilt everything since the last full success.
  - Fix: the record now holds the head plus a `pending=` list of failed services. Every completed run moves the head. Only the failed services are retried. A `--services` run clears the ones it deployed.
  - Evidence: `test_failed_service_is_retried_without_replanning_the_rest`, `test_services_run_clears_what_it_deployed_from_pending`.
- Finding: a linked-worktree run could write the shared record and hide a missed pull.
  - Fix: only a run whose git dir is the common git dir records.
  - Evidence: `test_linked_worktree_run_never_records`.
- Finding: after a lock timeout, `check_rebuild_log --mark` could mark the still-live run DIED.
  - Fix: skip `--mark` after a timeout.
- Finding: an unopenable lock file failed open without a word.
  - Fix: a `LOCK_UNAVAILABLE` log line.
- Finding: test gaps. The lock-acquired path, the NOTHING path record, and the migration step's widened base were untested.
  - Fix: added `test_waiting_run_pins_the_head_it_finds_after_the_wait` and `test_nothing_to_rebuild_still_records_the_head`. The catch-up test now asserts `--migrations --base <deployed>`.
- Finding (nit): fd 9 is inherited by every child.
  - Fix: a comment at the lock says any detached child must close it. No such child exists today besides the sandbox retry, which closes it.
- Not changed: unpinned runs still never record (see Risks).

## Restart required

```text
No restart required. The script is read from the primary checkout on the next `git pull`.
```

## Risks / concerns

- Severity: low. Concern: a service that fails on every run (for example a consumer of a migration the applier refuses) is retried on every pull. Mitigation: only that service is retried, not everything since. The deploy gate still refuses it until its migration lands. `PENDING_RETRY` names it in the log.
- Severity: low. Concern: an unpinned run (for example `git fetch` failed) never records, so the next pull may re-plan its range. Mitigation: that costs only redundant rebuilds.
- Severity: low. Concern: the lock makes a pull that lands mid-run wait (synchronous hook). Mitigation: the earlier run's remaining `up`s fail fast once main moves, so the wait is normally short. The wait is bounded by `ORION_REBUILD_LOCK_WAIT_S`, and on timeout the run proceeds as before.
- Severity: low. Concern: until the first full success after deploy, there is no record, so behavior is unchanged. Mitigation: by design ("first run behaves as today").
- Issues: Fixes #3704. Opened: none.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
