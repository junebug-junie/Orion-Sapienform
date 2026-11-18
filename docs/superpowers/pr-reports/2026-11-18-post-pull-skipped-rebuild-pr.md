## Summary

- A pull whose rebuild never ran no longer loses its services. Each fully successful deploy run now writes the commit it deployed to `~/.orion/post_pull_deployed_head`.
- The next run plans from that recorded commit when it is older than the previous pull (`ORIG_HEAD`). This only widens the diff. It never narrows it.
- A failed or partial run does not move the record. The next pull retries the failed services together with its own changes.
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

Second gap found in the same log: at 05:10:07Z a pull landed while the 05:07Z run was mid-flight. Its own run diffed 02e006803..8893fee9c and logged `NOTHING`. Meanwhile the first run's remaining 13 `up`s were refused because main had moved. Those services were lost the same way. The record (not advanced by a failed run) plus the new lock cover that case.

## Current architecture

`scripts/git_hooks/post-merge` (opt-in marker) runs `scripts/rebuild_services_from_git_diff.sh` synchronously. The script asks `scripts/rebuild_affected_services.py --plan` for the services changed between `ORIG_HEAD` (or MERGE_HEAD, `HEAD@{1}`, `HEAD~1`) and `HEAD`. There was no memory of what was actually deployed, so a pull without a completed run lost its changes. Runs could also overlap.

## Architecture touched

- `scripts/rebuild_services_from_git_diff.sh` only. The planner, the deploy wrapper, the migration applier order, pinned-head rules and sandbox sync are unchanged.

## Files changed

- `scripts/rebuild_services_from_git_diff.sh`: deployed-head record (read before planning, written after a full success), `CATCHUP` log line, `base=deployed:<sha>` in RUN/NOTHING lines, run lock (`LOCK_WAIT`/`LOCK_ACQUIRED`/`LOCK_TIMEOUT`), and the detached sandbox retry no longer inherits the lock fd.
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
9 passed
(same file against the pre-patch script: 5 failed, including test_skipped_pull_services_are_rebuilt_by_the_next_pull)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/scripts/ tests/test_deploy_migration_gate.py tests/test_post_pull_apply_migrations.py -q
(see PR comment / CI)
```

## Evals run

```text
No eval harness covers deploy scripts; the regression tests drive the real script end to end against throwaway git repos.
```

## Docker/build/smoke checks

```text
None run. No real rebuild or deploy (forbidden for this task). The first live proof is the next post-pull run: ~/.orion/post_pull_deployed_head appears after it ends with DONE failed= (empty).
```

## Review findings fixed

- See the PR conversation (code review subagent run before push).

## Restart required

```text
No restart required. The script is read from the primary checkout on the next `git pull`.
```

## Risks / concerns

- Severity: low. Concern: a service that fails on every run keeps the record pinned, so each later pull re-plans everything since the last full success. Mitigation: that is the intended "retry until deployed" behavior. The `CATCHUP` line names the base. Finish the failing service with `--services`, then the next full success moves the record.
- Severity: low. Concern: the lock makes a pull that lands mid-run wait (synchronous hook). Mitigation: the earlier run's remaining `up`s fail fast once main moves, so the wait is normally short. The wait is bounded by `ORION_REBUILD_LOCK_WAIT_S`, and on timeout the run proceeds as before.
- Severity: low. Concern: until the first full success after deploy, there is no record, so behavior is unchanged. Mitigation: by design ("first run behaves as today").
- Issues: Fixes #3704. Opened: none.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
