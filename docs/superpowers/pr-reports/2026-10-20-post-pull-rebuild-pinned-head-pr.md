## Summary

- A post-pull rebuild run now judges every service bring-up against the commit the run started on, not against an `origin/main` that moved during the run.
- `scripts/rebuild_services_from_git_diff.sh` fetches once at start. If main equals `origin/main` then, it exports that commit as `ORION_DEPLOY_PINNED_HEAD`. If main was already stale, no pin is set and the old refusal stands.
- `scripts/safe_docker_build.sh`, when pinned: main must still equal the pin, and the pin must be in `origin/main`'s history. Behind is accepted (with a stderr note). Ahead or diverged is refused.
- Without the pin, behavior is unchanged. Worktree, branch, dirty-tree and migration gates are untouched and run before the pin check.

## Outcome moved

Before: one PR merging on GitHub mid-run refused every remaining service in about 1 s. Live 2026-10-20, `~/.orion/post_pull_rebuild.log` pid=1322001 head=1609caeef: 9 rebuilt, then 55 `rc=1` from 18:48:29Z, right after PR #2805 merged at 18:48:13Z. It recurred in pids 2024537 and 2244733. After: the run finishes deploying the commit it pulled.

## Current architecture

The wrapper fetched `origin/main` on every `up` and required `HEAD == origin/main`. The rebuild script calls the wrapper once per service, sequentially, over many minutes.

## Architecture touched

Deploy policy only: `scripts/safe_docker_build.sh` section 2.1 and the start of `scripts/rebuild_services_from_git_diff.sh`. No services, bus, or schema changes.

## Files changed

- `scripts/safe_docker_build.sh`: honors `ORION_DEPLOY_PINNED_HEAD` (exact HEAD match + ancestor of origin/main).
- `scripts/rebuild_services_from_git_diff.sh`: sets the pin only when the run starts in sync; skipped for `--dry-run`/`--list-only`.
- `tests/test_deploy_migration_gate.py`: regression for remote-advanced-mid-run, plus refusals for moved-off-pin, ahead, diverged, dirty, worktree; fixture pops the pin from the inherited env.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: every wrapper call gets the start HEAD; a run started on stale main gets no pin.
- `CLAUDE.md` (= `AGENTS.md`) section 8: one sentence on the pin.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: wrapper accepts a pinned main that is behind origin/main.
- Compatibility notes: unpinned callers see no change.

## Env/config changes

- Added keys: none in any `.env_example`. `ORION_DEPLOY_PINNED_HEAD` is a process env var set by the rebuild script, not an operator config key.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest tests/test_deploy_migration_gate.py tests/scripts/test_rebuild_services_from_git_diff.py \
  tests/scripts/test_safe_docker_build_heartbeat.py tests/scripts/test_safe_docker_build_compose_file.py
97 passed
New pinned tests against the pre-fix scripts: 5 failed, 2 passed (dirty/worktree refusals already held).
sh -n / bash -n on both scripts: ok
```

## Evals run

```text
None. This is deploy tooling with no eval harness; the behavior is fully covered by the git-repo-backed tests above.
```

## Docker/build/smoke checks

```text
None run. The change takes effect on the next post-pull rebuild. UNVERIFIED live until a run spans a mid-run merge.
```

## Review findings fixed

- Finding (MUST): the rebuild script pinned HEAD unconditionally, so a run started on an already-stale main would deploy it.
  - Fix: fetch at start; pin only if HEAD == origin/main then.
  - Evidence: `test_run_started_on_stale_main_is_not_pinned`.
- Finding (NIT): wrapper test fixture inherited `ORION_DEPLOY_PINNED_HEAD` from the environment.
  - Fix: `env.pop` in `_wrapper_repo`.
  - Evidence: fixture change; suite green.
- Finding (SHOULD, not taken): a human can set the pin by hand to deploy a behind main.
  - Reason: any marker env var is equally settable by hand, so it adds ceremony, not safety. Setting it by hand is a conscious override, like the other `ORION_ALLOW_*` hatches, and it still cannot deploy ahead/diverged/dirty/worktree code.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: `scripts/apply_post_pull_migrations.py`'s deferred (after-backup) apply also compares HEAD to the local `origin/main` ref, which the wrapper's fetch can move mid-run. Mitigation: not changed here (out of scope); follow-up if a deferred apply is seen refused for this reason.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2818
