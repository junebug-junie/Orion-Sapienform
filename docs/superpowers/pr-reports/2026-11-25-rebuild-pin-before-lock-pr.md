## Summary

- The post-pull rebuild takes its deploy pin (the commit it started on, in sync with origin/main) before it waits for the rebuild lock, not after.
- If main itself moved during the wait, or there was no pin yet (fetch failed), the run re-judges after the wait.
- The post-pull migration applier honours the same pin: main must equal the pin, and the pin may be behind origin/main, never ahead or diverged (the rule `safe_docker_build.sh` already uses).

## Outcome moved

Live 2026-11-25 00:42Z: a `git pull` in sync with origin/main waited 4 min for the lock; four PRs merged meanwhile. After the wait the run saw a stale main, refused all 13 `up`s and the applier refused the waiting migration (`DONE rebuilt=0 failed=...`, pull exit 0, nothing deployed). With this fix that run would have deployed its 13 services and applied its migration.

## Current architecture

`scripts/rebuild_services_from_git_diff.sh` waited on `~/.orion/post_pull_rebuild.lock`, then fetched and pinned only if HEAD == origin/main. `scripts/apply_post_pull_migrations.py` `checkout_problem` required HEAD == origin/main and ignored the pin. Both readings happen after other runs' `safe_docker_build.sh` calls fetch and move origin/main.

## Architecture touched

Deploy tooling only: the rebuild runner and the post-pull migration applier. No service, schema, bus or env change.

## Files changed

- `scripts/rebuild_services_from_git_diff.sh`: `in_sync_head`; pin before the lock wait; re-judge after it when HEAD moved or no pin.
- `scripts/apply_post_pull_migrations.py`: `checkout_problem(repo, env)` honours `ORION_DEPLOY_PINNED_HEAD`.
- `tests/scripts/test_rebuild_deployed_head_catchup.py`: a merge during the lock wait keeps the pin (and the recorded deployed head); main moving during the wait is re-judged.
- `tests/test_post_pull_apply_migrations.py`: a pinned run applies after origin moved ahead; a pin main left, or one origin never had, is refused.

## Schema / bus / API changes

- Added / Removed / Renamed: none.
- Behavior changed: a deploy run that started in sync no longer refuses when origin/main moves while it waits.
- Compatibility notes: unpinned runs behave exactly as before.

## Env/config changes

- None. `.env_example` updated: no. Local `.env` sync: not needed.

## Tests run

```text
pytest tests/test_post_pull_apply_migrations.py tests/scripts/test_rebuild_deployed_head_catchup.py tests/scripts/test_rebuild_services_from_git_diff.py tests/test_deploy_migration_gate.py -> 264 passed
mutation: the new lock-wait test fails on the old rebuild script (origin/main), passes on this branch
```

## Evals run

```text
None: deploy tooling has no eval harness; the timed lock-wait test replays the incident.
```

## Docker/build/smoke checks

```text
Not run: no image changes. Live check = the next post-pull run that waits for the lock while a PR merges.
```

## Review findings fixed

- Finding (material): the migration applier never read the pin, so after a wait it still refused and the migration gate refused its consumers.
  - Fix: `checkout_problem` honours the pin with the wrapper's rule.
  - Evidence: `test_a_pinned_run_applies_after_origin_moved_ahead`, `test_a_pin_main_has_left_or_origin_never_had_is_refused`.
- Finding: a fetch failure before the wait was never retried after it.
  - Fix: re-judge when there is no pin, as well as when HEAD moved.
- Finding: the main test did not check the recorded deployed head.
  - Fix: asserts the state file records the start head; second test asserts it really waited.

## Restart required

```text
No restart required. The next `git pull` uses the new scripts.
```

## Risks / concerns

- Severity: low. Concern: the new tests use a 0.7 s timer against a 2 s lock holder; a very loaded CI box can only make them fail falsely, never pass falsely.
- Issues: Fixes #3880.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
