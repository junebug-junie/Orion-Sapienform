# Refresh Orion's sandbox after every deploy

## Summary

- Orion's own tools (dig, body atlas) run from Orion's sandbox copy of the repo, not from the deployed image. The sandbox only moved forward when Juniper opened or refreshed a Hub browser tab. After a merge and pull, an open tab kept Orion on old code.
- The post-pull rebuild now moves the sandbox to `origin/main` at the end of every successful run. It uses the existing sync code (rescue dirty work, then reset), run inside the Hub container.
- The sync never touches the sandbox while a turn runs. It waits up to 2 minutes. If a long turn holds the sandbox longer, a detached retry waits up to 90 minutes more, so `git pull` never waits on a turn.
- The sync never fails the rebuild. Each run writes one `SANDBOX_SYNC ok|skipped|failed` line to `~/.orion/post_pull_rebuild.log`. Kill switch: `ORION_POST_PULL_SANDBOX_SYNC=0`.
- Docs that told agents to check a dig change "inside the governor" now say that turns read the sandbox first, and how to check it.

## Outcome moved

Failure mode closed: "deployed, but Orion still runs the old code". Live case on 2026-11-04: the sandbox stayed at `4e2586c82` (#3107) while #3109 and #3112 were deployed. Orion's turn `91ef7dd3` (07:30Z) printed "5 hold" from the old code (deployed code says 4) and wrote a capacity sentence that the deployed code marks STALE.

## Current architecture

- A chat turn runs `claude -p` with `PYTHONPATH` = sandbox first, then `/app` (`orion/fcc/context_budget.py`, `extend_fcc_subprocess_env`). `python -P` drops only the current folder, not `PYTHONPATH`. Verified live in the governor container: with `PYTHONPATH=/mnt/orion-fcc/repo:/app`, `python3 -P -c "import orion.dig"` loads `/mnt/orion-fcc/repo/orion/dig/__init__.py`.
- `orion/fcc/sandbox_sync.py` resets the sandbox to `origin/main`. It first rescues dirty work into a stash or a rescue branch. It holds an exclusive file lock (`orion/fcc/turn_lock.py`); turns hold a shared lock, so a sync never runs during a turn.
- Only Hub's WebSocket connect called it. `GET /api/fcc-sandbox-sync` only reports the last result; it does not trigger a sync.
- `scripts/rebuild_services_from_git_diff.sh` runs after `git pull` on the primary checkout. It did nothing with the sandbox.

## Architecture touched

- Seam chosen: `docker exec <hub container> python3 -m orion.fcc.sandbox_sync`, from a small host script. Why this seam:
  - There is no trigger endpoint. The Hub route only reports state.
  - Running the module on the host fails: 19,038 files in the sandbox are owned by root (turns write them from containers), and the host user cannot rewrite them.
  - The Hub container already runs this exact sync as root, has the sandbox mount and the `~/.fcc/.env` mount (push auth), and its image carries the code from the same deploy.
  - The Hub container is found by its compose labels (`orion-hub` / `hub-app`), not by name, so no `PROJECT` lookup. No Hub container (for example on circe) means `skipped reason=hub_not_running`.
- Mid-turn decision: sync after **every** successful deploy run, including runs that rebuilt nothing. Not only when `orion-harness-governor` was rebuilt. Reasons:
  - Turns import the sandbox's whole `orion/` tree and read the whole repo, not one image's files. The rebuild plan does not say which of those changed.
  - Mid-turn safety does not depend on the governor restart. The turn lock already makes the sync wait for every running turn, in hub and in the governor.
  - The rescue keeps any dirty work either way.
- The sync is skipped when any stack failed (`reason=rebuild_failures`). The operator finishes that deploy with `--services`, and that run's success moves the sandbox.
- The sync runs only when the run was pinned to `origin/main` (`ORION_DEPLOY_PINNED_HEAD`). The helper also re-checks: primary checkout, on `main`, clean, `HEAD == origin/main`.

## Files changed

- `orion/fcc/sandbox_sync.py`: a `main()` command line around the existing `sync_fcc_sandbox`. It retries while a turn holds the lock (bounded by `--wait-s`), prints one verdict line, and exits 0 only when the sandbox is at `origin/main`.
- `scripts/sync_fcc_sandbox_after_deploy.sh` (new): checks the checkout, finds the Hub container, runs the sync inside it, and prints one `ok|skipped|failed` line. It always exits 0. `--log-to` makes the detached retry write its own log line.
- `scripts/rebuild_services_from_git_diff.sh`: `sync_fcc_sandbox` runs at the end of a successful run and of a "nothing to rebuild" run. It logs `SANDBOX_SYNC`, starts the detached retry when a turn holds the lock, and has the kill switch.
- `tests/scripts/test_rebuild_services_from_git_diff.py`, `tests/scripts/test_sync_fcc_sandbox_after_deploy.py`, `orion/fcc/tests/test_sandbox_sync.py`: tests.
- `CLAUDE.md` (section 8), `orion/dig/README.md`, `orion/body_atlas/README.md`, `services/orion-harness-governor/README.md`, `scripts/service_rebuild_paths.yaml`: turns read the sandbox first; how to check it (`git -C /mnt/orion-fcc/repo log -1 --oneline`). The dig README said `-P` makes Orion run `/app/orion`. That is true without `PYTHONPATH`, but false inside a real turn. It is now corrected.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the post-pull rebuild now resets Orion's sandbox to `origin/main`.
- Compatibility notes: if a Hub image is older than this PR, it has no `main()`. The module then prints nothing and the run logs `SANDBOX_SYNC failed reason=no_verdict`. This PR changes `orion/fcc/sandbox_sync.py`, so its own deploy rebuilds Hub first.

## Env/config changes

- Added keys: none in any `.env_example`. The rebuild script reads three shell environment switches, like `ORION_POST_PULL_APPLY_MIGRATIONS`. They are documented in the script header and in CLAUDE.md section 8:
  - `ORION_POST_PULL_SANDBOX_SYNC` (default on; `0` = off)
  - `ORION_SANDBOX_SYNC_WAIT_S` (default 120)
  - `ORION_SANDBOX_SYNC_RETRY_WAIT_S` (default 5400)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -q \
  tests/scripts/test_sync_fcc_sandbox_after_deploy.py tests/scripts/test_rebuild_services_from_git_diff.py \
  orion/fcc/tests/test_sandbox_sync.py tests/scripts/test_check_rebuild_log.py tests/test_post_pull_apply_migrations.py
232 passed

New coverage: sync runs after a successful run; also after a nothing-to-rebuild run; a failing or silent
helper never changes the exit code; kill switch skips; a failed stack skips; a stale-main run and a
dry run never call it; a turn holding the lock starts a detached retry that logs its own line; helper
skips on a linked worktree, a dirty checkout, main ahead of origin, or no Hub; CLI waits out a turn
and exits non-zero.

orion-static-gates.yml, every run step executed locally: 40 steps, 0 failed
```

## Evals run

```text
No eval harness: this is deploy plumbing, not cognition. The live check below is the evidence.
```

## Docker/build/smoke checks

Live check of the sync step only (no rebuild run). The Hub image does not have the new CLI yet, so this branch's `orion/fcc/sandbox_sync.py` was piped into the running Hub container. That runs the same code the helper runs, against the real sandbox.

```text
# before
$ git -C /mnt/orion-fcc/repo status --porcelain
(empty: clean)
$ git -C /mnt/orion-fcc/repo log -1 --oneline
4e2586c82 Merge pull request #3107 from junebug-junie/feat/claims-hold-when-evidence-checks
primary checkout / origin/main: 8e139dc5d

# first try, 2 min wait: two agent turns held the sandbox lock (governor pid 372395, claude -p running 40+ min)
$ docker exec -i orion-athena-hub python3 - --wait-s 120 < orion/fcc/sandbox_sync.py
result=skipped_turn_in_flight_external workspace=/mnt/orion-fcc/repo     (rc=1, tree untouched)

# the long-wait path (what the detached retry does), started 08:14:40Z
$ docker exec -i orion-athena-hub python3 - --wait-s 5400 --poll-s 15 < orion/fcc/sandbox_sync.py
INFO fcc_sandbox_sync_ok workspace=/mnt/orion-fcc/repo rescued=False method=none
result=synced workspace=/mnt/orion-fcc/repo head=8e139dc5d134 commits_advanced=32
rc=0, finished 08:55:59Z

# after
$ git -C /mnt/orion-fcc/repo log -1 --oneline
8e139dc5d Merge pull request #3114 from junebug-junie/fix/pr-field-note-turn-budget
```

The first try is the reason for the detached retry. On a real morning, turns held the sandbox for 41 minutes, much longer than a 2-minute wait. The full helper (`docker exec ... python3 -m orion.fcc.sandbox_sync`) is UNVERIFIED live until this PR's deploy rebuilds Hub. The first post-pull run after merge writes the first real `SANDBOX_SYNC` line.

## Review findings fixed

The review ran in a subagent against `origin/main...HEAD`.

- Finding: the docs said `git pull` never waits on a turn. That was false: the first try waits up to 2 min in the foreground.
  - Fix: CLAUDE.md and the script comment now say `git pull` waits at most about 2 min.
  - Evidence: the diff of the reworded text.
- Finding: several pulls during one long turn would each start a new 90-minute retry.
  - Fix: a retry first takes a host file lock (`<rebuild log>.sandbox-retry.lock`). A second retry logs `skipped reason=retry_already_pending`.
  - Evidence: `test_second_pending_retry_is_skipped`.
- Finding (nit): a timed-out docker client looked like an "old image" result, and the comment overstated what the timeout stops.
  - Fix: new `failed reason=timeout rc=124` verdict, and the comment now says the timeout bounds only how long the caller waits.
  - Evidence: `test_docker_client_timeout_is_named`.
- Finding (nit): the `nohup` fallback was not really detached.
  - Fix: removed. Without `setsid`, the log line says `retry=unavailable_no_setsid`.
  - Evidence: the diff.
- Finding (nit): a turn in flight was logged as `failed`.
  - Fix: it now logs `skipped reason=turn_in_flight`.
  - Evidence: `test_turn_in_flight_reads_as_deferred_not_failed`.
- Finding (nit): the detached-retry test polled for only 5 s.
  - Fix: it now polls for up to 20 s.
  - Evidence: the diff.
- Not changed: the helper's `main == origin/main` check uses the ref the rebuild already fetched. The sync then resets to the newest `origin/main`, the same as a Hub tab refresh. The docs say `origin/main`, not "the deployed commit".

## Restart required

```text
No restart required. The deploy of this PR rebuilds orion-hub (sandbox_sync.py changed), which gives the Hub image the new CLI. The next post-pull run uses it.
```

## Risks / concerns

- Severity: low. Concern: a sync between two turns moves Orion off a working branch onto `main`. Mitigation: this is the same behavior as a Hub tab refresh. Branch refs survive, and dirty work is rescued to a stash or rescue branch.
- Severity: low. Concern: the Hub status route (`/api/fcc-sandbox-sync`) shows only syncs made by the Hub process, not these. Mitigation: the rebuild log has the `SANDBOX_SYNC` line, and `git -C /mnt/orion-fcc/repo log -1` is the direct check.
- Severity: low. Concern: when turns hold the lock for more than 92 minutes, the sandbox stays stale until the next Hub connect or deploy. Mitigation: the retry logs `SANDBOX_SYNC retry failed ... skipped_turn_in_flight_external`, so the miss is visible.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3117

🤖 Generated with [Claude Code](https://claude.com/claude-code)
