# Hub-refresh → FCC sandbox sync: unwedge the guard that never let go

## Summary

- The connect-time sandbox sync was firing correctly on every browser refresh. The
  *guard* was the bug: any dirty worktree returned `skipped_dirty_worktree` and
  returned, with no path back to a synced state. Orion left work in the sandbox on
  2026-08-13; every refresh after that skipped, leaving the checkout 294 commits
  behind `main` with nothing able to clear it.
- Replaced refuse-on-dirty with **rescue-then-reset**: dirty state is stashed under a
  findable `orion-sandbox-autorescue/<branch>/<utc-stamp>` label, then the sync
  proceeds. Only a *failed* rescue declines the reset.
- Unpushed commits no longer block the sync either. `checkout main` does not move
  `refs/heads/<branch>`, so those commits stay reachable by branch name in the same
  clone; the sync logs which branch it stepped off and proceeds.
- Added a status surface (`GET /api/fcc-sandbox-sync`). The wedge was invisible for
  two days because the only evidence was a log line inside the hub container.
- Fixed a regression this patch introduced and caught live: the hub test suite drives
  the real `websocket_endpoint()`, so `pytest` reached the sync hook with the real
  `/mnt/orion-fcc/repo` workspace and stashed Orion's actual checkout. Guarded on
  `PYTEST_CURRENT_TEST`.

## Outcome moved

A Hub browser refresh now actually refreshes Orion's FCC sandbox to `origin/main`
instead of silently no-op'ing. Orion stops reasoning and editing against a checkout
hundreds of commits stale, and a sandbox that *does* decline to sync now says so on an
HTTP surface rather than only in container logs.

## Current architecture

`/mnt/orion-fcc/repo` (nvme0n1p1) is a standalone clone — not a worktree — shared
read-write by `orion-athena-hub` and `orion-athena-harness-governor`
(`HUB_AGENT_CLAUDE_WORKSPACE` / `HARNESS_FCC_WORKSPACE`, both
`/mnt/orion-fcc/repo`). Only Hub runs the sync, on WebSocket connect, fire-and-forget,
serialized by `_fcc_sandbox_sync_lock` and skipped while an FCC turn is in flight.
Governor has no sync code and does not need one — same path, one syncer.

Live state before this patch (read via the *container's* git, see Risks):

```
branch: test/orion-live-push-proof-aacd41ef @ df6b3021c   (294 behind origin/main)
status: ~200 modified, ~150 untracked
last 3 WS connects: all -> skipped_dirty_worktree
```

## Architecture touched

- `orion/fcc/sandbox_sync.py` — the sync's decision logic and a new recorded verdict.
- `services/orion-hub/scripts/websocket_handler.py` — connect-time hook only. The
  connection lifecycle, `connection_ready` frame, and the history rehydration added in
  `44ff583b2` are untouched; diff is confined to `_sync_fcc_sandbox_background`.
- `services/orion-hub/scripts/api_routes.py` — one read-only GET route.

## Files changed

- `orion/fcc/sandbox_sync.py`: rescue-then-reset; unpushed branches no longer block;
  every attempt records an inspectable verdict via `last_sync_state()`.
- `orion/fcc/tests/test_sandbox_sync.py`: the two tests that encoded the wedge as
  correct behaviour now assert the rescue; added untracked-file, rescue-failure, and
  verdict-recording coverage.
- `services/orion-hub/scripts/websocket_handler.py`: `PYTEST_CURRENT_TEST` guard; both
  turn-in-flight skips now record their verdict.
- `services/orion-hub/scripts/api_routes.py`: `GET /api/fcc-sandbox-sync`.
- `services/orion-hub/tests/test_fcc_sandbox_sync_test_process_guard.py`: regression
  test for the live incident below.

## Schema / bus / API changes

- Added: `GET /api/fcc-sandbox-sync` → `{workspace, result, at, branch, head,
  behind_main, rescued_from}`. Read-only, no auth change, no bus involvement.
- Behaviour changed: `sync_fcc_sandbox` return values. `skipped_dirty_worktree` and
  `skipped_unpushed_branch` are gone; `synced_after_rescue`,
  `skipped_dirty_rescue_failed`, and `skipped_test_process` are new. Values are
  log/telemetry strings only — nothing parses them across a service boundary.
- Compatibility: none needed. No schema, channel, or payload touched.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable — no env surface changed.
- local `.env` synced: not applicable, nothing to sync.
- skipped keys requiring operator action: none.

## Tests run

```text
$ pytest orion/fcc/tests -q
71 passed in 5.20s

$ pytest services/orion-hub/tests/test_fcc_sandbox_sync_test_process_guard.py -q
3 passed in 4.43s

$ pytest services/orion-hub/tests -q --tb=no -p no:randomly     # branch
59 failed, 1189 passed, 1 skipped, 2 errors in 191.62s

$ pytest services/orion-hub/tests -q --tb=no -p no:randomly     # main baseline (f835c737c)
59 failed, 1186 passed, 1 skipped, 2 errors in 203.05s
```

The hub suite's 59 failures are pre-existing on `main` and unrelated to this patch —
it is order-dependent and depends on live services. Branch and main differ by exactly
one test in each direction, both in
`test_substrate_mutation_manual_route_routing.py`, and both pass in isolation on both
sides: a flaky live-surface pair, not a regression.

Two traps worth recording, because both produced misleading results first:

- A fresh worktree has no gitignored `.env`, so `Settings()` fails validation and
  tests fail for reasons that look like code regressions. The first branch-vs-main
  comparison was confounded this way (35 vs 59) until `.env` was copied in. Verified
  still gitignored and unstaged afterwards.
- pytest's raw output had to be redirected to a file; the filtered tool view
  truncated the failure list to 3 of 37 lines, which would have hidden the comparison
  entirely.

Gates:

```text
$ git diff --check                                        -> clean
$ python scripts/check_fcc_context_parity.py              -> ok motor_ctx=65536 profile_max_ctx=131072
$ python scripts/check_service_env_compose_parity.py orion-hub -> N/A (env_file: declared)
```

`scripts/check_env_template_parity.py`, `check_schema_registry.py`, and
`check_bus_channels.py` are named in CLAUDE.md §17 but do not exist in the repo, and
there is no `agent-check` make target. Not introduced by this patch; flagged as a
contract/reality gap.

## Evals run

No eval harness exists for `orion/fcc` or `services/orion-hub`'s sandbox seam. Nothing
here is a quality/behaviour surface an eval would measure — the sync is a deterministic
git state machine, fully covered by the gate tests above. No eval added.

## Docker/build/smoke checks

`orion/` is baked into the hub image (`/app/orion/...`), not bind-mounted — confirmed
by `docker inspect` (only `static/` and `templates/` are mounted rw). So this patch is
inert until `orion-hub` is rebuilt. See Restart required.

Live evidence gathered against the running deployment:

```text
# the trigger fires; the guard was the bug
$ docker logs orion-athena-hub | grep fcc_sandbox
06:47:58 fcc_sandbox_sync_skipped_dirty branch=test/orion-live-push-proof-aacd41ef
06:50:45 fcc_sandbox_sync_skipped_dirty branch=test/orion-live-push-proof-aacd41ef
08:12:42 fcc_sandbox_sync_skipped_dirty branch=test/orion-live-push-proof-aacd41ef
   (3 WebSocket accepts, 3 sync attempts, 3 skips)

# sandbox unwedged (stashed first, per operator approval)
before: 356 dirty entries, test/orion-live-push-proof-aacd41ef @ df6b3021c, 294 behind
after:    0 dirty entries, main @ b4a697a9d, 0 behind
branch refs preserved: 4/4, including fix/harness-finalize-reflection-pydantic-fallback @ c4a375088

# PYTEST_CURRENT_TEST guard proven live: full hub suite run, sandbox untouched
after suite: 0 dirty, branch=main, 0 behind, stashes=2
```


## Review findings fixed

<filled in below>

## Restart required

```bash
<filled in below>
```

## Risks / concerns

- Severity: medium. Concern: the host user's view of the sandbox is *wrong*.
  `git status --porcelain` run as `athena` reported 2 dirty files; the same command
  inside `orion-athena-hub` (root, and the git that actually runs the sync) reported
  ~350. `.git` is root-owned, so the host user cannot refresh the index and gets a
  stale answer with a zero exit code. Mitigation: inspect the sandbox through
  `docker exec orion-athena-hub git -C /mnt/orion-fcc/repo ...`, never from the host
  shell. This is not fixed by this patch, only documented.
- Severity: low. Concern: rescue stashes accumulate one entry per dirty episode and
  are never pruned. Mitigation: the disk is 3.4T with 682M used; a clean tree produces
  no stash, so this only grows when Orion actually leaves work behind.
