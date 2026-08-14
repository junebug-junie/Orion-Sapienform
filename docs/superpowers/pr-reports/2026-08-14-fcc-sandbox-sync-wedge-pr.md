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

- `orion/fcc/turn_lock.py` (new): `flock`-based cross-container advisory lock on the
  shared `/mnt/orion-fcc` mount. Turns shared, sync exclusive/non-blocking.
- `orion/harness/fcc_motor.py`: governor turns hold the shared lock for their duration.
- `services/orion-hub/scripts/fcc_claude_bridge.py`: hub turns do the same.
- `orion/fcc/sandbox_sync.py`: rescue-then-reset; unpushed branches no longer block;
  stash-refusal falls back to a rescue branch; detached HEAD gets a ref; stale
  `index.lock` cleared; every attempt records an inspectable verdict via
  `last_sync_state()`.
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

A subagent review found 11 issues. The first is one this patch *introduced* and is the
reason the patch grew a cross-container lock.

- **Finding (critical): hub's sync could reset the sandbox under a running governor
  turn.** The turn interlock was `fcc_claude_bridge.active_turns()` — a dict local to
  the *hub process*. It cannot see a turn spawned by `orion-harness-governor`, which
  runs claude in the same directory via `orion/harness/fcc_motor.py`.
  - Verified worse than reported: commit `e6136c092` established that live chat turns
    dispatch through the governor's bus RPC path, not hub's in-process bridge, so that
    dict is empty during essentially *every* real turn. Confirmed on the running
    deployment — 8 `harness governor reply received` in hub logs, 0 in-process bridge
    turns. Before this patch the dirty-worktree refusal masked the hole; rescue-then-
    reset removed that accidental protection.
  - Fix: `orion/fcc/turn_lock.py` — an `flock` on the shared `/mnt/orion-fcc` mount,
    which is the only thing that can coordinate two containers. Turns take it
    **shared** (they are not exclusive with each other), the sync takes it
    **exclusive, non-blocking** and records `skipped_turn_in_flight_external`. Wired
    into both writers: `fcc_motor.py` (governor) and `fcc_claude_bridge.py` (hub).
    Lock file lives beside the repo, never inside it, so `git clean -fd` cannot delete
    the lock coordinating the sync that is running it.
  - Evidence: `test_sync_yields_to_a_turn_holding_the_shared_lock` — sync returns
    `skipped_turn_in_flight_external` while a turn holds the lock, and `synced` the
    moment it releases.

- **Finding (should): a conflicted merge was a brand-new permanent freeze.** `git
  stash` refuses on unmerged paths, so `_rescue_dirty_worktree` would fail on every
  subsequent sync — the same wedge, relabelled. And the log line was unreadable: git
  writes the refusal to stdout, so the one verdict meaning "a human is needed" logged
  literally `err=`.
  - Fix: log `stdout or stderr`, and add `_rescue_to_branch` — commit the tree onto a
    rescue branch when stash refuses, aborting merge/rebase state only *after* the
    content is safely committed.
  - Evidence: `test_stash_refusal_falls_back_to_a_rescue_branch` builds a real
    conflict, asserts stash genuinely refuses, then asserts the content survives on a
    rescue branch and that a second run is a clean no-op rather than another decline.

- **Finding (should): detached HEAD orphaned commits with no ref at all.** `git branch
  --show-current` is empty when detached, so the branch-preservation path never ran;
  commits survived only in HEAD's reflog until `git gc` expired them, while the sync
  reported a cheerful `synced`.
  - Fix: `_rescue_detached_head` mints `orion-sandbox-autorescue-detached/<stamp>`
    before the checkout, skipped when the branch fallback already moved HEAD onto its
    own ref.
  - Evidence: `test_detached_head_commits_get_a_real_ref` asserts `for-each-ref
    --contains <sha>` is non-empty after the sync.

- **Finding (should): the status endpoint reported `workspace: null` on a cold hub.**
  `{"workspace": settings...., **last_sync_state()}` let the state's own `None`
  override the configured value — in exactly the post-restart moment an operator
  checks it and the sandbox is most likely stale. There was also no test for the route
  at all.
  - Fix: spread first, expose `configured_workspace`. Two route tests added.
  - Evidence: `test_status_endpoint_reports_the_configured_workspace_on_a_cold_hub`.

- **Finding (should): `_LAST_SYNC` was mutated non-atomically, and skips erased the
  last success.** `clear()` then `update()` leaves a window where the threadpool-run
  route sees `{}`; and a skip overwrote `synced` so the surface could no longer answer
  "is the sandbox current?" — the question it exists for.
  - Fix: build the record and rebind in one assignment; keep `last_attempt` and
    `last_success` as separate slots.
  - Evidence: `test_last_sync_state_keeps_attempt_and_success_apart`.

- **Finding (should): the `PYTEST_CURRENT_TEST` guard was defeatable and had a real
  gap.** The marker is set per test *item*, so a task outliving the item — precisely
  what this fire-and-forget task is — sees it unset.
  - Fix: `or "pytest" in sys.modules`, which is stable for the whole process.
  - Evidence: `test_guard_holds_when_only_the_env_marker_is_cleared`.

- **Finding (should): running the tests wrote the operator's real GitHub PAT into
  `/tmp`.** `_configure_push_auth` falls back to `~/.fcc/.env`, which holds a live
  token, and every test builds a clone and syncs it. Confirmed: 26 files under
  `/tmp/pytest-of-athena/` contained the real token, retained across three run
  directories.
  - Fix: autouse fixture pointing `HARNESS_FCC_ENV_PATH` at a nonexistent tmp file;
    the two push-auth tests override it with their own fixture.
  - Evidence: purged the existing tmpdirs, re-ran the suite, and grepped for the real
    token by exact value — 0 files. The 6 remaining matches are the deliberate fake
    `ghp_test_token_123`.

- **Finding (should): a killed git left `.git/index.lock` with nothing to clear it.**
  Another silent recurring freeze of the same family.
  - Fix: `_clear_stale_index_lock`, run only while holding the exclusive turn lock
    (which rules out a live git in the other container) and only past the git timeout.
  - Evidence: `test_stale_index_lock_does_not_freeze_the_sync_forever` plus
    `test_fresh_index_lock_is_left_alone`, which pins the age check as load-bearing.

- **Finding (note): `behind_main` was 0 by construction on the success path**, and the
  test asserted that constant — an assertion that could not fail.
  - Fix: replaced with `commits_advanced` (start HEAD → new HEAD), the number that
    says the sync actually did something. `behind_main` is kept only on the decline
    paths, where it is real.

- **Finding (note): gitignored files are destroyed with no stash.** True — `status
  --porcelain` does not report them and `stash -u` would not capture them.
  - Fix: not code. The module docstring now states the carve-out explicitly instead of
    claiming "never destroys work".

- **Finding (note): unused `last_sync_state` import in `websocket_handler`.** Removed;
  the hub test imports it from `orion.fcc.sandbox_sync` directly.

One review point deliberately **not** treated as a defect: the reviewer noted the
status endpoint has no UI consumer and called it "a log line with extra steps". Fair,
and worth being honest about rather than quietly wiring a panel to look responsive: its
consumer is the operator and the restart-verification step in this report, not a
dashboard. If it is still unqueried a month from now it should be deleted, not
decorated.

## Restart required

Both containers, not just hub: `orion/harness/fcc_motor.py` changed, so the governor
must pick up the turn lock. A hub that takes the lock while a governor still ignores it
is worse than neither — the sync would believe it is interlocked when it is not.
Rebuild the **governor first**, so no window exists where hub resets a tree the
governor is writing without holding the lock.

```bash
cd /mnt/scripts/Orion-Sapienform-fcc-sandbox-sync-wedge
scripts/safe_docker_build.sh orion-harness-governor build
scripts/safe_docker_build.sh orion-harness-governor up -d
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

Verify (a browser refresh should now produce a `synced` verdict):

```bash
curl -fsS http://localhost:8080/api/fcc-sandbox-sync
docker logs --since 5m orion-athena-hub | grep fcc_sandbox
docker exec orion-athena-hub git -C /mnt/orion-fcc/repo rev-list --count HEAD..origin/main
ls -l /mnt/orion-fcc/.fcc-turn.lock    # created on first turn or sync
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
