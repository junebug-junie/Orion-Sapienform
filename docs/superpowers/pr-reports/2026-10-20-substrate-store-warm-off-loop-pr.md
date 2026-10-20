## Summary

- Each time orion-substrate-runtime starts, its event loop freezes for about 3 minutes. While it is frozen, every bus subscription times out, and chat replies go out without reflection.
- Cause: `_get_substrate_graph_store` had no lock. Six tick threads each built their own Falkor store, and building one loads the whole graph. Loop-thread callers, such as the walkway_seen loop, built one inline on the loop.
- Fix: the store is now built once, behind a lock. A caller on the loop thread never builds the store or waits for the build. It gets None, which callers already treat as "skip", and the build starts in a background thread. `worker.start()` starts the build at boot.
- The walkway_seen store fetch and the post-turn closure handler now run in a worker thread. They wait for the store instead of skipping. The closure's Falkor write also no longer blocks the loop.
- Companion to #2779, which makes listeners resubscribe after the freeze. This PR removes the freeze itself.

## Outcome moved

At boot the store loads once, off the loop, instead of 6 overlapping loads plus one on the loop. Expected result: listeners never time out at boot, and chat keeps reflection right after a restart. UNVERIFIED until this is deployed.

## Current architecture

`BiometricsSubstrateWorker._get_substrate_graph_store` was an unlocked check-then-build. `FalkorSubstrateStore.__init__` loads the whole graph synchronously. Live boot at 05:09:34 showed 6 `falkor_substrate_hydrate_stats` lines between 05:12:21 and 05:12:26 (ok_total=1 each, so 6 separate stores), and no log lines from the loop between 05:09:34 and 05:12:21.

## Architecture touched

orion-substrate-runtime only: the worker's store getter, the walkway_seen loop and the post-turn closure listener.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`:
  - a build lock, plus a separate warm-flag lock so the loop never waits on the build
  - `_on_event_loop_thread()`
  - `_start_substrate_graph_store_warm()`
  - a call to start the build from `start()`
- `services/orion-substrate-runtime/app/walkway_seen_loop.py`: the store fetch runs in a worker thread.
- `services/orion-substrate-runtime/app/post_turn_closure_listener.py`: `on_closure` runs in a worker thread.
- `services/orion-substrate-runtime/tests/test_substrate_store_warm.py`: 5 regression tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: a loop-thread caller gets None (and logs `substrate_store_warming caller=...`) until the background build finishes. All of today's callers run off the loop, so in practice they wait for the build and do not skip.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-substrate-runtime/tests/test_substrate_store_warm.py -> 5 passed
  against origin/main code: 3 of the first 4 tests fail (concurrent build, inline loop build, closure on loop)
pytest services/orion-substrate-runtime/tests (PYTHONPATH=.:services/orion-substrate-runtime)
  branch: 17 failed, 475 passed, 1 collection error
  main  : 17 failed, 470 passed, 1 collection error   (same pre-existing failures)
```

## Evals run

```text
No eval harness covers boot latency. Live check after deploy, below.
```

## Docker/build/smoke checks

```text
After deploy: `docker logs -t orion-athena-substrate-runtime` should show
- exactly one falkor_substrate_hydrate_stats line at boot
- no multi-minute gap in log lines
- no "listener stopped" lines
- "Application startup complete" within seconds of "Waiting for application startup"
```
UNVERIFIED until this is deployed.

## Review findings fixed

- Finding: the loop-thread warm path took the same lock that the build holds for the whole load. A loop caller that arrived during a build would block, which is the original freeze.
  - Fix: a separate warm-flag lock. The loop never touches the build lock.
  - Evidence: `test_loop_caller_returns_at_once_while_a_build_is_in_progress`.
- Finding: the warming flag was reset outside a lock.
  - Fix: it is now reset under the warm lock.
  - Evidence: same tests.
- Finding: the comment overstated what is class-level.
  - Fix: the comment is reworded.

Not fixed (LOW, pre-existing): `_get_sql_engine` has the same unlocked check-then-create. The worst case is one orphaned engine pool.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```
Run this from the primary main checkout after merge. Merge #2779 first or together.

## Risks / concerns

- Severity: low
  - Concern: while the single load runs, about 15 tick threads wait on the lock. The thread pool allows 32 workers on this 96-CPU host, so this should not starve it.
  - Mitigation: none needed today.
- Severity: low
  - Concern: the single load can still be slow. The patch removes the 6x contention and the loop block, not the load cost itself.
  - Mitigation: measure after deploy.

## PR link

<link>
