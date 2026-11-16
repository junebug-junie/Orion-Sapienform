## Summary

Camera-ask bids were still missing their 0.5 s answer window. This happened when the market recomputed its 30-day family weights at the same moment a burst of bids arrived. Fixes #3382 (second cause).

- The weights recompute ("replay", ~2-3 s of pure-Python CPU work) now runs in one separate worker process (`app/weights_worker.py`). It cannot hold the lock that the bid code needs (the Python GIL).
- Only the background refresher starts a replay once weights exist. The refresher does three steps in order: the personality read, the history read, then a check for a new UTC day or personality read (`reconcile_weights`). A bid never starts a replay or a personality read. It uses the last weights it has.
- A bid now builds the appetite for its own family only, not all 39 families. That is the only one its decision reads. Finding the previous close now uses a binary search, not a scan over every act. On live data, one bid went from ~0.22 s to ~0.03 s.
- If a replay hangs, the worker process is killed and a new worker starts. `/health` now shows which snapshot, UTC day and personality read the weights belong to.

## Outcome moved

On the live 30-day history (5,292 acts, 5,286 uses, 189 choices; read-only, 2026-11-16):

- Per-bid time (`Market.bid`, vision.ask want, live state, fakes for Redis/publish): **0.22-0.25 s -> 0.025-0.035 s**. Before, ~85% of the time was spent building the appetite for all 39 families. The two largest families took ~135 ms each, because finding each act's previous close was quadratic.
- The replay itself: 1.8-2.3 s here (3.3 s live). In the worker process, the bid loop's worst timer lag during a replay is <=4 ms. Before, with the replay in a thread, bids in a test ran 1.8-2.1x slower.
- Pickling: one `pickle.dumps` of the replay inputs took ~85 ms (8.6 MB, mostly `cost_json` text the replay never reads), all in one uninterruptible call. Now `cost` is dropped and the pickling is done in a thread, in chunks of 250 records. Packing takes ~63 ms in total, and the loop waits at most ~12 ms. The result is 31 kB and takes ~1 ms to pickle.

## Current architecture

#3468 made the history cache stale-while-revalidate (a bid gets the last snapshot while a fresh one loads in the background). But the weights replay still ran in `asyncio.to_thread`, and any bid that saw a changed weights key started it. On 11-16 07:00Z, the first bid after a personality refresh started a 3.3 s replay. Bids 2-6 of that burst took 0.45-0.70 s.

## Architecture touched

- `services/orion-durable-runs`: the market's weights replay seam, the refresher, the per-bid context, `/health` fields.
- `orion/physiology/choice.py` `build_context(only_families=...)` (optional; the default is unchanged, and every other caller uses the default).
- `orion/physiology/appetite.py` `build_family_appetite` previous-close lookup (same result: verified identical on all 39 live families).

## Files changed

- `services/orion-durable-runs/app/weights_worker.py`: new. Contains the replay function, the thread-side chunked packing, the worker-side unpacking, the spawn executor, and `kill`.
- `services/orion-durable-runs/app/physiology_market.py`: replay via the worker; `reconcile_weights`; the refresher order; bids no longer start replays or personality reads; per-family context; bounded replay; health fields.
- `services/orion-durable-runs/app/main.py`: passes the worker factory; personality TTL = history TTL x 0.8, so the read happens every refresher cycle.
- `services/orion-durable-runs/app/personality_values.py`: docstring.
- `orion/physiology/choice.py`, `orion/physiology/appetite.py`: as above.
- `services/orion-durable-runs/README.md`: the bid market's reads paragraph.
- `services/orion-durable-runs/tests/test_physiology_market_replay_off_loop.py`: new tests.
- `services/orion-durable-runs/tests/test_physiology_market_warm_cache.py`: two tests rewritten for "only the refresher replays".

## Schema / bus / API changes

- Added: `/health` -> `physiology_market.weights_history_at`, `weights_day`, `weights_personality_at`, `weights_worker` (debug fields).
- Removed / renamed: none.
- Behavior changed: bids no longer start a weights replay or a personality read. The decision logic is unchanged.
- Compatibility notes: none.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs: PYTHONPATH=<repo> pytest tests -q          -> 1012 passed, 149 skipped (twice)
orion/physiology/tests                                                  -> 352 passed
orion/curiosity/tests/test_replica_answers.py                           -> 11 passed
services/orion-sql-writer tests test_camera_ask_acts.py                 -> 23 passed
services/orion-mind tests/test_prime_loop.py                            -> 49 passed
services/orion-dream tests/test_compaction_consent.py, test_physiology_bind.py -> 18 + 21 passed
scripts/analysis/tests (body economy + outreach binding replay)         -> 5 passed
```

The key regression test, `test_a_cpu_bound_replay_in_the_worker_process_does_not_slow_the_bids`: a replay that busy-loops for 1.5 s runs while 7 bids go through. It asserts that each bid's wall time / CPU time is < 1.3 (the bid never waited for the GIL). Its control, `test_control_the_same_replay_in_a_thread_does_slow_the_bids`, uses the same setup with a thread executor and asserts > 1.6. The control read 1.8-2.1 in 40 of 40 runs, so the main test can fail. A further test checks that a personality snapshot change makes no bid start a replay.

## Evals run

```text
curiosity_binding_replay, reading_binding_replay, event_wants_replay --camera-asks,
event_rank_grading_mutations, ask_bind_mutations --lane durable-runs, cooling_act_replay,
body_binding_replay, prime_charge_replay, consent_gate_fixture        -> all exit 0
hold_fairness.py: needs CI's isolated Postgres (ORION_ADMISSION_TEST_DSN); unrelated GPU-pool eval, left to CI.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile -t orion-durable-runs:test-weights-off-loop .  -> ok
In-image: Market(weights_executor=process_executor)._compute_weights_entry on 600 acts
  -> "replay in worker process ok: 38 families 0.70s ProcessPoolExecutor"
uvicorn's console script has an `if __name__ == '__main__'` guard, so spawn's re-import of it is safe.
(Test image removed afterwards; the live image tag was not touched.)
```

Not done: live deploy. A live run under `uvicorn app.main:app` is UNVERIFIED until deploy.

## Review findings fixed

- Finding: the regression test could not catch a thread replay (a +30 ms slack hid a 2x slowdown).
  - Fix: heavier bids (6,000 acts); the measure is each bid's wall/CPU ratio, which is not affected by host speed changes; a thread control must fail; best of 3 for the process run (on a loaded host the OS sometimes puts the worker on the loop's core); skipped below 3 CPUs.
  - Evidence: process 1.0 in 8/8 unpinned runs; thread 1.8-2.1 in 40/40 runs.
- Finding: at boot the refresher waited for a cold personality read before the first history snapshot.
  - Fix: with no snapshot yet, the personality read starts without being awaited.
  - Evidence: `test_boot_does_not_wait_on_a_slow_personality_read`.
- Finding: the personality read ran only every ~8 min (TTL 300 s vs a 240 s refresher cycle).
  - Fix: personality TTL = history TTL x `REFRESH_FRACTION`.
  - Evidence: `main.py`.
- Finding: a stuck worker would block every later replay, because nothing killed it.
  - Fix: the replay is bounded by `REFRESH_TIMEOUT_SEC`. On timeout or a broken pool, `weights_worker.kill` terminates the worker, and the next replay starts a new one.
  - Evidence: `test_a_stuck_worker_is_killed_...`, `test_kill_terminates_a_busy_worker_process`. This test caught that `shutdown()` clears `_processes`: the helper now reads the processes first.
- Finding: whether the holdback arm gets its own weights depended on object identity surviving a pickle.
  - Fix: with no choices, both weights are set to the same object explicitly.
  - Evidence: `test_no_choices_through_a_real_worker_...`.
- Finding: nothing showed live that only the refresher replays.
  - Fix: `/health` weights key fields.
- Finding: no test for an unknown or non-bidding family with `only_families`.
  - Fix: `test_an_unknown_or_non_bidding_family_...`.

Also fixed: a pre-existing "coroutine `Market._refresh` was never awaited" warning when a refresh task is cancelled before it starts.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run this from the clean primary `main` checkout after merge (the post-pull rebuild does it automatically).

## Risks / concerns

- Severity: low. Concern: the worker is a second process (~0.2 s start, one Python's memory). Mitigation: one worker, started lazily, killed on shutdown or when it hangs.
- Severity: low. Concern: a bid-path history refresh (only when the refresher has failed for over a TTL) still includes a replay. Mitigation: that replay is now off the loop too.
- Severity: low. Concern: the remaining per-bid cost is ~20-40 ms in `tier1_reserved_share` (6 lanes x all acts). Mitigation: far inside the budget; no ticket.
- Live proof: the next camera-ask burst after deploy should have zero `physiology_bid_unavailable ... TimeoutError`. Check with `/health` `weights_worker: true` and `ps` in the container. Tracked on #3382 (verify-live).
- Issues opened: none. Issues closed: #3382 (on merge; live check to be posted there).

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
