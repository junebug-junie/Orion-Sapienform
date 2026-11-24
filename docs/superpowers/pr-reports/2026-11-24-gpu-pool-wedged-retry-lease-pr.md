## Summary

- After the planned mesh shutdown on 2026-11-24, one durable-run hold (the PR field note for #3804) sat in "waiting to retry" for 3+ hours and never moved. All 14 later field notes were stuck behind it.
- Cause, part 1: one scheduler pass both **expired** the hold (its heartbeat window had passed) and **recalled** it (it looked like it had held the slot past `background_max_hold_sec`, counted from before the outage). A recall on an expired hold is an illegal move.
- Cause, part 2: the lease graph rejected that move *inside* its `wait` node. The rejection left an error in the saved checkpoint, so the thread had no next step. Every later event (requeue, release, deadline) was then dropped by `_resume_inner` with no log line.
- Fix: the scheduler no longer recalls a lease it expires in the same pass. The runtime checks each move with the pure `transition()` before running the graph, so a rejected move never touches the checkpoint. A thread already wedged this way repairs itself on its next event.

## Outcome moved

A host outage can no longer leave a GPU lease stuck forever. Before: one bad pass froze the lease and its durable run until someone cancelled it by hand. After: the lease expires once, re-queues, and gets granted again (regression test replays the outage).

## Current architecture

`orion/gpu_pool/scheduler.py` computes decisions from one snapshot of the live leases. `services/orion-gpu-pool/app/runtime.py` applies them in order through the LangGraph lease graph (`orion/gpu_pool/lease_graph.py`), whose `wait` node raises `InvalidTransition` on an illegal (status, event) pair. `_resume_inner` returned `None` whenever the checkpoint had no next step.

## Architecture touched

gpu-pool scheduler (recall set) and runtime (resume path). No contract, schema, bus, or config change.

## Files changed

- `orion/gpu_pool/scheduler.py`: track `expiring` in section 1. Leave those leases out of `active`, and out of the operator-seat max_hold recall.
- `services/orion-gpu-pool/app/runtime.py`: check the move with `transition()` before `ainvoke`. Add `_heal_thread` for a non-final thread with no next step. Its log line is `gpu_pool_lease_thread_healed`.
- `orion/gpu_pool/tests/test_scheduler.py`: one pass never emits both Expire and Recall for one lease.
- `services/orion-gpu-pool/tests/test_runtime.py`: the outage replay, a rejected move leaving the thread usable, and an already-wedged thread healing.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: an illegal lease event is rejected before the graph runs it, not inside it. The log line is the same (`gpu_pool_transition_rejected`).
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
orion/gpu_pool/tests: 586 passed
services/orion-gpu-pool/tests: 205 passed, 15 skipped
The 4 new tests fail on origin/main code (checked by swapping the old scheduler.py/runtime.py back in): 4 failed
```

## Evals run

```text
services/orion-gpu-pool/evals/run_pool_day_eval.py: VERDICT: PASS
```

## Docker/build/smoke checks

```text
Not run. This patch is pure Python in the existing image. Live proof comes after deploy (see Risks).
```

## Review findings fixed

- Finding: the operator-only seat's max_hold recall (`scheduler.py` ~1055) reads `leases`, not `active`, so it was the one recall path not covered by the exclusion. It can't be reached today, because operator holds have no expiry.
  - Fix: added `and l.lease_id not in expiring` there too.
  - Evidence: commit 30e01e8aa; 586 scheduler tests pass.
- The reviewer confirmed the rest. The pre-check matches the graph, because `_TABLE` lookup ignores config. The heal can't fire on dead_letter, unavailable, released, or mid-run threads. The heal keeps the lease history intact.

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

The post-pull rebuild does this on merge.

## Risks / concerns

- Severity: low. Concern: `_heal_thread` discards the rejected event. That matches the old intent (rejected means not applied). Mitigation: it logs a WARNING for every heal.
- Live proof: UNVERIFIED until deploy. The lease that wedged live on 2026-11-24 (`7a3d623fce444e5c9401381f1585fb2b`) is still wedged in the pool: its run was cancelled by hand at 22:05Z, so it is now only a ghost queue entry. After deploy it should log `gpu_pool_lease_thread_healed`, then end through the normal path.
- Issues opened (PR field-note failure modes found in the same investigation): #3843 (a PR is dropped forever after 2 failed runs, silently), #3844 (a backlog pushes PRs past the 72 h lookback, silently), #3845 (a paused run blocks every later note).
- Issues closed: none.

## PR link

(this PR)
