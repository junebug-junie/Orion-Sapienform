# GPU pool: say what holds the lock

## Summary

- Every lease verb and the tick now take the runtime lock through `_locked(op)`. That times the wait and the hold, per operation.
- Waits or holds of 250 ms or more are logged as `gpu_pool_slow_lock`. The log line names the operation and how long each phase took: probe, resolve, live_leases, schedule, resume, start_thread, bus_publish, publish_state.
- `GET /v1/lock-stats` returns per-operation count, worst wait, worst hold and slow count since the last call, then resets.

## Outcome moved

Lease RPCs stalled in bursts after stage 3: acquire normally takes about 60 ms, but bursts reached about 2 s per call, and 4 callers gave up in 8 minutes. The pool had no way to say why. It took an hour of outside sampling to find the cause: Postgres I/O saturation from the substrate reconcile sweeps, which is fixed in a separate PR. The next stall names itself.

## Current architecture

There was one `asyncio.Lock` with no visibility into who held it or for how long.

## Architecture touched

`services/orion-gpu-pool` only: runtime lock wrapper, phase timers, one read-only HTTP route.

## Files changed

- `services/orion-gpu-pool/app/runtime.py`: `_locked`, `LockStats`, and phase timers.
- `services/orion-gpu-pool/app/main.py`: `GET /v1/lock-stats`.
- `services/orion-gpu-pool/README.md`: how to read it.
- `services/orion-gpu-pool/tests/test_runtime.py`: a slow holder gets named and counted.

## Schema / bus / API changes

- Added: `GET /v1/lock-stats` (HTTP, read-only).
- No bus or schema change.

## Env/config changes

None.

## Tests run

```text
services/orion-gpu-pool: 36 passed, 5 skipped (the Postgres tests skip locally without GPU_POOL_TEST_POSTGRES_URI; CI runs them)
```

## Evals run

```text
N/A: observability only; scheduling is unchanged.
```

## Docker/build/smoke checks

```text
UNVERIFIED until deploy: curl localhost:<pool port>/v1/lock-stats
```

## Review findings fixed

- Small, self-reviewed diff.

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: timers add microseconds per operation.
  - Mitigation: none needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
