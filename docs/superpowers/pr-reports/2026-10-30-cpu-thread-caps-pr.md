## Summary

- heartbeat now runs its math library (OpenBLAS) on one thread. Its matrices are tiny, so extra threads only spin-wait.
- vector-host now caps PyTorch at 8 threads per embedding (`OMP_NUM_THREADS=8`).
- vector-host now runs one embedding at a time, on its own single-worker executor, instead of `asyncio.to_thread`.
- Regression tests in both services fail if either cap is removed.

## Outcome moved

All numbers measured live on athena (96 logical cores, shared with sql-db, substrate-runtime and 32 dojo workers):

| | before | after |
|---|---|---|
| heartbeat, tiny-matrix benchmark in its own image | 0.61 s wall, 9.76 s CPU | 0.62 s wall, 0.62 s CPU |
| heartbeat, live container CPU samples | 0% -> 630% -> 1,524% -> 1,606% -> 271% | expected near 0% (UNVERIFIED until deployed) |
| vector-host, one bge-large embed (130 tokens) | 48 threads: 4.6 s, 146 CPU-s | 8 threads: 0.36 s, 2.8 CPU-s |
| vector-host, live burst | 965% CPU, 454 `pt_main_thread` threads | expected ~8 torch threads (UNVERIFIED until deployed) |

## Current architecture

- heartbeat: an ensemble of 8 quimb matrix-product states (`app/substrate/mps_state.py`: 10 sites, bond dim 4, phys dim 4). `absorb()` and `decay_reheat_tick()` run via `asyncio.to_thread`. Its numpy/scipy OpenBLAS used its default thread pool. 16 threads burned ~30% each in lockstep while the event loop thread was idle.
- vector-host: `VECTOR_HOST_EMBED_BACKEND=hf`, bge-large on CPU. `Embedder.embed()` used `asyncio.to_thread`. The default executor allows up to 32 workers. Each worker thread gets its own OpenMP team sized to the 48 physical cores, so a burst of bus messages stacked about 10 teams.

## Architecture touched

- `services/orion-heartbeat/Dockerfile`, `services/orion-vector-host/Dockerfile`, `services/orion-vector-host/app/embedder.py`. No contract, bus, schema or env-template change.

## Files changed

- `services/orion-heartbeat/Dockerfile`: `ENV OPENBLAS_NUM_THREADS=1`.
- `services/orion-vector-host/Dockerfile`: `ENV OMP_NUM_THREADS=8`.
- `services/orion-vector-host/app/embedder.py`: single-worker `ThreadPoolExecutor` for HF embeds; shut down in `close()`.
- `services/orion-heartbeat/tests/test_openblas_thread_cap.py`: gate on the Dockerfile cap.
- `services/orion-vector-host/tests/test_embedder_thread_caps.py`: 8 concurrent embeds never overlap; the Dockerfile cap is present.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: vector-host HF embeds are serialized. A burst of N requests now finishes in about N x 0.36 s, instead of all of them running at once at about 4.6 s each.
- Compatibility notes: none.

## Env/config changes

- Added keys: none in any `.env_example`. The caps are fixed image `ENV` values, not operator config.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
# inside each service's live image, worktree mounted
orion-vector-host: 17 passed
orion-heartbeat:   150 passed, 2 skipped
# regression proof
heartbeat test with the ENV line removed:          1 failed
vector-host tests with embed() reverted to to_thread: 2 failed, 1 passed
```

## Evals run

```text
No eval harness exists for either service's thread behaviour. The before/after
benchmarks above (tiny-matrix loop in the heartbeat image; bge-large embed at
48/16/8/4 threads in the live vector-host container) stand in for one.
```

## Docker/build/smoke checks

```text
docker build services/orion-heartbeat/Dockerfile   -> OK; OPENBLAS_NUM_THREADS=1 in image
docker build services/orion-vector-host/Dockerfile -> OK; torch.get_num_threads() == 8 in image
```

## Review findings fixed

Code review ran in a subagent: no correctness bugs.

- Finding: the heartbeat Dockerfile test matched the full `CMD` line, so any CMD change would crash it with `ValueError`.
  - Fix: the test now finds the first line starting with `CMD`.
  - Evidence: `pytest services/orion-heartbeat/tests/test_openblas_thread_cap.py` passes; it still fails when the ENV line is removed.
- Noted, no change: `embed()` after `close()` raises `RuntimeError`. The old `to_thread` path had the same exposure; uvicorn drains requests before lifespan shutdown.
- Noted, no change: `OMP_NUM_THREADS=8` also caps vector-host's numpy (OpenBLAS falls back to it). Its numpy work is 16x16x8 grids, too small to matter.

## Restart required

```bash
scripts/safe_docker_build.sh orion-heartbeat up -d --build && scripts/safe_docker_build.sh orion-vector-host up -d --build
```

Run from the clean primary `main` checkout after merge (the post-pull rebuild also picks these up).

## Risks / concerns

- Severity: low
- Concern: serialized embeds add queueing latency to a large burst on the RPC path (`orion:embedding:generate`).
- Mitigation: each embed is now about 13x faster, so a 10-message burst clears in about 3.6 s, versus about 4.6 s for every message before.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2984

🤖 Generated with [Claude Code](https://claude.com/claude-code)
