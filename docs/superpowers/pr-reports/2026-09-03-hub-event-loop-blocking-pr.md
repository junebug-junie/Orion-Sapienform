# The hub was blocked ~70% of the time by its own cognition loop

## Summary

- Orion's substrate mutation cycle — a **synchronous** function taking **64–91
  seconds** — ran inline on the hub's event loop every ~105s. While it ran the
  hub served nothing at all, including static assets.
- Correlation with the operator's symptom is exact: every stall over 50s begins
  **0–9 seconds** after a cycle start.
- 24 further route handlers did synchronous DB work inline, including
  `/api/self-brain/frames/tail` — the hub's most-polled endpoint (every 3s) —
  which also built a **new SQLAlchemy engine per request**.
- All of it now runs off the loop. A new CI gate
  (`scripts/check_async_routes_not_blocking.py`) makes the pattern fail the
  build instead of being found by an operator noticing slow tabs.
- Review found a route **this patch had edited** that was still blocking, and
  that the gate certified it clean. Both fixed; the gate now has 12 tests.

## Outcome moved

Operator report, after the previous fix (PR #2063) deployed: *"they do
eventually load, just not instant like it used to be."*

Sampling a static JS file once per second against the live hub — 150 samples,
nothing to do with the database:

```
p50 = 0.0004s
stalls > 1s:  60.0  7.0  13.4  60.0  7.2  60.0  10.5  60.0  7.2
285s stalled out of ~600s — and every 60.0 is the client cap, so a floor
```

Against the mutation cycle's own tick log:

```
stall start   dur     cycle start   offset
06:31:26     60.0s    06:31:17      +9s
06:33:03     60.0s    06:33:02      +1s
06:34:40     60.0s    06:34:40      +0s
```

A static asset cannot be slow because of a database query — unless the loop
serving it is blocked.

## Current architecture (before this patch)

`services/orion-hub/scripts/main.py`:

```python
async def _run_substrate_autonomy_scheduler() -> None:
    while True:
        api_routes_runtime.execute_substrate_mutation_scheduled_cycle()  # plain def
        await asyncio.sleep(interval_sec)
```

`execute_substrate_mutation_scheduled_cycle` is synchronous. Configured
interval 30s; ticks actually landed 94–121s apart, implying 64–91s of blocking
work per cycle. The decay scheduler directly below it already used
`asyncio.to_thread` correctly — this loop simply never did.

## Architecture touched

- `orion-hub` — scheduler execution model; 24 route handlers moved off the loop;
  three lazily-built engines given locks and connect timeouts; shutdown ordering.
- `orion/substrate` — an `RLock` and a snapshot accessor on `SubstrateMutationStore`.
- CI — one new static gate, plus a Makefile target.

## Files changed

- `services/orion-hub/scripts/main.py`: cycle via `asyncio.to_thread`, shielded
  from an uncancellable cancel; per-tick wall-clock logged.
- `services/orion-hub/scripts/api_routes.py`: per-phase timing in the cycle.
- `services/orion-hub/scripts/self_brain_routes.py`: process-wide cached engine
  (was one per request), three routes threaded, connect timeout.
- `services/orion-hub/scripts/substrate_observability_routes.py`: six loaders in
  one dispatch instead of six inline queries.
- `services/orion-hub/scripts/chat_turn_trace_routes.py`: three sync loaders threaded.
- `services/orion-hub/scripts/field_channel_glossary_routes.py`,
  `substrate_lattice_routes.py`: threaded, engine locked, connect timeout.
- 10 × `substrate_*_routes.py`: 21 call sites threaded.
- `orion/substrate/mutation_queue.py`: `RLock` + `cognition_view_snapshot()`.
- `services/orion-hub/scripts/mutation_cognition_context.py`: one locked snapshot
  instead of six live dict iterations.
- `scripts/check_async_routes_not_blocking.py`, `Makefile`,
  `.github/workflows/orion-static-gates.yml`: the gate.
- Tests: `test_check_async_routes_not_blocking.py` (12),
  `test_mutation_store_concurrent_reads.py` (5).

## Schema / bus / API changes

None. No channel, schema-registry, or response-shape change. The mutation
cycle's `mutation_scheduler_cycle_finished` payload gains a `phase_sec` field.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not required — no env key changed.
- local `.env` synced: run, no changes reported.

## Tests run

```text
services/orion-hub/tests/test_check_async_routes_not_blocking.py
orion/substrate/tests/test_mutation_store_concurrent_reads.py     17 passed
7 × changed route-file suites                        1 failed, 83 passed
orion/substrate/tests -k mutation                                 90 passed
services/orion-hub/tests -k "mutation_cognition|observability|
                             chat_turn_trace|grammar_atlas"        39 passed
node --test services/orion-hub/static/js/*.test.js       113 passed, 0 failed
12/12 CI static gates                                             all OK
```

The one failure, `test_channels_endpoint_returns_38_raw_channels_plus_1_derived`
(48 vs 39), is pre-existing: it fails identically with this patch's
`field_channel_glossary_routes.py` replaced by the unmodified `HEAD` copy.

### Mutation testing

The gate is tested against every shape it has historically missed:

```text
inline blocking call in the route                     -> caught
blocking one hop away in a sync helper                -> caught  (v1 missed)
two-hop helper chain                                  -> caught
helper reached through a LOOP VARIABLE                -> caught  (v2 missed)
awaited async helper that blocks internally           -> caught  (v2 missed)
blocking call as a to_thread ARGUMENT                 -> caught
missing scan root (discovery broken)                  -> exits 1, not 0
correct to_thread usage                               -> not flagged
nested def handed to to_thread                        -> not flagged
awaited async client (bus.connect)                    -> not flagged
@app.on_event lifecycle hook                          -> not flagged
noqa on a multi-line call's closing line              -> respected
```

## Evals run

No eval harness exists for these hub routes. Not claiming eval coverage; the
live latency correlation above is the substitute evidence.

## Docker/build/smoke checks

**Not deployed.** `UNVERIFIED` for every post-change runtime claim. The live
container still carries the old inline call and has no `_phase_sec`.

The measurements quoted are all *pre-change* observations of the live system
(latency sampling, tick-log correlation, `pg_stat_statements`, engine-build
timing), not claims about the patched behaviour.

## Review findings fixed

- **Finding:** `substrate_observability_routes.observability_summary` still ran
  six sequential Postgres queries inline — in a file this patch had edited. The
  patch threaded `_engine()`, which is lazy and never blocks.
  - **Fix:** all six in one `to_thread` dispatch.
- **Finding:** the gate certified that route clean, because the callee is a loop
  variable. Third time this checker shipped green over its own target.
  - **Fix:** resolves any bare NAME reached on the loop, not just direct calls.
  - **Evidence:** `test_flags_helper_reached_through_a_loop_variable`.
- **Finding:** an `async def` helper that blocks internally was invisible. Live
  example: `get_fused_chat_turn_trace` calls three sync loaders inline.
  - **Fix:** async helpers enter the helper map; loaders threaded.
- **Finding:** `_is_route` matched `@app.on_event`, so a startup-time engine
  build would turn CI red on correct code. Fixed.
- **Finding:** nested `def` bodies were inlined into their parent, falsely
  flagging `grammar_atlas_routes._with_session` — correct code. Fixed; a gate
  that cries wolf is a gate people learn to ignore.
- **Finding:** `# noqa: async-blocking` checked only a call's first line, so it
  was inert on all 55 multi-line call sites — the only documented escape hatch
  had never worked. Fixed.
- **Finding:** two lazy engines became thread-reachable without a lock, and
  `self_brain`'s new cached engine skipped the `connect_timeout` this repo
  already documents in `mutation_control_surface.py`. Both fixed.
- **Finding:** `asyncio.to_thread` is not cancellable, so shutdown returned
  immediately while a 64–91s cycle kept mutating state underneath it. Shielded.
- **Finding (headline):** moving the cycle to a thread arms a real race. It
  writes `_proposals`/`_trials` while the **chat path** reads them on the loop
  at `api_routes.py:3326`, outside any `try` — a `RuntimeError` there is a 500
  on Orion's main chat endpoint. Dormant only because `proposals_created` is
  currently 0.
  - **Fix:** `RLock` on the store; the reader takes one locked snapshot.
  - **Evidence:** a test that first *provokes* the `RuntimeError` on the real
    shape, then shows the snapshot surviving identical load.
  - **Note:** the first version of that test was **vacuous** —
    `list(d.values())` is GIL-atomic and never raises, so it passed with the
    lock removed. The docstring now separates what the copy guarantees (no
    crash) from what the lock guarantees (one instant across six dicts).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

Then confirm the fix on the live rail:

```bash
# should show no stalls; before, 3 of every 10 samples hit 7-60s
for i in $(seq 1 120); do \
  printf '%s %s\n' "$(date -u +%H:%M:%S)" \
  "$(curl -s -o /dev/null -w '%{time_total}' http://localhost:8080/static/js/turn-timer.js)"; \
  sleep 1; done | awk '$2>1.0'

# and read the cycle cost that is no longer hidden by the freeze
docker logs orion-athena-hub --since 10m 2>&1 | grep -E "autonomy_scheduler_tick_done|phase_sec"
```

## Risks / concerns

- **Severity: medium.** The cycle now runs concurrently with async routes
  instead of taking turns with them. The store race that exposes is fixed and
  tested, but the audit covered the objects the reviewer and I could enumerate —
  a `SubstrateMutationStore` field added later without the lock reopens it.
- **Severity: medium.** `UNVERIFIED` — not deployed. Everything rests on
  pre-change measurement plus tests.
- **Severity: low.** A 64–91s cycle is still a 64–91s cycle. Off the loop it
  stops announcing itself by freezing the UI, which is why per-phase timing
  ships with it: quiet is not fixed. The breakdown is a log line only; nothing
  alerts on it yet.
- **Severity: low.** The gate scans `services/orion-hub/scripts` only. Other
  services are unchecked.

## PR link

(filled in on open)
