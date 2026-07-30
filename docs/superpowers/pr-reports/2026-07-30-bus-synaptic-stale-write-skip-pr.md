# Bus synaptic prediction_error stale-write-skip fix

## Summary

- `_bus_synaptic_tick()` (`services/orion-substrate-runtime/app/worker.py`) only wrote `node:substrate.bus_synaptic`'s durable node when the freshly-computed error was `> 0.0`, so a calm tick simply skipped the write instead of refreshing the node to reflect true current state.
- Live-verified: the node was frozen at `prediction_error=1.0` for hours while the real tick logged fresh, varying, non-saturated values (0.000-0.144) every 30s the whole time.
- `orion-equilibrium-service`'s `_bus_synaptic_poll_loop` polls this node's raw value with zero staleness check, so the frozen value produced recurring false "Bus Anomaly Detected" alerts.
- Fix: split the gate so the receipt write (audit trail) stays gated on `error > 0.0`, but the node write (polled current-state) now runs unconditionally every tick.
- Added a README entry documenting the incident and fix, per this service's established convention.
- **Found but not fixed in this patch:** a second, independent bug in `orion/substrate/reconcile.py`'s `merge_node()` (used by `orion-cortex-exec-background`'s concept-induction materializer) re-locks this same node's `prediction_error` on a much faster cadence than this fix's 30s tick, so the false alerts will likely persist until that's also fixed. See "Known related risk" below.

## Outcome moved

`node:substrate.bus_synaptic`'s durable node can now read a genuine calm (`prediction_error=0.0`) state on any tick where the real signal is calm, instead of only ever ratcheting up to the last nonzero value and staying there forever. This directly targets the metric-quality-gate failure mode CLAUDE.md names explicitly (a metric that can vary but structurally can't return to calm).

## Current architecture

`orion-substrate-runtime`'s `_bus_synaptic_tick()` runs every `SUBSTRATE_BUS_SYNAPTIC_TICK_INTERVAL_SEC` (default 30s), reads live `gap_zscore`/`latency_zscore` edges from the `orion_bus_synapse` FalkorDB graph (written by `orion-bus-mirror`), computes `bus_synaptic_prediction_error()`, and — previously only when that error was `> 0.0` — wrote a receipt and durably upserted `node:substrate.bus_synaptic` in the `orion_substrate` FalkorDB graph. `orion-equilibrium-service`'s `_bus_synaptic_poll_loop` separately polls that same node's raw `prediction_error` property on its own interval and fires a `transport` metacog trigger (`MetacogTriggerV1`, `reason="transport:bus_synaptic:error=..."`) when it's at or above `EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD` (default `1.0`).

## Architecture touched

`services/orion-substrate-runtime` only. No contract/schema/bus changes — this is a pure bugfix to an existing write path's gating condition.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: `_bus_synaptic_tick()` — moved the `_write_prediction_error_node(...)` call out of the `if error > 0.0:` block so it runs on every tick; the `save_receipt(...)` call stays gated.
- `services/orion-substrate-runtime/tests/test_worker_bus_synaptic_tick.py`: renamed/rewrote `test_bus_synaptic_tick_no_edges_writes_nothing` → `test_bus_synaptic_tick_no_edges_still_writes_calm_node`, asserting the node write now happens (with `error=0.0`) while the receipt still doesn't.
- `services/orion-substrate-runtime/README.md`: added a dated entry describing the incident, the fix, the deliberately-out-of-scope sibling risk (4 other `_*_tick` methods share the same gate shape), and the separate `reconcile.py`/materializer root cause found during live verification.

## Schema / bus / API changes

None.

## Env/config changes

None. No `.env_example` changes; nothing to sync.

## Tests run

```
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_bus_synaptic_tick.py -q
9 passed, 3 warnings in 2.75s

$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-substrate-runtime/tests -q --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
16 failed, 192 passed, 15 warnings in 7.38s
```

The 16 failures are pre-existing and unrelated (cursor-reset auth, quarantine truth, reducer-health cross-test pollution, etc.) — confirmed by running the identical command against `main` and diffing: byte-for-byte the same 16 test names, same 192/16 split. This patch introduces zero new failures. `test_grammar_consumer_integration.py` fails to *collect* on `main` too (`ModuleNotFoundError: app.models`), unrelated to this change.

## Evals run

No dedicated eval harness exists for this narrow tick; not applicable.

## Docker/build/smoke checks

```
$ bash scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
... build succeeded, container recreated and started ...
```

Live-verified post-deploy:
- `docker logs orion-athena-substrate-runtime` shows `substrate_prediction_error_node_written node_id=node:substrate.bus_synaptic error=0.000 reducer_key=bus_synaptic` on calm ticks (previously no log line / no write at all on those ticks).
- Confirmed via direct FalkorDB `GRAPH.QUERY` against `orion_substrate` that the write path itself is correct and reaches durable storage when nothing else is racing it.
- **Also confirmed the write is currently still getting raced and overwritten** by the separate `reconcile.py`/materializer bug described above — traced via `redis-cli CLIENT LIST`/`MONITOR` on the FalkorDB container to a persistent connection from `orion-athena-cortex-exec-background`, re-touching this exact node every few seconds with a stale, self-reinforced `prediction_error=1.0`. This fix is real and correct in isolation (verified the write call itself lands when nothing else interferes, and the receipt/write split behaves exactly as intended per the passing tests), but the live symptom (false "Bus Anomaly Detected" alerts) will likely persist until the second bug is separately fixed.

## Review findings fixed

- Finding: No dated README entry for this incident/fix, despite this file's established convention of one per prior bus_synaptic change.
  - Fix: Added the entry (see Files changed).
  - Evidence: `services/orion-substrate-runtime/README.md` diff.
- Finding (confirmed, correctly left as-is): 4 sibling `_*_tick` methods (biometrics, execution, chat, route) share the identical `if error > 0.0:` gate around both their receipt and node write, so they carry the same latent staleness risk.
  - Fix: Not fixed in this patch — deliberately scoped to `bus_synaptic` only, per the task. Documented as a follow-up in the README and this report.
  - Evidence: `worker.py` lines ~735 (biometrics), ~2142 (execution), ~2202 (chat), ~2255 (route), all unchanged.
- No other material findings from the review pass (correctness, test coverage, and downstream consumer behavior all verified sound).

## Restart required

```
bash scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Already run once during this session's live verification; the running container reflects this fix. No further restart needed unless this branch is rebuilt from a fresh checkout.

## Risks / concerns

- Severity: Medium
- Concern: This fix alone does not resolve the live "Bus Anomaly Detected" false-alarm symptom, because a separate, independently-discovered bug in `orion/substrate/reconcile.py`'s `merge_node()` (invoked by `orion-cortex-exec-background`'s concept-induction materializer path) re-locks `node:substrate.bus_synaptic`'s `prediction_error` on a faster cadence than this tick's 30s interval, via an unconditional `merged_metadata = {**incoming.metadata, **existing.metadata}` (existing always wins, including reducer-owned keys) followed by an `upsert_node()` call with no `skip_metadata_keys` protection.
- Mitigation: Documented in this report and the README with exact file/line evidence and live-verification method (FalkorDB `CLIENT LIST`/`MONITOR` tracing). Needs its own fix and its own review, since `merge_node()` is shared by any concept merge in the repo, not just this one node — deliberately not attempted in this patch without sign-off given the wider blast radius.
- Severity: Low
- Concern: 4 sibling domains (biometrics, execution, chat, route) share the exact same write-skip-on-zero gate shape fixed here for bus_synaptic.
- Mitigation: Documented as a known follow-up; not fixed here to keep this patch scoped and reviewable.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/bus-synaptic-stale-write-skip
