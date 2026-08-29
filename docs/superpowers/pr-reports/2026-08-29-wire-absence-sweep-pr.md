# PR: absence detection goes live (phase 2)

## Summary

- Wires `sweep_absent_nodes()` into `orion-substrate-runtime`'s biometrics tick. PR #1935
  made node absence *expressible*; nothing called it, so nothing detected anything. This
  makes it **reachable**.
- Adds `build_absence_trigger_event()`: the synthetic trigger a silent node cannot send for
  itself, in the `biometrics.node:{node}:{ts}` shape the organ already parses.
- The sweep runs **before** `_tick`'s `if not events` early return -- a total outage produces
  no biometrics events, so a sweep placed after it would be skipped in exactly the incident
  it exists to catch.

## Outcome moved

circe was gone 2026-08-29 00:01:16Z -> 00:47:04Z (~45 min) with `expected_online: true` and
five declared capabilities, and `last_accepted_at["availability"]` never moved off
`2026-08-19T03:13:33Z`. `invoke_biometrics_pressure()` resolves its subject from the incoming
event, so a node that stops reporting is never assessed at all.

After this patch a node silent past `BIOMETRICS_NODE_STALE_AFTER_SEC` produces
`node_availability_concern` + `node_capability_absent`, and `capability_impacts` names the
real capabilities lost. **This is the change that would have caught the incident that started
this thread.**

## Architecture touched

`services/orion-substrate-runtime/app/worker.py` only, plus one helper in
`orion/substrate/biometrics_loop/pressure_organ.py`. No new service, channel, table or env
key.

Synthetic triggers go through the **same** `process_biometrics_grammar_events` pipeline as
real events with `enable_node_reducer=False`, so the organ, pressure reducer, receipts and
publish path are all shared. Only the node-biometrics projection is skipped: a synthetic
trigger is not a report and must never refresh `last_seen_at`, or the sweep would mark the
node fresh and stop detecting its own outage.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: `_absence_sweep()` + tick wiring
- `orion/substrate/biometrics_loop/pressure_organ.py`: `build_absence_trigger_event()`
- `tests/test_absence_sweep_wiring.py`: 5 tests (new)

## Schema / bus / API changes

None. Uses the roles and reducer arms PR #1935 already added.

## Env/config changes

None. Uses the service's existing `biometrics_node_stale_after_sec`
(`BIOMETRICS_NODE_STALE_AFTER_SEC`), closing the "phase 2 must thread the setting" caveat
#1935 flagged. Gated by the existing `enable_biometrics_pressure_organ`.

## Tests run

```text
pytest tests/test_absence_sweep_wiring.py -q                    -> 5 passed
pytest tests/test_absence_sweep_wiring.py tests/test_capability_absence_signal.py \
       tests/test_biometrics_pressure_organ.py tests/test_node_pressure_reducer.py -q
                                                                -> 40 passed
pytest services/orion-substrate-runtime/tests -q --continue-on-collection-errors
                                                                -> 305 passed, 17 failed, 1 error
```

The 17 failures + 1 error are **pre-existing**: verified by reverting both source files to
`main` in place and re-running the identical invocation -- byte-identical
`17 failed, 305 passed, 1 error`.

Mutation-tested: moving the sweep after the early return fails
`test_sweep_runs_before_the_no_events_early_return`.

**A real bug this caught:** my first cut imported `build_absence_trigger_event` /
`sweep_absent_nodes` from `...biometrics_loop.pipeline` instead of `...pressure_organ`.
`worker.py` would not have imported at all -- the service would have crash-looped on
startup. Found by diffing the suite against baseline (25 failed / 124 passed / **33 errors**
vs baseline's 17 / 305 / 1), not by any local check, because the module imports fine in
isolation.

## Docker/build/smoke checks

```text
Not run -- no Dockerfile, compose, dependency or port changed.
```

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Verify live -- this is the acceptance check, not a formality:

```bash
docker logs --since 10m orion-athena-substrate-runtime | grep biometrics_absence_sweep
# then, once a node has been stale past the threshold:
psql -h localhost -p 55432 -U postgres -d conjourney -Atc \
  "select jsonb_pretty(projection_json->'nodes'->'circe'->'capability_impacts')
     from substrate_active_node_pressure_projection order by generated_at desc limit 1"
# expect a non-empty list -- it has been [] in every row ever written.
```

## Risks / concerns

- Severity: medium. This fires on **every tick** while a node is stale, not once per outage.
  The pressure reducer's per-node+`pressure_kind` merge window (300s default) absorbs the
  repeats, but a long outage still writes a delta every 5 minutes per absent node. Bounded
  and small; worth watching on the first real outage.
- Severity: medium. `atlas` is `expected_online: false` and permanently silent, so it must
  never be swept. Covered by a test, and Rule A (`node_pressure_suppressed`) is the existing
  guard, but it is the obvious way this could become a permanent alarm.
- Severity: low. Wrapped in `except Exception` so a failure in the absence path cannot take
  down the ordinary tick for nodes that ARE reporting.
- Severity: low, NOT closed. This produces a **signal**, not a notification. Nothing yet
  turns a capability transition into something that reaches Juniper -- and `notify_attempts`
  has 0 rows ever, so that sink is unverified. Next step.
- Severity: low, unchanged from #1935. A node that has NEVER reported is absent from the
  projection entirely and still cannot be swept (`prometheus`). Needs a catalog sweep.
