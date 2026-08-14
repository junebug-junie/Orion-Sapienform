# Turn the fleet-power path on: `BIOMETRICS_MODE=both` on athena

## Summary

- Flipped athena to `BIOMETRICS_MODE=both`, which starts the cluster aggregator that had never
  run. This is the operator half of ROADMAP B1: the schema, producer, reducer, storage and
  cognition consumer all merged in #1650/#1652/#1659 into a path whose top link was switched off.
- Documented the key in `services/orion-biometrics/.env_example` as what it actually is — a
  **per-node role**, not a fleet-wide setting — with the exactly-one-hub constraint and the full
  chain it gates, so the next operator does not have to re-derive it from a `PUBSUB NUMSUB`.
- Corrected `orion/bus/channels.yaml`: `orion:spark:signal` claimed `orion-state-service` as its
  consumer. It has none. This was the question that prompted the whole trace.
- Corrected a now-false note in `orion/inner_state_registry.py` that called `biometrics_cluster.v1`
  "dark".

## Outcome moved

Orion can see what it costs to run, in watts, for the first time.

Before the flip, measured live on the running bus:

```text
orion:biometrics:cluster    2 subscribers    0 messages in 18 s
```

Two consumers subscribed and starved for as long as the fleet has been running.

After:

```json
{"status":"fresh","constraint":"POWER","strain":0.18,"homeostasis":0.82,
 "stability":0.97,"fleet_watts":927,"fleet_watts_partial":["circe"],"freshness_s":7}
```

That is `_metacog_biometrics_cue` rendered inside the live `orion-athena-cortex-exec` container
against live `state-service` state — the actual string that reaches metacognition.

## Current architecture

Every node ran `BIOMETRICS_MODE=agent`: measure yourself, publish
`orion:biometrics:summary` and `:induction`. Nothing ran `hub`, so nothing subscribed to those
channels fleet-wide, aggregated them, or published `orion:biometrics:cluster`.

`BiometricsHub.publish_cluster` — including the `aggregate_fleet_measurements` reducer merged in
#1659 — was reachable code that no configuration reached.

## Architecture touched

No code paths changed. One env value changed on one host, closing this chain:

```text
orion-biometrics (athena, mode=both)
  Hunter subscribes orion:biometrics:summary + :induction   <- all 3 nodes
  BiometricsHubWorker every CLUSTER_PUBLISH_INTERVAL=15 s
    -> aggregate_fleet_measurements()   sum extensive / max intensive
    -> orion:biometrics:cluster
         -> orion-state-service    main.py:209 -> store.py:207
              -> BiometricsContext.cluster
                   -> orion-cortex-exec _metacog_biometrics_cue  executor.py:761
         -> orion-hub biometrics_cache.py:110
```

Every link but the first already existed and was subscribed.

## Files changed

- `services/orion-biometrics/.env_example`: document `BIOMETRICS_MODE` as a per-node role, the
  exactly-one-hub constraint, and the chain it gates. Value stays `agent` — see below.
- `orion/bus/channels.yaml`: `orion:spark:signal` `consumer_services` `["orion-state-service"]` ->
  `[]`, with the evidence inline.
- `orion/inner_state_registry.py`: `biometrics_cluster.v1` note said "dark ... the hub fallback is
  what's actually live". Both halves are now false.
- `docs/superpowers/pr-reports/2026-08-14-biometrics-hub-mode-flip-pr.md`: this report.

## Why `.env_example` keeps `agent`

Deliberate, and the one judgement call here. `BIOMETRICS_MODE` is among the few keys whose correct
value differs by host: `agent` is right for atlas and circe and right for any new node; `both` is
right only for athena. `.env_example` is the template copied onto every node, so writing `both`
there would make a blindly-copied deploy a second hub — two publishers aggregating the same fleet
into the same channel, with consumers silently keeping whichever arrived last and no error
anywhere. The safe default plus a loud documented exception beats a default that is correct for
one host in three.

`scripts/sync_local_env_from_example.py` only fills in missing keys and never clobbers an existing
local value, so athena's `both` will not be reverted by a later sync.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `orion:biometrics:cluster` now carries traffic (every 15 s from athena) where
  it previously carried none. Consumers were already subscribed and already handled the payload;
  they simply start receiving it.
- Compatibility notes: `orion:spark:signal`'s `consumer_services` correction is a documentation
  fix to the contract, not a behavior change — nothing was consuming it before or after.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes — comment only, value unchanged.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes. No key changes to
  propagate; run reported only pre-existing unrelated divergences.
- **Operator change made directly**: `services/orion-biometrics/.env` on athena,
  `BIOMETRICS_MODE=agent` -> `both`. Applied to the primary checkout (the canonical live copy) and
  to this worktree's deploy copy. Not applied to atlas or circe, and must not be.
- skipped keys requiring operator action: none.

## Tests run

```text
$ .venv/bin/python -m pytest tests/test_fleet_measurements.py \
    tests/test_single_consumer_channels_gate.py \
    services/orion-cortex-exec/tests/test_metacog_biometrics_fleet_watts.py -q
45 passed, 13 warnings in 4.31s

$ ORION_BUS_URL=redis://100.92.216.81:6379/0 \
    .venv/bin/python scripts/check_single_consumer_channels.py
single_consumer gate OK: 31 channel(s) checked, 4 warning(s)
```

## Evals run

```text
None. services/orion-biometrics has no evals/ directory. This change ships no new
logic to evaluate -- the reducer it switches on was covered by the 45 tests above
when it merged in #1659. Follow-up noted below.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-biometrics up -d --build
Image orion-biometrics-biometrics Built
Container orion-athena-biometrics Recreated / Started

# artefact verified, not the build log (this repo has shipped a stale layer before):
$ docker exec orion-athena-biometrics python -c "import orion.telemetry.biometrics_pipeline as p; ..."
aggregate present: True
FLEET_SUM_KEYS: ('chassis_watts', 'gpu_watts_total', 'gpu_count', 'cpu_cores')

$ curl -fsS http://localhost:8100/health
"mode": "both"   "node": "athena"   "cluster_channel": "orion:biometrics:cluster"

# hub ingesting all three nodes:
$ docker logs orion-athena-biometrics | grep "Hunter intake"
channel=orion:biometrics:summary ... node='athena'
channel=orion:biometrics:summary ... node='atlas'
channel=orion:biometrics:summary ... node='circe'

# reached the consumer:
$ curl -fsS http://localhost:8270/state/latest    (orion-state-service)
biometrics.cluster.sources              ['athena', 'atlas', 'circe']
biometrics.cluster.measurements         chassis_watts 939.0, gpu_watts_total 384.6,
                                        gpu_count 6.0, cpu_cores 248.0,
                                        temp_c_max 72.0, fan_pct_max 56.0
biometrics.cluster.measurements_missing chassis_watts ['circe'], fan_pct_max ['circe']
biometrics.cluster.constraint           POWER

# reached cognition:
$ docker exec orion-athena-cortex-exec python -c "_metacog_biometrics_cue(live_state)"
{"status":"fresh","constraint":"POWER","strain":0.18,"homeostasis":0.82,
 "stability":0.97,"fleet_watts":927,"fleet_watts_partial":["circe"],"freshness_s":7}
```

Note the fleet figure moved from 663 W (this morning's reading) to 927-939 W. Both exclude circe,
which has no reachable BMC, and `fleet_watts_partial` says so rather than letting the number pass
as a total.

## Review findings fixed

- Finding: `orion:spark:signal` named `orion-state-service` as consumer; state-service consumes
  `orion:spark:state:snapshot` and never references `:signal`.
  - Fix: `consumer_services: []` with the evidence inline.
  - Evidence: live `PUBSUB NUMSUB orion:spark:signal` = 0; `rg spark services/orion-state-service/`
    returns only `spark.state.snapshot` and a README snippet.
- Finding: `inner_state_registry.py` described `biometrics_cluster.v1` as dark and orion-hub's
  weighting fallback as "what's actually live". This flip falsifies both.
  - Fix: note rewritten to say the duplication is now active, not dormant, and to carry the live
    verification.
  - Evidence: `fleet_watts=927`, `sources=[athena,atlas,circe]` above.

## Restart required

Already applied on athena. For any operator reproducing this from a fresh checkout:

```bash
# athena only -- set BIOMETRICS_MODE=both in services/orion-biometrics/.env, then:
scripts/safe_docker_build.sh orion-biometrics up -d --build
curl -fsS http://localhost:8100/health | grep '"mode"'
```

No restart needed on atlas or circe, and they must stay `agent`.

## Risks / concerns

- Severity: low. Concern: `strain` in the cue is the known `sum(strain_inputs) / 7` dilution bug in
  `orion/telemetry/biometrics_pipeline.py` — a fixed divisor of 7 regardless of how many inputs are
  present, so it reads low whenever some are absent. Now that the cluster is live this diluted
  number reaches cognition where before it did not. Mitigation: it was already reaching cognition
  via the per-node path; the flip changes which aggregate carries it, not whether it is wrong. Fix
  is parked as ROADMAP B2 and should land before anything reasons off `strain` quantitatively.
- Severity: low. Concern: role weighting is now a live triple-implementation
  (`CLUSTER_ROLE_WEIGHTS` here, field-topology edges, orion-hub's `BIOMETRICS_ROLE_WEIGHTS_JSON`).
  Mitigation: resolution already specced in
  `docs/notes/2026-07-12-phase4-cluster-weighting-research.md`; registry note updated to say the
  duplication is now active.
- Severity: low. Concern: one more publisher every 15 s on athena, the node already CPU-contended.
  Mitigation: it is one aggregation over three cached summaries; measured no change in the
  container's steady state.
- Severity: informational. Concern: `services/orion-biometrics` has no `evals/` harness, so the
  quality of the fleet aggregate is covered only by unit tests. Follow-up: fold a fleet-coverage
  eval into ROADMAP B3 rather than opening a bare eval harness now.

## PR link

<to be filled after push>
