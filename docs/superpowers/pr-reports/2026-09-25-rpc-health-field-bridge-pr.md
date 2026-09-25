# PR: RPC delivery reaches the field (rpc-health -> capability:transport reliability_pressure)

## Summary

- Orion's sense of "can my parts reach each other" (`capability:transport`) had a
  reliability channel that could never move. Its only input was the bus-observer's own
  failure count, which has read exactly 0 for weeks. Meanwhile every service already
  counts, per request channel, how many of its requests were answered and how many hit
  their deadline, and publishes that every 30 s on `orion:rpc_health:snapshot`.
- orion-substrate-runtime now listens to those snapshots. Every 30 s it reads the last
  10 minutes and reports the worst bus request channel's share of calls that timed
  out: `timeouts / max(calls, 10)`. A channel only counts once it has 2 or more
  timeouts in the window, so a single stray timeout reads 0. A channel reads 1.0 only
  after 10 unanswered calls with no success.
- That reading lands on a new pseudo-node, `node:substrate.rpc_delivery`
  (`rpc_timeout_pressure`), through the normal receipt -> field-digester path. A new
  topology edge carries it to `capability:transport` `reliability_pressure`.
- cortex-exec's current-turn probe now tags its LLM call with its own health label.
  The probe has a 3-second deadline that is shorter than normal LLM latency on purpose,
  and it made 247 of the 311 `LLMGatewayService` RPC timeouts on 2026-09-22..25.
  Without the tag the reading could not rest at 0 (see the metric gate).
- If the reading stops arriving for 2 minutes (bridge down, flag rolled back, bus
  outage), the field drops it. Every consumer then sees "not measured" instead of a
  frozen last value.
- orion-equilibrium-service also stops alerting on the probe's hop (its exclude list
  gains `current_turn_probe`).
- Two flags, off in code, on in `.env_example` and the local `.env`:
  `SUBSTRATE_RPC_DELIVERY_BRIDGE_ENABLED` (substrate-runtime) and
  `ENABLE_RPC_DELIVERY_FIELD_DIGESTION` (field-digester).

## Outcome moved

Before (live, last 24 h up to 2026-09-25 06:23 UTC): 40,316 field ticks, and
`capability:transport` `reliability_pressure` had a maximum of exactly 0, attributed to
`node:athena`. After deploy, a request path that stops answering raises it, and the
receipt names the channel. On the 2 h live replay (shipped config), it reads a measured
0.0 on 96% of ticks. It reads 0.0625 while a real 2-timeout `LLMGatewayService` episode
sits in the window.

## Current architecture

- `RpcHealthSnapshotV1` (`orion/core/bus/rpc_health.py`, `rpc_health_publish.py`) is
  published by 14 producer instances today (cortex-exec x4, cortex-orch, hub, actions,
  durable-runs, thought, mind, execution-dispatch-runtime, harness-governor, gpu-pool,
  embodiment), all with the per-hop `channel_latency` field.
- Consumers were orion-signal-gateway (`rpc_transport_health` signals) and
  orion-equilibrium-service (transport metacog trigger + latency EWMA gate,
  `orion/metacog/transport_baseline.py`). Nothing reached the field or attention.
- `orion/substrate/signal_bridge.py` turns `OrionSignalV1` into substrate *molecules*
  (not field receipts), supports three unrelated signal kinds, and its worker is never
  started. It was not revived: it is the wrong output shape, and it would have added a
  hop through signal-gateway for data already on the bus.
- `capability:transport` inputs: `node:athena` `observer_failure_pressure ->
  reliability_pressure`, `catalog_drift_pressure -> contract_pressure`;
  `node:substrate.bus_synaptic` `prediction_error -> pressure`.

## Architecture touched

- **New pure reducer** `orion/substrate/rpc_delivery.py`: rolling window, per-hop sums
  across producers, worst-hop ratio with a denominator floor, replay guard per
  (service, instance), bounded memory, thread lock (listener on the event loop, tick on
  a worker thread). Reuses `transport_baseline.is_excluded` and `_parse_ts`.
- **orion-substrate-runtime**: `_rpc_health_listener_loop` (bus subscribe, same shape
  as the vision listener) and `_rpc_delivery_tick_loop` (writes the receipt). Both
  start only when the bus is up and the flag is on.
- **orion-field-digester**: `rpc_timeout_pressure` node channel (replace mode, not
  decayed), `rpc_delivery` branch in `delta_to_perturbations`, per-lane gate in
  `delta_digestion_enabled`.
- **Topology**: `node:substrate.rpc_delivery -> capability:transport`
  (`rpc_timeout_pressure: reliability_pressure`, weight 0.85). Diffusion takes the
  max of this and `node:athena`'s `observer_failure_pressure`, so neither masks the
  other.
- **cortex-exec**: `health_label="current_turn_probe"` on the probe RPC.
- **Bus contract**: `orion:rpc_health:snapshot` gains consumer
  `orion-substrate-runtime`.

## Metric quality gate: `rpc_timeout_pressure`

1. **Provenance.** `OrionBusAsync.rpc_request()` records each call's outcome
   (`RpcHealthAggregator.record_success/record_timeout`, per hop key
   `channel[#label]`), `rpc_health_publish` drains and publishes it every 30 s as
   `channel_latency[hop].{success_count,timeout_count}`. Then:
   `worker._handle_rpc_health_message` -> `RpcDeliveryWindow.fold` ->
   `RpcDeliveryWindow.reading` (per-hop sums, worst `hop_pressure`) ->
   `rpc_delivery_receipt` -> field-digester `state_deltas.py` (`rpc_delivery`) ->
   `diffusion.py` over the new edge.
2. **Independence.**
   - vs `observer_failure_pressure` (same target channel): that counts the
     bus-observer's own tick failures. Different sensor, different event. It has
     been 0 all month while RPC timeouts happened daily.
   - vs `bus_synaptic` `prediction_error` (`capability:transport` `pressure`):
     inter-service message timing gaps from the bus mirror. It measures cadence
     surprise, not whether a request got its reply. Different target channel too.
   - **Overlap disclosed (review finding 3):** a caller deadline fires on a lost
     message and on a slow callee alike. Live, nearly every nonzero reading names
     `LLMGatewayService` or `orion:cortex:request`, i.e. slow inference or cortex work.
     So when inference is slow, this co-moves with `capability:llm_inference` load
     (`gpu_pressure`). The claim is therefore stated as "asked and got no answer in
     time", not "the network lost it". That wording is in the glossary, the edge
     comment and the module docstring. It stays on `capability:transport` because that
     is the caller-side delivery surface. The receipt names the hop, so the two causes
     can be told apart.
   - vs `inference_failure_pressure` (llm-gateway lane, #2327, on
     `capability:llm_inference`): that is the backend's outcome, as the gateway sees
     it: calls that came back with `[Error: ...]`. This is the caller's view:
     replies that did not arrive before the caller's own deadline. The two sets of
     events are disjoint by construction. An error reply arrives in time, so it counts
     as a *success* here. A caller timeout on a request the backend later serves counts
     as `served` there. A sick backend can move both, but through different evidence.
   - vs equilibrium's transport metacog trigger: that is a consumer of the same
     snapshots, raising a metacognition event per timeout or latency episode. It does
     not feed the field. This bridge does not reuse its EWMA (see 3).
   - Not a monotonic transform of anything already in the field.
3. **Theory anchor.** This is the error rate in the RED / SRE golden-signals sense,
   measured where the error happens: at the caller, against the caller's own
   deadline. A timeout ratio has a natural zero (a working request path does not time
   out), so there is **no learned baseline**. The equilibrium EWMA exists because
   latency has no natural zero. Putting one here would teach the field that a steady
   10% failure rate is calm. The denominator floor (10) is a chosen constant, not a
   calibrated one (see the sensitivity row below).
4. **Live-data sanity.** Neither signal-gateway nor equilibrium persists the
   snapshots. They were recovered from equilibrium's per-hop `transport_baseline_obs`
   log (2 h 03 min, 1,074 producer snapshots) and a 3.5 min raw capture off
   `orion:rpc_health:snapshot` (93 payloads, 14 producers). Both are committed as eval
   fixtures and replayed with 30 s ticks:

   | replay | ticks | exactly 0.0 | p50 | p90 | max | nonzero worst hops |
   |---|---|---|---|---|---|---|
   | shipped (probe labelled, >=2 timeouts) | 247 | **96.4%** | 0.0 | 0.0 | 0.0625 | LLMGatewayService 9 |
   | no hysteresis (lone timeouts count) | 247 | 70% | 0.0 | 0.056 | 0.10 | LLMGatewayService 53, cortex:request 21 |
   | probe unlabelled (before this patch) | 247 | 36.8% | 0.071 | 0.14 | 0.23 | LLMGatewayService 156 |
   | raw wire capture (pre-label producers) | 8 | 0% | 0.19 | 0.20 | 0.30 | LLMGatewayService 8 |

   - **Calm:** the shipped config rests at a measured 0.0. That is not "no data": there
     were 0 unmeasured ticks and a median of about 120 counted calls per window. A live
     in-process smoke ran the branch code against the production bus (06:20-06:26 UTC,
     14 producers) and read 0.0 on every tick. Caveat: the replay fixture moved 46 probe
     timeouts onto the probe's hop key by timestamp matching. The probe's successes could
     not be separated, so 96% is an optimistic bound. Resting at 0 live is
     **UNVERIFIED until the cortex-exec label is deployed**.
   - **Non-calm:** real timeouts move it. In the shipped config, a genuine 2-timeout
     `LLMGatewayService` episode read 0.0625 for 9 ticks. The wire capture reads 0.3 on
     real published payloads. Those were mostly probe timeouts, because the producers
     predate the label, but it shows the reducer moving on the real wire shape.
   - **The hysteresis (>=2 timeouts):** without it, 30% of ticks carried one lone
     timeout. Each read 0.05-0.1, then stepped back to 0 exactly 10 minutes later. That
     step is a clock artifact, and feedback credit would score it as a reliability
     improvement for whatever action was in flight (review finding 2).
   - **The floor the probe created:** without the label, the probe's timeouts kept the
     reading off zero on 63% of ticks, even with the hysteresis. This is the
     structural-floor failure named in CLAUDE.md §0A, so the label is part of this
     patch, not a follow-up.
   - **Denominator floor sensitivity** (max, lone timeouts counted): n0=1 -> 0.143,
     5 -> 0.143, 10 -> 0.10, 20 -> 0.0625.
   - **UNVERIFIED live:** a large reading (an actual outage, >0.5) has not happened
     since the bridge existed. It is proven by tests only (`10/10 -> 1.0`,
     `dead hop among 500 healthy calls -> 0.6`).
   - Note on scope: in 04:00-05:59 today there were ~250 `orion:gpu_pool:lease` /
     `state` RPC timeouts. They came from orion-llm-gateway, which does **not**
     publish rpc-health, so this bridge cannot see them. See concerns.
5. **Existing-mechanism check.** signal-gateway `rpc_transport_health`: its level is
   pooled success/(success+timeout) per snapshot, with no window, no hop, no floor,
   and it only reaches OTel/signals, not the field. Equilibrium transport baseline:
   latency EWMA plus episodes, feeding metacog only. `signal_bridge.py`: wrong output,
   dead worker. `transport_loop` / `node:rpc_timeout`: removed in #2323 because it
   counted timeout atoms with no denominator. Reused from existing code:
   `transport_baseline.is_excluded` and the equilibrium default exclusion labels
   (`log_orion_metacognition`, `gpu_pool_wait`).
6. **Reversibility.** Cheap. Either flag off stops it. The channel is one entry in
   `channels.py`, the glossary and the topology, plus one edge. No table, no
   migration, no autoencoder manifest change (the anomaly model has a fixed trained
   input width). Removing it also means dropping the pseudo-node from live
   `node_vectors` (add to `RETIRED_PSEUDO_NODES`). The metric lock was re-locked:
   one routing change (new consumer) and one added field channel.

## Files changed

- `orion/substrate/rpc_delivery.py`: new reducer + receipt builder.
- `services/orion-substrate-runtime/app/{worker.py,settings.py}`, `.env_example`,
  `docker-compose.yml`, `README.md`: listener, tick, flag and knobs.
- `services/orion-substrate-runtime/tests/test_worker_rpc_delivery_tick.py`: wire
  envelope -> receipt, replay, unmeasured, flag off, store failure.
- `services/orion-substrate-runtime/evals/{run_rpc_delivery_eval.py,test_rpc_delivery_eval.py,fixtures/*.jsonl.gz}`:
  live replay eval (numbers above).
- `services/orion-field-digester/app/{tensor/channels.py,ingest/state_deltas.py,worker.py,settings.py}`,
  `.env_example`, `docker-compose.yml`, `README.md`: channel, perturbation, gate.
- `services/orion-field-digester/tests/test_field_rpc_delivery_perturbations.py`:
  real receipt -> reconcile -> diffusion to `capability:transport`.
- `config/field/orion_field_topology.v1.yaml`: node channel + edge.
- `config/field/field_channel_glossary.v1.yaml`, `tests/test_field_channel_glossary.py`: glossary entry.
- `services/orion-cortex-exec/app/current_turn_llm_signals.py` + test: probe hop label.
- `services/orion-field-digester/app/digestion/decay.py`, `app/tensor/update_rules.py`:
  `EXPIRING_NODE_CHANNELS` / `expire_unrefreshed_channels` (drop after 120 s unwritten).
- `services/orion-field-digester/app/tensor/channels.py`: `rpc_timeout_pressure` is a
  single-observer channel owned by `node:substrate.rpc_delivery`;
  `tests/test_reconcile_single_observer_channels.py` updated for a non-athena owner.
- `services/orion-equilibrium-service/{app/settings.py,.env_example,docker-compose.yml,README.md}`
  + `tests/test_transport_baseline_gate.py`: `current_turn_probe` excluded. The test was
  already failing on main: it pinned the list to `["log_orion_metacognition"]`.
- `orion/bus/channels.yaml`: consumer added.
- `scripts/sync_local_env_from_example.py`: `SUBSTRATE_RPC_DELIVERY_`,
  `SUBSTRATE_RPC_HEALTH_`, `ENABLE_RPC_DELIVERY_` prefixes.
- `config/metrics/metric_definitions.lock.json`: re-lock.
- `tests/test_rpc_delivery_reducer.py`: reducer unit tests (on validated
  `RpcHealthSnapshotV1` payloads).

## Schema / bus / API changes

- Added: `StateDeltaV1.target_kind="rpc_delivery"` (free-string field, no schema
  change); consumer `orion-substrate-runtime` on `orion:rpc_health:snapshot`; field
  node channel `rpc_timeout_pressure`; pseudo-node `node:substrate.rpc_delivery`.
- Removed / Renamed: none.
- Behavior changed: cortex-exec's probe RPC is recorded under hop key
  `orion:exec:request:LLMGatewayService#current_turn_probe`. Pooled rpc-health fields
  are unchanged. Equilibrium's transport baseline sees a new key, which cold-starts
  per its warm-up, and the old `LLMGatewayService` key stops accruing probe timeouts.
- Compatibility notes: `RpcHealthSnapshotV1` unchanged. Snapshots without
  `channel_latency` are ignored, never guessed from pooled fields.

## Env/config changes

- Changed default: equilibrium `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS` gains
  `current_turn_probe`. Local `.env` was edited by hand for that one key, because the
  sync script leaves diverged keys alone and `--force` would flatten secrets.
- Added keys: substrate-runtime `SUBSTRATE_RPC_DELIVERY_BRIDGE_ENABLED` (code false,
  example true), `SUBSTRATE_RPC_DELIVERY_TICK_INTERVAL_SEC=30.0`,
  `SUBSTRATE_RPC_DELIVERY_WINDOW_SEC=600.0`, `SUBSTRATE_RPC_DELIVERY_MIN_DENOMINATOR=10`,
  `SUBSTRATE_RPC_DELIVERY_EXCLUDE_LABELS=log_orion_metacognition,gpu_pool_wait,current_turn_probe`,
  `SUBSTRATE_RPC_HEALTH_SNAPSHOT_CHANNEL=orion:rpc_health:snapshot`; field-digester
  `ENABLE_RPC_DELIVERY_FIELD_DIGESTION` (code false, example true).
- Removed / renamed keys: none.
- `.env_example` updated: yes (both services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (run
  from the worktree; all 7 keys present in the primary checkout's `.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_rpc_delivery_reducer.py                                   16 passed
services/orion-substrate-runtime/tests/test_worker_rpc_delivery_tick.py  7 passed
services/orion-substrate-runtime tests+evals: 13 failed / 11 errors, identical set on
  origin/main (diffed in a clean detached worktree); the rest pass
services/orion-field-digester/tests (excl. cwd-sensitive heartbeat_chassis) 247 passed
services/orion-equilibrium-service/tests (PYTHONPATH=repo root)      188 passed
root tests/test_field_* / attention_field / feedback / credit / proposal (40 files):
  7 failed + 1 collection error, identical set on origin/main (diffed)
services/orion-cortex-exec/tests/test_current_turn_llm_signals.py   26 passed
CI static-gates pytest steps (env sync, parity, agent trace registry, ladder
  liveness, grammar producer catalog) + glossary                  109 passed
static gates: metric_lineage, definition_drift, inner_state_registry, stdlib_shadow,
  hostname_refs, compose mounts (x2), journal dispatch, schedule collisions,
  sentience_instruments, system_health_producers, control_surface_parity,
  async_routes, chat_route_poachers, env_template_parity, bus_reply_channels,
  env_key_single_source, metric_dead_wiring -> all exit 0
git diff --check -> clean
Pre-existing, unrelated: tests/test_field_topology_config.py (canonical 8 edges vs
  legacy alias 6 on main already) and tests/test_field_deterministic_replay.py
  (stale run_digestion_tick signature) fail on main too; not in CI.
```

## Evals run

```text
python services/orion-substrate-runtime/evals/run_rpc_delivery_eval.py   (table above)
pytest services/orion-substrate-runtime/evals/test_rpc_delivery_eval.py   6 passed
Live in-process smoke (branch reducer, production bus, 06:20-06:26 UTC):
  14 producers folded, every tick pressure 0.0, receipts schema-valid.
```

## Docker/build/smoke checks

```text
docker compose config (primary .env files) for orion-substrate-runtime and
  orion-field-digester: all new keys resolve (true/600/10/30/labels/channel).
Import smoke in the current orion-substrate-runtime image with branch code mounted:
  app.worker + orion.substrate.rpc_delivery import, bridge flag default False.
No image built or deployed (no deploys from this task). Runtime effect UNVERIFIED
until deployed.
```

## Review findings fixed

Review by a separate subagent over `git diff origin/main...HEAD`. There were no
blockers and 3 material findings; all were fixed.

- Finding (material): a stopped bridge held its last value forever. Feedback's
  write-backed check (`credit_integrity.channel_write_backed`) read it as fresh, and
  the listener never resubscribed after an error.
  - Fix: field-digester drops `rpc_timeout_pressure` when it has not been written for
    120 s (`EXPIRING_NODE_CHANNELS`, run before diffusion). The channel is
    single-observer, so it is never seeded as 0.0 on other nodes. The listener
    resubscribes with exponential backoff (up to 60 s).
  - Evidence: `test_stopped_bridge_expires_to_unmeasured_not_held` (the value, the
    capability and its provenance all clear at 121 s),
    `test_channel_is_never_seeded_on_other_nodes`,
    `test_listener_resubscribes_after_a_subscription_failure`.
- Finding (material): when the 10-minute window rolled off a single timeout, feedback
  credited that as a reliability improvement. This happened on about 30% of ticks.
  - Fix: `min_timeouts=2` hysteresis per hop.
  - Evidence: eval `no_hysteresis` 70% zero vs `shipped` 96.4% zero;
    `test_hysteresis_removes_lone_timeout_steps`,
    `test_a_lone_timeout_is_below_the_hysteresis_and_two_are_not`.
- Finding (material): what "transport reliability" measures. It is mostly slow
  LLM/cortex work, which overlaps with inference.
  - Fix: the claim is restated as "no answer before the caller's deadline" in the
    glossary, the topology edge comment, the module docstring and the field-digester
    README. The overlap is recorded in gate step 2.
  - Evidence: `nonzero_worst_hops` in the eval.
- Finding (minor): the probe's new hop key would open `zero_success` episodes in
  equilibrium once emit is turned on.
  - Fix: `current_turn_probe` added to the equilibrium exclude defaults (settings,
    compose, example, README, local `.env`).
  - Evidence: `test_default_exclude_labels_cover_metacog_self_loop`.
- Finding (minor): the eval wording overclaimed ("no synthetic rows").
  - Fix: reworded, and resting at 0 live is marked UNVERIFIED until cortex-exec is
    deployed.
- Nits:
  - The replay guard now includes `node`
    (`test_same_service_on_two_nodes_has_separate_replay_guards`).
  - The `_overflow` hop is now logged instead of being skipped silently.
  - The calm-provenance test now removes `node:athena`'s input and asserts the bridge
    node exactly.
  - Before review, the live smoke itself caught a calm reading that named a healthy
    hop as "worst". Fixed: `worst_hop` is None when nothing timed out.

## Restart required

Order matters: cortex-exec first, so the probe label is live before the field reads.

```bash
cd /mnt/scripts/Orion-Sapienform   # after merge + pull
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
```

orion-equilibrium-service also needs a restart to pick up the new exclude label
(`scripts/safe_docker_build.sh orion-equilibrium-service up -d --build`). Each command
needs to run from a worktree on merged main, per the wrapper's policy. No SQL migration.

Proof after deploy:

```bash
# 1. bridge ticking (every 30 s; 'unmeasured' only if no bus RPC in 10 min)
docker logs --since 5m orion-athena-substrate-runtime 2>&1 | grep substrate_rpc_delivery_tick
# 2. probe on its own hop key
docker logs --since 10m orion-athena-equilibrium 2>&1 | grep -c 'LLMGatewayService#current_turn_probe'
```

```sql
-- 3. receipts written
SELECT created_at, receipt_json->'state_deltas'->0->'after'->'reading'
FROM substrate_reduction_receipts WHERE receipt_id LIKE 'receipt:rpc_delivery:%'
ORDER BY created_at DESC LIMIT 5;
-- 4. field node + capability moved (provenance names the source)
SELECT generated_at,
  field_json->'node_vectors'->'node:substrate.rpc_delivery'->>'rpc_timeout_pressure' AS node_val,
  field_json->'capability_vectors'->'capability:transport'->>'reliability_pressure' AS cap_val,
  field_json->'capability_provenance'->'capability:transport'->>'reliability_pressure' AS src
FROM substrate_field_state ORDER BY generated_at DESC LIMIT 5;
-- 5. after a day: fraction of ticks at 0 vs nonzero (expect mostly 0, some >0)
SELECT count(*) FILTER (WHERE (field_json->'capability_vectors'->'capability:transport'->>'reliability_pressure')::float = 0) AS zero,
       count(*) FILTER (WHERE (field_json->'capability_vectors'->'capability:transport'->>'reliability_pressure')::float > 0) AS nonzero,
       max((field_json->'capability_vectors'->'capability:transport'->>'reliability_pressure')::float)
FROM substrate_field_state WHERE generated_at > now() - interval '24 hours';
```

## Risks / concerns

- Severity: medium. Concern: orion-llm-gateway does not publish rpc-health, so its
  `gpu_pool:lease`/`state` RPC timeouts (~250 in 04:00-05:59 today, found in grammar
  and traced to the gateway's logs) are invisible to this bridge. cortex-exec's own
  gpu-pool lease client hop also never appears in its snapshots, so its bus instance
  may not be drained. Mitigation: follow-up to add the gateway as a publisher and to
  check cortex-exec's lease-client bus. Both are producer patches outside this bridge.
- Severity: low. Concern: a caller timeout mixes "not delivered" with "the work ran
  past the caller's deadline" (e.g. `orion:cortex:request` at 420 s). From Orion's
  side both mean "I asked and got no answer in time", which is what this channel
  claims, but it is not only network delivery. Mitigation: the receipt names the hop.
  `verb:` / `fcc:` long-work hops are excluded.
- Severity: medium. Concern: deploy order. If substrate-runtime and field-digester are
  rebuilt before cortex-exec, the unlabelled probe keeps the reading off 0 on about 63%
  of ticks (p50 0.07). Mitigation: the restart order below puts cortex-exec first.
- Severity: low. Concern: when no bus call happens for 10+ minutes, the channel expires
  to "not measured" after 120 s. The capability's `reliability_pressure` then falls back
  to `node:athena`'s input. That is honest, but a real outage that also stops every
  caller reads as unmeasured, not as 1.0.
- Severity: low. Concern: `capability:transport` `reliability_pressure` feeds
  `proposal_risk` (+0.10 at >= 0.5) and feedback credit (a drop counts as an
  improvement). Replayed readings never reached 0.5, so that bump only fires in a real
  multi-timeout outage. With the hysteresis, a drop is always the end of a 2+ timeout
  episode, never a single stray timeout ageing out.
- Severity: low. Concern: `node:substrate.rpc_delivery` is a new `node_vectors` entry.
  The review found that attention (an explicit domain map) and endogenous curiosity
  (which reads FalkorDB) do not pick it up. It reaches Orion through
  `capability:transport`.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
