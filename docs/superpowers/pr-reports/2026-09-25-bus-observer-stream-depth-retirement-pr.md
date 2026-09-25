## Summary

- Orion's bus observer used to measure "how backed up is the queue" by counting entries (`XLEN`) in two Redis Streams that belong to one periodic news job (world_pulse). A live scan of the whole bus found only 5 Redis Streams exist, and the one with a live consumer has nothing waiting (lag 0, pending 0). The dead-letter stream it also watched does not exist at all.
- Decision: **retire, don't widen.** Widening to every stream with a consumer group would measure one healthy queue plus one abandoned for 61 days, so it would read "calm" by construction too.
- Removed end to end: the depth and backpressure atoms, `BUS_STREAM_DEPTH_WARNING/CRITICAL`, eight `TransportBusStateV1` fields, five derived pressures, three field channels, a lattice edge that carried nothing, and every reader (hub card, relational adapter, consolidation motif, coherence rule, polarity sets, glossary, decay lists).
- Two of the five "stream" signals (`stream_backlog_health`, `delivery_confidence`) turned out not to read any stream. They were the observer's own Redis `PING`, and that could never report a failure: the atom saying "PING failed" goes out over the same Redis. `reliability_pressure` keeps exactly the same values; the ping input is now folded in directly.
- Migration safety: saved projections from before this change still load, and field reconcile removes the retired channel names from every saved vector, including a stale `transport_pressure` left behind by a July rename.

## Outcome moved

Failure mode removed: three field channels that were flat by construction (0.0016, 1.0, 1.0 on every one of 24,633 ticks) no longer sit in the field vectors, where attention, coherence and consolidation could read them as a real calm bus. Where transport health lives now: `capability:transport.pressure` (bus_synaptic), `catalog_drift_pressure` (mesh-wide census), `observer_failure_pressure` / `reliability_pressure`, and RPC health.

## Live stream inventory (read-only, `ORION_BUS_URL`, 2026-09-25 ~06:00Z)

| Stream | XLEN | Last write | Group | Consumers | Pending | Lag |
|---|---|---|---|---|---|---|
| `orion:stream:world_pulse:run:result` | 160 | ~18 h | `cg:concept-induction` | 1 | 0 | 0 |
| `orion:queue:spark:introspection` | 22,165 | ~61 days | `spark-introspector-workers` | 343 | 0 | 0 |
| `orion:pad:events` | 500 | ~59 days | none | | | |
| `orion:pad:frames` | 501 | ~59 days | none | | | |
| `orion:curiosity:help:request` | 1 | ~10 days | none | | | |

`orion:stream:world_pulse:run:result:dlq`: key does not exist.

## Metric quality gate (CLAUDE.md 0A)

No metric added or widened. One kept with a changed input path, `reliability_pressure`:

1. Provenance: `orion/substrate/transport_loop/extract.py::compute_transport_pressures`, `max(observer_failure_pressure, ping_pressure)`; `ping_pressure` comes from `bus_health_observed`'s `redis_ping_ok` (`services/orion-bus/app/grammar_emit.py::record_health_observed`).
2. Independence: same inputs as before (observer failure + PING); no new causal chain. The retired `delivery_confidence` was exactly `1 - reliability_pressure`, so it was redundant, not independent.
3. Theory anchor: unchanged. It is the observer's own liveness, not bus health.
4. Live sanity: values unchanged by construction (parametrized equivalence test over all 6 ping/failure cases). Its non-calm states (observer failure) are real and reachable. The retired `PING` pair failed this step: a failed `PING` can't be delivered. The retired depth trio failed it too: 157-160 entries against a 100,000 threshold.
5. Existing mechanism: the successors already exist (bus_synaptic, census, RPC health). Nothing new was built.
6. Reversibility: cheap. Pure deletion plus a load-time key drop.

## Current architecture

`bus-observer` (orion-bus) ran PING + `XLEN` on `BUS_OBSERVER_STREAMS` every 10 s and emitted grammar atoms. `transport_bus_reducer` (orion-substrate-runtime) turned them into `TransportBusStateV1` pressures. The field digester turned `pressure_hints` into `node:athena` channels and diffused `stream_backlog_pressure` into `capability:orchestration.pressure`. It also had a `capability:transport -> capability:orchestration` edge on a channel nothing wrote.

## Architecture touched

orion-bus (producer), `orion/substrate/transport_loop` (reducer), orion-substrate-runtime (worker, settings), orion-field-digester (channels, reconcile, decay, ingest), `config/field/*` lattice + glossary, `orion/field`, `orion/attention`, `orion/consolidation`, relational adapter, orion-hub lattice card.

## Files changed

- `services/orion-bus/app/bus_observer.py`, `grammar_emit.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, `SERVICE_PORTS.yaml`, `SUBSTRATE_TRACE_MAP.md`, `AGENT_CONTEXT.md`: producer side removed; `BUS_OBSERVER_STREAMS` now documented as catalog/schema-sample only.
- `orion/grammar/atom_signals.py`: `uncertainty_from_backpressure` removed.
- `orion/schemas/transport_projection.py`: 8 fields removed; `mode="before"` validator drops exactly those names so persisted rows load under `extra="forbid"`.
- `orion/substrate/transport_loop/{extract,reducer,pipeline,constants}.py`: family removed; retired roles explicitly ignored; `reliability_pressure` unchanged.
- `services/orion-substrate-runtime/app/{worker,settings}.py`, `.env_example`, `docker-compose.yml`: `BUS_STREAM_DEPTH_CRITICAL` removed; `backpressure` out of the incident-log set.
- `services/orion-field-digester/app/tensor/{channels,reconcile}.py`: channels retired (`None` = no successor); new `RETIRED_CAPABILITY_CHANNELS` prune; single-observer map now empty (mechanism kept, tested with a synthetic channel).
- `services/orion-field-digester/app/{ingest/state_deltas,digestion/decay,digestion/diffusion}.py`: hints no longer injected; decay lists trimmed; docstring examples updated.
- `config/field/orion_field_topology.v1.yaml`, `biometrics_lattice.yaml`: channels removed; `stream_backlog_pressure -> pressure` mapping and the dead transport->orchestration edge deleted.
- `config/field/field_channel_glossary.v1.yaml`: 3 entries removed.
- `orion/field/pressure.py`, `orion/attention/field_attention/selectors.py`: polarity sets trimmed (kept in sync).
- `orion/field_coherence.py`: rule on the retired pair removed.
- `orion/consolidation/motif.py`, `config/consolidation/consolidation_policy.v1.yaml`: `transport_healthy_idle` keyed on `max_pressure` (`capability:transport.pressure`); unread `min_stream_backlog_health` removed.
- `orion/substrate/relational/adapters/transport_ctx.py`: salience/confidence from `reliability_pressure`/`contract_pressure` (confidence value identical).
- `services/orion-hub/static/js/substrate-lattice.js`, `scripts/substrate_lattice_routes.py`: M3 card shows `reliability_pressure`/`redis_ping_ok`.
- `scripts/smoke_orion_bus_transport_full_stack.sh`: reads `reliability_pressure`.
- `config/metrics/metric_definitions.lock.json`: re-locked (3 removed URNs).
- Tests across all of the above; `docs/superpowers/specs/2026-09-25-bus-observer-stream-depth-retirement.md`.

## Schema / bus / API changes

- Added: none.
- Removed: grammar roles `bus_stream_depth_observed`, `bus_backpressure_observed`. `TransportBusStateV1.{total_stream_depth, max_stream_depth, backpressure_count, stream_backlog_health, delivery_confidence, stream_depth_pressure, backpressure, stream_backlog_pressure}`. Field channels `stream_backlog_pressure` (node + capability), `stream_backlog_health`, `delivery_confidence`.
- Renamed: consolidation condition `max_stream_backlog_pressure` -> `max_pressure`.
- Behavior changed: `capability:orchestration.pressure` loses a 0.0016 input from `node:athena` (the max with `cpu_pressure` dominates, so this has no practical effect).
- Compatibility notes: no channel or registry change (grammar atoms ride `orion:grammar:event`). Persisted projection rows with retired keys still load (verified against live Postgres). Pre-deploy receipts carrying retired hints produce no perturbation.

## Env/config changes

- Added keys: none.
- Removed keys: `BUS_STREAM_DEPTH_WARNING`, `BUS_STREAM_DEPTH_CRITICAL` (orion-bus); `BUS_STREAM_DEPTH_CRITICAL` (orion-substrate-runtime).
- Renamed keys: none.
- `.env_example` updated: yes (both services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran (it adds, never deletes); the three retired keys were removed by hand from the primary checkout's `services/orion-bus/.env` and `services/orion-substrate-runtime/.env`.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest tests/test_transport_substrate_reducer.py tests/test_transport_projection_schemas.py tests/test_transport_rpc_timeout_not_a_bus.py tests/test_field_channel_glossary.py tests/test_attention_field_selectors.py tests/test_consolidation_transport_motifs.py orion/grammar/tests orion/attention/tension/tests orion/substrate/relational/tests ...  -> green except 15 pre-existing failures, identical set on main (test_dispatch_starvation x11, test_consolidation_policy_loader x4)
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_* services/orion-field-digester/tests -> 251 passed, 5 failed (identical 5 fail on main)
services/orion-bus: pytest tests -> 49 passed
services/orion-substrate-runtime: failure set diffed vs main -> identical apart from the rewritten incident-log test (now passing)
services/orion-hub: lattice/glossary/mood-arc route tests -> 1 failed, identical on main
node --check substrate-lattice.js, bash -n smoke script, YAML parse -> ok
```

## Evals run

```text
No eval harness exists for the transport reducer or the bus observer; this is a retirement, not new behavior.
Substitute live check: scripts/check_substrate_projection_schema_drift.py against live Postgres -> OK (live pre-retirement row validates).
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-bus/docker-compose.yml config            -> exit 0, no BUS_STREAM_DEPTH_*
docker compose ... -f services/orion-substrate-runtime/docker-compose.yml config -> exit 0, no BUS_STREAM_DEPTH_*
Static gates: check_metric_lineage --gate PASS, check_definition_drift --gate PASS (after --update), check_env_template_parity PASS,
check_inner_state_registry OK, check_sentience_instruments --static-only OK, check_service_hostname_refs OK, check_compose_no_relative_mounts PASS
No deploy performed.
```

## Review findings fixed

Review (orion-repo-agent subagent, full diff vs origin/main): no critical or material findings. It independently confirmed the value equivalence for `reliability_pressure` and adapter confidence, the migration path (nested load, JSON load, constructor, and non-retired keys still rejected), reconcile running before perturbation, and env parity. Four minor findings, all fixed:

- Finding: the retired-field drop runs on every construction, so a live writer of retired keys would go unnoticed, and two fixtures passed retired keys without testing anything.
  - Fix: the validator now logs `transport_bus_state_retired_fields_dropped fields=[...]` (expected once per old row; repeated = a live writer). Retired keys removed from the zero-evidence fixture. The verbatim live-phantom fixture keeps them on purpose.
  - Evidence: `tests/test_transport_rpc_timeout_not_a_bus.py`, `tests/test_transport_projection_schemas.py` green.
- Finding: the tension baselines (`tension_baseline_mu/var/n`, keyed `node\x1fchannel`) kept entries for retired channels forever.
  - Fix: reconcile's `_prune_retired_tension_baselines` drops keys whose channel is in `RETIRED_NODE_CHANNELS`.
  - Evidence: `test_retired_channels_are_pruned_from_tension_baselines`.
- Finding: the claim that adapter confidence is unchanged was only tested in the healthy case, using a fixture value that can't occur.
  - Fix: parametrized test over reducer-produced states. Expected 1.0 / 0.5 / 0.7 / 0.7, green.
  - Side note, pre-existing: a bus whose ping failed reports confidence 0.7, higher than an unknown bus at 0.5. That is the `or 0.7` fallback, not caused by this PR, and worth a follow-up.
- Finding: two generic diffusion tests used the deleted transport->orchestration `stream_backlog_pressure` edge as their fixture.
  - Fix: fixture channel renamed to `synthetic_cap_pressure` in `tests/test_causal_geometry_report.py` and `tests/test_field_digestion_rules.py`.
- Reviewer UNVERIFIED (permission-denied read): live field-state row. I checked it directly earlier this session: `node:athena` held 1.0/1.0/0.0016, and both capability vectors held `stream_backlog_pressure=0.0` and a stale `transport_pressure=0.0`.

## Restart required

Deploy order: readers first, then the reducer, then the producer. New reader code handles old rows. Old reader code would fill the missing fields with their 0.5 defaults once the reducer rewrites the row, so readers must be rebuilt first.

```bash
# readers of the transport projection / field channels (shared orion/ package)
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-consolidation-runtime up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# reducer, then producer
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-bus up -d --build   # bus-observer sidecar
```

Live checks after deploy:

```bash
# 1. no retired atoms after the bus-observer restart (expect 0)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*) from grammar_events where source_service='orion-bus' and created_at > now() - interval '5 minutes' and event_json->'atom'->>'semantic_role' in ('bus_stream_depth_observed','bus_backpressure_observed')"
# 2. projection row lost the retired keys (expect f)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select projection_json->'buses'->'bus:athena' ? 'stream_backlog_pressure' from substrate_transport_bus_projection"
# 3. field vectors pruned (expect empty / nulls)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select field_json->'node_vectors'->'node:athena'->'stream_backlog_health', field_json->'capability_vectors'->'capability:orchestration'->'transport_pressure' from substrate_field_state order by generated_at desc limit 1"
```

## Risks / concerns

- Severity: low. Concern: `BUS_OBSERVER_STREAMS` still drives `contract_pressure` via a schema sample of the same two world_pulse keys (one of which doesn't exist), so that signal is equally narrow. Mitigation: flagged in the spec as the next instrument to judge; not changed here.
- Severity: low. Concern: overlap with sibling branches. `chore/transport-lattice-semantics` (topology edge, motifs, proposal templates): this PR deletes the dead transport->orchestration edge and rekeys `transport_healthy_idle`, so rebase onto this. `watch_transport_backpressure` proposal template (name only, empty dimensions) is left for that branch. `fix/transport-split-batch-fake-health` touches the same `extract.py`/`reducer.py`: its "split between atoms writes 0.5 defaults" concern now only affects `reliability_pressure`. Mitigation: announced on the agent board; neither sibling had commits when this was cut.
- Severity: low. Concern: constructing `TransportBusStateV1` in code with a retired kwarg is silently dropped, not rejected. Mitigation: the drop list is closed (8 names); any other unknown key still raises (tested).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2341

🤖 Generated with [Claude Code](https://claude.com/claude-code)
