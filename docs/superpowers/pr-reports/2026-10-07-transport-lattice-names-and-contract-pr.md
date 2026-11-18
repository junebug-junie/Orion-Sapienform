## Summary

- Every transport-lattice policy row now says which real reading it is (`source:`). The Hub reads values only through it, and its hand-kept name table is gone. A new static CI gate refuses a row that points at nothing.
- Kept the row id `bus_synaptic_pressure`. It is also the Redis key for 160 hours of learned thresholds, and renaming it would reset them for a day. Its declared source is `capability:transport.pressure`.
- Fixed the mind recall line. It printed a raw fraction next to thresholds on a different scale (×0.85); both are now on the same scale.
- Retired the bus observer's two-stream schema check (`contract_pressure`) end to end: producer, atom, reducer fields, node-level field channel, policy row, Hub contract gate, relational adapter and incident log.
- Kept `capability:transport.contract_pressure`. It is really catalog drift under a misleading name, and removing it would change attention on about 2.4% of ticks. Options are written up for Juniper (decision D3).

## Outcome moved

- One address per transport signal. Drift between the policy label and the field metric now fails CI.
- A metric that read 0.0 on 123,412 of 123,412 ticks no longer exists.
- Orion's recall sentence now states thresholds on the same scale as the value it reports.

## Current architecture

- The field called the signal `pressure` and the policy called it `bus_synaptic_pressure`. The Hub (`_CHANNEL_VALUE_SOURCES`) and orion-mind each kept their own mapping between the two names.
- The bus observer sampled 5 entries from each of 2 world_pulse streams and validated them, producing per-bus `contract_pressure` on the reducer and on `node:athena`.
- The Hub's "contract" gate actually read `capability:transport.contract_pressure`, which the topology fills from catalog drift.

## Architecture touched

orion-bus (bus observer), the transport reducer and `TransportBusStateV1`, orion-field-digester (channels, decay, ingest, topology), orion-hub (lattice routes and JS), orion-mind (recall resolver), the relational adapter, the substrate-runtime incident log, the glossary and the metric lock.

## Files changed

- `config/substrate-lattice/transport_lattice_policy.v1.yaml`: `source:` per row; `contract_pressure` row deleted.
- `services/orion-hub/scripts/substrate_lattice_routes.py`: resolves values through `source`, never a guessed key; contract gate deleted; the pressure gate's observer half no longer depends on the bus row.
- `services/orion-hub/static/js/substrate-lattice.js`: M3 card no longer shows `contract_pressure`.
- `services/orion-mind/app/recall_signal_resolver.py`: threshold ladder converted to the scale of the stated value.
- `tests/test_transport_lattice_policy_sources.py` (new) and `.github/workflows/orion-static-gates.yml`: static gate.
- `services/orion-bus/app/{bus_observer,grammar_emit,settings}.py`, `.env_example`, `docker-compose.yml`, docs: schema sample removed.
- `orion/grammar/atom_signals.py`: unused helper removed.
- `orion/schemas/transport_projection.py`: two fields retired and dropped on read.
- `orion/substrate/transport_loop/{extract,reducer}.py`: retired atom ignored; hint removed.
- `services/orion-field-digester/app/{tensor/channels,digestion/decay,ingest/state_deltas}.py` and `config/field/orion_field_topology.v1.yaml`: node-level channel retired and pruned.
- `orion/substrate/relational/adapters/transport_ctx.py`, `services/orion-substrate-runtime/app/worker.py`: field removed from salience and incident log.
- `config/field/field_channel_glossary.v1.yaml` and `config/metrics/metric_definitions.lock.json`: honest capability-only meaning, re-locked.
- Tests across the touched services; spec at `docs/superpowers/specs/2026-10-07-transport-lattice-names-and-contract.md`.

## Schema / bus / API changes

- Added: `source:` block on lattice policy rows.
- Removed:
  - `TransportBusStateV1.contract_pressure` and `.schema_mismatch_stream_count`.
  - The `bus_schema_validation_failed` atom.
  - Node-level field channel `contract_pressure`.
  - Hub gate `contract`.
- Renamed: none.
- Behavior changed:
  - The mind recall line's threshold numbers are now ÷0.85; the decision to render the line is unchanged.
  - Hub `value_source` comes from the policy.
- Compatibility notes:
  - Persisted rows carrying the retired fields load and drop them (verified against the live row).
  - Old atoms still in the backlog are ignored.
  - Deploy order is safe in either direction.

## Env/config changes

- Added keys: none.
- Removed keys: `BUS_OBSERVER_SCHEMA_SAMPLE_COUNT` (orion-bus).
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran; the primary checkout's `services/orion-bus/.env` never had the key.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-field-digester/tests: 275 passed, 6 skipped
services/orion-bus/tests (PYTHONPATH=services/orion-bus:.): 36 passed
hub test_substrate_lattice_routes + hub_tab + tests/test_transport_lattice_policy_sources.py: 76 passed
orion-mind test_recall_signal_resolver: 24 passed
root transport/field/proposal/relational/atom_signals suites: 174+19 passed
substrate-runtime transport tests: 18 passed
Pre-existing failures, identical on main: test_transport_perturbations_diffuse_to_transport_capability,
  test_reasoning_load_diffuses_to_orchestration, hub glossary 52-vs-39 count, substrate-runtime cursor tests (need local Postgres)
```

## Evals run

```text
Replay: 2,044 real bus-observer traces (last 6 h of grammar_events) through the main and branch reducers:
  0 differences in catalog_drift/observer_failure/reliability; old contract_pressure non-zero on 0 traces.
Metric gate, live data:
  - 72 h: 24,570 ticks, 0 schema-failure atoms; field node:athena.contract_pressure 0.0 on 123,412/123,412.
  - 120 s mesh-wide PSUBSCRIBE: 9,498 cataloged messages, 0 mismatches.
  - 72 h container logs: 0 publish-side rejects.
  - All 172 retained world_pulse entries valid.
Attention impact of the kept capability channel: it wins capability:transport's proxy on 314/13,134 sampled ticks (2.4%).
```

## Docker/build/smoke checks

```text
Not built or deployed (deploy is Juniper's call).
check_substrate_projection_schema_drift against live Postgres: OK (transport projection validates with the new schema).
check_env_template_parity: PASS.
check_definition_drift --gate: PASS.
check_metric_lineage --gate: PASS.
```

## Review findings fixed

- Finding: the pressure gate went "unknown" as a whole when the bus row lacked a source, which dropped the independent observer check.
  - Fix: the observer half is always evaluated.
  - Evidence: an extended hub test asserts "watch" on the observer alone.
- Finding: the glossary filter `"node" not in e` was a no-op.
  - Fix: filter on `e.get("node")`.
  - Evidence: policy-source tests pass.
- Finding: the gate mutation test counted problems only.
  - Fix: assert exactly one problem per bad row.
- Finding: the mind scale used `.get(...) or 1.0`, a silent fallback.
  - Fix: direct index.
- Finding: a stale Hub comment.
  - Fix: updated.
- Finding (noted, not changed): the new assertions in `test_transport_perturbations_diffuse_to_transport_capability` never run, because that test already fails on main.
  - Coverage: the same behavior is pinned by the reconcile test and the ratchet test.

## Restart required

Not deployed. Order: no migrations. Build bus first, then the readers:

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-bus up -d --build bus-observer
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: `capability:transport.contract_pressure` is still a misleading name for catalog drift.
  - Mitigation: documented in the glossary; decision D3 recommends renaming it to `catalog_drift_pressure` at capability level (same values).
- Severity: low.
  - Concern: `BUS_OBSERVER_STREAMS` now feeds only a dormant catalog-drift fallback.
  - Mitigation: flagged as the next instrument to judge.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2532

🤖 Generated with [Claude Code](https://claude.com/claude-code)
