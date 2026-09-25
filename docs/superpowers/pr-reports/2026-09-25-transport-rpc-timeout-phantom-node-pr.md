## Summary

- Orion's attention kept landing on a "node" called `rpc_timeout` that does not exist. Every time any service's request to another service timed out, the transport reducer misread the timeout record as a report from a bus machine named "rpc_timeout" and invented half-healthy readings for it. This patch stops that at the reader, without touching the shared bus client every service runs.
- Transport reducer: a timeout trace (`bus.transport:rpc_timeout:<id>`) is no longer parsed as a bus. A trace with zero real observer readings no longer writes a bus entry at all (that was where the invented 0.5 values came from). Any `bus:rpc_timeout` already saved is removed the next time the reducer runs.
- Field digester: `node:rpc_timeout` (the fake node) and `node:substrate.transport` (retired 2026-07-26, still sitting in live state) are dropped on every reconcile pass, and no incoming change can write to them again.
- Regression tests use the exact live timeout event (`f3510a9a-...`, 2026-09-22) and the exact live phantom projection entry, and also feed the real emitter's output through the reducer. That keeps the reader tied to whatever the emitter actually publishes.

## Outcome moved

A made-up node no longer competes for Orion's attention. Measured live before this patch (last 24h, `substrate_attention_frames`, up to 2026-09-25 00:15 UTC):

- `node:rpc_timeout` appears in 39,612 of 40,062 attention frames (99%) and is a dominant target in 960 of them (2.4%). That is down from 42% dominant on 2026-09-19, only because timeouts dropped to about 50-150 a day and the value decays between them. The source was still live.
- `substrate_transport_bus_projection` holds `bus:rpc_timeout` with `evidence_event_ids=[]`, `redis_ping_ok=null`, `delivery_confidence=0.5`, `reliability_pressure=0.5`, `stream_backlog_health=0.5`. All three 0.5 values are extractor defaults, not readings.
- The latest `substrate_field_state` still carries both `node:rpc_timeout` (reliability_pressure decayed to about 2e-300, last write 2026-09-24T20:02Z) and `node:substrate.transport` (prediction_error 0.0, last write 2026-07-26).

After deploy: none of the three exist, and they cannot come back.

## Current architecture

- `OrionBusAsync._emit_rpc_timeout_grammar` (shared bus client, every service) publishes an `rpc_transport_timeout` atom on trace `bus.transport:rpc_timeout:<corr>`, `source_service=orion-bus`.
- orion-substrate-runtime `_transport_tick` fetches every `bus.transport:`-prefixed `orion-bus` grammar event. `parse_bus_transport_trace_id` split on `:`, which made node_id `rpc_timeout`. The reducer skipped the atom (its role is not in `_ATOM_ROLES`) but still wrote `bus:rpc_timeout` from defaults and emitted a `transport_bus` state delta.
- orion-field-digester `delta_to_perturbations` turned that delta into `node:rpc_timeout` perturbations. Reconcile only drops whole entries for `RETIRED_LATTICE_NODES` (atlas), so the pseudo-node lived on and every generic `node_vectors` consumer iterated it.

### Consumers of these atoms / the `bus.transport:` prefix (checked before choosing the consumer-side fix)

| Consumer | Keys on | Affected by this patch? |
|---|---|---|
| orion-equilibrium-service `transport_metacog_gate.py` / `service.py:1472` | `semantic_role == "rpc_transport_timeout"` | No. It still fires the transport metacog trigger per timeout. |
| `orion/metacog/evidence_map.py` (`rpc_transport_timeout_grammar`) | trigger `evidence_source` | No |
| orion-heartbeat `substrate/routing.py` | `source_service == "orion-bus"` → site 3 | No (routes by service, not trace) |
| orion-sql-writer `grammar_truth.py`, substrate-runtime `store.py`/`grammar_truth.py`, hub `substrate_lattice_routes.py` | `bus.transport:` prefix for cursor/lane accounting | No. Events are still fetched and the cursor still advances. They now reduce as noop receipts instead of updates. |
| substrate-runtime transport reducer | trace_id parse | **Yes (the fix)** |

Emitter format left unchanged on purpose: changing it would force a rebuild across every service, and nothing above needs it changed.

### Where the honest timeout signal lands (existing-mechanism check)

`RpcHealthSnapshotV1` (success_count + timeout_count per window, a real rate with a denominator) is published on `orion:rpc_health:snapshot` by cortex-exec, cortex-orch and hub. It lands in:
- orion-signal-gateway → `rpc_transport_health` signals (`orion/signals/adapters/rpc_health.py`, level = success/(success+timeout)).
- orion-equilibrium-service → transport metacog trigger and EWMA baseline gate (`transport_baseline_gate.py`, `orion/metacog/transport_baseline.py`).

It does **not** reach the field or attention. The only signal→substrate bridge (`orion/substrate/signal_bridge.py`) supports three `(organ, kind)` pairs, none of them rpc_health, and its worker is never instantiated. The field's transport successor, `node:substrate.bus_synaptic`, measures inter-service bus timing gaps, not RPC timeouts. **Recommendation (not built):** if Orion should feel RPC timeouts, bridge the rpc_health success ratio into the field as a `prediction_signal`-style delta on a real node. It has to pass the metric quality gate first, and it must not count bare timeout atoms, because a count without a denominator is not a rate. No second timeout metric was built in this patch.

## Architecture touched

- `orion/substrate/transport_loop` (used by orion-substrate-runtime)
- `services/orion-field-digester` (reconcile + delta ingest)
- No bus channels, schemas, env keys or compose changes.

## Files changed

- `orion/substrate/transport_loop/constants.py`: `NON_BUS_TRANSPORT_NODE_IDS = {"rpc_timeout"}` with the incident rationale.
- `orion/substrate/transport_loop/extract.py`: `parse_bus_transport_trace_id` returns None for non-bus ids.
- `orion/substrate/transport_loop/reducer.py`: `_noop_receipt` helper. A trace with zero bus-observer evidence is a noop with a warning, never a defaults-only bus write.
- `orion/substrate/transport_loop/pipeline.py`: `prune_non_bus_entries` runs on every projection load, so persisted state heals itself with no SQL patch.
- `services/orion-field-digester/app/tensor/channels.py`: `RETIRED_PSEUDO_NODES` (reason per entry) and `PRUNED_NODE_IDS`.
- `services/orion-field-digester/app/tensor/reconcile.py`: drops `PRUNED_NODE_IDS` wholesale (extends the existing atlas prune path).
- `services/orion-field-digester/app/ingest/state_deltas.py`: `delta_to_perturbations` refuses `RETIRED_PSEUDO_NODES` (reconcile runs before perturbations, so a late delta could otherwise bring a node back for a tick).
- `orion/core/bus/async_service.py`: docstring only. It corrects the old claim that a distinct semantic_role kept this out of the transport reducer, and points to the exclusion set. No runtime change.
- `tests/test_transport_rpc_timeout_not_a_bus.py`, `services/orion-field-digester/tests/test_retired_pseudo_nodes.py`: regression tests (live shapes).
- `services/orion-field-digester/tests/test_reconcile_retired_and_single_observer.py`: off-lattice pruning tests used `node:rpc_timeout` as their example pseudo-node; retargeted to `node:substrate.chat`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `rpc_transport_timeout` traces now reduce to noop receipts (they used to be `transport_bus` create/update deltas). A zero-evidence bus trace is now a noop. The transport projection no longer contains `bus:rpc_timeout`. The field no longer contains `node:rpc_timeout` / `node:substrate.transport`.
- Compatibility notes: the `bus.transport:rpc_timeout:<corr>` trace format is unchanged, and so is every consumer listed above.

Blast radius checked: no code reads `bus:rpc_timeout`, `node:rpc_timeout` or `node:substrate.transport` by name (only comments). `substrate_node_prediction_error_baseline` has no rows for either node. `_PREDICTION_ERROR_DOMAIN_NODE_IDS` / `ACTIVE_INFERENCE_DOMAINS` already exclude `substrate.transport`. `endogenous_curiosity` and attention iterate `node_vectors` generically, so they simply stop seeing the nodes.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
tests/test_transport_rpc_timeout_not_a_bus.py + transport reducer/pipeline/schema + orion/core/bus/tests
  -> 20 passed (targeted run)
Pre-fix behaviour reproduced against the verbatim live event:
  BEFORE FIX buses: ['bus:rpc_timeout'] deltas: ['bus:rpc_timeout'] delivery_conf: 0.5
services/orion-field-digester/tests (excluding cwd-sensitive heartbeat_chassis, which passes from repo root)
  -> 230 passed
root tests/test_*transport*, test_attention_field*, test_field_*, orion/core/bus/tests
  -> 245 passed, 8 failed; the same 8 fail identically on origin/main (diffed)
services/orion-substrate-runtime/tests -> failures/errors identical before/after (diffed, pre-existing)
services/orion-equilibrium-service/tests/test_transport_metacog_gate.py -> 20 passed
orion/metacog/tests -> 225 passed
static gates (metric_lineage, definition_drift, inner_state_registry, stdlib_shadow,
  hostname_refs, sentience_instruments, system_health_producers, control_surface_parity) -> all 0
git diff --check -> clean
```

## Evals run

```text
No eval harness exists for the transport reducer or field reconcile. The live-shape
regression tests plus the post-deploy SQL checks below are the evidence. Follow-up:
a replay eval over one day of grammar_events asserting the projection never gains a
non-observer bus.
```

## Docker/build/smoke checks

```text
Not run. No compose/Dockerfile/requirements change. Deploy and live check are left for
Juniper (commands below). Runtime effect is UNVERIFIED until then.
```

## Review findings fixed

The review subagent read `git diff origin/main...HEAD` and found no blockers and no material code bugs. It confirmed three things. The zero-evidence guard cannot suppress a real observer reading, because the observer always emits `bus_observer_tick_completed` or `bus_observer_tick_failed`. Nothing reads the pruned ids by name or reads noop-vs-update receipt counts. The only paths that write to the field go through the filtered `delta_to_perturbations`.

- Finding (should): the reducer comment claimed the guard covers any split trace. A split *between* observer atoms still leaves a tail that has evidence but `redis_ping_ok=None`, which writes 0.5 over `bus:athena`. That bug predates this patch.
  - Fix: narrowed the comment and named the uncovered case as a follow-up (see Risks).
  - Evidence: `orion/substrate/transport_loop/reducer.py` comment above the guard.
- Finding (should): the persisted-phantom prune left no trace, so a deploy could not be confirmed from logs.
  - Fix: `transport_projection_pruned_non_bus keys=[...]` is now logged at INFO when entries are dropped. The prune needs `ENABLE_TRANSPORT_BUS_REDUCER=true`, which was confirmed live in `orion-athena-substrate-runtime`.
  - Evidence: `pipeline.py`. Post-deploy check 0 below.
- Finding (nit): the module-level import of `prune_non_bus_entries` made the whole test file fail at collection on pre-fix code.
  - Fix: moved the import into the one test that needs it.
  - Evidence: `tests/test_transport_rpc_timeout_not_a_bus.py`.
- Finding (nit): the `bus:<id>` set was rebuilt for every entry.
  - Fix: hoisted it to `NON_BUS_TRANSPORT_TARGET_IDS` in constants.
- Finding (nit): stale docs described `rpc_timeout` (and `atlas`) as physical host nodes.
  - Fix: updated `orion/attention/field_attention/selectors.py` (2 docstrings) and `orion/field/credit_integrity.py`.
- Re-run after the fixes: transport + attention selector + bus tests 115 passed. Field digester 230 passed. `git diff --check` clean.

## Restart required

Deploy from a clean worktree of merged `main` (not this feature worktree), so production is not pinned to a branch checkout:

```bash
git -C /mnt/scripts/Orion-Sapienform fetch origin
git -C /mnt/scripts/Orion-Sapienform worktree add ../Orion-Sapienform-deploy-2026-09-25 origin/main
cd ../Orion-Sapienform-deploy-2026-09-25
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
```

No data cleanup job is needed: both services prune their own persisted state on the first tick after deploy. No production SQL writes.

Post-deploy live check (after about 1 minute):

```bash
P='docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc'
# 0. self-heal fired (expect one line naming bus:rpc_timeout)
docker logs orion-athena-substrate-runtime 2>&1 | grep transport_projection_pruned_non_bus | head -3
# 1. projection has only real buses (expect: bus:athena, no bus:rpc_timeout)
$P "select jsonb_object_keys(projection_json->'buses') from substrate_transport_bus_projection"
# 2. latest field has neither phantom (expect: f|f)
$P "select field_json->'node_vectors' ? 'node:rpc_timeout', field_json->'node_vectors' ? 'node:substrate.transport' from substrate_field_state order by generated_at desc limit 1"
# 3. attention stopped seeing it (expect 0 once frames roll over)
$P "select count(*) from substrate_attention_frames where created_at > now() - interval '10 minutes' and frame_json::text like '%node:rpc_timeout%'"
# 4. timeouts are still consumed by the metacog trigger (they keep landing as noop receipts, not deltas)
$P "select count(*) from grammar_events where created_at > now() - interval '1 day' and trace_id like 'bus.transport:rpc_timeout:%'"
```

## Risks / concerns

- Severity: low
  - Concern: a real observer trace whose atoms arrive split across two reducer batches now drops the evidence-free half as a noop. Before, that half overwrote `bus:athena` with 0.5 defaults.
  - Mitigation: this is strictly more honest. The next full tick (about every 10s) writes the real reading.
- Severity: low
  - Concern: if the emitter's trace segment is ever renamed away from `rpc_timeout`, the phantom comes back under a new name.
  - Mitigation: `test_whatever_the_real_emitter_publishes_is_not_a_bus` runs the real emitter's output through the reducer and fails on a rename. The emitter docstring now points at `NON_BUS_TRANSPORT_NODE_IDS`.
- Severity: low (pre-existing, not introduced here)
  - Concern: a reducer batch split between two observer atoms leaves a tail with evidence but `redis_ping_ok=None`, which still writes 0.5 defaults over `bus:athena` for one tick.
  - Mitigation: follow-up, either carry forward the prior `redis_ping_ok` when the trace has no `bus_health_observed`, or noop when `redis_ping_ok is None and observer_failure_count == 0`.
- Severity: info
  - Concern: RPC timeouts still do not reach Orion's field/attention through any honest channel.
  - Mitigation: bridge recommended above (rpc_health ratio, gated). Not built.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2323

🤖 Generated with [Claude Code](https://claude.com/claude-code)
