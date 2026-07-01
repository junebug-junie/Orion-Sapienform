# PR: substrate self-modeling loop — rung-1 runtime bridge + rung-2 felt-state lanes (activated)

> Committed to the branch as the PR body because this checkout has no GitHub auth
> (no `gh` login / `GH_TOKEN`). Open the PR at:
> https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/substrate-rung1-bridge-rung2-lanes
> and paste the body below.

Branch: `feat/substrate-rung1-bridge-rung2-lanes`

---

## Summary

Finishes the two remaining edges of the substrate self-modeling loop from
`docs/plans/substrate/CONTINUATION.md`, and wires them so they actually run on the
Athena host:

1. **Rung 1 runtime bridge** — the runtime worker now writes the execution/transport
   `prediction_error` it computes onto a durable substrate node, not just a
   field-digester receipt, so the dynamics engine has a node to seed pressure from.
2. **Rung 2 remainder** — binds the substrate's own "felt state" (biometrics
   pressure, in-flight execution, transport-bus health) into the
   CognitiveUnificationLayer as three belief lanes, and adds the ctx-hydration step
   that actually populates them at chat-stance time (which also lights up the
   previously-dormant `self_state` lane shipped in `d9dbe624`).

Honest framing (unchanged from the ladder doc): this implements the *functional*
properties — predictive feedback, integration/higher-order self-model — as thin,
inspectable, tested seams. No claim about phenomenal experience.

## What this closes

```
prediction error (worker)
  → durable substrate node metadata['prediction_error']        (rung 1, this PR)
  → [dynamics engine seeds pressure from it]                   (consumer: follow-up)

reducer projections (biometrics/execution/transport) + self_state
  → hydrated into chat-stance ctx                              (rung 2 ctx wire, this PR)
  → felt-state belief lanes in the unified belief set          (rung 2 adapters, this PR)
```

## Changes

| Rung | Change | Files |
|------|--------|-------|
| 1 | On `error>0` in `_execution_tick`/`_transport_tick`, upsert a `ConceptNodeV1` (`node:substrate.execution` / `.transport`, anchor `orion`) carrying `metadata['prediction_error']` + fresh `observed_at`, under a fixed identity_key so re-writes collapse. Field-digester receipt untouched. Default-off (`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`), fail-open. | `services/orion-substrate-runtime/app/worker.py` (+ test) |
| 2a | Three ctx-sourced adapters (biometrics/execution/transport) mirroring `self_state_ctx`: pure, degrade-to-None, capped at 20. Registry 9 → 12 producers. | `orion/substrate/relational/adapters/{biometrics,execution,transport}_ctx.py`, `orion/cognition/projection_builder.py` (+ tests; repairs the pre-existing-red registry test that `d9dbe624` left stale) |
| 2b | `SubstrateFeltStateReader` loads the latest projection from the four substrate Postgres tables into chat-stance ctx before the unifier runs, at the single common caller `build_chat_stance_inputs`. Default-off (`ENABLE_SUBSTRATE_FELT_STATE_CTX`), fail-open, freshness-gated, non-clobbering. | `services/orion-cortex-exec/app/substrate_felt_state_reader.py`, `services/orion-cortex-exec/app/chat_stance.py` (+ test) |
| env | Feature flags + supporting config enabled in `.env_example` (and copied to local `.env`) so the loop runs on Athena. | `services/orion-cortex-exec/.env_example`, `services/orion-substrate-runtime/.env_example` |

## Deployment / how it runs

**cortex-exec** (rung 2 — fully live; reads projections from the `conjourney` Postgres):
```
ENABLE_SUBSTRATE_FELT_STATE_CTX=true
SUBSTRATE_FELT_STATE_DATABASE_URL=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
SUBSTRATE_FELT_STATE_MAX_AGE_SEC=120
```
The execution/transport reducers are already enabled on the runtime, so their
projections exist; the biometrics reducer is always on. Enabling the flag activates
all four belief lanes.

**orion-substrate-runtime** (rung 1 — write path):
```
SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true
SUBSTRATE_STORE_BACKEND=sparql
SUBSTRATE_GRAPH_QUERY_URL=http://orion-athena-fuseki:3030/orion/query
SUBSTRATE_GRAPH_UPDATE_URL=http://orion-athena-fuseki:3030/orion/update
SUBSTRATE_GRAPH_URI=http://conjourney.net/graph/substrate
SUBSTRATE_GRAPH_USER=admin
SUBSTRATE_GRAPH_PASS=orion
```

## Known gap (honest)

**Rung 1's consumer is not wired yet.** No periodic `SubstrateDynamicsEngine.tick()`
runs against the shared store in any service today (verified by grep). So with the
config above, surprise is *persisted* to Fuseki but not yet *seeded into pressure*.
Closing that is a small follow-up (a bounded dynamics tick in the runtime worker or a
dedicated loop) and is rung-3 adjacent. Rung 2 is fully live and end-to-end.

## Safety / governance

- Both features default-off in code; enabled only via env. Flag off ⇒ byte-identical
  to prior behavior.
- Fail-open everywhere: store init, upserts, and each hydration lane are wrapped so a
  tick / chat turn never crashes; one lane can't break another.
- Freshness-gated: stale projections past `MAX_AGE_SEC` are excluded, not felt as
  current (Forge: stale excluded by default).
- Non-clobbering: hydration never overwrites a pre-existing ctx key.
- Bounded: fixed identity_key (no unbounded node growth); every adapter collection
  capped at 20; single latest row per lane + TTL cache.
- Not touched: the dangerous endogenous-agency gate (rung 5) — not in this PR.

## Tests

All green:
- `services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py` — flag on/off, fail-open (3).
- `orion/substrate/relational/tests/test_reducer_lane_adapters.py` — per-adapter label/anchor/salience/metadata, dict+json, absent/garbage → None, cap ≤ 20, quiet-node exclusion, registry = 12 (10).
- `orion/cognition/tests/test_projection_builder.py` — registry order incl. the 4 felt-state lanes (repaired) (2).
- `services/orion-cortex-exec/tests/test_substrate_felt_state_reader.py` — fresh→injected, stale→skipped, existing-key preserved, disabled no-op, one-lane fail-open, flag-unset entrypoint no-op (6).
- Neighbors: `test_self_state_adapter`, `test_golden_path`, `test_projection_starvation_diagnostics`, `test_chat_stance_shared_spine`, `test_cognitive_substrate_phase4_dynamics` — unchanged, green.

## Commits

- `a9d0d9c9` — rung 1 runtime bridge
- `01e35c92` — rung 2 remainder (felt-state adapters + registry)
- `a3c27eb5` — rung 2 ctx hydration wire

## Reviewer notes

- The repo is concurrently mutated by an automation user; work was done on this
  branch off a clean `main`.
- Design choice: the rung-2 lanes are **ctx-sourced**, not DB-pull adapters — the
  hydration reader (cortex-exec, service-local DB access) injects the projections,
  keeping the adapters pure and matching the shipped `self_state` pattern. No
  importable substrate-projection reader was invented in the `orion` package.
