# orion-substrate-runtime

Event-native substrate worker for the biometrics closed loop:

```text
grammar_events (orion-biometrics) → node biometrics projection
  → biometrics_pressure organ → pressure reducer → active pressure projection

grammar_events (orion-cortex-exec, cortex.exec:*) → execution trajectory projection
  → execution_trajectory_reducer → StateDeltaV1(target_kind=execution_run)
  → substrate_reduction_receipts → orion-field-digester

grammar_events (orion-bus, bus.transport:*) → transport bus projection
  → transport_bus_reducer → StateDeltaV1(target_kind=transport_bus)
  → substrate_reduction_receipts → orion-field-digester (when ENABLE_TRANSPORT_FIELD_DIGESTION=true)

grammar_events (orion-cortex-orch, orch.route:*) → route arbitration projection
  → route_grammar_reducer → StateDeltaV1(target_kind=route_arbitration_run)
  → substrate_reduction_receipts
  (default on: PUBLISH_CORTEX_ORCH_GRAMMAR=true on orion-cortex-orch AND
  ENABLE_ROUTE_GRAMMAR_REDUCER=true here, both true by default now.
  manual_migration_route_substrate_loop.sql must be applied first.)
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_transport_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_route_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reducer_quarantine_v1.sql
# Self-observability v2 (coalition dwell log + endogenous curiosity candidates):
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_coalition_dwell_v1.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_curiosity_candidates_v1.sql
cp services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/.env
python scripts/sync_local_env_from_example.py orion-substrate-runtime
```

`POSTGRES_URI` must reach the **conjourney** database (`orion-athena-sql-db:5432`). A stale `orion:orion@postgres:5432/orion` DSN breaks `/grammar/truth` and all reducers.

Set `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true` after cortex-exec grammar publish is enabled (`PUBLISH_CORTEX_EXEC_GRAMMAR=true` on orion-cortex-exec). Checked-in `.env_example` default is `true`.

Set `ENABLE_TRANSPORT_BUS_REDUCER=true` after orion-bus transport traces are publishing (`PUBLISH_ORION_BUS_GRAMMAR=true`). Checked-in `.env_example` default is `true`.

`ENABLE_ROUTE_GRAMMAR_REDUCER` and orch's `PUBLISH_CORTEX_ORCH_GRAMMAR` both default `true` now (graduated out of shadow mode, matching the `chat_grammar`/`execution_trajectory` precedent in commit `044d5318`). **`manual_migration_route_substrate_loop.sql` must be applied before this reducer can write** -- it will error on every tick against a fresh DB until `substrate_route_arbitration_projection` exists. Projection (`active_route_arbitration`) is capped the same way `active_execution_trajectory` is (`ROUTE_ARBITRATION_MAX_RUNS=2000`, `ROUTE_ARBITRATION_MAX_AGE_SEC=86400`, LRU by `last_updated_at`) -- not settings-configurable yet, unlike execution's cap; revisit if this lane needs a different cap once it's run at real volume.

## Health monitoring -> hub pending-attention box

`GET /grammar/truth`'s `degraded`/`degraded_reasons` used to be manual-curl-only. `app/health_monitor.py::HealthMonitor` (mirrors `orion-self-state-runtime`'s pattern) polls it every `SUBSTRATE_RUNTIME_HEALTH_CHECK_INTERVAL_SEC` (default `900.0`) and fires an `orion-notify` attention request -- which surfaces as a card in orion-hub's pending-attention UI -- on a healthy->unhealthy transition only (not every tick), plus a recovery note on the way back. Requires `NOTIFY_BASE_URL` (default `http://orion-athena-notify:7140`) to actually reach `orion-notify`; fails open (logs and retries next tick) if unreachable.

## Run

```bash
cd services/orion-substrate-runtime
docker compose up --build
```

Health: `GET http://localhost:8115/health`

## Idempotent committed deltas

`ReductionReceiptV1.receipt_id` and `StateDeltaV1.delta_id` are derived from reducer inputs via `orion.substrate.ids` so replay/retry does not mint duplicate committed identities. Postgres inserts use `ON CONFLICT (receipt_id) DO NOTHING`.

Pressure receipts also include `emission_id` in the receipt preimage (organ still mints a fresh `emission_id` per invocation). Field digester dedupe should key on **`delta_id`** (event-scoped), not `receipt_id`, when comparing across emissions.

Hub debug (node-scoped lineage):

- `GET /api/substrate/biometrics-node/{node_id}/latest`
- `GET /api/substrate/receipts/{receipt_id}`

### Projection debug reads (this service, internal)

Same response contract on all three: `{"ok": false, "reason": "no_projection"}` if the reducer hasn't written yet, `{"ok": true, "projection": {...}}` otherwise.

- `GET /projections/execution_trajectory` — `active_execution_trajectory`
- `GET /projections/chat_session` — `active_chat_session`
- `GET /projections/route_arbitration` — `active_route_arbitration`

## Grammar production observe

Truth: `GET http://localhost:8115/grammar/truth`

Includes per-reducer health (`reducer_health_by_name`), pending backlog, and stream lag
(cursor vs latest grammar event). Execution and transport reducers run on independent poll
loops with configurable batch limits so transport catch-up is not starved by biometrics.

Post-deploy smoke (from repo root):

```bash
./scripts/grammar_production_truth.sh
```

### Cursor recovery (internal operator endpoint)

**Not exposed via hub/Caddy.** Requires `SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN` and header `X-Orion-Operator-Token`.

See also: [docs/grammar_production_observe_deploy.md](../../docs/grammar_production_observe_deploy.md)

```bash
# Replay from earliest matching grammar event
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=earliest'

# Jump to newest row (operator acknowledges skip; marks /grammar/truth degraded)
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=tail'

# Set cursor to last event at/before a timestamp (timezone-aware ISO required)
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=timestamp&at=2026-06-01T00:00:00Z'
```

Known cursors: `biometrics_grammar_consumer`, `execution_grammar_reducer`, `transport_grammar_reducer`, `route_grammar_consumer`.

### Poison quarantine acknowledgement (internal operator endpoint)

When a reducer quarantines a poison event, `/grammar/truth` stays **degraded** with
`reducer_quarantine_present:<cursor_name>` until an operator acknowledges the quarantine.
Quarantine state is durable in Postgres (`substrate_reducer_quarantine`); acknowledgement
sets `acknowledged_at` but preserves the audit row.

```bash
# Acknowledge a single quarantined event
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/quarantine/ack?cursor_name=transport_grammar_reducer&event_id=gev_x'

# Acknowledge all unacked quarantine for a cursor
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/quarantine/ack?cursor_name=transport_grammar_reducer&ack_all=true'
```

Accepted-pressure reducer output publishes to `orion:grammar:accepted-pressure` (not canonical `orion:grammar:event`).

## Dynamics tick (rung-1 pacemaker)

`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` writes execution/transport prediction-error
values onto durable substrate graph nodes (`metadata['prediction_error']`), but by itself
nothing ever reads them back — the seeded surprise just sits inert on the node. Set
`SUBSTRATE_DYNAMICS_TICK_ENABLED=true` to run a periodic, bounded, fail-open
`SubstrateDynamicsEngine.tick()` against the same shared substrate graph store, which
seeds and propagates activation pressure from those `prediction_error` values.

- `SUBSTRATE_DYNAMICS_TICK_ENABLED` (default `false`): enable the tick loop.
- `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` (default `30.0`): tick cadence. Deliberately slower
  than `GRAMMAR_POLL_INTERVAL_SEC` because each tick issues a bounded but real query
  (`snapshot()`, `limit_nodes=500`) against the configured store backend, not an in-memory read.

Only meaningful once `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` and
`SUBSTRATE_STORE_BACKEND=sparql` (Fuseki) are set — with the in-memory default store the
prediction-error nodes are process-local and this tick has nothing durable to consume.

## Unified turn bus listeners

Besides grammar poll reducers, substrate-runtime subscribes to RPC-style harness channels:

| Listener | Channel | Env keys |
|----------|---------|----------|
| Finalize appraisal (5a) | `orion:substrate:finalize_appraisal:request` | `CHANNEL_FINALIZE_APPRAISAL_REQUEST`, `CHANNEL_FINALIZE_APPRAISAL_RESULT_PREFIX` |
| Post-turn closure (step 7) | `orion:substrate:post_turn_closure` | `CHANNEL_POST_TURN_CLOSURE`, `ENABLE_POST_TURN_CLOSURE_LISTENER` |

Post-turn closure logs `post_turn_closure received …` on ingest. When
`surprise_unresolved=true`, it attempts a rung-1 prediction-error graph write only if
`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` (logs `post_turn_closure_prediction_error_write`
or `…_skipped` otherwise). Full Task 21 strain reducers are not implemented here.

**Reverie semantic lift:** unresolved closures also upsert human referent rows into
`substrate_turn_referent` via `turn_referent_store.persist_turn_referent`. Apply
`services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql` before enabling
`ORION_REVERIE_SEMANTIC_LIFT_ENABLED` on orion-thought.

## Downstream of this service: Layers 6-11

This service (Layers 1-5: grammar events → reducers → `substrate_field_state` via
`orion-field-digester`) is the root of a full substrate ladder. Each layer below is its
own service, one Postgres table, one poll loop. This section documents the whole chain
so nobody has to re-derive it from scratch again — see the 2026-07-12 archaeology in
`docs/notes/2026-07-12-metrics-swamp-arsonist-review.md` for how this was traced.

```text
L1-5  substrate_field_state          orion-field-digester        (this service's output)
  |     decay 0.92/tick, node/capability lattice, real biometrics + cortex-exec receipts
  v
L6    substrate_self_state           orion-self-state-runtime    "how is Orion doing"
  |     13 named 0-1 dimensions, memoryless recompute every 2s (see below)
  v
L7    substrate_proposal_frames      orion-proposal-runtime      self_state -> POSSIBLE actions
  v
L8    substrate_policy_decision_frames orion-policy-runtime      proposals -> governed decision
  v
L9    ExecutionDispatchFrameV1       orion-execution-dispatch-   policy+proposal+self_state ->
                                      runtime                     dispatch envelope
                                                                   (EXECUTION_DISPATCH_MODE=
                                                                    dry_run by default -- L9 has
                                                                    never mutated the real world)
  v
L10   substrate_feedback_frames      orion-feedback-runtime      scores the dry-run outcome
  v
L11   consolidation windows/motifs   orion-consolidation-runtime  folds L6-L10 history into
                                                                   memory windows; consumed by
                                                                   orion-dream/compaction_applier.py
```

Each layer's own README states its literal `X + Y -> service -> Z` data flow; this
diagram just makes the whole chain visible in one place. Whether L11's output actually
changes anything a human or Orion would notice (vs. terminating in Hub debug tiles) was
still an open question as of 2026-07-12 -- verify live before assuming either way.

### L6 (`SelfStateV1`) metric shape and mechanics

Schema: `orion/schemas/self_state.py`. Computation: `orion/self_state/{builder,scoring,
prediction}.py`. Tuning surface: `config/self_state/self_state_policy.v1.yaml` (weights,
channel->dimension map, thresholds -- config, not code).

**How a score is built:** raw named channels (`cpu_pressure`, `execution_load`,
`staleness`, `repair_pressure`, ...) come from `substrate_field_state` node/capability
vectors and from attention targets' channels (weighted by
`salience x attention_target_weights[target_kind]`), merged by `max()`. Those channels
route into dimensions via `policy.channel_dimension_map`, again via `max()` when several
channels land on one dimension. A few dimensions get bespoke formulas on top:

| Dimension | Formula |
|---|---|
| `coherence` | `Σ(stabilizing_channel × weight) − 0.25×Σ(penalty_channels)` |
| `uncertainty` | `overall_salience × (1 − coherence)` |
| `field_intensity` | `0.6×overall_salience + 0.4×recent_perturbation_saturation` |
| `agency_readiness` | `coherence − 0.25×execution_p − 0.35×reliability_p − 0.25×uncertainty − 0.15×resource_p` |
| `resource/execution/reasoning/reliability/continuity/introspection/social_pressure` | pure channel-map `max()`, no extra math |
| `policy_pressure` | hardcoded `0.0`, always (see Known issues) |
| `transport_integrity` (conditional 13th dim) | separate formula from field `capability:transport` hints, gated by `ENABLE_TRANSPORT_SELF_STATE_INFLUENCE` |

`overall_intensity` = weighted average of the 12 core dims per `dimension_weights`.
`overall_condition` buckets `overall_intensity` via `condition_thresholds`
(quiet/steady/loaded/strained/unstable).

**How it evolves tick-to-tick:** Layer 6 itself is memoryless -- every poll
(`SELF_STATE_POLL_INTERVAL_SEC=2.0`) fully re-derives from the current field+attention
snapshot. Decay lives one layer down (this service, `BIOMETRICS_FIELD_DECAY_RATE`), not
here. The only inter-tick state Layer 6 keeps is:

- **Trajectory**: per-dimension delta vs. `previous_self_state` (age-gated ≤300s), kept
  if `|delta|≥0.02`; weighted net delta vs. `trajectory_threshold=0.03` →
  `improving/degrading/stable`.
- **Prediction/surprise** (`orion/self_state/prediction.py`): each tick emits a naive
  one-step linear extrapolation (`score + last_delta`). Next tick compares actual vs.
  that prediction → `prediction_error_scores` (kept if ≥0.01), and
  `overall_surprise = max(...)` (deliberately max, not mean -- one badly-mispredicted
  dimension is enough to call the whole self "surprised"). This is the real
  predictive-coding mechanism in the ladder and feeds the `predictive` drive downstream.

Retention: `SELF_STATE_RETENTION_HOURS=72`, pruned hourly.

### Known issues in the L6 metric shape (2026-07-12 audit)

- **`policy_pressure` is a dead dimension** -- full schema slot, `0.00` weight in the
  policy, hardcoded `0.0` in `builder.py`, no producer anywhere. Keyword-cathedral entry:
  a label with zero runtime behavior.
- **`confidence` and `reasons` carry no per-dimension signal.** Every dimension in a
  given tick gets the *same* `overall_confidence`
  (`0.5 + 0.5×(len(dominant_targets)/5)` -- a proxy for "how many attention targets
  fired," unrelated to the specific dimension), and `reasons` is the fixed template
  `f"{dim_id} from field+attention channel synthesis"` for all 13 dimensions. Anything
  reading these fields expecting real differentiation is reading a constant.
- **`transport_integrity` doesn't affect `overall_intensity`.** It's not in
  `ALL_DIMENSION_IDS` or `dimension_weights` -- a display-only side channel dressed as a
  13th dimension.
- **`max()`-everywhere aggregation likely explains the "resource_pressure saturated at
  1.0" finding** from the scarcity-economy audit
  (`docs/notes/2026-07-12-metrics-swamp-arsonist-review.md`). `resource_pressure` is fed
  by `max(cpu_pressure, gpu_pressure, memory_pressure, disk_pressure, thermal_pressure,
  "pressure", transport_pressure)`, both at the channel-merge stage and again at the
  channel→dimension mapping stage. Any single spiking channel pins the whole dimension
  to 1.0 regardless of the other six being calm -- `resource_pressure` currently measures
  "worst channel," not an aggregate, everywhere it's computed this way.

### Downstream drive taxonomy overlap

Two independently-computed drive-pressure vectors exist over the same 6-key taxonomy
(`coherence, continuity, capability, relational, predictive, autonomy`):
`orion/spark/concept_induction/drives.py` (`DriveEngine`, fed by L6 self-state tensions +
biometrics/mesh/failure signals) and `orion/autonomy/reducer.py` (`AutonomyStateV2`, fed
by chat-turn evidence only -- a test explicitly bans it from importing self_state/phi).
Both import the shared `orion/autonomy/signal_drive_map.py` taxonomy config, but each
still keeps its own local pressure computation -- not yet one shared store. See the
arsonist review doc for the full merge/burn assessment.
