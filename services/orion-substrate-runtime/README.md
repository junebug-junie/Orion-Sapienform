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

On a *fresh* unhealthy transition (no orion-notify alert already open for it), the monitor waits `SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC` (default `15.0`) and rechecks once before paging. This exists because a single degraded `/grammar/truth` observation can be a self-healing blip -- e.g. `reducer_cursor_commit_failing:biometrics_grammar_consumer` can be set by one cursor-advance write losing a race with transient Postgres load and then clear itself on the very next 1s poll tick, long before the next 900s health-check tick would have seen it recover. The recheck confirms the condition is still true before escalating; a genuinely sustained incident is still unhealthy 15s later and pages exactly as before.

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

`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` writes execution/transport/biometrics
prediction-error values onto durable substrate graph nodes (`metadata['prediction_error']`),
but by itself
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

**Write guard:** `SubstrateDynamicsEngine.tick()` (`orion/substrate/dynamics.py`) only calls
`store.upsert_node()` for a node when its activation crossed the existing `1e-6` change
threshold, its dormancy state flipped, or its pressure actually moved — not unconditionally
for every node on every tick. On the SPARQL backend each `upsert_node()` is a full
DELETE+INSERT transaction; with the tick's default 30s cadence, writing every node
regardless of change was found (2026-07-16) driving ~2.5 SPARQL updates/sec against Fuseki
sustained (78 substrate nodes / 30s), which is indistinguishable from real data growth on
disk until the next `tdbcompact` and was a material contributor to `orion-rdf-store` running
out of compaction headroom. If you add new per-node fields to the tick, route their writes
through this same changed-gate rather than bypassing it.

Gotcha the guard exposed: `recency_score` decays every tick regardless of whether activation
changed, so a node whose activation has ratcheted to a peak (common — `decay_half_life_seconds`
defaults to `None`, i.e. no time-based decay) can go indefinitely without a real write once the
guard skips it. The dormancy check in the same loop must read *this tick's* freshly computed
recency (`activations[f"{node_id}:recency"]`), not `node.signals.activation.recency_score` off
the top-of-tick store snapshot — otherwise a node can never cross into dormant via pure recency
decay once persistence stops, since the stale stored value never moves. Fixed in the same patch
that added the guard; if you touch this loop again, keep the dormancy decision reading fresh
recency even though the stored copy may lag.

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

`_write_prediction_error_node()` (`app/worker.py`) writes to a single, fixed `node_id` per
lane (`node:substrate.execution`, `node:substrate.transport`, `node:substrate.biometrics`,
`node:substrate.chat`, `node:substrate.route`, `node:substrate.harness_closure` for post-turn
closure), so repeat writes collapse instead of spawning a new node per event.

`node:substrate.biometrics` (`biometrics_prediction_error()`, `orion/substrate/
prediction_error.py`, wired into `_tick()`) is the third instrument in this family, shadow-
built 2026-07-21 per the Sentience Striving Program charter §6 item 3. Unlike execution's
fixed four-key `pressure_hints` diff, biometrics `pressure_hints` keys vary per node role
(GPU nodes carry `gpu`/`strain`; orchestration nodes carry `disk_pressure`/
`memory_pressure`/`thermal_pressure` — confirmed live against real
`substrate_node_biometrics_projection` data), so this instrument diffs the union of keys
present on either side of a given node rather than a fixed list. See `docs/superpowers/
specs/2026-07-21-biometrics-prediction-error-shadow-design.md` for the full metric-quality-
gate writeup.

`node:substrate.chat` (`chat_prediction_error()`, wired into `_chat_tick()`) and
`node:substrate.route` (`route_prediction_error()`, wired into `_route_tick()`) are the
fourth and fifth instruments in this family, shadow-built 2026-07-21, closing charter §6
item 3's producer-instrumentation sweep across all five named domains. `chat_prediction_
error()` diffs `compute_chat_pressure_hints()` (`orion/substrate/chat_loop/
grammar_extract.py:114` — `conversation_load`/`repair_pressure`/`topic_coherence`, computed
transiently, not persisted on `ChatTurnStateV1`) across successive turn states, same
fixed-key/`_THRESHOLD`-scaled shape as execution/transport. `route_prediction_error()` is
**deliberately different in shape**: `RouteArbitrationRunStateV1`'s decision fields
(`lane`, `lane_reason`, `output_mode`, `mind_requested`) are categorical, not continuous, so
it scores a per-field mismatch rate (1.0 if a field differs, else 0.0, averaged across
compared fields and matched runs) instead of an absolute-value delta, and does **not** apply
the module's `_THRESHOLD = 0.30` scaling — that scaling exists to saturate an unbounded
continuous delta, and a mismatch rate is already bounded to `[0, 1]` by construction. See
`docs/superpowers/specs/2026-07-21-chat-route-prediction-error-shadow-design.md` for the
full metric-quality-gate writeup and the reasoning for route's different scoring shape.
Repeat writes to the same node accumulate bounded per-event attribution in
`metadata['contributing_turn_ids']` (capped at 20, deduped, oldest dropped) — this is
attribution, not the pressure/dormancy state itself, which `DYNAMICS_ENGINE_OWNED_METADATA_
KEYS` (`orion/substrate/falkor_codec.py`) separately carries forward on every rewrite so
`SubstrateDynamicsEngine.tick()`'s computed state doesn't get clobbered. `contributing_turn_
ids` is durable against the live Falkor backend via `orion/substrate/falkor_codec.py`'s
`DYNAMICS_METADATA_KEYS` allowlist (JSON-string-encoded as `contributing_turn_ids_json`,
same pattern as `taxonomy_path_json`) and is consumed by `orion/substrate/attention_
broadcast.py::substrate_pressure_signals()` (`evidence_refs`) and `orion/substrate/
attention_self_model.py::reduce_attention_self_model()`'s optional `harness_closure_signal`
narrative enrichment — see `docs/superpowers/specs/2026-07-18-prediction-error-attribution-
wiring-design.md`.

**Fixed: execution/route prediction-error was structurally always 0.0, not "quiet" (2026-07-22).**
`execution_prediction_error()` and `route_prediction_error()` (`orion/substrate/prediction_error.py`)
both diffed a `curr` run against the `prev` run sharing its exact `trace_id`, skipping any run with
no match. Traced live 2026-07-21/22 while investigating why `node:substrate.execution`/
`node:substrate.transport` looked "3.6-3.7d stale" for a since-in-flight attention-salience
replacement design (branch `docs/attention-salience-tentative-plan`, not yet merged as of this
fix — see that branch's `docs/superpowers/specs/2026-07-21-attention-salience-cathedral-
replacement-tentative-plan.md` if it has landed by the time you're reading this) work:
`substrate_reduction_receipts` itself was healthy and flowing continuously for every
domain, but 26/26 sampled live `execution_trajectory_reducer` receipts were `operation: "create"`
with a unique `target_id` each — real cortex-exec runs are single-shot (created once, never
revised), so an exact `trace_id` match structurally never occurs. `execution_prediction_error()`
therefore returned `0.0` in perpetuity regardless of real execution volume — an instrument defect,
not a data-scarcity or dormancy signal. `route_prediction_error()` shares the identical trace_id-
match design (same live one-shot-per-turn shape, sparser sample). **`transport_prediction_error()`
did not have this bug** — `transport_bus_reducer` genuinely revises the same `bus_id` (`bus:athena`)
in place every tick, so its trace_id-equivalent match works; its low variance is a real quiet bus
(confirmed live: `bus_health`/`delivery_confidence` each took exactly 1 distinct value across the
last 500 live receipts checked), not an instrument bug. **Correction, 2026-07-22 (see the "transport
domain scope" note below): "quiet bus" is true for `bus_health`/`delivery_confidence` specifically,
but `transport_pressure`'s own quietness is not about the bus being calm — it's about what this
instrument is structurally capable of observing at all.**

Fix: when no exact `trace_id` match exists, both functions now fall back to diffing against
`prev`'s most-recently-updated run (by `last_updated_at`) instead of contributing nothing — the
best available "what did we expect" reference, equivalent to comparing this tick's freshest
observation against last tick's freshest one. A run that genuinely does get revised in place still
prefers its own exact match (regression-tested — see `_prefers_exact_trace_id_match_over_fallback`
in `orion/substrate/tests/test_prediction_error.py`). This does not change `biometrics_prediction_
error()`, which keys off persistent node identities that already recur across polls in real
traffic.

Implication for any future replay/precision work reading these instruments' history (e.g. the
attention-salience-cathedral candidates above): execution/route history *before* this fix landed
is contaminated with false zeros from the unmatched-trace_id bug, not real "no surprise" ticks —
exclude pre-fix rows the same way `field_channel_corpus.v1`'s own training-quality cutoff excludes
its pre-fix window (see `orion-field-digester`'s README).

**Fixed: `chat_prediction_error()` was structurally always 0.0 too, same symptom as execution/route
but a different root cause (2026-07-22).** Chat turns are single-shot bursts the same shape as
execution/route's runs (`build_chat_turn_grammar_events` in `services/orion-hub/scripts/
grammar_emit.py` shares one `trace_id` across every layer of a turn — trace_started, chat root,
context, raw_input, repair_signal, stance_disposition, trace_ended — emitted together), so a
`turn_id` is created once and never revisited. The bug here wasn't an unmatched-key comparison —
it was that a brand-new `turn_id` (the only kind that ever appears in a fresh tick, since
`updated.turns` is a persistent, cumulative dict of 241+ turns and everything else is unchanged
between `prev`/`curr`) hit `prev_turn is None: continue` and was silently skipped, never
contributing to the surprise score. Confirmed live: `node:substrate.chat` had never been written
in `substrate_field_state.node_vectors` despite `substrate_chat_session_projection` holding 241
real turns accumulated since 2026-06-19 — a genuine instrument defect, not a sign chat's
substrate-grammar reduction itself was broken (it wasn't; the 241 turns are real). Fix: reuse the
identical `_latest_run()` fallback pattern from execution/route, applied to `prev.turns` instead
of `prev.runs` — see `docs/superpowers/specs/2026-07-22-chat-route-prediction-error-audit.md`.

`route_prediction_error()`'s live value is currently a subnormal float (~3e-322) in
`substrate_field_state.node_vectors` — traced as far as confirming `orion/substrate/
pressure.py::prediction_error_pressure()` does **not** explain it (that function recomputes fresh
from `node.metadata['prediction_error']` on every call rather than persisting a decayed value), so
the actual mechanism is still open — see the audit doc's Missing Questions. Not fixed in this
patch.

**The "transport domain" is one queue, not the bus (found 2026-07-22, while re-verifying the "quiet
bus" claim above against real numbers).** `bus_health`/`delivery_confidence`/`transport_pressure`
(`orion/substrate/transport_loop/extract.py::compute_transport_pressures()`) are computed entirely
from whatever streams `orion-bus`'s bus-observer role is configured to watch
(`BUS_OBSERVER_STREAMS`, `services/orion-bus/app/settings.py`/`.env_example`). That config, and its
own inline comment, are blunt about the scope: *"`orion:evt:gateway`/`orion:bus:out` were
placeholder names from the original bus-observer commit (`ee810551`, 2026-05-25), never once
wired [to real streams]... These are the only two real Streams in the current architecture, so
they are now the entire default"* — `BUS_OBSERVER_STREAMS=orion:stream:world_pulse:run:result,
orion:stream:world_pulse:run:result:dlq`. Almost all of Orion's actual inter-service traffic runs
over Redis pub/sub channels, which have no XLEN-style depth/backlog concept the way a Stream does
— `world_pulse`'s result queue is the *only* thing on the entire bus built as a Stream, so it is
structurally the only thing this instrument can ever measure. "Transport domain" and "bus health"
are the wrong names for what this actually is: **whether one specific service's result queue is
backing up.**

Confirmed live (2026-07-22): `redis-cli -h <bus host> XLEN orion:stream:world_pulse:run:result` →
`91`; `:dlq` → `0`. `services/orion-bus/.env_example`'s own `BUS_STREAM_DEPTH_CRITICAL=100000`
divides this 91 down to `0.00091` (`orion/substrate/transport_loop/extract.py:86`,
`stream_depth_pressure = min(max_stream_depth / critical, 1.0)`) — a ~1,099x ratio between the real
number and the "critical" threshold, which reads as "quiet" no matter what the real queue is doing,
short of it growing to tens of thousands of messages. Checked against the training corpus
(`/mnt/telemetry/field_channels/corpus/field_channels.jsonl`): before the second cutoff
(2026-07-22T04:35:01Z, PR #1248's `mode="add"` fix), this same value was pinned at ~100,000 for 80%
of the corpus's history — that saturation was the already-documented accumulation bug, not evidence
this queue is normally busy. After the fix, it has been *exactly* 90 or 91, with zero movement, for
the entire ~18h observed post-fix window — consistent with `orion:stream:world_pulse:run:result`
having 91 messages sitting permanently unconsumed the whole time, not a healthy, actively-drained
queue. Whether that backlog itself is expected (nobody reads this queue by design) or a dead
consumer is a separate, not-yet-investigated question.

**Implication for the charter and for `mood-arc`:** Sentience Striving Program charter §9b item 3
(Predictive Processing/Active Inference) names "transport" as one of five domains with "equivalent
shadow-measurement instrumentation" to execution/biometrics/chat/route — true in the sense that
code exists and runs, misleading in the sense that this domain's real coverage is one queue, not
general bus stress across services. `prediction_error` (the `max()`-merge across all five domains'
instruments, `orion/field/pressure.py::collect_field_channel_pressures()`) is also one of
`field_channel_corpus.v1`'s trained mood-arc channels — this domain's contribution to that merge
has the same narrow-scope caveat. See `orion/mood_arc/README.md` and this document's "fourth
training-data quality cutoff" section for the corpus-facing side of this.

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
