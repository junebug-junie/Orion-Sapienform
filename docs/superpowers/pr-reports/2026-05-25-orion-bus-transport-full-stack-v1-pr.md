# PR: Orion Bus Transport Full Substrate Integration v1

## Summary

Finishes `orion-bus` onboarding from trace-only (PR #630) through field/attention-visible M5, with gated scaffolding for self/proposal/policy/dispatch/feedback/consolidation (Layers 6–11).

## Previous live state

PR #630:

```text
orion-bus → bus.transport GrammarEventV1 → grammar_events
```

## This PR

```text
bus.transport traces
→ transport_bus reducer (Layer 3)
→ StateDeltaV1 / ReductionReceiptV1
→ field digestion → capability:transport (Layer 4)
→ attention visibility (Layer 5)
→ gated self/proposal/policy/dispatch/feedback/consolidation (Layers 6–11)
```

## Flags (defaults)

| Flag | Default |
|------|---------|
| `ENABLE_TRANSPORT_BUS_REDUCER` | `false` |
| `ENABLE_TRANSPORT_FIELD_DIGESTION` | `false` |
| `ENABLE_TRANSPORT_ATTENTION_VISIBILITY` | `false` |
| `ENABLE_TRANSPORT_SELF_STATE_INFLUENCE` | `false` |
| `ENABLE_TRANSPORT_PROPOSALS` | `false` |
| `TRANSPORT_PROPOSAL_MODE` | `read_only` |
| `TRANSPORT_SUBSTRATE_MATURITY` | `trace_only` (logging) |

Layers 8–11 use global frame runtimes; transport safety is enforced via proposal templates, policy `hard_blocks`, and dispatch dry-run constraints (no separate `ENABLE_TRANSPORT_*` on those services in v1).

## Implemented layers

| Layer | Status | Default |
|-------|--------|---------|
| 3 reducer | implemented | disabled |
| 4 field | implemented | disabled |
| 5 attention | implemented | disabled |
| 6 self-state | implemented/gated | disabled |
| 7 proposal | implemented/gated | disabled |
| 8 policy | read-only rules + hard_blocks | global runtime |
| 9 dispatch | dry-run constraints | global runtime |
| 10 feedback | dry-run outcomes | global runtime |
| 11 consolidation | 2 transport motifs | global runtime |

## Tests run

```text
PYTHONPATH=. pytest tests/test_transport_projection_schemas.py \
  tests/test_transport_substrate_reducer.py \
  tests/test_transport_substrate_pipeline.py \
  tests/test_attention_transport_visibility.py \
  tests/test_self_state_transport_dimension.py \
  tests/test_proposal_transport_readonly_candidates.py \
  tests/test_policy_transport_gates.py \
  tests/test_execution_dispatch_transport_dry_run.py \
  tests/test_feedback_transport_outcomes.py \
  tests/test_consolidation_transport_motifs.py -q
→ 17 passed

PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_transport_perturbations.py -q
→ 3 passed
```

## Live proof (enable progressively)

1. `psql … -f services/orion-sql-db/manual_migration_transport_substrate_loop.sql`
2. `ENABLE_TRANSPORT_BUS_REDUCER=true` → verify `substrate_reduction_receipts` (`target_kind=transport_bus`, `catalog_drift_pressure=1.0` for current athena observer)
3. `ENABLE_TRANSPORT_FIELD_DIGESTION=true` → `capability:transport` in `substrate_field_state`
4. `ENABLE_TRANSPORT_ATTENTION_VISIBILITY=true` → attention item for `capability:transport` or `node:athena`

Smoke: `./scripts/smoke_orion_bus_transport_full_stack.sh --mode m3|m4|m5`

## Non-goals respected

- No packet logging or raw Redis payload ingestion
- No bus restart/replay/purge/catalog writes
- Field digester does not read `grammar_events`
- No automatic policy mutation or habit execution

## Known follow-ups

- Add remaining consolidation motifs (`transport_backpressure_loop`, `transport_observer_failure_loop`, etc.)
- Optional dedicated `ENABLE_TRANSPORT_*` flags on policy/dispatch/feedback/consolidation runtimes
- Decide whether legacy observer streams belong in `orion/bus/channels.yaml` vs intentional contract drift signal
- `substrate_transport_bus_cursor` table reserved; grammar cursor uses `substrate_reduction_cursor`

## Doc conflicts

None observed; implementation follows controlling plan `docs/superpowers/plans/2026-05-25-orion-bus-transport-full-stack-v1.md`.
