# Orion Bus Transport Substrate — Live Proof Ladder

PR #648 — Phases/Layers 1–11.

---

## How to prove the stack is live

Each milestone (M) has a feature flag, a database artifact to check, and a smoke command.
Compose defaults are **off**, but as of 2026-09-22 every flag below is **on in production**
(checked with `docker exec <container> env`):

| Container | Setting |
|-----------|---------|
| `orion-athena-substrate-runtime` | `ENABLE_TRANSPORT_BUS_REDUCER=true`, `TRANSPORT_SUBSTRATE_MATURITY=full_observe` |
| `orion-athena-field-digester` | `ENABLE_TRANSPORT_FIELD_DIGESTION=true` |
| `orion-athena-proposal-runtime` | `ENABLE_TRANSPORT_PROPOSALS=true`, `TRANSPORT_PROPOSAL_MODE=read_only` |
| `orion-athena-execution-dispatch-runtime` | `EXECUTION_DISPATCH_MODE=dispatch_read_only` |

A flag being on is not proof a layer is live: M5 was dead 2026-09-20 21:58Z to 2026-09-23 with
every flag on (a `FieldStateV1` schema-skew rejection). Check freshness, not flags. Audit:
`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md`.

---

## M3 — Bus reducer + projection

**Flag:** `ENABLE_TRANSPORT_BUS_REDUCER=true` in `services/orion-substrate-runtime/.env`

**What happens:** `orion-substrate-runtime` reads `bus.transport:*` grammar events,
runs `transport_bus_reducer`, writes a row to `substrate_reduction_receipts`
(with `reducer_name = 'transport_bus_reducer'`), and upserts `substrate_transport_bus_projection`.

**Proof command:**

```bash
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m3
```

**Expected:** rows with `reducer_name = transport_bus_reducer` in receipts, and a
`substrate_transport_bus_projection` row with `buses` containing `bus:athena`.

**Note:** The receipts table stores `target_kind` / `target_id` inside `receipt_json`, not as
top-level columns. Use `WHERE reducer_name = 'transport_bus_reducer'` to filter.

---

## M4 — Field vector includes `capability:transport`

**Flag:** `ENABLE_TRANSPORT_FIELD_DIGESTION=true` in `services/orion-field-digester/.env`

**What happens:** `orion-field-digester` digests the transport state delta into the field tensor,
creating a `capability:transport` entry in `substrate_field_state.field_json -> 'capabilities'`.

**Proof command:**

```bash
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m4
```

**Expected:** a row with `capability_id = 'capability:transport'` in the result.

---

## M5 — Attention frame includes transport

**Flag:** none. `ENABLE_TRANSPORT_ATTENTION_VISIBILITY` was removed 2026-07-30
(`services/orion-attention-runtime/.env_example`). Capability targets, including
`capability:transport`, are scored by novelty against the previous frame
(`orion/attention/field_attention/selectors.py::select_capability_targets` ->
`_novelty_targets`), so transport surfaces when its field vector changes, not when it is high.

**M5 is satisfied if `capability:transport` appears in ANY bucket.**

```
suppressed = healthy / quiet transport (valid passing state)
capability = above min_salience, not dominant
dominant   = transport is high-priority right now
```

**Proof command:**

```bash
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m5
```

**Expected:** `capability:transport` in at least one of `dominant_targets`,
`capability_targets`, or `suppressed_targets` in the most recent frame.

---

## Layers 7–11 — Full observe

Layers 7–11 activate once M3/M4/M5 are proven. Each layer reads the previous layer's DB output.

There is no L6 any more. L6 was the `transport_integrity` dimension of `SelfStateV1`, written by
`orion-self-state-runtime`; that service and `config/self_state/` were deleted 2026-07-22 (commit
`bcc72f6a0`). M5 feeds L7 directly. `orion/schemas/self_state.py` is still imported by other modules
and was not deleted.

| Layer | Service | Flag | Evidence table |
|-------|---------|------|----------------|
| L7 proposals | `orion-proposal-runtime` | `ENABLE_TRANSPORT_PROPOSALS=true` + `TRANSPORT_PROPOSAL_MODE=read_only` | `substrate_proposal_frames` — transport inspect candidates, no destructive actions |
| L8 policy | `orion-policy-runtime` | *(no env flag; controlled by policy YAML)* | `substrate_policy_decision_frames` — approved transport inspect decisions |
| L9 dispatch | `orion-execution-dispatch-runtime` | *(no env flag; `EXECUTION_DISPATCH_MODE`, production = `dispatch_read_only`)* | `substrate_execution_dispatch_frames` — dispatch_mode must be a read-only/dry-run mode |
| L10 feedback | `orion-feedback-runtime` | *(no env flag)* | `substrate_feedback_frames` — outcome_status reflects dry_run result |
| L11 consolidation | `orion-consolidation-runtime` | *(no env flag)* | `substrate_consolidation_frames` — `transport_contract_drift_loop` motif (needs contract_pressure ≥ 0.70; 7-day max observed 0.018, never fired in 30 days — see the audit) |

**Safety constraint:** `TRANSPORT_PROPOSAL_MODE=read_only` (default) blocks
`restart_bus`, `purge_stream`, `replay_stream`, `change_catalog`, `change_bus_config` proposals.
Do not change this to `unrestricted` without explicit operator sign-off.

**Proof command:**

```bash
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=full-observe
```

Layers 8–11 have no dedicated transport env flags because their gating is handled by policy
templates and the `dry_run` dispatch mode, not by feature switches.

---

## Quick operator runbook

```bash
# 1. Enable M3
echo "ENABLE_TRANSPORT_BUS_REDUCER=true" >> services/orion-substrate-runtime/.env
docker compose -f services/orion-substrate-runtime/docker-compose.yml up -d --force-recreate
docker exec ${PROJECT}-substrate-runtime env | grep -E 'TRANSPORT|ENABLE_TRANSPORT|BUS_STREAM_DEPTH'
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m3

# 2. Enable M4
echo "ENABLE_TRANSPORT_FIELD_DIGESTION=true" >> services/orion-field-digester/.env
docker compose -f services/orion-field-digester/docker-compose.yml up -d --force-recreate
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m4

# 3. M5 has no flag; just check it
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m5

# 4. Layers 7–11 — enable proposals, then full-observe
echo "ENABLE_TRANSPORT_PROPOSALS=true" >> services/orion-proposal-runtime/.env
docker compose -f services/orion-proposal-runtime/docker-compose.yml up -d --force-recreate
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=full-observe
```
