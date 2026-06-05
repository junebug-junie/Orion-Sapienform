# Orion Bus Transport Substrate — Live Proof Ladder

PR #648 — Phases/Layers 1–11.

---

## How to prove the stack is live

Each milestone (M) has a feature flag, a database artifact to check, and a smoke command.
All flags default **off**; set them in `services/<service>/.env` and force-recreate the container.

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

**Flag:** `ENABLE_TRANSPORT_ATTENTION_VISIBILITY=true` in `services/orion-attention-runtime/.env`

**What happens when flag is OFF:** `capability:transport` is removed from `dominant_targets`
and `capability_targets` and placed into `suppressed_targets`. This is normal and expected
when transport is healthy and below salience threshold.

**What happens when flag is ON:** transport can surface in `dominant_targets` or
`capability_targets` if salience warrants it; otherwise stays in `suppressed_targets`.

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

## Layers 6–11 — Full observe

Layers 6–11 activate once M3/M4/M5 are proven. Each layer reads the previous layer's DB output.

| Layer | Service | Flag | Evidence table |
|-------|---------|------|----------------|
| L6 self-state | `orion-self-state-runtime` | `ENABLE_TRANSPORT_SELF_STATE_INFLUENCE=true` | `substrate_self_state.self_state_json -> 'dimensions' -> 'transport_integrity'` |
| L7 proposals | `orion-proposal-runtime` | `ENABLE_TRANSPORT_PROPOSALS=true` + `TRANSPORT_PROPOSAL_MODE=read_only` | `substrate_proposal_frames` — transport inspect candidates, no destructive actions |
| L8 policy | `orion-policy-runtime` | *(no env flag; controlled by policy YAML)* | `substrate_policy_decision_frames` — approved transport inspect decisions |
| L9 dispatch | `orion-execution-dispatch-runtime` | *(no env flag; uses `EXECUTION_DISPATCH_MODE=dry_run`)* | `substrate_execution_dispatch_frames` — dispatch_mode must be `dry_run` |
| L10 feedback | `orion-feedback-runtime` | *(no env flag)* | `substrate_feedback_frames` — outcome_status reflects dry_run result |
| L11 consolidation | `orion-consolidation-runtime` | *(no env flag)* | `substrate_consolidation_frames` — `transport_contract_drift_loop` motif |

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

# 3. Enable M5
echo "ENABLE_TRANSPORT_ATTENTION_VISIBILITY=true" >> services/orion-attention-runtime/.env
docker compose -f services/orion-attention-runtime/docker-compose.yml up -d --force-recreate
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=m5

# 4. Layers 6–11 — enable self-state influence + proposals, then full-observe
echo "ENABLE_TRANSPORT_SELF_STATE_INFLUENCE=true" >> services/orion-self-state-runtime/.env
echo "ENABLE_TRANSPORT_PROPOSALS=true" >> services/orion-proposal-runtime/.env
docker compose -f services/orion-self-state-runtime/docker-compose.yml up -d --force-recreate
docker compose -f services/orion-proposal-runtime/docker-compose.yml up -d --force-recreate
./scripts/smoke_orion_bus_transport_full_stack.sh --mode=full-observe
```
