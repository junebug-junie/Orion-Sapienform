# orion-policy-runtime

Layer 8 substrate service: evaluates `ProposalFrameV1` candidates against `SubstratePolicyV1` and persists **governed decisions** (`PolicyDecisionFrameV1`). Policy is not execution.

## Data flow

```text
substrate_proposal_frames
  → orion-policy-runtime
  → PolicyDecisionFrameV1
  → substrate_policy_decision_frames
```

2026-07-22 (SelfStateV1 burn): decisions are evaluated directly off
`ProposalFrameV1` (which already carries `source_field_tick_id`) -- no
separate self-state dependency or load.

## Non-goals

- No cortex-exec, bus publish, operator notifications, settings mutation, or LLM calls in the
  policy-decision data flow above -- `PolicyDecisionFrameV1` is Postgres-only, not a bus event.
- `execution_constraints` on decisions are instructions for Layer 9 only.

## Liveness telemetry

Separately from the policy-decision data flow (still Postgres-only, no change), this service
publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s) via its own independent Redis connection -- part of the
repo-wide service-heartbeat rollout
(docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md), not a policy
bus-publish.

## Idempotency

One policy decision frame per `source_proposal_frame_id`. Re-running the worker for the same proposal frame is a no-op.

## Pending-marker reconciler (bounded, 2026-09-25)

`policy_pending` is cleared in the same transaction as the policy decision frame insert. A safety-net sweep sets it
back to `true` for any of the proposals whose marker is `false` but whose policy decision frame does not
exist -- it can only add work, never remove it. Shared implementation:
`orion/substrate/pending_marker_reconcile.py`.

- Every `POLICY_RECONCILE_INTERVAL_SEC` (900): one short UPDATE over rows generated in the last
  `POLICY_RECONCILE_WINDOW_SEC` (7200) -- index scan on `generated_at` plus a per-row index probe.
- At most once per `POLICY_RECONCILE_FULL_SWEEP_INTERVAL_SEC` (86400; 0 disables), only during UTC
  hour `POLICY_RECONCILE_FULL_SWEEP_HOUR_UTC` (9 = 03:00 MDT; -1 = any hour): one read-only
  whole-history anti-join SELECT, then UPDATEs in batches of 5000 ids that re-check the condition.
- Logs: `policy_pending_reconciled requeued=N scope=window|full` (WARNING, only when work was recovered) and
  `policy_pending_full_sweep_done candidates=N batches=N requeued=N elapsed_ms=N` (INFO, every full sweep).

Before this, the sweep was an unbounded anti-join UPDATE every 15 min and was one of the three
top I/O statements on athena's Postgres while never finding anything.

## Run

```bash
cp -n .env_example .env
docker compose up -d --build
curl -s http://localhost:8120/health
curl -s http://localhost:8120/latest | jq
```

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < ../../services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql
```

## Smoke

From repo root:

```bash
./scripts/smoke_policy_decision_frame_v1.sh
```
