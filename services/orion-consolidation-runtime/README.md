# orion-consolidation-runtime

Layer 11 of the Orion cognition substrate: aggregates Layers 5–10 substrate history over a time window and persists `ConsolidationFrameV1` motif snapshots.

**Consolidation is pattern observation, not learning.** This service does not mutate policy, invoke an LLM, or change runtime behavior. The one exception is infra telemetry: it publishes a
bus-native `SystemHealthV1` heartbeat to `orion:system:health` every `HEARTBEAT_INTERVAL_SEC`
(default 10s), on its own independent bus connection -- process liveness, not cognition.

## Inputs

- `substrate_attention_frames`
- `substrate_proposal_frames`
- `substrate_policy_decision_frames`
- `substrate_execution_dispatch_frames`
- `substrate_feedback_frames`

## Outputs

- `substrate_consolidation_frames`

## Port

`8123` (`CONSOLIDATION_RUNTIME_PORT`)

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_consolidation_v1.sql
```

## Run

```bash
cd services/orion-consolidation-runtime
cp -n .env_example .env
docker compose up -d --build
```

## Smoke

```bash
./scripts/smoke_consolidation_v1.sh
```

## Idempotency

One frame per `(window_start, window_end, policy_id)` via stable `frame_id`. Re-runs use `INSERT ... ON CONFLICT (frame_id) DO NOTHING`.

## Causal feedback window

Execution dispatch first writes `substrate_execution_dispatch_inflight`; its database timestamp
becomes the frame's `generated_at`, and the real frame atomically replaces the receipt after the
RPC/result path. That receipt is the producer-visibility watermark even if the process pauses or a
result write blocks. `window.source_visibility_lag_sec` (900s against the enforced 720s maximum
RPC timeout) remains defense in depth. The oldest `feedback_pending` dispatch is the second
closure watermark.
Consolidation does not freeze an hourly window until both have crossed its end, so delayed
consequences cannot arrive after the stable frame has already been written. Dispatch rows are selected by their own
`generated_at` window; feedback is then joined through `source_execution_dispatch_frame_id`,
regardless of when the worker generated it. This prevents an hour of Layer 11 from combining
current dispatches with consequences belonging to an earlier hour—or permanently dropping a late
consequence. Compact empty-dispatch control receipts are aggregate bookkeeping, not feedback
frames, and are intentionally excluded.
