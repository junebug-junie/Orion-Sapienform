# orion-feedback-runtime

Layer 10 of the Orion cognition substrate: observes `ExecutionDispatchFrameV1` outcomes and persists `FeedbackFrameV1` consequence snapshots.

**Feedback is consequence capture, not learning.** This service does not consolidate patterns, mutate policy, approve actions, retry actions, or invoke an LLM.

## Inputs

- `substrate_execution_dispatch_frames`
- `substrate_dispatch_results` — real cortex-exec results for evidenced `dispatched`
  candidates (P1 of the motor-nerve spec; `load_cortex_result_evidence` was a stub
  returning `[]` before this, so `cortex_result`-kind observations were previously
  never produced from a live dispatch)
- `substrate_policy_decision_frames` (optional linkage)
- `substrate_proposal_frames` (optional linkage)
- `substrate_field_state` (before + first field tick after dispatch timestamp --
  2026-07-22, SelfStateV1 burn: replaces the old `substrate_self_state` read;
  `field_pressures()` (`orion/field/pressure.py`) provides the same real,
  non-hand-tuned channel comparison self-state used to pass through)

## Outputs

- `substrate_feedback_frames`
- Bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
  `HEARTBEAT_INTERVAL_SEC` (default 10s), on its own independent bus connection, separate
  from the worker's own `FEEDBACK_BUS_CHANNEL` publish connection

## Port

`8122` (`FEEDBACK_RUNTIME_PORT`)

## Bus / Redis

| Env | Default | Purpose |
|-----|---------|---------|
| `ORION_BUS_URL` | `redis://bus-core:6379/0` (compose) | Redis URL for Orion bus publish |
| `ORION_BUS_ENABLED` | `true` | Disable bus connect/publish when `false` |
| `FEEDBACK_BUS_CHANNEL` | `orion:feedback:frame` | Channel for `feedback.frame.v1` envelopes |

Bring up with root `.env` so `ORION_BUS_URL` matches the live stack:

```bash
docker compose --env-file ../../.env --env-file .env -f docker-compose.yml up -d --build
```

The last one adds the `*_pending` marker this service's work lookup depends on. Skip it and
every tick raises `UndefinedColumn`, which the poll loop swallows -- `/health` stays green and
nothing is produced. That file ends with a VERIFY block; run it and read it.

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_feedback_frame_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_feedback_frame_v2_drop_self_state.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_substrate_frames_created_at_index.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_substrate_pending_markers.sql
```

## Run

```bash
cd services/orion-feedback-runtime
cp -n .env_example .env
docker compose --env-file ../../.env --env-file .env up -d --build
```

## Smoke

```bash
./scripts/smoke_feedback_frame_v1.sh
```

## Hub (optional)

```bash
curl -s http://localhost:8080/api/substrate/feedback/latest | jq
```

### Visual execution outcomes

Dispatch persists the explicit `visual_outcome` in
`substrate_dispatch_results.result_json`. `load_cortex_result_evidence` retains
that field and measured latency; `normalize_cortex_result_evidence` preserves it
for both the feedback builder and the effect resolver. Transport `success` or
`ok=True` does not establish visual production. The builder records `produced`
as completed, thermal/busy deferrals as deferred, `already_satisfied` as not
attempted, `failed` as an operational failure, and absent/unknown visual receipts
as unknown. Correlation and artifact references remain available without copying
source text into feedback.

Only `produced` can qualify for a visual effect observation, and it must still
carry a justified declared signal and measured field window. Baseline runs use
`expected_effect=None`, so even production never advances an effect posterior.
Historical `render_scene` resource-pressure claims are rejected at resolution
(including queued dispatches); existing rows remain historical data. Deferrals,
already-satisfied, failed and unknown results do not become treated observations
or manufacture an untreated control tick. Failure latency remains recorded.

Focused regression: `tests/test_feedback_runtime_store.py` exercises persisted
result rows through SQL evidence loading, normalization, feedback and resolution;
`tests/test_action_outcome_resolution.py` covers field-present non-observation and
retired claims. These are deterministic tests, not live runtime verification.
This service has no periodic eval harness; follow-up: replay recorded visual
outcome sequences through feedback after an approved deployment and confirm no
posterior changes on non-observations. The visual-baseline integration eval is
owned by the visual producer/dispatch patch.
