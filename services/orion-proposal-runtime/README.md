# orion-proposal-runtime

Layer 7 substrate service: converts `SelfStateV1` (+ optional attention/field context) into **possible actions** (`ProposalFrameV1`), not automatic actions.

## Data flow

```text
substrate_self_state
+ substrate_attention_frames
+ substrate_field_state
  → orion-proposal-runtime
  → ProposalFrameV1
  → substrate_proposal_frames
```

## Non-goals

- No policy approval, cortex-exec, bus publish, operator notifications, or LLM calls.
- `execution_intent` on candidates is descriptive only.

## Idempotency

One proposal frame per `source_self_state_id`. Re-running the worker for the same self-state snapshot is a no-op. Policy/template changes do not regenerate until a new self-state row exists (v1 semantics).

## Attention-bound proposals (P5)

`ProposalTemplateV1.target_binding` lets a template point at a live field on
the inbound context instead of a fixed target. The only binding implemented is
`ATTENTION_FIRST_TARGET_BINDING = "self_state.dominant_attention_targets[0]"`
(`orion/proposals/builder.py`): `_resolve_binding_target()` reads
`SelfStateV1.dominant_attention_target_details[0]` and only resolves when its
`target_kind` is one of `node`, `capability`, `field`, `system` (the exact
intersection of `AttentionTargetSummaryV1.target_kind` and
`ProposalCandidateV1.target_kind`'s allowed values). `_resolve_binding_target`
never raises -- an empty attention list, an unbound template, or an
unsupported `target_kind` all fall through to the template's existing static
target with no candidate produced for that template. `ProposalCandidateV1`
gains `binding_resolved_from` so a resolved candidate is traceable back to the
attention target it bound to.

`config/proposals/proposal_policy.v1.yaml`'s `inspect_attended_target`
template uses this binding and ships **live** (`base_priority: 0.34`, not
dark-shipped at 0.0). The YAML comment on that template documents a 7-day kill
criterion. `orion/autonomy/evals/run_attention_bound_proposal_eval.py` checks
that criterion against real proposal-frame data and reports "insufficient
data" gracefully if the template hasn't accumulated enough candidates yet:

```bash
python orion/autonomy/evals/run_attention_bound_proposal_eval.py
```

## Run

```bash
cp -n .env_example .env
docker compose up -d --build
curl -s http://localhost:8119/health
curl -s http://localhost:8119/latest | jq
```

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < ../../services/orion-sql-db/manual_migration_proposal_frame_v1.sql
```

## Smoke

From repo root:

```bash
./scripts/smoke_proposal_frame_v1.sh
```
