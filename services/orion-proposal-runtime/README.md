# orion-proposal-runtime

Layer 7 substrate service: converts `FieldStateV1` (+ optional attention context) into **possible
actions** (`ProposalFrameV1`), not automatic actions.

**Correction (2026-07-28, doc drift fix — the code changed 2026-07-22, this README didn't):** this
used to say `SelfStateV1` was the primary input. That was already wrong by the time it was
written down here — `orion/proposals/builder.py::build_proposal_frame()`'s 2026-07-22
"SelfStateV1 burn" made `field: FieldStateV1` load-bearing and dropped `SelfStateV1` from the
function signature entirely (it was previously received but discarded — "reserved for continuity
in later revisions"). `field_pressures()` computed from `FieldStateV1` is the real, direct input;
`substrate_self_state` is not read by this service's core proposal-building path.

## Data flow

```text
substrate_field_state
+ substrate_attention_frames (optional)
  → orion-proposal-runtime
  → ProposalFrameV1
  → substrate_proposal_frames
```

## Non-goals

- No policy approval, cortex-exec, bus publish, operator notifications, or LLM calls in the
  proposal-generation data flow above -- `ProposalFrameV1` is Postgres-only, not a bus event.
- `execution_intent` on candidates is descriptive only.

## Liveness telemetry

Separately from the proposal-generation data flow (still Postgres-only, no change), this service
publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s) via its own independent Redis connection -- part of the
repo-wide service-heartbeat rollout
(docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md), not a proposal
bus-publish.

## Idempotency

One proposal frame per `source_field_tick_id` (+ `attention_frame_id` + `policy_id`,
`stable_proposal_frame_id()`). Re-running the worker for the same field tick is a no-op.
**Correction (2026-07-28, same doc-drift fix as above):** this used to key on `source_self_state_id`
— stale since the 2026-07-22 burn moved the real identity key to the field tick.

## Attention-bound proposals (P5)

`ProposalTemplateV1.target_binding` lets a template point at a live field on
the inbound context instead of a fixed target. The only binding implemented is
`ATTENTION_FIRST_TARGET_BINDING = "attention.dominant_targets[0]"`
(`orion/proposals/builder.py`) — **renamed 2026-07-22 (SelfStateV1 burn) from
`"self_state.dominant_attention_targets[0]"`; attention targets were always
`FieldAttentionFrameV1.dominant_targets` underneath, `self_state` was a lossy pass-through hop,
not the real source. `config/proposals/proposal_policy.v1.yaml`'s `target_binding` literal was
updated to match in the same changeset — this README wasn't, until now.** `_resolve_binding_target()`
reads `FieldAttentionFrameV1.dominant_targets[0]` and only resolves when its
`target_kind` is one of `node`, `capability`, `field`, `system` (the exact
intersection of `FieldAttentionTargetV1.target_kind` and
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

## Precision-weighted dimension confidence (2026-07-28, PR #1442)

`orion/proposals/scoring.py::dimension_confidence()` (consumed by `proposal_confidence()`, one of
`proposal_priority()`'s three terms) used to be a binary data-presence flag — `1.0` if a
`field_pressures` dimension was present this tick, `0.0` otherwise, wearing a name that implied
real epistemic confidence it didn't compute. It's now a genuine precision estimate: a per-dimension
EWMA baseline (`orion/bus/ewma.py::compute_ewma_update`), updated once per digestion tick by
`orion-field-digester` (`app/digestion/precision.py`, see that service's own README for the full
account and the real historical-variance measurement that justified it), scored as deviation from
each dimension's own recent normal instead of a flat presence check. This service's own scoring
math (`template_match_score`/`proposal_urgency`/`proposal_priority`/`proposal_risk`) is otherwise
unchanged by this patch — `proposal_priority()`'s fixed `0.4/0.2/0.1` weighting and every
template's `base_priority`/`base_risk` in `config/proposals/proposal_policy.v1.yaml` are still
static, hand-typed, and not calibrated against `FeedbackFrameV1`'s real recorded outcomes — a
known, named, separate follow-up (`docs/superpowers/specs/2026-07-28-precision-weighted-proposal-scoring-design.md`'s
Missing Question 4), not done in this patch.

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
