# Testing and Live Proof Guide

This guide defines the minimum proof standard for substrate trace and Layer 1–11 work.

A PR report should not say a layer is live unless there is observed proof from runtime tables, logs, or API responses.

## Unit tests

For substrate trace emission:

```bash
PYTHONPATH=. pytest services/<service>/tests/test_<service>_grammar_emit.py -q
PYTHONPATH=. pytest services/<service>/tests/test_<service>_grammar_publish_fail_open.py -q
```

Required assertions:

- events validate against `GrammarEventV1`;
- only closed `GrammarEventKind` values are used;
- only closed `RelationType` values are used;
- semantic roles match the local trace map;
- redaction rules are enforced;
- publishing exceptions are fail-open;
- disabled publishing is a no-op.

## Reducer tests

```bash
PYTHONPATH=. pytest tests/test_<domain>_substrate_reducer.py -q
PYTHONPATH=. pytest tests/test_<domain>_substrate_pipeline.py -q
```

Required assertions:

- trace grouping works;
- extraction works;
- malformed events are handled;
- stable delta ids are stable;
- receipts persist;
- cursor behavior is safe;
- no raw payload leakage.

## Frame runtime tests

```bash
PYTHONPATH=. pytest tests/test_<frame>_schemas.py -q
PYTHONPATH=. pytest tests/test_<frame>_policy_loader.py -q
PYTHONPATH=. pytest tests/test_<frame>_builder.py -q
```

Runtime services should also test store and worker behavior.

## Compile check

```bash
PYTHONPATH=. python -m compileall \
  orion/<domain> \
  orion/schemas/<schema_file>.py \
  services/<service> \
  -q
```

## Live proof: Layer 1/2 traces

```sql
select
    created_at
  , source_service
  , trace_id
  , event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
  , event_json::jsonb #>> '{atom,summary}' as summary
from grammar_events
where source_service = '<service>'
order by created_at desc
limit 20;
```

## Live proof: Layer 3 reduction

```sql
select
    created_at
  , receipt_id
  , delta_id
  , target_kind
  , target_id
  , operation
  , delta_json::jsonb #> '{pressure_hints}' as pressure_hints
from substrate_reduction_receipts
where target_kind = '<target_kind>'
order by created_at desc
limit 20;
```

## Live proof: Layer 4 field

```sql
select
    generated_at
  , field_json::jsonb -> 'nodes' as nodes
  , field_json::jsonb -> 'capabilities' as capabilities
  , field_json::jsonb -> 'recent_perturbations' as recent_perturbations
from substrate_field_state
order by generated_at desc
limit 1;
```

## Live proof: Layer 5 attention

```sql
select
    generated_at
  , frame_id
  , source_field_tick_id
  , frame_json #>> '{overall_salience}' as overall_salience
  , frame_json #> '{dominant_targets}' as dominant_targets
from substrate_attention_frames
order by generated_at desc
limit 1;
```

## Live proof: Layer 6 self-state

```sql
select
    generated_at
  , self_state_id
  , self_state_json #>> '{overall_condition}' as overall_condition
  , self_state_json #>> '{overall_intensity}' as overall_intensity
  , self_state_json #> '{summary_labels}' as summary_labels
from substrate_self_state
order by generated_at desc
limit 1;
```

## Live proof: Layer 7 proposal

```sql
select
    generated_at
  , frame_id
  , proposal_frame_json #>> '{overall_action_pressure}' as overall_action_pressure
  , proposal_frame_json #>> '{policy_required}' as policy_required
  , proposal_frame_json #> '{candidates}' as candidates
from substrate_proposal_frames
order by generated_at desc
limit 1;
```

## Live proof: Layer 8 policy

```sql
select
    generated_at
  , frame_id
  , policy_decision_frame_json #>> '{execution_allowed}' as execution_allowed
  , policy_decision_frame_json #>> '{operator_review_required}' as operator_review_required
  , jsonb_array_length(policy_decision_frame_json #> '{approved_decisions}') as approved_count
  , jsonb_array_length(policy_decision_frame_json #> '{review_required_decisions}') as review_count
from substrate_policy_decision_frames
order by generated_at desc
limit 1;
```

## Live proof: Layer 9 dispatch

```sql
select
    generated_at
  , frame_id
  , dispatch_frame_json #>> '{dispatch_mode}' as dispatch_mode
  , dispatch_frame_json #>> '{dispatch_attempted}' as dispatch_attempted
  , dispatch_frame_json #>> '{dispatch_count}' as dispatch_count
  , jsonb_array_length(dispatch_frame_json #> '{candidates}') as candidate_count
  , jsonb_array_length(dispatch_frame_json #> '{blocked_candidates}') as blocked_count
from substrate_execution_dispatch_frames
order by generated_at desc
limit 1;
```

## Live proof: Layer 10 feedback

```sql
select
    generated_at
  , frame_id
  , feedback_frame_json #>> '{outcome_status}' as outcome_status
  , feedback_frame_json #>> '{outcome_score}' as outcome_score
  , jsonb_array_length(feedback_frame_json #> '{observations}') as observation_count
  , feedback_frame_json #> '{absence_evidence}' as absence_evidence
from substrate_feedback_frames
order by generated_at desc
limit 1;
```

## Live proof: Layer 11 consolidation

```sql
select
    generated_at
  , frame_id
  , window_start
  , window_end
  , consolidation_frame_json #> '{dominant_motifs}' as dominant_motifs
  , consolidation_frame_json #> '{motif_observations}' as motif_observations
  , consolidation_frame_json #> '{source_counts}' as source_counts
from substrate_consolidation_frames
order by generated_at desc
limit 1;
```

## PR report proof levels

Use these labels:

```text
unit_tested
integration_tested
live_smoked
observed_in_sql
observed_in_hub
not_verified_live
```

Do not claim `live` without observed runtime evidence.

## Failure triage

If a downstream table is empty, check in order:

1. upstream source table has rows;
2. runtime container exists;
3. env flag is enabled;
4. migration was applied;
5. logs show save or failure;
6. policy/config path exists inside container;
7. source id has not already been processed idempotently;
8. Hub route was rebuilt if API proof is expected.