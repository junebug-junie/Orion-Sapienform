select
    journal_id,
    run_id,
    journaled_at,
    (journaled_at at time zone 'UTC')::date as journal_date,
    run_type,
    offered_concept_count,
    available_concept_count,
    offered_relation_count,
    available_relation_count,
    harness_step_count,
    grounding_status,
    whole_turn_seconds,
    harness_seconds,
    graph_write_status
from {{ source('curiosity_safe', 'curiosity_run_journals') }}
