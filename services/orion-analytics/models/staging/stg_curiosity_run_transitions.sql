select
    transition_id,
    run_id,
    workflow,
    node,
    next_node,
    status,
    resumed_from_node,
    event_at,
    (event_at at time zone 'UTC')::date as event_date,
    stored_at,
    run_type,
    attempt_count,
    error_present_flag
from {{ source('curiosity_safe', 'curiosity_run_transitions') }}
