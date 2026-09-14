select
    transition_id,
    run_id,
    workflow,
    node,
    next_node,
    status,
    resumed_from_node,
    event_at,
    event_date,
    case when status = 'failed' then 1 else 0 end::smallint as failed_transition_flag,
    case when status = 'resumed' then 1 else 0 end::smallint as resumed_transition_flag,
    error_present_flag
from {{ ref('stg_curiosity_run_transitions') }}
