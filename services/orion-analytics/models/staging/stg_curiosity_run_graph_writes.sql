select
    graph_write_id,
    journal_id,
    run_id,
    journaled_at,
    (journaled_at at time zone 'UTC')::date as journal_date,
    run_type,
    element_type,
    element_family,
    element_count
from {{ source('curiosity_safe', 'curiosity_run_graph_writes') }}
