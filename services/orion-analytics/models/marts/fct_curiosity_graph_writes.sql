select
    graph_write_id,
    journal_id,
    run_id,
    journaled_at as event_at,
    journal_date as event_date,
    run_type,
    element_type,
    element_family,
    element_count
from {{ ref('stg_curiosity_run_graph_writes') }}
