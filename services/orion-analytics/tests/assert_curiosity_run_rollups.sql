with expected_transitions as (
    select
        run_id,
        count(*)::bigint as transition_count,
        count(*) filter (where status = 'failed')::bigint as failed_transition_count,
        count(*) filter (where status = 'resumed')::bigint as resumed_transition_count,
        case when count(*) filter (where status = 'completed') > 0 then 1 else 0 end::smallint
            as completed_flag
    from {{ source('curiosity_safe', 'curiosity_run_transitions') }}
    group by run_id
),
expected_journals as (
    select run_id, count(*)::bigint as journal_record_count
    from {{ source('curiosity_safe', 'curiosity_run_journals') }}
    group by run_id
),
expected_graph as (
    select run_id, sum(element_count)::bigint as graph_element_count
    from {{ source('curiosity_safe', 'curiosity_run_graph_writes') }}
    group by run_id
)

select actual.run_id
from {{ ref('fct_curiosity_runs') }} actual
left join expected_transitions transitions using (run_id)
left join expected_journals journals using (run_id)
left join expected_graph graph using (run_id)
where actual.transition_count is distinct from coalesce(transitions.transition_count, 0)
   or actual.failed_transition_count is distinct from coalesce(transitions.failed_transition_count, 0)
   or actual.resumed_transition_count is distinct from coalesce(transitions.resumed_transition_count, 0)
   or actual.completed_flag is distinct from coalesce(transitions.completed_flag, 0)
   or actual.journal_record_count is distinct from coalesce(journals.journal_record_count, 0)
   or (
        actual.graph_write_status = 'wrote_elements'
        and actual.graph_element_count is distinct from graph.graph_element_count
   )
   or (
        actual.graph_write_status = 'wrote_nothing'
        and actual.graph_element_count is distinct from 0
   )
