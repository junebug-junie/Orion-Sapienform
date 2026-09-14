with counts as (
    select
        (select count(*) from {{ ref('fct_curiosity_run_transitions') }}) as transition_rows,
        (
            select count(*)
            from {{ ref('fct_curiosity_run_transitions') }} child
            join {{ ref('fct_curiosity_runs') }} parent using (run_id)
        ) as joined_transition_rows,
        (select count(*) from {{ ref('fct_curiosity_graph_writes') }}) as graph_rows,
        (
            select count(*)
            from {{ ref('fct_curiosity_graph_writes') }} child
            join {{ ref('fct_curiosity_runs') }} parent using (run_id)
        ) as joined_graph_rows,
        (select count(*) from {{ ref('fct_curiosity_material_pool') }}) as material_rows,
        (
            select count(*)
            from {{ ref('fct_curiosity_material_pool') }} child
            join {{ ref('fct_curiosity_runs') }} parent using (run_id)
        ) as joined_material_rows
)

select *
from counts
where transition_rows <> joined_transition_rows
   or graph_rows <> joined_graph_rows
   or material_rows <> joined_material_rows
