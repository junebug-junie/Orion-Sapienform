with counts as (
    select
        (select count(*) from {{ source('curiosity_safe', 'curiosity_run_transitions') }})
            as source_transition_rows,
        (select count(*) from {{ ref('fct_curiosity_run_transitions') }})
            as fact_transition_rows,
        (select count(distinct transition_id) from {{ ref('fct_curiosity_run_transitions') }})
            as distinct_fact_transition_ids,
        (
            select count(*)
            from (
                select run_id from {{ source('curiosity_safe', 'curiosity_run_transitions') }}
                union
                select run_id from {{ source('curiosity_safe', 'curiosity_run_journals') }}
            ) observed
        ) as source_observed_runs,
        (select count(*) from {{ ref('fct_curiosity_runs') }}) as fact_observed_runs,
        (select count(distinct run_id) from {{ ref('fct_curiosity_runs') }})
            as distinct_fact_run_ids,
        (select count(*) from {{ source('curiosity_safe', 'curiosity_run_graph_writes') }})
            as source_graph_rows,
        (select count(*) from {{ ref('fct_curiosity_graph_writes') }}) as fact_graph_rows,
        (select count(distinct graph_write_id) from {{ ref('fct_curiosity_graph_writes') }})
            as distinct_fact_graph_ids,
        (select count(*) from {{ source('curiosity_safe', 'curiosity_run_material_pool') }})
            as source_material_rows,
        (select count(*) from {{ ref('fct_curiosity_material_pool') }}) as fact_material_rows,
        (select count(distinct material_pool_id) from {{ ref('fct_curiosity_material_pool') }})
            as distinct_fact_material_ids
)

select *
from counts
where source_transition_rows <> fact_transition_rows
   or fact_transition_rows <> distinct_fact_transition_ids
   or source_observed_runs <> fact_observed_runs
   or fact_observed_runs <> distinct_fact_run_ids
   or source_graph_rows <> fact_graph_rows
   or fact_graph_rows <> distinct_fact_graph_ids
   or source_material_rows <> fact_material_rows
   or fact_material_rows <> distinct_fact_material_ids
