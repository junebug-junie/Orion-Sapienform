with counts as (
    select
        (select count(*) from {{ source('orion_operational', 'substrate_reverie_chain') }}) as source_rows,
        (select count(*) from {{ ref('fct_reverie_chains') }}) as fact_rows,
        (select count(distinct reverie_chain_id) from {{ ref('fct_reverie_chains') }}) as distinct_fact_ids
)

select *
from counts
where source_rows <> fact_rows
   or fact_rows <> distinct_fact_ids
