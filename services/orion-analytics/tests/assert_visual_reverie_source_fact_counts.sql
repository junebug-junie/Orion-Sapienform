with counts as (
    select
        (select count(*) from {{ source('orion_operational', 'reverie_visual_chain') }})
            as source_chain_rows,
        (select count(*) from {{ ref('fct_visual_reverie_chains') }})
            as fact_chain_rows,
        (select count(distinct visual_chain_id) from {{ ref('fct_visual_reverie_chains') }})
            as distinct_fact_chain_ids,
        (select count(*) from {{ source('orion_operational', 'reverie_visual_artifact') }})
            as source_artifact_rows,
        (select count(*) from {{ ref('fct_visual_reverie_artifacts') }})
            as fact_artifact_rows,
        (select count(distinct visual_artifact_id) from {{ ref('fct_visual_reverie_artifacts') }})
            as distinct_fact_artifact_ids,
        (select coalesce(sum(artifact_count), 0) from {{ ref('fct_visual_reverie_chains') }})
            as rolled_up_artifact_rows
)

select *
from counts
where source_chain_rows <> fact_chain_rows
   or fact_chain_rows <> distinct_fact_chain_ids
   or source_artifact_rows <> fact_artifact_rows
   or fact_artifact_rows <> distinct_fact_artifact_ids
   or fact_artifact_rows <> rolled_up_artifact_rows
