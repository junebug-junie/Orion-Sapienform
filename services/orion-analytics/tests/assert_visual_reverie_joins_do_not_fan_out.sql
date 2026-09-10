with joined_chains as (
    select
        f.visual_chain_id,
        o.outcome_key as matched_outcome_key,
        d.date_key as matched_date_key
    from {{ ref('fct_visual_reverie_chains') }} f
    left join {{ ref('dim_reverie_outcomes') }} o on f.outcome_key = o.outcome_key
    left join {{ ref('dim_reverie_dates') }} d on f.event_date = d.date_key
),

joined_artifacts as (
    select
        a.visual_artifact_id,
        c.visual_chain_id as matched_chain_id,
        d.date_key as matched_date_key
    from {{ ref('fct_visual_reverie_artifacts') }} a
    left join {{ ref('fct_visual_reverie_chains') }} c
      on a.visual_chain_id = c.visual_chain_id
    left join {{ ref('dim_reverie_dates') }} d on a.event_date = d.date_key
),

counts as (
    select
        (select count(*) from {{ ref('fct_visual_reverie_chains') }}) as chain_rows,
        (select count(*) from joined_chains) as joined_chain_rows,
        (select count(distinct visual_chain_id) from joined_chains) as distinct_joined_chain_ids,
        (select count(*) from joined_chains where matched_outcome_key is null or matched_date_key is null)
            as unmatched_chain_rows,
        (select count(*) from {{ ref('fct_visual_reverie_artifacts') }}) as artifact_rows,
        (select count(*) from joined_artifacts) as joined_artifact_rows,
        (select count(distinct visual_artifact_id) from joined_artifacts)
            as distinct_joined_artifact_ids,
        (select count(*) from joined_artifacts where matched_chain_id is null or matched_date_key is null)
            as unmatched_artifact_rows
)

select *
from counts
where chain_rows <> joined_chain_rows
   or joined_chain_rows <> distinct_joined_chain_ids
   or unmatched_chain_rows <> 0
   or artifact_rows <> joined_artifact_rows
   or joined_artifact_rows <> distinct_joined_artifact_ids
   or unmatched_artifact_rows <> 0
