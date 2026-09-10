with joined as (
    select
        f.reverie_chain_id,
        o.outcome_key as matched_outcome_key,
        d.date_key as matched_date_key
    from {{ ref('fct_reverie_chains') }} f
    left join {{ ref('dim_reverie_outcomes') }} o
      on f.outcome_key = o.outcome_key
    left join {{ ref('dim_reverie_dates') }} d
      on f.event_date = d.date_key
),
counts as (
    select
        (select count(*) from {{ ref('fct_reverie_chains') }}) as fact_rows,
        count(*) as joined_rows,
        count(distinct reverie_chain_id) as distinct_joined_ids,
        count(*) filter (where matched_outcome_key is null or matched_date_key is null) as unmatched_rows
    from joined
)

select *
from counts
where fact_rows <> joined_rows
   or joined_rows <> distinct_joined_ids
   or unmatched_rows <> 0
