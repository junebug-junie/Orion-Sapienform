-- Read-only evaluation. Success means all three deltas are zero.
begin transaction read only;

with source_count as (
    select count(*)::bigint as rows
    from public.substrate_reverie_chain
),
fact_count as (
    select count(*)::bigint as rows,
           count(distinct reverie_chain_id)::bigint as distinct_ids
    from analytics.fct_reverie_chains
),
joined_count as (
    select count(*)::bigint as rows
    from analytics.fct_reverie_chains f
    left join analytics.dim_reverie_outcomes o on f.outcome_key = o.outcome_key
    left join analytics.dim_reverie_dates d on f.event_date = d.date_key
),
daily as (
    select d.date_key, count(f.reverie_chain_id)::bigint as chain_count
    from analytics.dim_reverie_dates d
    left join analytics.fct_reverie_chains f on f.event_date = d.date_key
    group by d.date_key
)
select
    source_count.rows as source_rows,
    fact_count.rows as fact_rows,
    fact_count.distinct_ids as distinct_fact_ids,
    joined_count.rows as joined_rows,
    source_count.rows - fact_count.rows as source_fact_delta,
    fact_count.rows - fact_count.distinct_ids as duplicate_fact_delta,
    joined_count.rows - fact_count.rows as join_fanout_delta,
    count(*) filter (where daily.chain_count = 0) as zero_activity_days
from source_count, fact_count, joined_count, daily
group by source_count.rows, fact_count.rows, fact_count.distinct_ids, joined_count.rows;

rollback;
