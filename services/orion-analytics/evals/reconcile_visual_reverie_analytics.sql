-- Read-only evaluation. Every delta must be zero.
begin transaction read only;

with source_chain_count as (
    select count(*)::bigint as rows
    from public.reverie_visual_chain
),
fact_chain_count as (
    select
        count(*)::bigint as rows,
        count(distinct visual_chain_id)::bigint as distinct_ids,
        coalesce(sum(artifact_count), 0)::bigint as rolled_up_artifacts,
        sum(missing_artifact_flag)::bigint as chains_without_artifact,
        count(continuity_used_flag)::bigint as chains_with_continuity_marker,
        sum(continuity_used_flag)::bigint as chains_using_continuity,
        sum(late_interval_flag)::bigint as late_intervals,
        round(extract(epoch from (current_timestamp - max(event_at)))::numeric / 60.0, 2)
            as minutes_since_last_chain
    from analytics.fct_visual_reverie_chains
),
source_artifact_count as (
    select count(*)::bigint as rows
    from public.reverie_visual_artifact
),
fact_artifact_count as (
    select
        count(*)::bigint as rows,
        count(distinct visual_artifact_id)::bigint as distinct_ids,
        sum(captioned_flag)::bigint as captioned_artifacts
    from analytics.fct_visual_reverie_artifacts
),
joined_artifact_count as (
    select count(*)::bigint as rows
    from analytics.fct_visual_reverie_artifacts a
    left join analytics.fct_visual_reverie_chains c
      on a.visual_chain_id = c.visual_chain_id
    left join analytics.dim_reverie_dates d
      on a.event_date = d.date_key
)
select
    source_chain_count.rows as source_chain_rows,
    fact_chain_count.rows as fact_chain_rows,
    fact_chain_count.distinct_ids as distinct_fact_chain_ids,
    source_chain_count.rows - fact_chain_count.rows as chain_source_fact_delta,
    fact_chain_count.rows - fact_chain_count.distinct_ids as chain_duplicate_delta,
    source_artifact_count.rows as source_artifact_rows,
    fact_artifact_count.rows as fact_artifact_rows,
    fact_artifact_count.distinct_ids as distinct_fact_artifact_ids,
    source_artifact_count.rows - fact_artifact_count.rows as artifact_source_fact_delta,
    fact_artifact_count.rows - fact_artifact_count.distinct_ids as artifact_duplicate_delta,
    fact_artifact_count.rows - fact_chain_count.rolled_up_artifacts as artifact_rollup_delta,
    joined_artifact_count.rows - fact_artifact_count.rows as artifact_join_fanout_delta,
    fact_chain_count.chains_without_artifact,
    fact_chain_count.chains_with_continuity_marker,
    fact_chain_count.chains_using_continuity,
    fact_artifact_count.captioned_artifacts,
    fact_chain_count.late_intervals,
    fact_chain_count.minutes_since_last_chain
from source_chain_count, fact_chain_count, source_artifact_count,
     fact_artifact_count, joined_artifact_count;

rollback;
