with chains as (
    select *
    from {{ ref('stg_visual_reverie_chains') }}
),

artifact_counts as (
    select
        visual_chain_id,
        count(*)::bigint as artifact_count
    from {{ ref('stg_visual_reverie_artifacts') }}
    group by visual_chain_id
),

with_previous as (
    select
        chains.*,
        lag(event_at) over (order by event_at, visual_chain_id) as previous_chain_at
    from chains
)

select
    with_previous.visual_chain_id,
    with_previous.event_at,
    with_previous.event_date,
    with_previous.outcome_key,
    with_previous.continuity_used_flag,
    coalesce(artifact_counts.artifact_count, 0)::bigint as artifact_count,
    case when artifact_counts.visual_chain_id is null then 1 else 0 end::smallint as missing_artifact_flag,
    extract(epoch from (with_previous.event_at - with_previous.previous_chain_at)) / 60.0
        as minutes_since_previous_chain,
    case
        when extract(epoch from (with_previous.event_at - with_previous.previous_chain_at)) / 60.0
             > {{ var('visual_chain_late_interval_minutes', 45) }}
            then 1
        else 0
    end::smallint as late_interval_flag
from with_previous
left join artifact_counts using (visual_chain_id)
