with expected_continuity as (
    select
        chain_id as visual_chain_id,
        case
            when jsonb_typeof(chain_json -> 'continuity_streak') = 'number'
                then case
                    when (chain_json ->> 'continuity_streak')::numeric > 0 then 1
                    else 0
                end
            else null
        end::smallint as expected_continuity_used_flag
    from {{ source('orion_operational', 'reverie_visual_chain') }}
)

select
    fact.visual_chain_id,
    fact.missing_artifact_flag,
    fact.late_interval_flag,
    fact.continuity_used_flag,
    expected_continuity.expected_continuity_used_flag
from {{ ref('fct_visual_reverie_chains') }} fact
join expected_continuity using (visual_chain_id)
where fact.missing_artifact_flag <> case when fact.artifact_count = 0 then 1 else 0 end
   or fact.late_interval_flag <> case
       when fact.minutes_since_previous_chain > {{ var('visual_chain_late_interval_minutes', 45) }}
           then 1
       else 0
   end
   or fact.continuity_used_flag is distinct from expected_continuity.expected_continuity_used_flag
