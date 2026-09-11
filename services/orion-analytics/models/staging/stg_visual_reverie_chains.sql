select
    chain_id as visual_chain_id,
    created_at as event_at,
    (created_at at time zone 'UTC')::date as event_date,
    terminal_reason as outcome_key,
    case
        when jsonb_typeof(chain_json -> 'continuity_streak') = 'number'
            then case when (chain_json ->> 'continuity_streak')::numeric > 0 then 1 else 0 end
        else null
    end::smallint as continuity_used_flag,
    stored_at
from {{ source('orion_operational', 'reverie_visual_chain') }}
