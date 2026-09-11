select
    sha256 as visual_artifact_id,
    chain_id as visual_chain_id,
    step_index,
    bytes as artifact_bytes,
    case
        when description is not null and btrim(description) <> '' then 1
        else 0
    end::smallint as captioned_flag,
    created_at as event_at,
    (created_at at time zone 'UTC')::date as event_date
from {{ source('orion_operational', 'reverie_visual_artifact') }}
