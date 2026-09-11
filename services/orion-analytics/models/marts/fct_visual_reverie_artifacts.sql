select
    visual_artifact_id,
    visual_chain_id,
    step_index,
    artifact_bytes,
    captioned_flag,
    event_at,
    event_date
from {{ ref('stg_visual_reverie_artifacts') }}
