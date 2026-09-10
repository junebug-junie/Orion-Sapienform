select
    reverie_chain_id,
    event_at,
    event_date,
    outcome_key
from {{ ref('stg_reverie_chains') }}
