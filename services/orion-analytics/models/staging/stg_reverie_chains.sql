select
    chain_id as reverie_chain_id,
    created_at as event_at,
    (created_at at time zone 'UTC')::date as event_date,
    terminal_reason as outcome_key,
    stored_at
from {{ source('orion_operational', 'substrate_reverie_chain') }}
