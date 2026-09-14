select
    material_pool_id,
    journal_id,
    run_id,
    journaled_at as event_at,
    journal_date as event_date,
    run_type,
    material_kind,
    available_count
from {{ ref('stg_curiosity_run_material_pool') }}
