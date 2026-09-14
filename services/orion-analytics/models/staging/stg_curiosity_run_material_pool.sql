select
    material_pool_id,
    journal_id,
    run_id,
    journaled_at,
    (journaled_at at time zone 'UTC')::date as journal_date,
    run_type,
    material_kind,
    available_count
from {{ source('curiosity_safe', 'curiosity_run_material_pool') }}
