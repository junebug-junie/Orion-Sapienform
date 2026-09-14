select journal_id::text as unsafe_id, 'grounding_status'::text as unsafe_field
from {{ source('curiosity_safe', 'curiosity_run_journals') }}
where grounding_status is not null
  and grounding_status not in (
      'grounded',
      'partial',
      'failed',
      'empty_draft',
      'invalid_request',
      'fcc_bad_model_label',
      'fcc_lane_context_too_small',
      'fcc_spawn_failed',
      'fcc_stream_stalled',
      'fcc_timeout',
      'fcc_stream_line_limit',
      'fcc_draft_length_ceiling_exceeded',
      'fcc_nonzero_exit',
      'fcc_context_overflow',
      'fcc_mcp_github_missing',
      'other'
  )

union all

select graph_write_id::text, 'element_type'
from {{ source('curiosity_safe', 'curiosity_run_graph_writes') }}
where element_type !~ '^((-> )?[A-Za-z][A-Za-z0-9_]{0,62})$'

union all

select material_pool_id::text, 'material_kind'
from {{ source('curiosity_safe', 'curiosity_run_material_pool') }}
where material_kind !~ '^[a-z][a-z0-9_]{0,62}$'
