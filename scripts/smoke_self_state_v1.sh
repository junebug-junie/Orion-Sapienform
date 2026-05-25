#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest attention frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_field_tick_id,
    frame_json #>> '{overall_salience}' as overall_salience,
    frame_json #> '{dominant_targets}' as dominant_targets
from substrate_attention_frames
order by generated_at desc
limit 1;
"

echo ""
echo "=== Latest self-state ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    self_state_id,
    source_field_tick_id,
    source_attention_frame_id,
    self_state_json #>> '{overall_condition}' as overall_condition,
    self_state_json #>> '{overall_intensity}' as overall_intensity,
    self_state_json #> '{dimensions}' as dimensions,
    self_state_json #> '{dominant_attention_targets}' as dominant_attention_targets,
    self_state_json #> '{summary_labels}' as summary_labels
from substrate_self_state
order by generated_at desc
limit 1;
"
