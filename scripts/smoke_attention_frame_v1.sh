#!/usr/bin/env bash
set -euo pipefail
DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest field state ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    tick_id,
    field_json #> '{node_vectors,node:athena}' as athena,
    field_json #> '{capability_vectors,capability:orchestration}' as orchestration
from substrate_field_state
order by generated_at desc
limit 1;"

echo "=== Latest attention frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_field_tick_id,
    frame_json #> '{dominant_targets}' as dominant_targets,
    frame_json #>> '{overall_salience}' as overall_salience
from substrate_attention_frames
order by generated_at desc
limit 1;"
