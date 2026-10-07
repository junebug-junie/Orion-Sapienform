#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest proposal frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_field_tick_id,
    proposal_frame_json #>> '{overall_action_pressure}' as overall_action_pressure,
    proposal_frame_json #>> '{overall_risk}' as overall_risk,
    proposal_frame_json #>> '{policy_required}' as policy_required,
    proposal_frame_json #> '{candidates}' as candidates
from substrate_proposal_frames
order by generated_at desc
limit 1;
"
