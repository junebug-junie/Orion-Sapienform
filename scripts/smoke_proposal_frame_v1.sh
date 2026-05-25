#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest self-state ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    self_state_id,
    self_state_json #>> '{overall_condition}' as overall_condition,
    self_state_json #>> '{overall_intensity}' as overall_intensity,
    self_state_json #> '{summary_labels}' as summary_labels
from substrate_self_state
order by generated_at desc
limit 1;
"

echo ""
echo "=== Latest proposal frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_self_state_id,
    proposal_frame_json #>> '{overall_action_pressure}' as overall_action_pressure,
    proposal_frame_json #>> '{overall_risk}' as overall_risk,
    proposal_frame_json #>> '{policy_required}' as policy_required,
    proposal_frame_json #> '{candidates}' as candidates
from substrate_proposal_frames
order by generated_at desc
limit 1;
"
