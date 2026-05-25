#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest policy decision frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_proposal_frame_id,
    policy_decision_frame_json #>> '{execution_allowed}' as execution_allowed,
    policy_decision_frame_json #> '{approved_decisions}' as approved_decisions,
    policy_decision_frame_json #> '{review_required_decisions}' as review_required_decisions
from substrate_policy_decision_frames
order by generated_at desc
limit 1;
"

echo ""
echo "=== Latest execution dispatch frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_policy_frame_id,
    dispatch_frame_json #>> '{dispatch_mode}' as dispatch_mode,
    dispatch_frame_json #>> '{dispatch_attempted}' as dispatch_attempted,
    dispatch_frame_json #>> '{dispatch_count}' as dispatch_count,
    dispatch_frame_json #> '{candidates}' as candidates,
    dispatch_frame_json #> '{blocked_candidates}' as blocked_candidates
from substrate_execution_dispatch_frames
order by generated_at desc
limit 1;
"
