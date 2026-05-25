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
    source_self_state_id,
    proposal_frame_json #>> '{overall_action_pressure}' as overall_action_pressure,
    proposal_frame_json #> '{candidates}' as candidates
from substrate_proposal_frames
order by generated_at desc
limit 1;
"

echo ""
echo "=== Latest policy decision frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_proposal_frame_id,
    policy_decision_frame_json #>> '{overall_risk}' as overall_risk,
    policy_decision_frame_json #>> '{operator_review_required}' as operator_review_required,
    policy_decision_frame_json #>> '{execution_allowed}' as execution_allowed,
    policy_decision_frame_json #> '{approved_decisions}' as approved_decisions,
    policy_decision_frame_json #> '{review_required_decisions}' as review_required_decisions,
    policy_decision_frame_json #> '{rejected_decisions}' as rejected_decisions
from substrate_policy_decision_frames
order by generated_at desc
limit 1;
"
