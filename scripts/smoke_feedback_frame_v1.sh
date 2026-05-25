#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-orion-athena}"
DB="${PROJECT}-sql-db"
PSQL=(docker exec -i "$DB" psql -U postgres -d conjourney -v ON_ERROR_STOP=1)

echo "=== Latest execution dispatch frame ==="
"${PSQL[@]}" -c "
select
    generated_at,
    frame_id,
    dispatch_frame_json #>> '{dispatch_mode}' as dispatch_mode,
    dispatch_frame_json #>> '{dispatch_attempted}' as dispatch_attempted,
    dispatch_frame_json #> '{candidates}' as candidates
from substrate_execution_dispatch_frames
order by generated_at desc
limit 1;
"

echo "=== Latest feedback frame ==="
"${PSQL[@]}" -c "
select
    generated_at,
    frame_id,
    source_execution_dispatch_frame_id,
    feedback_frame_json #>> '{outcome_status}' as outcome_status,
    feedback_frame_json #>> '{outcome_score}' as outcome_score,
    feedback_frame_json #> '{observations}' as observations,
    feedback_frame_json #> '{positive_evidence}' as positive_evidence,
    feedback_frame_json #> '{negative_evidence}' as negative_evidence,
    feedback_frame_json #> '{absence_evidence}' as absence_evidence
from substrate_feedback_frames
order by generated_at desc
limit 1;
"
