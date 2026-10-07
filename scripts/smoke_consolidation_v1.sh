#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-orion-athena}"
DB="${PROJECT}-sql-db"
PSQL=(docker exec -i "$DB" psql -U postgres -d conjourney -v ON_ERROR_STOP=1)

echo "=== Latest feedback frame ==="
"${PSQL[@]}" -c "
select
    generated_at,
    frame_id,
    feedback_frame_json #>> '{outcome_status}' as outcome_status,
    feedback_frame_json #>> '{outcome_score}' as outcome_score,
    jsonb_array_length(feedback_frame_json #> '{observations}') as observation_count,
    feedback_frame_json #> '{absence_evidence}' as absence_evidence
from substrate_feedback_frames
order by generated_at desc
limit 1;
"

echo "=== Latest consolidation frame ==="
"${PSQL[@]}" -c "
select
    generated_at,
    frame_id,
    window_start,
    window_end,
    consolidation_frame_json #> '{dominant_motifs}' as dominant_motifs,
    consolidation_frame_json #> '{motif_observations}' as motif_observations,
    consolidation_frame_json #> '{source_counts}' as source_counts
from substrate_consolidation_frames
order by generated_at desc
limit 1;
"
