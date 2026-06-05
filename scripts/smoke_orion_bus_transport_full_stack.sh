#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  echo "Usage: $0 --mode m3|m4|m5|full-observe"
  exit 1
}

[[ "$MODE" == --mode=* ]] || usage
MODE="${MODE#--mode=}"

DB="${DB:-orion-athena-sql-db}"

case "$MODE" in
  m3)
    echo "== m3: bus traces + transport_bus receipts =="
    docker exec -i "$DB" psql -U postgres -d conjourney -c "
      select count(*) as bus_traces
      from grammar_events
      where source_service = 'orion-bus' and trace_id like 'bus.transport:%';"
    docker exec -i "$DB" psql -U postgres -d conjourney -c "
      select created_at, target_kind, target_id,
             receipt_json::jsonb #>> '{state_deltas,0,after,pressure_hints,catalog_drift_pressure}' as catalog_drift
      from substrate_reduction_receipts
      where target_kind = 'transport_bus'
      order by created_at desc limit 5;"
    ;;
  m4)
    echo "== m4: field capability:transport =="
    docker exec -i "$DB" psql -U postgres -d conjourney -c "
      select generated_at,
             c ->> 'id' as capability_id,
             c -> 'channels' as channels
      from substrate_field_state
      cross join lateral jsonb_array_elements(field_json::jsonb -> 'capabilities') c
      where c ->> 'id' = 'capability:transport'
      order by generated_at desc limit 3;"
    ;;
  m5)
    echo "== m5: attention transport visibility =="
    docker exec -i "$DB" psql -U postgres -d conjourney -c "
      select generated_at,
             item ->> 'target_id' as target_id,
             item ->> 'salience_score' as salience,
             item -> 'dominant_channels' as channels
      from substrate_attention_frames
      cross join lateral jsonb_array_elements(frame_json #> '{capability_targets}') item
      where item ->> 'target_id' in ('capability:transport', 'node:athena')
      order by generated_at desc limit 10;"
    ;;
  full-observe)
    echo "== full-observe: layers 6-11 (requires flags enabled) =="
    for q in \
      "select generated_at, self_state_json #> '{dimensions,transport_integrity}' from substrate_self_state order by generated_at desc limit 1" \
      "select generated_at, proposal_frame_json #> '{candidates}' from substrate_proposal_frames order by generated_at desc limit 1" \
      "select generated_at, policy_decision_frame_json #> '{approved_decisions}' from substrate_policy_decision_frames order by generated_at desc limit 1" \
      "select generated_at, dispatch_frame_json #>> '{dispatch_mode}' from substrate_execution_dispatch_frames order by generated_at desc limit 1" \
      "select generated_at, feedback_frame_json #>> '{outcome_status}' from substrate_feedback_frames order by generated_at desc limit 1" \
      "select m ->> 'label' from substrate_consolidation_frames cross join lateral jsonb_array_elements(consolidation_frame_json #> '{motif_observations}') m where m ->> 'label' like 'transport_%' order by generated_at desc limit 5"; do
      docker exec -i "$DB" psql -U postgres -d conjourney -c "$q" || true
    done
    ;;
  *)
    usage
    ;;
esac

echo "smoke_orion_bus_transport_full_stack: $MODE complete"
