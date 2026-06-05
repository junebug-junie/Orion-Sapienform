#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  echo "Usage: $0 --mode m3|m4|m5|full-observe"
  echo ""
  echo "  m3          transport_bus_reducer receipts + substrate_transport_bus_projection"
  echo "  m4          capability:transport in substrate_field_state"
  echo "  m5          capability:transport in any attention bucket (dominant/capability/suppressed)"
  echo "  full-observe layers 6-11: self_state, proposals, policy, dispatch, feedback, consolidation"
  exit 1
}

[[ "$MODE" == --mode=* ]] || usage
MODE="${MODE#--mode=}"

DB="${DB:-orion-athena-sql-db}"

# Wrapper so each query fails loudly but we can catch specific failures.
psql_run() {
  docker exec -i "$DB" psql -U postgres -d conjourney -c "$1"
}

case "$MODE" in
  m3)
    echo "== m3: transport_bus_reducer receipts =="
    # substrate_reduction_receipts uses reducer_name as a physical column (added in migration_receipt_retention_v1).
    # target_kind / target_id live inside receipt_json, not as top-level columns.
    psql_run "
      SELECT
        created_at,
        reducer_name,
        receipt_kind,
        receipt_status,
        receipt_json::jsonb #>> '{state_deltas,0,target_kind}' AS target_kind,
        receipt_json::jsonb #>> '{state_deltas,0,target_id}'   AS target_id,
        receipt_json::jsonb #>> '{state_deltas,0,after,pressure_hints,catalog_drift_pressure}'
          AS catalog_drift,
        receipt_json::jsonb #>> '{state_deltas,0,after,pressure_hints,transport_pressure}'
          AS transport_pressure
      FROM substrate_reduction_receipts
      WHERE reducer_name = 'transport_bus_reducer'
      ORDER BY created_at DESC
      LIMIT 5;"

    echo ""
    echo "== m3: current substrate_transport_bus_projection =="
    psql_run "
      SELECT
        projection_id,
        updated_at,
        projection_json::jsonb #>> '{schema_version}' AS schema_version,
        jsonb_object_keys(projection_json::jsonb -> 'buses') AS bus_ids
      FROM substrate_transport_bus_projection
      ORDER BY updated_at DESC
      LIMIT 3;"
    ;;

  m4)
    echo "== m4: capability:transport in substrate_field_state =="
    psql_run "
      SELECT
        generated_at,
        c ->> 'id'       AS capability_id,
        c -> 'channels'  AS channels
      FROM substrate_field_state
      CROSS JOIN LATERAL jsonb_array_elements(field_json::jsonb -> 'capabilities') AS c
      WHERE c ->> 'id' = 'capability:transport'
      ORDER BY generated_at DESC
      LIMIT 3;"
    ;;

  m5)
    echo "== m5: capability:transport in attention frame buckets =="

    # Check all three buckets; suppressed is valid when transport is healthy but below salience threshold.
    for bucket in dominant_targets capability_targets suppressed_targets; do
      echo ""
      echo "-- bucket: $bucket --"
      psql_run "
        SELECT
          generated_at,
          '$bucket'                           AS bucket,
          item ->> 'target_id'                AS target_id,
          item ->> 'salience_score'           AS salience,
          item ->> 'suggested_observation_mode' AS obs_mode,
          item -> 'dominant_channels'         AS dominant_channels
        FROM substrate_attention_frames
        CROSS JOIN LATERAL jsonb_array_elements(frame_json::jsonb -> '$bucket') AS item
        WHERE item ->> 'target_id' = 'capability:transport'
        ORDER BY generated_at DESC
        LIMIT 5;"
    done

    echo ""
    echo "== m5: summary — which bucket(s) contain capability:transport (most recent frame) =="
    psql_run "
      WITH latest AS (
        SELECT frame_json::jsonb AS fj
        FROM substrate_attention_frames
        ORDER BY generated_at DESC
        LIMIT 1
      ),
      buckets AS (
        SELECT 'dominant_targets'    AS bucket, item FROM latest, jsonb_array_elements(fj -> 'dominant_targets')    AS item
        UNION ALL
        SELECT 'capability_targets'  AS bucket, item FROM latest, jsonb_array_elements(fj -> 'capability_targets')  AS item
        UNION ALL
        SELECT 'suppressed_targets'  AS bucket, item FROM latest, jsonb_array_elements(fj -> 'suppressed_targets')  AS item
      )
      SELECT bucket, item ->> 'target_id' AS target_id, item ->> 'salience_score' AS salience
      FROM buckets
      WHERE item ->> 'target_id' = 'capability:transport';"
    ;;

  full-observe)
    echo "== full-observe: layers 6-11 (requires all flags enabled) =="
    echo ""

    echo "-- L6 self_state: transport_integrity dimension --"
    psql_run "
      SELECT generated_at,
             self_state_json::jsonb #> '{dimensions,transport_integrity}' AS transport_integrity
      FROM substrate_self_state
      ORDER BY generated_at DESC
      LIMIT 1;" || true

    echo ""
    echo "-- L7 proposals: transport candidate count (bounded) --"
    psql_run "
      SELECT generated_at,
             jsonb_array_length(proposal_frame_json::jsonb -> 'candidates') AS total_candidates,
             (
               SELECT count(*)
               FROM jsonb_array_elements(proposal_frame_json::jsonb -> 'candidates') AS c
               WHERE c ->> 'proposal_id' ILIKE '%transport%'
             ) AS transport_candidates
      FROM substrate_proposal_frames
      ORDER BY generated_at DESC
      LIMIT 1;" || true

    echo ""
    echo "-- L8 policy: transport-related approved decisions (bounded) --"
    psql_run "
      SELECT generated_at,
             (
               SELECT count(*)
               FROM jsonb_array_elements(policy_decision_frame_json::jsonb -> 'approved_decisions') AS d
               WHERE d ->> 'proposal_id' ILIKE '%transport%'
             ) AS transport_approved
      FROM substrate_policy_decision_frames
      ORDER BY generated_at DESC
      LIMIT 1;" || true

    echo ""
    echo "-- L9 dispatch: dispatch_mode --"
    psql_run "
      SELECT generated_at,
             dispatch_frame_json::jsonb #>> '{dispatch_mode}' AS dispatch_mode
      FROM substrate_execution_dispatch_frames
      ORDER BY generated_at DESC
      LIMIT 1;" || true

    echo ""
    echo "-- L10 feedback: outcome_status --"
    psql_run "
      SELECT generated_at,
             feedback_frame_json::jsonb #>> '{outcome_status}' AS outcome_status
      FROM substrate_feedback_frames
      ORDER BY generated_at DESC
      LIMIT 1;" || true

    echo ""
    echo "-- L11 consolidation: transport motif observations (bounded) --"
    psql_run "
      SELECT generated_at,
             m ->> 'label' AS motif_label,
             m ->> 'strength' AS strength
      FROM substrate_consolidation_frames
      CROSS JOIN LATERAL jsonb_array_elements(
        consolidation_frame_json::jsonb -> 'motif_observations'
      ) AS m
      WHERE m ->> 'label' ILIKE 'transport_%'
      ORDER BY generated_at DESC
      LIMIT 10;" || true
    ;;

  *)
    usage
    ;;
esac

echo ""
echo "smoke_orion_bus_transport_full_stack: $MODE complete"
