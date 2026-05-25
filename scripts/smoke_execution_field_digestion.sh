#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

run_sql() {
  docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "$1"
}

echo "=== Orion execution → field digestion smoke (read-only) ==="
echo "DB=$DB PGDATABASE=$PGDATABASE PGUSER=$PGUSER"
echo ""

echo "--- 1. Latest cortex-exec grammar events ---"
run_sql "
select created_at, event_id, trace_id,
       event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
from grammar_events
where source_service = 'orion-cortex-exec'
  and trace_id like 'cortex.exec:%'
order by created_at desc
limit 10;
"

echo "--- 2. Latest execution_run receipts ---"
run_sql "
select r.created_at, r.receipt_id, d ->> 'delta_id' as delta_id,
       d ->> 'target_kind' as target_kind,
       d ->> 'target_id' as target_id,
       d #> '{after,pressure_hints}' as pressure_hints
from substrate_reduction_receipts r
cross join lateral jsonb_array_elements(
  coalesce(r.receipt_json::jsonb -> 'state_deltas', '[]'::jsonb)
) d
where d ->> 'target_kind' = 'execution_run'
order by r.created_at desc
limit 10;
"

echo "--- 3. Applied execution deltas ---"
run_sql "
with exec_deltas as (
  select r.receipt_id, d ->> 'delta_id' as delta_id
  from substrate_reduction_receipts r
  cross join lateral jsonb_array_elements(
    coalesce(r.receipt_json::jsonb -> 'state_deltas', '[]'::jsonb)
  ) d
  where d ->> 'target_kind' = 'execution_run'
)
select a.applied_at, a.delta_id, a.receipt_id
from substrate_field_applied_deltas a
join exec_deltas e on e.delta_id = a.delta_id
order by a.applied_at desc
limit 10;
"

echo "--- 4. Latest field vector ---"
run_sql "
select generated_at,
       field_json::jsonb -> 'node_vectors' -> 'node:athena' as athena_vector,
       field_json::jsonb -> 'capability_vectors' -> 'capability:orchestration' as orchestration_vector,
       field_json::jsonb -> 'recent_perturbations' as recent_perturbations
from substrate_field_state
order by generated_at desc
limit 1;
"

if [[ "${RESET_FIELD_STATE:-0}" == "1" ]]; then
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "WARNING: RESET_FIELD_STATE=1 — destructive dev reset requested."
  echo "This deletes substrate_field_state, cursor, and applied execution deltas."
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  sleep 3
  run_sql "delete from substrate_field_applied_deltas where delta_id in (
    select d ->> 'delta_id'
    from substrate_reduction_receipts r
    cross join lateral jsonb_array_elements(
      coalesce(r.receipt_json::jsonb -> 'state_deltas', '[]'::jsonb)
    ) d
    where d ->> 'target_kind' = 'execution_run'
  );"
  run_sql "delete from substrate_field_state;"
  run_sql "delete from substrate_field_digester_cursor;"
  echo "Reset complete."
fi

echo ""
echo "Done. Optional destructive reset: RESET_FIELD_STATE=1 $0"
