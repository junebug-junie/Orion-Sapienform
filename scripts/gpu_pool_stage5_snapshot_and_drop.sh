#!/usr/bin/env bash
# GPU pool stage 5.6: snapshot, verify, then (only with --drop) drop the four dead legacy tables.
#
# Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md
# (Decision 5; "Juniper's answers" 3). Runs ONLY on Juniper's go. Backfill protocol (AGENTS.md s.14):
# snapshot first, progress log, report, before/after CSV, all under $OUT.
#
#   durable_gateway_permits, durable_resource_leases, durable_resource_demands, durable_elastic_slot
#   (+ the orphan sequence durable_resource_fencing_generation). durable_admission_runs and
#   durable_resource_events are the live run registry and are never touched.
#
# Usage (athena, from a worktree or the checkout -- it only reads the migration file):
#   scripts/gpu_pool_stage5_snapshot_and_drop.sh --cutoff '2026-09-29 22:45:00+00'          # snapshot + verify only
#   scripts/gpu_pool_stage5_snapshot_and_drop.sh --cutoff '2026-09-29 22:45:00+00' --drop   # ... then drop
#
# Refuses (exit 2, nothing dropped) when:
#   - any row in the four tables was written after --cutoff (permit granted/heartbeated, lease
#     granted/heartbeated, demand created, elastic slot touched), or a permit/lease is still active or a
#     demand still pending: something still writes them, so the 5.4 cutover is not finished;
#   - the gzipped dump's COPY row counts differ from the live counts;
#   - at drop time, under ACCESS EXCLUSIVE locks, the counts or the newest write moved since the dump.
# The drop is one transaction (lock -> re-check -> DROP); any failure rolls it back.
#
# Env overrides (tests): SQL_CONTAINER (orion-athena-sql-db), PGDATABASE_NAME (conjourney),
# PGUSER_NAME (postgres), OUT (/tmp/gpu-pool-stage5-drop).
set -euo pipefail

SQL_CONTAINER="${SQL_CONTAINER:-orion-athena-sql-db}"
DB="${PGDATABASE_NAME:-conjourney}"
PGU="${PGUSER_NAME:-postgres}"
OUT="${OUT:-/tmp/gpu-pool-stage5-drop}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATION="$REPO_ROOT/services/orion-sql-db/manual_migration_gpu_pool_stage5_drop_legacy_tables.sql"
TABLES=(durable_gateway_permits durable_resource_leases durable_resource_demands durable_elastic_slot)
SEQUENCE=durable_resource_fencing_generation

CUTOFF=""
DROP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cutoff) CUTOFF="${2:-}"; shift 2 ;;
    --drop) DROP=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
[[ -n "$CUTOFF" ]] || { echo "--cutoff '<timestamptz>' is required (e.g. the 5.4 deploy: '2026-09-29 22:45:00+00')" >&2; exit 64; }
[[ -f "$MIGRATION" ]] || { echo "missing $MIGRATION" >&2; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT/dump_verify.csv.partial"
LOG="$OUT/progress.log"
DUMP="$OUT/legacy_tables.sql.gz"
START=$(date +%s)
log() {  # event title | percent | elapsed | detail
  local line
  line="$(date -u +%FT%TZ) | $1 | ${2}% | elapsed=$(( $(date +%s) - START ))s | ${3:-}"
  echo "$line" | tee -a "$LOG"
}
refuse() { log "REFUSED" "$1" "$2"; echo "REFUSED: $2 -- nothing dropped" >&2; exit 2; }
psql_q() { docker exec -i "$SQL_CONTAINER" psql -U "$PGU" -d "$DB" -v ON_ERROR_STOP=1 -X -At "$@"; }

log "start" 0 "container=$SQL_CONTAINER db=$DB cutoff=$CUTOFF drop=$DROP out=$OUT"
psql_q -c "SELECT '$CUTOFF'::timestamptz" >/dev/null || refuse 0 "cutoff '$CUTOFF' is not a timestamptz or the database is unreachable"

present=()
for t in "${TABLES[@]}"; do
  [[ "$(psql_q -c "SELECT to_regclass('public.$t') IS NOT NULL")" == "t" ]] && present+=("$t")
done
if [[ ${#present[@]} -eq 0 ]]; then
  log "already_dropped" 100 "none of ${TABLES[*]} exist"
  exit 0
fi
[[ ${#present[@]} -eq ${#TABLES[@]} ]] || refuse 5 "partial state: only ${present[*]} exist; inspect by hand"

# Newest write per table, and anything still live. One SQL text, re-run under locks at drop time.
LATEST_SQL="SELECT greatest(
  (SELECT max(greatest(granted_at, heartbeat_at)) FROM durable_gateway_permits),
  (SELECT max(greatest(granted_at, heartbeat_at)) FROM durable_resource_leases),
  (SELECT max(created_at) FROM durable_resource_demands),
  (SELECT max(greatest(requested_at, resident_at, idle_since, last_restored_at)) FROM durable_elastic_slot))"
LIVE_SQL="SELECT (SELECT count(*) FROM durable_gateway_permits WHERE status='active')
  + (SELECT count(*) FROM durable_resource_leases WHERE status='active')
  + (SELECT count(*) FROM durable_resource_demands WHERE status='pending')"
COUNTS_SQL="SELECT (SELECT count(*) FROM durable_gateway_permits)||','||(SELECT count(*) FROM durable_resource_leases)
  ||','||(SELECT count(*) FROM durable_resource_demands)||','||(SELECT count(*) FROM durable_elastic_slot)"

latest="$(psql_q -c "$LATEST_SQL")"
after="$(psql_q -c "SELECT coalesce(($LATEST_SQL) > '$CUTOFF'::timestamptz, false)")"
live="$(psql_q -c "$LIVE_SQL")"
log "cutoff_check" 10 "newest_write=${latest:-none} after_cutoff=$after live_rows=$live"
[[ "$after" == "f" ]] || refuse 10 "a legacy row was written at $latest, after the cutoff $CUTOFF"
[[ "$live" == "0" ]] || refuse 10 "$live legacy permit/lease/demand rows are still active/pending"

counts="$(psql_q -c "$COUNTS_SQL")"
IFS=',' read -r -a before <<<"$counts"
log "counts" 20 "$(for i in "${!TABLES[@]}"; do printf '%s=%s ' "${TABLES[$i]}" "${before[$i]}"; done)"

dump_args=()
for t in "${TABLES[@]}"; do dump_args+=(-t "public.$t"); done
dump_args+=(-t "public.$SEQUENCE")
log "dump" 30 "pg_dump ${TABLES[*]} $SEQUENCE -> $DUMP"
docker exec "$SQL_CONTAINER" pg_dump -U "$PGU" -d "$DB" --no-owner --no-privileges "${dump_args[@]}" | gzip -c >"$DUMP.partial"
mv "$DUMP.partial" "$DUMP"
log "dump_written" 60 "bytes=$(stat -c %s "$DUMP")"

# Verify: every table's COPY block in the dump has exactly the live row count.
for i in "${!TABLES[@]}"; do
  t="${TABLES[$i]}"
  dumped="$(gzip -dc "$DUMP" | awk -v t="COPY public.$t " 'index($0, t) == 1 {on=1; next} on && $0 == "\\." {on=0} on {n++} END {print n+0}')"
  echo "$t,${before[$i]},$dumped" >>"$OUT/dump_verify.csv.partial"
  [[ "$dumped" == "${before[$i]}" ]] || refuse 70 "$t: dump has $dumped rows, table has ${before[$i]}"
done
mv "$OUT/dump_verify.csv.partial" "$OUT/dump_verify.csv"
log "dump_verified" 70 "COPY row counts match the live counts (dump_verify.csv: table,live,dumped)"

if [[ "$DROP" -ne 1 ]]; then
  log "snapshot_only" 100 "--drop not given: nothing dropped"
  {
    echo "# GPU pool stage 5.6 legacy-table snapshot"
    echo; echo "- verdict: SNAPSHOT_ONLY (no --drop)"; echo "- cutoff: $CUTOFF; newest legacy write: ${latest:-none}"
    echo "- dump: $DUMP ($(stat -c %s "$DUMP") bytes); verified row counts: dump_verify.csv"
  } >"$OUT/report.md"
  exit 0
fi

# Drop: one transaction. Lock every table, re-check nothing moved since the dump, then apply the
# checked-in migration. lock_timeout so a backup holding a lock fails this instead of hanging it.
log "drop" 80 "locking, re-checking, applying $(basename "$MIGRATION")"
{
  echo "\\set ON_ERROR_STOP on"
  echo "BEGIN;"
  echo "SET LOCAL lock_timeout = '30s';"
  echo "LOCK TABLE durable_gateway_permits, durable_resource_leases, durable_resource_demands, durable_elastic_slot IN ACCESS EXCLUSIVE MODE;"
  echo "DO \$\$ BEGIN"
  echo "  IF ($COUNTS_SQL) <> '$counts' THEN RAISE EXCEPTION 'legacy row counts moved since the dump'; END IF;"
  echo "  IF coalesce(($LATEST_SQL) > '$CUTOFF'::timestamptz, false) THEN RAISE EXCEPTION 'a legacy row was written after the cutoff'; END IF;"
  echo "END \$\$;"
  cat "$MIGRATION"
  echo "COMMIT;"
} | psql_q -f - >>"$LOG" 2>&1 || refuse 80 "drop transaction failed and rolled back (see $LOG)"

gone="$(psql_q -c "SELECT count(*) FROM unnest(ARRAY['${TABLES[0]}','${TABLES[1]}','${TABLES[2]}','${TABLES[3]}','$SEQUENCE']) n WHERE to_regclass('public.'||n) IS NOT NULL")"
kept="$(psql_q -c "SELECT (to_regclass('public.durable_admission_runs') IS NOT NULL AND to_regclass('public.durable_resource_events') IS NOT NULL)")"
[[ "$gone" == "0" && "$kept" == "t" ]] || { log "ANOMALY" 90 "remaining_legacy=$gone registry_kept=$kept"; exit 3; }

{
  echo "table,rows_before,rows_after"
  for i in "${!TABLES[@]}"; do echo "${TABLES[$i]},${before[$i]},dropped"; done
} >"$OUT/before_after.csv"
{
  echo "# GPU pool stage 5.6 legacy-table drop"
  echo; echo "- verdict: DROPPED"; echo "- cutoff: $CUTOFF; newest legacy write: ${latest:-none}"
  echo "- dump (restore with \`gzip -dc $DUMP | psql\`): $DUMP ($(stat -c %s "$DUMP") bytes)"
  echo "- rows: before_after.csv; dump verification: dump_verify.csv"
  echo "- durable_admission_runs / durable_resource_events: present"
  echo "- errors: 0; needs another pass: no"
} >"$OUT/report.md"
log "done" 100 "dropped ${TABLES[*]} $SEQUENCE; report $OUT/report.md"
