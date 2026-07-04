#!/usr/bin/env bash
set -u
set -o pipefail

# ==============================================================================
# Orion Mesh - Bring up all service docker-compose stacks (bus first, sliding pool)
#
# Parallel sliding-window variant of up_all_services.sh: orion-bus runs alone first,
# then remaining services start with at most BATCH_SIZE in flight at once (as one
# finishes, the next starts — no waiting for a full batch to drain).
#
# IMPORTANT:
# - This script intentionally does NOT set a docker compose project name (-p).
#   Your services already manage stable naming via .env + container_name/image tags.
#   Forcing -p causes orphan warnings + container_name conflicts (double-prefixes).
#
# Usage:
#   ./mesh-utilities/common/up_all_services_batched.sh
#
# Optional env vars:
#   WAIT_SEC=30
#   BATCH_SIZE=5                        # max concurrent bring-ups after bus (sliding)
#   BUS_WAIT_SEC=120                    # max seconds to wait for bus redis readiness
#   PROJECT_NAME=orion                  # informational only (NOT used for -p)
#   EXCLUDE_SERVICES="svc1 svc2"        # replaces defaults/file excludes
#   EXCLUDE_SERVICES_ADD="svc3 svc4"    # adds to excludes
#   EXCLUDE_FILE="path/to/exclude.txt"  # defaults to mesh-utilities/common/exclude_services.txt
#
# permissions ephemeral:
#   chmod +x mesh-utilities/common/up_all_services_batched.sh
# permissions git persisted:
#   git add --chmod=+x mesh-utilities/common/up_all_services_batched.sh
# ==============================================================================

WAIT_SEC="${WAIT_SEC:-30}"
BATCH_SIZE="${BATCH_SIZE:-5}"
BUS_WAIT_SEC="${BUS_WAIT_SEC:-120}"
PROJECT_NAME="${PROJECT_NAME:-${COMPOSE_PROJECT_NAME:-${PROJECT:-orion}}}"  # informational only

BUS_SERVICE_DIR="orion-bus"

SERVICES_DIR="services"
COMPOSE_BASENAME="docker-compose.yml"
ROOT_ENV=".env"

# --- Default hardcoded excludes (edit these) ---
DEFAULT_EXCLUDES=(
)

# --- Optional exclude file (one service name per line; supports # comments) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXCLUDE_FILE="$SCRIPT_DIR/exclude_services.txt"
EXCLUDE_FILE="${EXCLUDE_FILE:-$DEFAULT_EXCLUDE_FILE}"

# --- Optional runtime overrides ---
EXCLUDE_SERVICES="${EXCLUDE_SERVICES:-}"         # replace excludes entirely
EXCLUDE_SERVICES_ADD="${EXCLUDE_SERVICES_ADD:-}" # append excludes

# --- repo root detection (script is expected at root/mesh-utilities/common) ---
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ ! -d "$REPO_ROOT/$SERVICES_DIR" ]]; then
  if command -v git >/dev/null 2>&1; then
    GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${GIT_ROOT:-}" && -d "$GIT_ROOT/$SERVICES_DIR" ]]; then
      REPO_ROOT="$GIT_ROOT"
    fi
  fi
fi

cd "$REPO_ROOT" || exit 2

# cortex-exec multi-lane helpers (explicit build + fleet verification)
# shellcheck source=cortex_exec_fleet_helpers.sh
source "$SCRIPT_DIR/cortex_exec_fleet_helpers.sh"

if [[ ! -f "$ROOT_ENV" ]]; then
  echo "ERROR: missing $REPO_ROOT/$ROOT_ENV (run from repo root or ensure it exists)." >&2
  exit 2
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer (got: $BATCH_SIZE)" >&2
  exit 2
fi

if ! [[ "$BUS_WAIT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BUS_WAIT_SEC must be a positive integer (got: $BUS_WAIT_SEC)" >&2
  exit 2
fi

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
  echo "ERROR: Bash 4.3+ required (wait -n). Current: ${BASH_VERSION}" >&2
  exit 2
fi

# ------------------------------------------------------------------------------
# Build effective excludes
# ------------------------------------------------------------------------------
EFFECTIVE_EXCLUDES=()
EFFECTIVE_EXCLUDES+=("${DEFAULT_EXCLUDES[@]}")

if [[ -f "$EXCLUDE_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"             # strip comments
    line="$(echo "$line" | xargs)" # trim
    [[ -z "$line" ]] && continue
    EFFECTIVE_EXCLUDES+=("$line")
  done < "$EXCLUDE_FILE"
fi

# If EXCLUDE_SERVICES is set, it replaces everything
if [[ -n "${EXCLUDE_SERVICES// }" ]]; then
  EFFECTIVE_EXCLUDES=()
  for x in $EXCLUDE_SERVICES; do
    EFFECTIVE_EXCLUDES+=("$x")
  done
fi

# Append more
if [[ -n "${EXCLUDE_SERVICES_ADD// }" ]]; then
  for x in $EXCLUDE_SERVICES_ADD; do
    EFFECTIVE_EXCLUDES+=("$x")
  done
fi

is_excluded() {
  local svc="$1"
  for x in "${EFFECTIVE_EXCLUDES[@]}"; do
    [[ "$svc" == "$x" ]] && return 0
  done
  return 1
}

compose_cmd() {
  local svc="$1"
  local compose_file="$SERVICES_DIR/$svc/$COMPOSE_BASENAME"
  local svc_env="$SERVICES_DIR/$svc/.env"

  if [[ ! -f "$compose_file" ]]; then
    echo "ERROR: missing $compose_file" >&2
    return 2
  fi
  if [[ ! -f "$svc_env" ]]; then
    echo "ERROR: missing $svc_env" >&2
    return 2
  fi

  # IMPORTANT: no "-p ..." here. Must match your manual invocation pattern.
  echo "docker compose --env-file $ROOT_ENV --env-file $svc_env -f $compose_file"
}

up_one() {
  local svc="$1"
  local cmd
  cmd="$(compose_cmd "$svc")" || return 2

  if [[ "$svc" == "$CORTEX_EXEC_SERVICE_DIR" ]]; then
    set +e
    up_cortex_exec_fleet "$cmd" "$REPO_ROOT"
    local rc=$?
    set -e
    return $rc
  fi

  echo ""
  echo "=== [$svc] build + up ==="
  # Don't let one failure abort the whole run; collect failures.
  set +e
  # shellcheck disable=SC2086
  $cmd up -d --build
  local rc=$?
  set -e
  return $rc
}

up_one_bg() {
  local svc="$1"
  local fail_dir="$2"
  local log_file="$fail_dir/$svc.up.log"
  if ! up_one "$svc" > "$log_file" 2>&1; then
    echo "$svc" > "$fail_dir/$svc.fail"
  fi
  sed "s/^/[$svc] /" "$log_file"
}

collect_up_failures() {
  local fail_dir="$1"
  local f
  shopt -s nullglob
  for f in "$fail_dir"/*.fail; do
    FAILED_UP+=("$(basename "$f" .fail)")
  done
  shopt -u nullglob
}

bus_redis_port() {
  local bus_env="$SERVICES_DIR/$BUS_SERVICE_DIR/.env"
  local port="6379"
  if [[ -f "$bus_env" ]]; then
    local val
    val="$(grep -E '^REDIS_PORT=' "$bus_env" | tail -1 | cut -d= -f2- | tr -d "\"'" | xargs)"
    [[ -n "$val" ]] && port="$val"
  fi
  echo "$port"
}

wait_for_bus_ready() {
  local port max_wait interval elapsed
  port="$(bus_redis_port)"
  max_wait="$BUS_WAIT_SEC"
  interval=2
  elapsed=0

  echo ""
  echo "=== Waiting for bus readiness (redis PONG on localhost:$port, max ${max_wait}s) ==="

  while [[ "$elapsed" -lt "$max_wait" ]]; do
    if command -v redis-cli >/dev/null 2>&1; then
      if redis-cli -h localhost -p "$port" ping 2>/dev/null | grep -q PONG; then
        echo "✅ Bus ready (redis-cli PONG on port $port)"
        return 0
      fi
    fi

    local cmd core_id health
    cmd="$(compose_cmd "$BUS_SERVICE_DIR")" || return 1
    set +e
    # shellcheck disable=SC2086
    core_id=$($cmd ps -q bus-core 2>/dev/null | head -1)
    set -e
    if [[ -n "${core_id//[$'\n\r\t ']}" ]]; then
      health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$core_id" 2>/dev/null || echo "unknown")"
      if [[ "$health" == "healthy" || "$health" == "running" ]]; then
        echo "✅ Bus ready (bus-core container: $health)"
        return 0
      fi
    fi

    sleep "$interval"
    elapsed=$((elapsed + interval))
    echo "  ... still waiting (${elapsed}s / ${max_wait}s)"
  done

  echo "ERROR: bus not ready within ${max_wait}s" >&2
  return 1
}

ps_ids() {
  local svc="$1"
  local cmd
  cmd="$(compose_cmd "$svc")" || return 0
  set +e
  # shellcheck disable=SC2086
  $cmd ps -q 2>/dev/null
  set -e
}

print_failed_logs() {
  local cid="$1"
  echo "---- logs (tail 80) for container $cid ----"
  docker logs --tail 80 "$cid" 2>&1 || true
}

FAIL_TMP_DIR=""
cleanup_fail_tmp() {
  [[ -n "${FAIL_TMP_DIR:-}" && -d "$FAIL_TMP_DIR" ]] && rm -rf "$FAIL_TMP_DIR"
}
trap cleanup_fail_tmp EXIT INT TERM

# ------------------------------------------------------------------------------
# Discover candidates
# ------------------------------------------------------------------------------
set -e

mapfile -t CANDIDATES < <(
  for d in "$SERVICES_DIR"/*; do
    [[ -d "$d" ]] || continue
    svc="$(basename "$d")"
    [[ -f "$SERVICES_DIR/$svc/$COMPOSE_BASENAME" ]] && echo "$svc"
  done | sort
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "ERROR: No services found under $SERVICES_DIR/*/$COMPOSE_BASENAME" >&2
  exit 2
fi

echo "=== Repo root: $REPO_ROOT ==="
echo "=== Project name (informational): $PROJECT_NAME ==="
echo "=== Max concurrency (sliding, after bus): $BATCH_SIZE ==="
echo "=== Discovered services (have $COMPOSE_BASENAME) ==="
printf ' - %s\n' "${CANDIDATES[@]}"

echo ""
echo "=== Excluded services (effective) ==="
if [[ "${#EFFECTIVE_EXCLUDES[@]}" -gt 0 ]]; then
  printf ' - %s\n' "${EFFECTIVE_EXCLUDES[@]}"
else
  echo " - <none>"
fi

# Filter services we will run (must have .env and not excluded)
RUN_LIST=()
SKIPPED_NOENV=()
SKIPPED_EXCLUDED=()

for svc in "${CANDIDATES[@]}"; do
  if is_excluded "$svc"; then
    SKIPPED_EXCLUDED+=("$svc")
    continue
  fi
  if [[ ! -f "$SERVICES_DIR/$svc/.env" ]]; then
    SKIPPED_NOENV+=("$svc")
    continue
  fi
  RUN_LIST+=("$svc")
done

echo ""
echo "=== Will run (after filtering) ==="
printf ' - %s\n' "${RUN_LIST[@]}"

if [[ "${#SKIPPED_EXCLUDED[@]}" -gt 0 ]]; then
  echo ""
  echo "=== Skipped (excluded) ==="
  printf ' - %s\n' "${SKIPPED_EXCLUDED[@]}"
fi

if [[ "${#SKIPPED_NOENV[@]}" -gt 0 ]]; then
  echo ""
  echo "=== Skipped (missing services/<svc>/.env) ==="
  printf ' - %s\n' "${SKIPPED_NOENV[@]}"
fi

# ------------------------------------------------------------------------------
# Bring up: bus first (alone), then remaining services with sliding concurrency
# ------------------------------------------------------------------------------
FAILED_UP=()
BUS_READY=false

if [[ " ${RUN_LIST[*]} " =~ " ${BUS_SERVICE_DIR} " ]]; then
  echo ""
  echo "=== Critical path: $BUS_SERVICE_DIR (sequential, before sliding pool) ==="
  if ! up_one "$BUS_SERVICE_DIR"; then
    FAILED_UP+=("$BUS_SERVICE_DIR")
  elif wait_for_bus_ready; then
    BUS_READY=true
  else
    FAILED_UP+=("$BUS_SERVICE_DIR")
  fi
else
  echo ""
  echo "WARNING: $BUS_SERVICE_DIR not in run list (missing compose/.env or excluded)."
  BUS_READY=true
fi

REST_LIST=()
for svc in "${RUN_LIST[@]}"; do
  [[ "$svc" == "$BUS_SERVICE_DIR" ]] && continue
  REST_LIST+=("$svc")
done

if [[ "$BUS_READY" != true ]]; then
  echo ""
  echo "ERROR: critical path $BUS_SERVICE_DIR failed; skipping sliding pool" >&2
elif [[ "${#REST_LIST[@]}" -gt 0 ]]; then
  echo ""
  echo "=== Sliding pool: max $BATCH_SIZE in flight (${#REST_LIST[@]} services after bus) ==="

  FAIL_TMP_DIR="$(mktemp -d)"
  next_idx=0
  in_flight=0
  rest_total="${#REST_LIST[@]}"

  while [[ "$next_idx" -lt "$rest_total" ]] || [[ "$in_flight" -gt 0 ]]; do
    while [[ "$in_flight" -lt "$BATCH_SIZE" ]] && [[ "$next_idx" -lt "$rest_total" ]]; do
      svc="${REST_LIST[$next_idx]}"
      next_idx=$((next_idx + 1))
      up_one_bg "$svc" "$FAIL_TMP_DIR" &
      in_flight=$((in_flight + 1))
      echo "  -> started [$svc] ($in_flight/$BATCH_SIZE in flight, $next_idx/$rest_total started)"
    done

    [[ "$in_flight" -eq 0 ]] && break

    set +e
    wait -n
    wait_rc=$?
    set -e
    if [[ "$wait_rc" -eq 127 ]]; then
      echo "WARNING: wait -n found no child (in_flight=$in_flight); assuming pool drained" >&2
      in_flight=0
    else
      in_flight=$((in_flight - 1))
    fi
  done

  collect_up_failures "$FAIL_TMP_DIR"
  rm -rf "$FAIL_TMP_DIR"
  FAIL_TMP_DIR=""
fi

if [[ "$BUS_READY" == true ]]; then
  echo ""
  echo "=== Waiting $WAIT_SEC seconds for containers to settle... ==="
  sleep "$WAIT_SEC"
else
  echo ""
  echo "=== Skipping settle wait (bus not ready) ==="
fi

# ------------------------------------------------------------------------------
# Status report
# ------------------------------------------------------------------------------
echo ""
echo "=== STATUS REPORT (informational project: $PROJECT_NAME) ==="
printf "%-35s  %-12s  %s\n" "service" "running/total" "details"
printf "%-35s  %-12s  %s\n" "-------" "------------" "-------"

FAILED_SERVICES=()

for svc in "${RUN_LIST[@]}"; do
  ids="$(ps_ids "$svc" || true)"

  if [[ -z "${ids//[$'\n\r\t ']}" ]]; then
    printf "%-35s  %-12s  %s\n" "$svc" "0/0" "NO CONTAINERS (compose may have failed)"
    FAILED_SERVICES+=("$svc")
    continue
  fi

  total=0
  running=0
  details=()

  while IFS= read -r cid; do
    [[ -z "$cid" ]] && continue
    total=$((total+1))
    status="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || echo "unknown")"
    exitcode="$(docker inspect -f '{{.State.ExitCode}}' "$cid" 2>/dev/null || echo "?")"
    if [[ "$status" == "running" ]]; then
      running=$((running+1))
    else
      details+=("$cid:$status(exit=$exitcode)")
    fi
  done <<< "$ids"

  if [[ "$running" -eq "$total" ]]; then
    printf "%-35s  %-12s  %s\n" "$svc" "$running/$total" "OK"
  else
    printf "%-35s  %-12s  %s\n" "$svc" "$running/$total" "FAIL"
    FAILED_SERVICES+=("$svc")

    # Tail logs for non-running containers
    while IFS= read -r cid; do
      [[ -z "$cid" ]] && continue
      status="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || echo "unknown")"
      if [[ "$status" != "running" ]]; then
        print_failed_logs "$cid"
      fi
    done <<< "$ids"
  fi
done

echo ""
if [[ "${#FAILED_UP[@]}" -gt 0 ]]; then
  echo "⚠️  Bring-up failures (compose or readiness):"
  printf ' - %s\n' "${FAILED_UP[@]}"
  echo ""
fi

if [[ "$BUS_READY" != true ]]; then
  echo "❌ Critical path $BUS_SERVICE_DIR not ready; sliding pool was skipped." >&2
  exit 1
fi

if [[ "${#FAILED_SERVICES[@]}" -eq 0 ]]; then
  if [[ "${#FAILED_UP[@]}" -gt 0 ]]; then
    exit 1
  fi
  echo "✅ All services are running."
else
  echo "❌ Services with issues:"
  printf ' - %s\n' "${FAILED_SERVICES[@]}"
  exit 1
fi
